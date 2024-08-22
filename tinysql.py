#!/usr/bin/env python

from pathlib import Path
from types import SimpleNamespace
from typing import NamedTuple, Tuple, Callable, List, Dict, Union, get_type_hints, Type
import io
import sqlite3
import pickle
import numpy as np


TABLE_REGISTRY = {}


TYPE_MAPPING = {
    # standard types
    str:        'TEXT',
    int:        'INTEGER',
    float:      'REAL',

    # BOOLEAN will effectively be mapped to NUMERIC due to type affinity,
    # because sqlite does not have a native BOOL type (see
    # https://www.sqlite.org/datatype3.html for more details)
    bool:       'BOOLEAN',

    # special and custom types
    np.ndarray: 'ndarray',
}


class TableSpec:
    def __init__(self, tablename: str):
        self.name         = tablename
        self.fields       = SimpleNamespace()
        self.primary_keys = []
        self.foreign_keys = []

    def __repr__(self):
        field_str = [f"{fname}: {getattr(self.fields, fname)}" for fname in vars(self.fields)]
        field_str = ", ".join(field_str)
        pk_str    = ", ".join(self.primary_keys)
        fk_str    = ", ".join(self.foreign_keys)
        return f"TableSpec(name={self.name}, fields=[{field_str}], primary_keys=[{pk_str}], foreign_keys=[{fk_str}])"


class TableRegistryEntry(NamedTuple):
    tspec   : TableSpec
    init_fn : Callable | None
    cls     : Type


def db_table(tablename: str, primary_keys: List[str] | None = None, foreign_keys: List[Tuple[str, str]] | None = None, init_fn: Callable | None = None):
    def decorator(cls):
        ts = TableSpec(tablename)
        annotations = get_type_hints(cls)
        for field_name, field_type in annotations.items():
            if field_type in TYPE_MAPPING:
                setattr(ts.fields, field_name, TYPE_MAPPING[field_type])
            else:
                raise ValueError(f"Unsupported field type: {field_type} for field {field_name}")

            ts.primary_keys = primary_keys or []
            ts.foreign_keys = foreign_keys or []

        if tablename in TABLE_REGISTRY:
            raise ValueError(f"Duplicate table name detected: {tablename}")

        cls._tinysql_tspec   = ts
        cls._tinysql_init_fn = init_fn
        TABLE_REGISTRY[tablename] = TableRegistryEntry(ts, init_fn, cls)
        return cls
    return decorator


class db_enum_initfn:
    def __init__(self, tablename, values):
        self.tablename = tablename
        self.values = values

    def __call__(self, con):
        cur = con.cursor()
        cur.executemany(f"INSERT INTO {self.tablename} (value, name, description) VALUES (?, ?, ?)", self.values)
        con.commit()


def db_enum(tablename: str, descriptions: Dict[str, str] = {}):
    def decorator(cls):
        # test type of members. cannot work with mixed type enums
        types = [type(f.value) for f in cls]
        if types.count(types[0]) != len(types):
            raise TypeError("Mixed-type enums are not supported.")

        # turn enum into a tablespec
        ts = TableSpec(tablename)
        ts.fields.value       = TYPE_MAPPING[types[0]]
        ts.fields.name        = "TEXT"
        ts.fields.description = "TEXT"
        ts.primary_keys       = ["value"]

        if tablename in TABLE_REGISTRY:
            raise ValueError(f"Duplicate table name detected: {tablename}")

        cls._tinysql_tspec   = ts
        cls._tinysql_init_fn = db_enum_initfn(tablename, [(f.value, f._name_, descriptions.get(f._name_, "")) for f in cls])
        TABLE_REGISTRY[tablename] = TableRegistryEntry(ts, cls._tinysql_init_fn, cls)
        return cls
    return decorator


def get_tspec(cls: Type):
    if hasattr(cls, '_tinysql_tspec'):
        return cls._tinysql_tspec
    else:
        tspec = TABLE_REGISTRY.get(cls.__name__, None)
        if tspec is not None:
            return tspec
        raise RuntimeError(f"Type not mapped to database: {cls}.")


def sql_builder_mk_table(tspec: TableSpec):
    sql = "CREATE TABLE {} (".format(tspec.name)
    sql += ", ".join("{} {}".format(k, v) for k, v in tspec.fields.__dict__.items())
    if tspec.foreign_keys is not None and len(tspec.foreign_keys) > 0:
        sql += ", " + ", ".join("FOREIGN KEY ({}) REFERENCES {}".format(fk[0], fk[1]) for fk in tspec.foreign_keys)
    if tspec.primary_keys is not None and len(tspec.primary_keys) > 0:
        sql += ", PRIMARY KEY (" + ", ".join(tspec.primary_keys) + ")"
    sql += ");"
    return sql


def sql_builder_insert(tspec: TableSpec, replace_existing = False):
    modifier = "or REPLACE" if replace_existing else "" # "or IGNORE"
    sql = f"INSERT {modifier} INTO {tspec.name}"
    sql += "("
    sql += ", ".join(key for key in tspec.fields.__dict__.keys())
    sql += ") VALUES ("
    sql += ", ".join("?" for _ in range(len(tspec.fields.__dict__)))
    sql += ");"
    return sql


def dump(context, tspec, fieldname, data, obj, get_fn):
    if obj is None:
        return ""

    # get root / tspec_dir
    root = context.table_storage_root
    tspec_dir = Path(tspec.name)
    (root / tspec_dir).mkdir(parents=True, exist_ok=True)

    fname = "_".join(str(get_fn(data, key)) for key in tspec.primary_keys)
    fname = "_".join([fname, fieldname])

    if type(obj) == np.ndarray:
        fname = Path(fname)
        np.savez(root / tspec_dir / fname, fieldname=obj)
        fname = fname.with_suffix(".npz")
    else:
        fname = Path(fname + ".pkl")
        with open(root / tspec_dir / fname, 'wb') as file:
            pickle.dump(obj, file)

    return str(tspec_dir / fname)


def insert_impl(context, data, tspec: TableSpec, get_fn: Callable):
    sql = sql_builder_insert(tspec)
    _data = tuple()

    # build the data for the sqlite table
    for fname, ftype in tspec.fields.__dict__.items():
        if context.use_external_storage and ftype.lower() in ['blob', 'ndarray']:
            fpath = dump(context, tspec, fname, data, get_fn(data, fname), get_fn)
            _data = _data + (fpath, )
        else:
            _data = _data + (get_fn(data, fname), )

    # finally execute
    cur = context.con.cursor()
    cur.execute(sql, _data)
    context.con.commit()


def insert(context, data, tspec: TableSpec | None = None):
    if hasattr(data, '_tinysql_tspec'):
        insert_impl(context, data, data._tinysql_tspec, lambda d, k: getattr(d, k))

    elif isinstance(data, dict):
        if tspec is not None:
            insert_impl(context, data, tspec, lambda d, k: d[k])
        else:
            raise RuntimeError(f"TableSpec missing")

    else:
        raise RuntimeError(f"Type not mapped to database: {type(data)}")


def table_exists(con, tablename):
    query = f"SELECT name FROM sqlite_master WHERE type='table' AND name='{tablename}';";
    cur = con.cursor()
    cur.execute(query)
    row = cur.fetchone()
    return row is not None


def adapt_array(arr):
    out = io.BytesIO()
    np.save(out, arr)
    out.seek(0)
    return sqlite3.Binary(out.read())


def convert_array(text):
    out = io.BytesIO(text)
    out.seek(0)
    return np.load(out)


class DatabaseContext:
    def __init__(self, db_path: Path, table_storage_root: Path | None):
        self.db_path              = db_path
        self.table_storage_root   = table_storage_root
        self.use_external_storage = table_storage_root is not None
        if not self.use_external_storage:
            sqlite3.register_adapter(np.ndarray, adapt_array)
            sqlite3.register_converter("ndarray", convert_array)
        self.con = sqlite3.connect(db_path, detect_types=sqlite3.PARSE_DECLTYPES)

    def close(self):
        self.con.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, exc_traceback):
        self.close()


def setup_db(db_path: Path | str, table_storage_root: Path | str | None):
    db_path = Path(db_path) if isinstance(db_path, str) else db_path
    table_storage_root = Path(table_storage_root) if isinstance(table_storage_root, str) else table_storage_root

    context = DatabaseContext(db_path, table_storage_root)
    cur = context.con.cursor()
    for tabledef in TABLE_REGISTRY.values():
        tspec   = tabledef.tspec
        init_fn = tabledef.init_fn
        if table_exists(context.con, tspec.name):
            continue

        sql = sql_builder_mk_table(tspec)
        cur.execute(sql)
        context.con.commit()
        if init_fn is None:
            continue
        init_fn(context.con)

    return context

