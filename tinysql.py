#!/usr/bin/env python

from pathlib import Path
from types import SimpleNamespace
from typing import NamedTuple, Tuple, Callable, List, Dict, Any, get_type_hints, Type
import io
import sqlite3
import pickle
import numpy as np


__version__ = '0.2.1'


TABLE_REGISTRY = {}


class TypeMap(NamedTuple):
    sql_type: str
    is_blob : bool


TYPE_MAPPING = {
    # standard types
    str:        TypeMap('TEXT',    False),
    int:        TypeMap('INTEGER', False),
    float:      TypeMap('REAL',    False),

    # BOOLEAN will effectively be mapped to NUMERIC due to type affinity,
    # because sqlite does not have a native BOOL type (see
    # https://www.sqlite.org/datatype3.html for more details)
    bool:       TypeMap('BOOLEAN', False),

    # special and custom types that are considered blobs
    bytes:      TypeMap('BLOB',    True),
    bytearray:  TypeMap('BLOB',    True),
    memoryview: TypeMap('BLOB',    True),
    np.ndarray: TypeMap('ndarray', True),
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


class Condition:
    def build(self) -> Tuple[str, List[Any]]:
        raise NotImplementedError


class Equals(Condition):
    def __init__(self, column: str, value: Any):
        self.column = column
        self.value = value

    def build(self) -> Tuple[str, List[Any]]:
        return f"{self.column} = ?", [self.value]


class NotEquals(Condition):
    def __init__(self, column: str, value: Any):
        self.column = column
        self.value = value

    def build(self) -> Tuple[str, List[Any]]:
        return f"{self.column} != ?", [self.value]


class GreaterThan(Condition):
    def __init__(self, column: str, value: Any):
        self.column = column
        self.value = value

    def build(self) -> Tuple[str, List[Any]]:
        return f"{self.column} > ?", [self.value]


class LessThan(Condition):
    def __init__(self, column: str, value: Any):
        self.column = column
        self.value = value

    def build(self) -> Tuple[str, List[Any]]:
        return f"{self.column} < ?", [self.value]


class Between(Condition):
    def __init__(self, column: str, lower: Any, upper: Any):
        self.column = column
        self.lower = lower
        self.upper = upper

    def build(self) -> Tuple[str, List[Any]]:
        return f"{self.column} BETWEEN ? AND ?", [self.lower, self.upper]


class Like(Condition):
    def __init__(self, column: str, pattern: str):
        self.column = column
        self.pattern = pattern

    def build(self) -> Tuple[str, List[Any]]:
        return f"{self.column} LIKE ?", [self.pattern]


class In(Condition):
    def __init__(self, column: str, values: List[Any]):
        self.column = column
        self.values = values

    def build(self) -> Tuple[str, List[Any]]:
        placeholders = ', '.join(['?'] * len(self.values))
        return f"{self.column} IN ({placeholders})", self.values


class And(Condition):
    def __init__(self, *conditions: Condition):
        self.conditions = conditions

    def build(self) -> Tuple[str, List[Any]]:
        clauses = []
        parameters = []
        for condition in self.conditions:
            clause, params = condition.build()
            clauses.append(f"({clause})")
            parameters.extend(params)
        return " AND ".join(clauses), parameters


class Or(Condition):
    def __init__(self, *conditions: Condition):
        self.conditions = conditions

    def build(self) -> Tuple[str, List[Any]]:
        clauses = []
        parameters = []
        for condition in self.conditions:
            clause, params = condition.build()
            clauses.append(f"({clause})")
            parameters.extend(params)
        return " OR ".join(clauses), parameters


class DatabaseContext:
    def __init__(self, db_path: Path, table_storage_root: Path | None):
        self.registry             = {}
        self.db_path              = db_path
        self.table_storage_root   = table_storage_root
        self.use_external_storage = table_storage_root is not None
        if not self.use_external_storage:
            sqlite3.register_adapter(np.ndarray, adapt_array)
            sqlite3.register_converter("ndarray", convert_array)
        self.con = sqlite3.connect(db_path, detect_types=sqlite3.PARSE_DECLTYPES)

        self.insert_fn = insert
        self.select_fn = select

    def close(self):
        self.con.close()

    def insert(self, data, tspec: TableSpec | None = None, replace_existing=True):
        self.insert_fn(self, data, tspec, replace_existing)

    def select(self, cls: Type, condition: Condition | None = None, limit: int | None = None, offset: int | None = None):
        self.select_fn(self, cls, condition, limit, offset)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, exc_traceback):
        self.close()


def sql_builder_create_table(tspec: TableSpec) -> str:
    sql = f"CREATE TABLE {tspec.name} ("
    sql += ", ".join(f"{fname} {tmap.sql_type}" for fname, tmap in tspec.fields.__dict__.items())
    if tspec.foreign_keys is not None and len(tspec.foreign_keys) > 0:
        sql += ", " + ", ".join("FOREIGN KEY ({}) REFERENCES {}".format(fk[0], fk[1]) for fk in tspec.foreign_keys)
    if tspec.primary_keys is not None and len(tspec.primary_keys) > 0:
        sql += ", PRIMARY KEY (" + ", ".join(tspec.primary_keys) + ")"
    sql += ");"
    return sql


def sql_builder_insert(tspec: TableSpec, replace_existing : bool = False) -> str:
    modifier = "or REPLACE" if replace_existing else "" # "or IGNORE"
    sql = f"INSERT {modifier} INTO {tspec.name}"
    sql += "("
    sql += ", ".join(key for key in tspec.fields.__dict__.keys())
    sql += ") VALUES ("
    sql += ", ".join("?" for _ in range(len(tspec.fields.__dict__)))
    sql += ");"
    return sql


def sql_builder_select(tspec: TableSpec) -> str:
    fieldstr = ', '.join(vars(tspec.fields))
    sql = f"SELECT {fieldstr} FROM {tspec.name}"
    return sql


def register_tspec(registry, cls, tablename: str, tspec: TableSpec, init_fn: Callable | None):
    cls._tinysql_tspec = tspec
    cls._tinysql_init_fn = init_fn
    cls._tinysql_insert = sql_builder_insert(tspec, False)
    cls._tinysql_insert_replace = sql_builder_insert(tspec, True)
    cls._tinysql_select = sql_builder_select(tspec)
    registry[tablename] = TableRegistryEntry(tspec, init_fn, cls)
    return cls


def db_table(tablename: str, primary_keys: List[str] | None = None, foreign_keys: List[Tuple[str, str]] | None = None, init_fn: Callable | None = None,  context: DatabaseContext | None = None):
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

        registry = context.registry if context else TABLE_REGISTRY
        if tablename in registry:
            raise ValueError(f"Duplicate table name detected: {tablename}")

        cls = register_tspec(registry, cls, tablename, ts, init_fn)
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


def db_enum(tablename: str, descriptions: Dict[str, str] = {}, context: DatabaseContext | None = None):
    def decorator(cls):
        # test type of members. cannot work with mixed type enums
        types = [type(f.value) for f in cls]
        if types.count(types[0]) != len(types):
            raise TypeError("Mixed-type enums are not supported.")

        # turn enum into a tablespec
        ts = TableSpec(tablename)
        ts.fields.value       = TYPE_MAPPING[types[0]]
        ts.fields.name        = TYPE_MAPPING[str]
        ts.fields.description = TYPE_MAPPING[str]
        ts.primary_keys       = ["value"]

        registry = context.registry if context else TABLE_REGISTRY
        if tablename in registry:
            raise ValueError(f"Duplicate table name detected: {tablename}")

        init_fn = db_enum_initfn(tablename, [(f.value, f._name_, descriptions.get(f._name_, "")) for f in cls])
        cls = register_tspec(registry, cls, tablename, ts, init_fn)
        return cls
    return decorator


def get_tspec(cls: Type | str) -> TableSpec:
    if isinstance(cls, str):
        tentry = TABLE_REGISTRY.get(cls, None)
        if tentry is not None:
            return tentry.tspec
        raise RuntimeError(f"Type not mapped to database: {cls}.")

    elif hasattr(cls, '_tinysql_tspec'):
        return cls._tinysql_tspec

    else:
        tentry = TABLE_REGISTRY.get(cls.__name__, None)
        if tentry is not None:
            return tentry.tspec
        raise RuntimeError(f"Type not mapped to database: {cls}.")


def dump(context, tspec: TableSpec, fieldname: str, data, obj, get_fn: Callable) -> str:
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


def insert_impl(context: DatabaseContext, data, sql: str, tspec: TableSpec, get_fn: Callable):
    _data = tuple()

    # build the data for the sqlite table
    for fname, ftype in tspec.fields.__dict__.items():
        if ftype.is_blob and context.use_external_storage:
            fpath = dump(context, tspec, fname, data, get_fn(data, fname), get_fn)
            _data = _data + (fpath, )
        else:
            _data = _data + (get_fn(data, fname), )

    # finally execute
    cur = context.con.cursor()
    cur.execute(sql, _data)
    context.con.commit()


def insert_from_class(context: DatabaseContext, data: Type, replace_existing=True):
    insert_sql = data._tinysql_insert if not replace_existing else data._tinysql_insert_replace
    insert_impl(context, data, insert_sql, data._tinysql_tspec, lambda d, k: getattr(d, k))


def insert_from_dict(context: DatabaseContext, data: Dict, tspec: TableSpec, replace_existing=True):
    insert_sql = sql_builder_insert(tspec, replace_existing)
    insert_impl(context, data, insert_sql, tspec, lambda d, k: d[k])


def insert(context: DatabaseContext, data, tspec: TableSpec | None = None, replace_existing=True):
    if hasattr(data, '_tinysql_tspec'):
        insert_from_class(context, data, replace_existing)

    elif isinstance(data, dict):
        if tspec is None:
            raise RuntimeError(f"TableSpec must be provided for dictionary")
        insert_from_dict(context, data, tspec, replace_existing)

    else:
        raise RuntimeError(f"Type not mapped to database: {type(data)}")


def select(context: DatabaseContext, cls: Type, condition: Condition | None = None, limit: int | None = None, offset: int | None = None):
    if not hasattr(cls, '_tinysql_select'):
        raise TypeError("Type not mapped to database: {cls}")

    sql = cls._tinysql_select
    params = []
    if condition:
        where, params = condition.build()
        sql += f" WHERE {where}"

    if limit is not None:
        sql += f" LIMIT {limit}"
        if offset is not None:
            sql += f" OFFSET {offset}"

    cur = context.con.cursor()
    for row in cur.execute(sql, params):
        yield cls(*row)


def table_exists(con, tablename: str) -> bool:
    query = f"SELECT name FROM sqlite_master WHERE type='table' AND name='{tablename}';";
    cur = con.cursor()
    cur.execute(query)
    row = cur.fetchone()
    return row is not None


def adapt_array(arr: np.ndarray):
    out = io.BytesIO()
    np.save(out, arr)
    out.seek(0)
    return sqlite3.Binary(out.read())


def convert_array(text) -> np.ndarray:
    out = io.BytesIO(text)
    out.seek(0)
    return np.load(out)


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

        sql = sql_builder_create_table(tspec)
        cur.execute(sql)
        context.con.commit()
        if init_fn is None:
            continue
        init_fn(context.con)

    return context

