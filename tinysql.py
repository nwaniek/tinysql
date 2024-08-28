#!/usr/bin/env python

from enum import Flag, auto
from pathlib import Path
from types import SimpleNamespace
from typing import NamedTuple, Tuple, Callable, List, Dict, Any, get_type_hints, Type
from collections import defaultdict
import io
import sqlite3
import pickle
import numpy as np
from uuid import uuid4


__version__ = '0.2.8'


TABLE_REGISTRY = {}


class TypeFlags(Flag):
    NONE    = auto()
    BLOB    = auto()
    AUTOINC = auto()


class TypeMap(NamedTuple):
    sql_type: str
    flags   : TypeFlags


class autoinc(int):
    def __new__(cls, value=None):
        if value is None:
            obj = super().__new__(cls, 0)
        elif isinstance(value, int):
            obj = super().__new__(cls, value)
        else:
            raise TypeError(f"autoinc must be an integer or None, got {type(value)}")
        return obj

    def __repr__(self):
        return f"{super().__repr__()}"


def gen_uuid():
    return uuid4().hex


class uuid(str):
    def __new__(cls, value=None):
        if value is None:
            obj = super().__new__(cls, gen_uuid())
        elif isinstance(value, str):
            obj = super().__new__(cls, value)
        else:
            raise TypeError(f"uuid must be a str or None, got {type(value)}")
        return obj

    def __repr__(self):
        return f"{super().__repr__()}"


TYPE_MAPPING = {
    # standard types
    str:        TypeMap('TEXT',    TypeFlags.NONE),
    int:        TypeMap('INTEGER', TypeFlags.NONE),
    float:      TypeMap('REAL',    TypeFlags.NONE),

    # BOOLEAN will effectively be mapped to NUMERIC due to type affinity,
    # because sqlite does not have a native BOOL type (see
    # https://www.sqlite.org/datatype3.html for more details)
    bool:       TypeMap('BOOLEAN', TypeFlags.NONE),

    # special and custom types that are considered blobs
    bytes:      TypeMap('BLOB',    TypeFlags.BLOB),
    bytearray:  TypeMap('BLOB',    TypeFlags.BLOB),
    memoryview: TypeMap('BLOB',    TypeFlags.BLOB),
    np.ndarray: TypeMap('ndarray', TypeFlags.BLOB),

    # special types supported by tinysql and mapped to appropriate sqlite
    # representations
    autoinc:    TypeMap('INTEGER', TypeFlags.AUTOINC),
    uuid:       TypeMap('TEXT',    TypeFlags.NONE),
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


class Not(Condition):
    def __init__(self, condition: Condition):
        self.condition = condition

    def build(self) -> Tuple[str, List[Any]]:
        clause, params = self.condition.build()
        return " NOT (" + clause + ")", params


class DatabaseContext:
    def __init__(self, db_path: Path | str, table_storage_root: Path | str | None, use_global_registry: bool = True):
        # sanitize paths
        db_path = Path(db_path) if isinstance(db_path, str) else db_path
        table_storage_root = Path(table_storage_root) if isinstance(table_storage_root, str) else table_storage_root

        self.use_global_registry  = use_global_registry
        self.registry             = TABLE_REGISTRY if use_global_registry else {}
        self.db_path              = db_path
        self.table_storage_root   = table_storage_root
        self.use_external_storage = table_storage_root is not None
        if not self.use_external_storage:
            sqlite3.register_adapter(np.ndarray, adapt_array)
            sqlite3.register_converter("ndarray", convert_array)
        self.con                  = sqlite3.connect(db_path, detect_types=sqlite3.PARSE_DECLTYPES)

        self.insert_fn            = insert
        self.select_fn            = select
        self.insertmany_fn        = insertmany
        self.tables_initialized   = False

    def init_tables(self):
        if self.tables_initialized:
            return
        cur = self.con.cursor()
        for tabledef in self.registry.values():
            tspec   = tabledef.tspec
            init_fn = tabledef.init_fn
            if table_exists(self.con, tspec.name):
                continue

            sql = sql_builder_create_table(tspec)
            cur.execute(sql)
            self.con.commit()
            if init_fn is None:
                continue
            init_fn(self.con)
        self.tables_initialized = True

    def close(self):
        self.con.close()

    def insert(self, *args, **kwargs):
        self.insert_fn(self, *args, **kwargs)

    def insertmany(self, *args, **kwargs):
        self.insertmany_fn(self, *args, **kwargs)

    def select(self, cls: Type, *params, **kwargs):
        yield from self.select_fn(self, cls, *params, **kwargs)

    def __enter__(self):
        self.init_tables()
        return self

    def __exit__(self, exc_type, exc_value, exc_traceback):
        self.close()


def sql_builder_create_table(tspec: TableSpec) -> str:
    primary_keys = tspec.primary_keys or []
    # sqlite allows only one autoincrement field, and this is effectively also
    # the primary key. enforce this behavior by checking that there's only one
    # pk that is also autoinc (if there's autoinc)
    autoinc_field = None
    has_autoinc = False
    for fname, tmap in tspec.fields.__dict__.items():
        has_autoinc = TypeFlags.AUTOINC in tmap.flags
        if has_autoinc:
            autoinc_field = fname
            break
    if has_autoinc:
        if autoinc_field not in tspec.primary_keys:
            raise ValueError(f"Field {autoinc_field} is declared AUTOINCREMENT, but not a PRIMARY KEY. This is not supported by sqlite.")
        if len(tspec.primary_keys) > 1:
            raise ValueError(f"AUTOINCREMENT not supported on composite PRIMARY KEYs in sqlite. Affected field: {autoinc_field}.")

    sql = f"CREATE TABLE {tspec.name} ("
    sql += ", ".join(f"{fname} {tmap.sql_type}{' PRIMARY KEY' if fname in primary_keys and len(primary_keys) <= 1 else ''}{' AUTOINCREMENT' if TypeFlags.AUTOINC in tmap.flags else ''}" for fname, tmap in tspec.fields.__dict__.items())
    if tspec.foreign_keys is not None and len(tspec.foreign_keys) > 0:
        sql += ", " + ", ".join("FOREIGN KEY ({}) REFERENCES {}".format(fk[0], fk[1]) for fk in tspec.foreign_keys)
    if primary_keys is not None and len(primary_keys) > 1:
        sql += ", PRIMARY KEY (" + ", ".join(primary_keys) + ")"
    sql += ");"
    return sql


def sql_builder_insert(tspec: TableSpec, replace : bool = False) -> str:
    modifier = "or REPLACE" if replace else "" # "or IGNORE"

    # filter out autoincrement fields (they will be set by sqlite)
    fields = []
    for fname, tmap in tspec.fields.__dict__.items():
        if TypeFlags.AUTOINC in tmap.flags:
            continue
        fields.append(fname)

    sql = f"INSERT {modifier} INTO {tspec.name}"
    sql += "("
    sql += ", ".join(f for f in fields)
    sql += ") VALUES ("
    sql += ", ".join("?" for _ in range(len(fields)))
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


def get_tspec(cls: Type | str, context: DatabaseContext | None = None) -> TableSpec:
    registry = context.registry if context else TABLE_REGISTRY

    if isinstance(cls, str):
        tentry = registry.get(cls, None)
        if tentry is not None:
            return tentry.tspec
        raise RuntimeError(f"Type not mapped to database: {cls}.")

    elif hasattr(cls, '_tinysql_tspec'):
        return cls._tinysql_tspec

    else:
        tentry = registry.get(cls.__name__, None)
        if tentry is not None:
            return tentry.tspec
        raise RuntimeError(f"Type not mapped to database: {cls}.")


def get_tablename(cls: Type | str, context: DatabaseContext | None = None) -> str:
    tspec = get_tspec(cls, context)
    return tspec.name


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


def prepare_data_tuple(context: DatabaseContext, data, tspec: TableSpec, get_fn: Callable):
    data_tuple = tuple()
    for fname, ftype in tspec.fields.__dict__.items():
        if TypeFlags.AUTOINC in ftype:
            continue
        if TypeFlags.BLOB in ftype.flags and context.use_external_storage:
            fpath = dump(context, tspec, fname, data, get_fn(data, fname), get_fn)
            data_tuple = data_tuple + (fpath, )
        else:
            data_tuple = data_tuple + (get_fn(data, fname), )
    return data_tuple


def insert_impl(context: DatabaseContext, data, sql: str, tspec: TableSpec, get_fn: Callable):
    data_tuple = prepare_data_tuple(context, data, tspec, get_fn)
    cur = context.con.cursor()
    cur.execute(sql, data_tuple)
    context.con.commit()


def insert_from_class(context: DatabaseContext, data: Type, replace=True):
    insert_sql = data._tinysql_insert if not replace else data._tinysql_insert_replace
    insert_impl(context, data, insert_sql, data._tinysql_tspec, lambda d, k: getattr(d, k))


def insert_from_dict(context: DatabaseContext, data: Dict, tspec: TableSpec, replace=True):
    insert_sql = sql_builder_insert(tspec, replace)
    insert_impl(context, data, insert_sql, tspec, lambda d, k: d[k])


def insert(context: DatabaseContext, data, tspec: TableSpec | None = None, replace = True):
    if hasattr(data, '_tinysql_tspec'):
        insert_from_class(context, data, replace)

    elif isinstance(data, dict):
        if tspec is None:
            raise RuntimeError(f"TableSpec must be provided for dictionary")
        insert_from_dict(context, data, tspec, replace)

    else:
        raise RuntimeError(f"Type not mapped to database: {type(data)}")


def group_by_type(data: List[object]):
    grouped_data: Dict[type, List[Any]] = defaultdict(list)
    for item in data:
        item_type = type(item)
        grouped_data[item_type].append(item)
    return list(grouped_data.values())


def insertmany_impl(context: DatabaseContext, data: list, sql: str, tspec: TableSpec, get_fn: Callable):
    data_tuples = [prepare_data_tuple(context, item, tspec, get_fn) for item in data]
    cur = context.con.cursor()
    cur.executemany(sql, data_tuples)
    context.con.commit()


def insertmany_from_class(context: DatabaseContext, data: list[Type], replace: bool = True, keep_order: bool = False):
    groups = group_by_type(data)
    all_equal_type = len(groups) == 1
    if keep_order and not all_equal_type:
        for item in data:
            insert_from_class(context, item, replace)
    else:
        for group in groups:
            tspec = group[0]._tinysql_tspec
            sql   = group[0]._tinysql_insert if not replace else group[0]._tinysql_insert_replace
            insertmany_impl(context, group, sql, tspec, lambda d, k: getattr(d, k))


def insertmany_from_dict(context: DatabaseContext, data: list[Dict], tspec: TableSpec, replace: bool = True):
    sql = sql_builder_insert(tspec, replace)
    insertmany_impl(context, data, sql, tspec, lambda d, k: d[k])


def insertmany(context: DatabaseContext, data: list, tspec: TableSpec | None = None, replace: bool = True, keep_order: bool = False):
    if not len(data):
        return

    if hasattr(data[0], '_tinysql_tspec'):
        insertmany_from_class(context, data, replace, keep_order)

    elif isinstance(data[0], dict):
        if tspec is None:
            raise RuntimeError(f"TableSpec must be provided for list of dictionaries")
        insertmany_from_dict(context, data, tspec, replace)

    else:
        raise RuntimeError(f"Type not mapped to database: {type(data)}")


def update(context: DatabaseContext, cls: Type, updates, condition: Condition | None = None):
    if not hasattr(cls, '_tinysql_tspec'):
        raise TypeError("Type not mapped to database: {cls}")

    if not len(updates):
        return

    tspec = cls._tinysql_tspec
    sql = f"UPDATE {tspec.name} SET "
    sql += ", ".join([f"{field} = {value}" for (field, value) in updates])
    params = []
    if condition is not None:
        where, params = condition.build()
        sql += f" WHERE {where}"

    cur = context.con.cursor()
    cur.execute(sql, params)
    context.con.commit()


def execute(context: DatabaseContext, sql: str, parameters, /):
    cur = context.con.cursor()
    cur.execute(sql, parameters)
    context.con.commit()


def executemany(context: DatabaseContext, sql: str, parameters, /):
    cur = context.con.cursor()
    cur.executemany(sql, parameters)
    context.con.commit()


def select_base_impl(context: DatabaseContext, cls: Type, sql: str, params = (), limit: int | None = None, offset: int | None = None):
    if limit is not None:
        sql += f" LIMIT {limit}"
        if offset is not None:
            sql += f" OFFSET {offset}"

    cur = context.con.cursor()
    for row in cur.execute(sql, params):
        yield cls(*row)


def select_from_condition(context: DatabaseContext, cls: Type, condition: Condition | None = None, **kwargs):
    if not hasattr(cls, '_tinysql_select'):
        raise TypeError("Type not mapped to database: {cls}")

    sql = cls._tinysql_select
    params = []
    if condition:
        where, params = condition.build()
        sql += f" WHERE {where}"

    yield from select_base_impl(context, cls, sql, params, **kwargs)


def select_from_sql(context: DatabaseContext, cls: Type, where: str, params, **kwargs):
    if not hasattr(cls, '_tinysql_select'):
        raise TypeError("Type not mapped to database: {cls}")

    sql = cls._tinysql_select
    sql += " " + where
    yield from select_base_impl(context, cls, sql, params, **kwargs)


def select(context: DatabaseContext, cls: Type, *args, **kwargs):
    if len(args) > 2:
        raise TypeError("select() takes at most 2 positional argument")

    if len(args) == 0 or args[0] == None:
        yield from select_from_condition(context, cls, None, **kwargs)

    elif isinstance(args[0], Condition):
        print(".......")
        yield from select_from_condition(context, cls, args[0], **kwargs)

    elif isinstance(args[0], str):
        yield from select_from_sql(context, cls, args[0], args[1] if len(args) > 1 else (), **kwargs)

    else:
        raise ValueError("Argument of type {type(args[0])} not supported in select()")


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
    # this will use the global table registry, and also init the tables
    context = DatabaseContext(db_path, table_storage_root, use_global_registry=True)
    context.init_tables()
    return context
