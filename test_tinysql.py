#!/usr/bin/env python

from enum import Enum
from typing import NamedTuple
import tinysql
import numpy as np
import uuid

context = tinysql.DatabaseContext('test.sqlite', 'test_storage')

def gen_uuid():
    return uuid.uuid4().hex

@tinysql.db_table("AmazingValues", primary_keys=["id"], context=context)
class AmazingValues(NamedTuple):
    id:     tinysql.autoinc
    value0: str
    value1: float
    value2: np.ndarray


@tinysql.db_enum("MyEnum", descriptions={'One': 'First field of MyEnum', 'Two': 'Second field of MyEnum'}, context=context)
class MyEnum(Enum):
    One = "one"
    Two = "two"


def test_insert(context):
    print("Table 'AmazingValues' exists:", tinysql.table_exists(context.con, "AmazingValues"))

    # insert via object
    values = AmazingValues(tinysql.autoinc(), "hello, world!", 123.12, np.ones((4, 4)))
    context.insert(values)

    # insert as a dict with corresponding tablespec
    values = {'id': gen_uuid(),
              'value0': 'world, hello!',
              'value1': 71.71,
              'value2': np.eye(10)}
    context.insert(values, tinysql.get_tspec(AmazingValues))

    # print everything in the table
    sql = "SELECT * FROM AmazingValues"
    rows = context.con.execute(sql)
    for row in rows:
        print(row)


def test_enum(context):
    print("Table 'MyEnum exists:", tinysql.table_exists(context.con, "MyEnum"))
    sql = "SELECT value, name, description FROM MyEnum"
    rows = context.con.execute(sql)
    for row in rows:
        print(f"MyEnum.{row[1]} = '{row[0]}' # {row[2]}")


def test_select(context):
    results = tinysql.select(context, AmazingValues, tinysql.GreaterThan('value1', 70.0))
    for obj in results:
        print(obj)



def test_autoinc():
    ts = tinysql.TableSpec("AutoIncTable")
    ts.fields.id0 = tinysql.TYPE_MAPPING[tinysql.autoinc]
    ts.fields.id1 = tinysql.TYPE_MAPPING[int]
    ts.primary_keys = ['id0']
    sql = tinysql.sql_builder_create_table(ts)
    print(sql)


if __name__ == "__main__":
    with context:
        test_autoinc()
        test_insert(context)
        test_enum(context)
        test_select(context)
