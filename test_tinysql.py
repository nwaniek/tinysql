#!/usr/bin/env python

from enum import Enum
from typing import NamedTuple
import tinysql
import numpy as np

@tinysql.db_table("AmazingValues", primary_keys=["id"])
class AmazingValues(NamedTuple):
    id:     int
    value0: str
    value1: float
    value2: np.ndarray


@tinysql.db_enum("MyEnum", descriptions={'One': 'First field of MyEnum', 'Two': 'Second field of MyEnum'})
class MyEnum(Enum):
    One = "one"
    Two = "two"


def test_insert(context):
    print("Table 'AmazingValues' exists:", tinysql.table_exists(context.con, "AmazingValues"))

    # insert via object
    values = AmazingValues(1, "hello, world!", 123.12, np.ones((4, 4)))
    context.insert(values)

    # insert as a dict with corresponding tablespec
    values = {'id': 2,
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
    results = tinysql.select(context, AmazingValues)
    for obj in results:
        print(obj)


if __name__ == "__main__":
    with tinysql.setup_db('test.sqlite', 'test_storage') as context:
        # test_insert(context)
        # test_enum(context)
        test_select(context)
