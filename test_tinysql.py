#!/usr/bin/env python

from dataclasses import dataclass
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
    id:     str
    value0: str
    value1: float
    value2: np.ndarray


@tinysql.db_table("DataClassTest", primary_keys=["id"], context=context)
@dataclass
class DataClassTest:
    id:     str
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
    values = AmazingValues(gen_uuid(), "hello, world!", 123.12, np.ones((4, 4)))
    context.insert(values)

    # testing the dataclass
    context.insert(DataClassTest(gen_uuid(), "hello, world, this is a dataclass!", 123.12, np.ones((4, 4))))

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

    # print everything in the table
    sql = "SELECT * FROM DataClassTest"
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


def test_insertmany(context):
    values = [AmazingValues(gen_uuid(), "one", 1.0, np.ones((1,1))),
              AmazingValues(gen_uuid(), "two", 2.0, np.ones((2,2))),
              AmazingValues(gen_uuid(), "three", 3.0, np.ones((3,3)))]
    context.insertmany(values)
    sql = "SELECT * FROM AmazingValues"
    rows = context.con.execute(sql)
    for row in rows:
        print(row)


@tinysql.db_table("Employee", primary_keys=['id'], context=context)
class Employee(NamedTuple):
    id: tinysql.autoinc
    name: str
    salary: float


def test_update(context):
    # first write some data to the database
    context.insert(Employee(tinysql.autoinc(), "Alice",   1500.0))
    context.insert(Employee(tinysql.autoinc(), "Bob",     1000.0))
    context.insert(Employee(tinysql.autoinc(), "Eve",     1250.0))
    context.insert(Employee(tinysql.autoinc(), "Charlie", 1400.0))
    print("insert")
    results = context.select(Employee)
    for result in results:
        print("  ", result)

    # now we want to build an update, raising the salary for everyone who earns
    # less than 1400 by a certain factor
    tinysql.update(context, Employee, [('salary', 'salary * 1.1'), ('name', 'UPPER(name)')], tinysql.LessThan('salary', 1400.0))
    print("update typed")
    results = context.select(Employee)
    for result in results:
        print("  ", result)

    # direct sql via execute
    tablename = tinysql.get_tablename(Employee)
    tinysql.execute(context, f"UPDATE {tablename} SET salary = salary * 1.1, name = lower(name) WHERE salary < ?", (1400.0, ))
    print("update via execute")
    results = context.select(Employee)
    for result in results:
        print("  ", result)

    # direct sql via executemany
    tablename = tinysql.get_tablename(Employee)
    tinysql.executemany(context, f"UPDATE {tablename} SET salary = salary * 2.1, name = UPPER(name) WHERE id = ?", [(1,), (4,),])
    print("update via executemany")
    results = context.select(Employee)
    for result in results:
        print("  ", result)


if __name__ == "__main__":
    with context:
        test_autoinc()
        test_insert(context)
        test_enum(context)
        test_select(context)
        test_insertmany(context)
        test_update(context)
