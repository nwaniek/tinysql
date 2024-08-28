#!/usr/bin/env python

from dataclasses import dataclass, field
from enum import Enum
from typing import NamedTuple
import time
import numpy as np
import uuid
import tinysql



@tinysql.db_table("AmazingValues", primary_keys=["id"])
class AmazingValues(NamedTuple):
    id:     str
    value0: str
    value1: float
    value2: np.ndarray


@tinysql.db_table("DataClassTest", primary_keys=["id"])
@dataclass
class DataClassTest:
    id:     str
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
    values = AmazingValues(tinysql.gen_uuid(), "hello, world!", 123.12, np.ones((4, 4)))
    context.insert(values)

    # testing the dataclass
    context.insert(DataClassTest(tinysql.gen_uuid(), "hello, world, this is a dataclass!", 123.12, np.ones((4, 4))))

    # insert as a dict with corresponding tablespec
    values = {'id': tinysql.gen_uuid(),
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
    results = tinysql.select(context, AmazingValues, tinysql.Not(tinysql.GreaterThan('value1', 70.0)))
    for obj in results:
        print(obj)


def test_select_sql(context):
    results = tinysql.select(context, AmazingValues, "WHERE value1 >= 70.0 AND value1 < 150.0")
    for obj in results:
        print(obj)

    results = tinysql.select(context, AmazingValues, "WHERE value1 >= ? AND value1 < ?", (70.0, 120.0, ))
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
    values = [AmazingValues(tinysql.gen_uuid(), "one", 1.0, np.ones((1,1))),
              AmazingValues(tinysql.gen_uuid(), "two", 2.0, np.ones((2,2))),
              DataClassTest(tinysql.gen_uuid(), "out of order", 17.17, np.zeros((1,1))),
              AmazingValues(tinysql.gen_uuid(), "three", 3.0, np.ones((3,3)))]
    context.insertmany(values, keep_order=True)
    sql = "SELECT * FROM AmazingValues"
    rows = context.con.execute(sql)
    for row in rows:
        print(row)


@tinysql.db_table("Employee", primary_keys=['id'])
@dataclass
class Employee:
    id: tinysql.autoinc
    name: str
    salary: float


@tinysql.db_table("StrData", primary_keys=['id0', 'id1'])
@dataclass
class StrData:
    id0: tinysql.uuid
    id1: tinysql.uuid
    data: str


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

    # update one particular instance
    data = StrData(tinysql.uuid(), tinysql.uuid(), "This is old data")
    context.insert(data)
    for result in context.select(StrData):
        print(result)
    data.data = "This is new data!"
    context.update(data)
    for result in context.select(StrData):
        print(result)



@tinysql.db_table("DCWithMethods", primary_keys=["pk1", "pk2"])
@dataclass
class DCWithMethods:
    pk1: int
    pk2: int

    def __init__(self):
        # simulate a split primary key by splitting a time value into its parts
        # before and after the comma
        value = time.time()
        self.pk1 = int(value)
        self.pk2 = int((value - self.pk1) * 1_000_000)

    def to_timestr(self):
        value = float(self.pk1) + float(self.pk2) / 1_000_000.0
        tstruct = time.localtime(value)
        timestr = time.strftime("%Y-%m-%d %H:%M:%S", tstruct)
        return timestr


def test_dcmethods(context):
    dcm1 = DCWithMethods()
    dcm2 = DCWithMethods()
    context.insertmany([dcm1, dcm2])
    print(dcm1.pk2, dcm1.to_timestr(), dcm2.pk2, dcm2.to_timestr())



@tinysql.db_table("UUIDTest", primary_keys=["id"])
@dataclass
class UUIDTest:
    id: tinysql.uuid


def test_uuid(context):
    context.insert(UUIDTest(tinysql.uuid()))
    for obj in context.select(UUIDTest):
        print(obj)


if __name__ == "__main__":
    context = tinysql.DatabaseContext('test.sqlite', 'test_storage')
    with context:
        test_autoinc()
        test_insert(context)
        test_enum(context)
        test_select(context)
        test_select_sql(context)
        test_insertmany(context)
        test_update(context)
        test_dcmethods(context)
        test_uuid(context)
