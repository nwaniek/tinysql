#!/usr/bin/env python

from dataclasses import dataclass
import numpy as np

import tinysql
tinysql.configure(use_global_registry=False)


@tinysql.db_table("AmazingValues", primary_keys=["id"])
@dataclass
class AmazingValues:
    id:     str
    value0: str
    value1: float
    value2: np.ndarray


@tinysql.db_table("OtherValues", primary_keys=["id"])
@dataclass
class OtherValues:
    id:     str
    value0: float

# the following should print an empty dict
print(tinysql.TABLE_REGISTRY)

context1 = tinysql.DatabaseContext('db1.sqlite', None, classes=[AmazingValues])
context2 = tinysql.DatabaseContext('db2.sqlite', None, classes=[OtherValues])
context3 = tinysql.DatabaseContext('db3.sqlite', None, classes=[AmazingValues, OtherValues])

# this should print one entry per dictionary each
print(context1.registry)
print(context2.registry)

with context1:
    context1.insert(AmazingValues(tinysql.uuid(), 'value0', 1.23, np.zeros((3,3))))
    # the following will fail, context1 has no idea about OtherValues
    try:
        context1.insert(OtherValues(tinysql.uuid(), 7.71))
    except tinysql.TableNotMappedError:
        print("Caught a TableNotMappedError")
        pass

context2.open()
context2.insert(OtherValues(tinysql.uuid(), 7.71))
# same as above, other way around
try:
    context2.insert(AmazingValues(tinysql.uuid(), 'value0', 1.23, np.zeros((3,3))))
except tinysql.TableNotMappedError:
    print("Caught a TableNotMappedError")
    pass
context2.close()

# context 3 knows all the classes
with context3:
    context3.insert(AmazingValues(tinysql.uuid(), 'value0', 1.23, np.zeros((3,3))))
    context3.insert(OtherValues(tinysql.uuid(), 7.71))

