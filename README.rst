tinysql - A minimalistic object-relational mapper
===================================================

Introduction
------------

``tinysql`` is a lightweight Object-Relational Mapping (ORM) layer designed to facilitate the management of tabular data.
For instance, this relates to data from various domains such as computational neuroscience, data science, analytics, time series analysis, machine learning, and artificial intelligence.
It provides a minimalistic approach to map such data onto an SQLite database without obscuring the underlying SQL.
Binary data can be stored outside the database, because who doesn't need to share data with their colleagues but doesn't want to send or copy a terabyte-sized database.
In fact, it most likely shouldn't even be called an ORM.


Goal
----
The primary goal of ``tinysql`` is to offer a barebones ORM that maintains the expressive power of SQL.
By not abstracting away SQL, ``tinysql`` ensures that users can fully leverage SQL while benefiting from a simplified interface to map tabular/struct data onto a database.


Why tinysql?
------------
While more powerful alternatives like `SQLAlchemy <https://www.sqlalchemy.org>`_ or `DataJoint <https://www.datajoint.com/>`_ are available, they often come with additional complexity.
``tinysql`` addresses the need for a straightforward, minimalistic solution by focusing on:

* **Simplicity**: Avoids the overhead of complex ORM frameworks.
* **Direct SQL Access**: Retains the ability to execute custom SQL queries and perform explicit data manipulations.
* **Local and Portable Data Storage**: binary data (e.g. numpy arrays, BLOBs, etc) can be stored outside the database in the local file system, e.g. on disk, allowing exchange of such data with colleagues.


Features
--------
* **Minimalistic Design**: ``tinysql`` offers a simple and intuitive interface for database interactions.
* **Direct SQL Execution**: ``tinysql`` allows for writing and executing custom SQL scripts without restrictions.
* **Flexible Data Handling**: ``tinysql`` supports storing numpy arrays and other large data objects as BLOBs on disk, making data exchange easier.


Use Cases
---------
``tinysql`` is particularly useful for scenarios such as:

* **Local Data Analysis**: You need a lightweight tool for working with data locally without the overhead of larger frameworks.
* **Data Exchange**: You want to share specific data files (e.g. numpy arrays) without transferring entire databases.
* **Custom SQL**: You want to write and execute custom SQL queries while still benefiting from ORM features.


Installation
------------

At the moment, ``tinysql`` is not yet available on PyPI. To install it, you
therefore need to download or clone this repository and then use ``pip``.
Example:

.. code-block:: sh

    $ git clone https://github.com/nwaniek/tinysql.git
    $ cd tinysql
    $ pip install .

In the future, meaning as soon as ``tinysql`` is available on PyPI, you can
install it by simply running

.. code-block:: sh

    $ pip install tinysql


Usage
-----

To use ``tinysql``, you define your tables and interact with your database using minimalistic ORM methods.
A brief example could look as follows:

.. code-block:: python

    from typing import NamedTuple
    from tinysql import setup_db, Equals, Or, In, select, db_table

    # this is an example class that is derived from NamedTuple and mapped to the
    # database. It contains spiking data from neural recordings for a particular
    # animal
    @db_table('SpikeData', primary_keys=['id'])
    class SpikeData(NamedTuple):
        id: str # this could be a SHA1
        animal_name: str
        neuron_id: int
        spike_times: np.ndarray
        comment: str

    # The following will open an existing database, or create one if it does not
    # exist yet. If a second argument is given to setup_db, then tinysql will
    # assume storage of BLOBs and ndarrays should happen outside the database,
    # i.e. on disk or wherever the path points to.
    with setup_db('database.db', '/path/to/external/storage') as context:

        # load some data from, preprocess, etc...
        # once you have SpikeData with your data, we can insert it
        the_data = np.load('original_data_file_n123.npy')
        spikes = SpikeData(get_sha1('original_data_file_n123.npy'), 'Fievel', 123, the_data, "Data from Fievel's 123rd neuron")
        # we can either use the free function "insert", or the context method:
        context.insert(spikes)
        # is equvalent to: insert(context, spikes)

        # do something else, and now we want to analyse the data from Fievel and
        # Tanya. We can do so by using use some basic Conditionals (Equals, Or, ...)
        # to restrict results
        results = select(context, SpikeData, Or(Equals('animal_name', 'Fievel'), Equals('animal_name', 'Tanya')))
        for result in results:
            print(result)

        # tinysql supports most SQL WHERE conditionals, so instead of combining
        # an OR and two Equals, we could also do instead of the previous
        results = select(context, SpikeData, In('animal_name', ['Fievel', 'Tanya']))
        for result in results:
            print(result)

Enums
~~~~~

Of course, we also often use all kinds of enums to identify stuff or flag things.
And, obviously, you should map your enums to the database, too.
This is why ``tinysql`` supports all standard python enum types.

.. code-block:: python

    from tinysql import db_enum

    # for instance, we might want to use an enum to identify the brain region
    # in which the spike data was recorded in
    @db_enum("RecordingArea", descriptions={'PPC': 'Posterior Parietal Cortex', 'EC': 'Entorhinal Cortex', 'CA1': 'Cornu Ammonis 1', 'CA3': 'Cornu Ammonus 3'})
    class RecordingArea(Enum):
        PPC = "PPC"
        EC  = "EC"
        CA1 = "CA1"
        CA3 = "CA3"

    # db_enum doesn't care about the enum type, and you can also omit the
    # description if you don't want to document things in the database
    @dbenum('MyIntEnum')
    class MyIntEnum(IntEnum):
        One: auto()
        Two: auto()
        Three: auto()


Conditions
~~~~~~~~~~

Despite not really being a full-fledged ORM, ``tinysql`` provides a means to write
conditionals that are translated to SQL. In the spirit of ``tinysql``, they are
kept as minimalistic as possible and as close to SQL as it gets:

.. code-block:: python

    from tinysql import select, Not, GreaterThan

    results = select(context, AmazingValues, Not(GreaterThan('value1', 70.0)))
    for obj in results:
        print(obj)

``tinysql`` currently provides Equals, NotEquals, GreaterThan, LessThan, Between,
Like, In, And, Or, and Not. You can nest them arbitrarily and thereby build
complex expressions, but then again you might just simply drop into SQL to
achieve this, as will be shown next.


Direct SQL passthrough
~~~~~~~~~~~~~~~~~~~~~~

``tinysql`` does not hide the connection to the sqlite database it is connected to
(after using it as a context manager or runnning `init_tables`). It provides
some methods that you can use to fill specific objects like `select` where, you
can pass an SQL expression, and it will fill a particular class with the
results:

.. code-block:: python

    results = select(context, AmazingValues, "WHERE value1 >= ? AND value1 < ?", (70.0, 120.0, ))
    for obj in results:
        print(obj)

If you use select, or any other SQL passthrough method, it is up to you to make
sure that the result from the database can be accepted by the constructor of the
class that you pass in. That is, under the hood, ``tinysql`` merely forwards the
results via `cls(*row)`.

It is also possible to directly write SQL statements and execute them as you
usually would with sqlite:

.. code-block:: python

    with setup_db('mydatabase.sqlite') as context:
        cur = context.con.cursor()
        rows = cur.execute("SELECT * FOM AmazingValues")
        for row in rows:
            print(row)


Moreover, ``tinysql`` provides some methods like ``execute`` and ``executemany``,
that directly pass through to the connection and commits the statement, to save
you a few keystrokes:


.. code-block:: python

    with setup_db('mydatabase.sqlite') as context:
        context.executemany("INSERT INTO MyTable VALUES (?)", [("one",), ("two",)])

which is equivalent to

.. code-block:: python

    with setup_db('mydatabase.sqlite') as context:
        cur = context.con.cursor()
        cur.executemany("INSERT INTO MyTable VALUES (?)", [("one",), ("two",)])
        context.con.commit()

Does it save much? No. Is ist convenient? Yes.


Autoincrement
~~~~~~~~~~~~~

Sometimes there's a need for an autoincrement field. tinysql supports this, but
be aware that sqlite has special treatment for autoincrement. That is, an
autoinc field must be a primary key, and there can be only one primary key in
the table. If you attempt to create tinysql-mapped tables with autoinc fields
and more than one primary key, tinysql will raise an exception! Read more about
sqlite's autoinc in the `sqlite documentation <https://www.sqlite.org/autoinc.html>`_.

.. code-block:: python

    from tinysql import autoinc, db_table

    # to create an autoinc field, simply use tinysql's autoinc type
    @db_table('FancyData', primary_keys=['id'])
    class FancyData(NamedTuple):
        id : autoinc
        stuff: str

    # when creating a new instance of FancyData, you need to pass an instance of
    # autoinc to FancyData. tinysql will filter out autoinc fields when
    # inserting data into the database. when loading data, you'll get a regular
    # integer back.
    my_data = FancyData(autoinc(), 'really amazing data!')

There's another subtle issue with autoinc, namely when using tinysql with an
external storage for BLOBs. At the time of writing an entry into the
database, or more precisely before writing the data to the table, the value of
the autoinc field might not yet be determined. Yet, the primary key(s) of a
mapped/registered class will be used in the production of the filename where
the ndarray will be stored.

As a general recommendation: don't mix autoinc fields with BLOB fields in one
class. Rather, use another form of primary key, something that can be determined
at runtime before writing things to the database, such as a SHA1 over your data,
or a time-based UUID.


UUIDs
~~~~~

For convenience, and to ameliorate the situation regarding autoincrement and
external storage, ``tinysql`` provides a specific class ``uuid``. Well, it
really is just a wrapper around ``str`` and the function ``gen_uuid()``, which
in turn simply calls ``uuid4().hex`` from python's ``uuid`` module... The reason
``tinysql.uuid`` exists is to make this type somewhat explicit, with the goal to
improve the self-documentation level of code.

Here's how to use it:

.. code-block:: python

    from tinysql import db_table, uuid

    @db_table("UUIDTest", primary_keys=["id"])
    @dataclass
    class UUIDTest:
        id: uuid

    def test_uuid(context):
        context.insert(UUIDTest(uuid()))
        for obj in context.select(UUIDTest):
            print(obj)

As with anything else in ``tinysql``, it is kept as barebones as it gets. That
means that you have to specify the value itself during construction (see the
``context.insert(...)`` line).  You could also move this into a custom
constructor or use ``id: uuid = field(default_constructor = lambda: uuid())``,
but this would likely break ``tinysql``'s ``select`` statement, which merely
passes each result row from a database query to the constructor of a class.


Working with several databases
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Other times, you might want to work with several databases at the same time.
While this is possible with ``tinysql``, there are some limitations you need to be
aware of. To understand these limitations, it's necessary to look under the hood
of how ``tinysql`` manages tables.

When you use the ``db_enum`` or ``db_table`` decorator as in the examples above,
then ``tinysql`` will store an entry into its 'global table registry'. You can
inspect this registry if you want at runtime:

.. code-block:: python

    from typing import NamedTuple
    import tinysql

    @db_table(...) # map/register your class
    class MyData(NamedTuple):
        # ...

    # list all tables globally known to tinysql
    print(tinysql.TABLE_REGISTRY)


When you create/open a connection to a database using ``setup_db``, then the
DatabaseContext that is returned from the function call will inherit this global
registry.

To handle several databases, you need to register a class against a specific
context. You also need to initialize the tables by either using the context as a
context manager, or explicitly invoking its ``init_tables`` method. Here's an
example for all of this:

.. code-block:: python

    from typing import NamedTuple
    from tinysql import db_table, DatabaseContext

    # create two instances of DatabaseContext, each pointing to a particular
    # sqlite database, and telling them to *not* use the global registry.
    # If you wonder why tinysql defaults to a global registry? The reason is
    # that, at least in my use cases, I more often work with databases with
    # the same tables, or with just a single database connection. Using the
    # global registry by default improves terseness slightly.
    context1 = DatabaseContext('db1.sqlite', use_global_registry=False)
    context2 = DatabaseContext('db2.sqlite', use_global_registry=False)

    # register a table against a specific context.
    @db_table("StringData", context=context1)
    class StringData:
        data: str

    # register another table against the other context
    @db_table("FloatData", context=context2)
    class FloatData:
        data: float

    # at this point, StringData will be only known to context1, while
    # FloatData will only be known to context2. We need to make sure that the
    # tables get initialized. This can be done either via a context manager, or
    # explicitly:

    with context1:
        # do something with the context, like adding string data to this
        # database
        context1.insert(StringData("wow!"))

    # Note that the connection to the database will be closed once the context
    # manager goes out of context. That is, any further operation against the
    # database with context1 will now fail
    context1.insert(StringData("this will fail"))

    # the alternative is to explicitly initialize the tables.
    context2.init_tables()
    # and then use it
    context2.insert(FloatData(42.0))
    # make sure to close the context when you're done. This will close the
    # connection to the database
    context2.close()


Extending tinysql with other types
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

If you wish to extend ``tinysql`` with other types than the standard types that it
already supports, autoinc, np.ndarray, and other BLOBs, then best have a look at
``tinysql``'s ``TYPE_MAPPING`` variable. This is simply a dict which contains a map
from a type that you want to use in a type annotation to the sqlite database
type and some additional flag. You can either inject your own type mappings into
``TYPE_MAPPING``, or change it directly there (remember, tinysql is as basic as it
gets, and a 'single file package').


Contributing
------------
Contributions are welcome!
If you have suggestions, bug reports, or want to contribute code, please open an issue or submit a pull request on GitHub.


License
-------
``tinysql`` is licensed under the MIT License.
See the `LICENSE <LICENSE>`_ file for details.

