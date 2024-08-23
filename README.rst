`tinysql` - A minimalistic object-relational mapper
===================================================

Introduction
------------

`tinysql` is a lightweight Object-Relational Mapping (ORM) layer designed to facilitate the management of tabular data.
For instance, this relates to data from various domains such as computational neuroscience, data science, analytics, time series analysis, machine learning, and artificial intelligence.
It provides a minimalistic approach to map such data onto an SQLite database without obscuring the underlying SQL.
Binary data can be stored outside the database, because who doesn't need to share data with their colleagues but doesn't want to send or copy a terabyte-sized database.
In fact, it most likely shouldn't even be called an ORM.


Goal
----
The primary goal of `tinysql` is to offer a barebones ORM that maintains the expressive power of SQL.
By not abstracting away SQL, `tinysql` ensures that users can fully leverage SQL while benefiting from a simplified interface to map tabular/struct data onto a database.


Why `tinysql`?
--------------
While more powerful alternatives like `SQLAlchemy <https://www.sqlalchemy.org>`_ or `DataJoint <https://www.datajoint.com/>`_ are available, they often come with additional complexity.
`tinysql` addresses the need for a straightforward, minimalistic solution by focusing on:

* **Simplicity**: Avoids the overhead of complex ORM frameworks.
* **Direct SQL Access**: Retains the ability to execute custom SQL queries and perform explicit data manipulations.
* **Local and Portable Data Storage**: binary data (e.g. numpy arrays, BLOBs, etc) can be stored outside the database in the local file system, e.g. on disk, allowing exchange of such data with colleagues.


Features
--------
* **Minimalistic Design**: `tinysql` offers a simple and intuitive interface for database interactions.
* **Direct SQL Execution**: `tinysql` allows for writing and executing custom SQL scripts without restrictions.
* **Flexible Data Handling**: `tinysql` supports storing numpy arrays and other large data objects as BLOBs on disk, making data exchange easier.


Use Cases
---------
`tinysql` is particularly useful for scenarios such as:

* **Local Data Analysis**: You need a lightweight tool for working with data locally without the overhead of larger frameworks.
* **Data Exchange**: You want to share specific data files (e.g. numpy arrays) without transferring entire databases.
* **Custom SQL**: You want to write and execute custom SQL queries while still benefiting from ORM features.


Installation
------------

To install `tinysql`, simply run

.. code-block:: sh

    $ pip install tinysql


Usage
-----

To use `tinysql`, you define your tables and interact with your database using minimalistic ORM methods.
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

Of course, we also often use all kinds of enums to identify stuff or flag things.
And, obviously, you should map your enums to the database, too.
This is why `tinysql` supports all standard python enum types.

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


Sometimes there's a need for an autoincrement field. tinysql supports this, but
be aware that sqlite has special treatment for autoincrement. That is, an
autoinc field must be a primary key, and there can be only one primary key in
the table. If you attempt to create tinysql-mapped tables with autoinc fields
and more than one primary key, tinysql will raise an exception! Read more about
sqlite's autoinc in the `sqlite documentation <https://www.sqlite.org/autoinc.html>`_.

.. code-block:: python

    from tinysql import autoinc, db_table

    # to create an autoinc field, simply use tinysql's autoinc type
    @db_table('FancyData', primary_key['id'])
    class FancyData(NamedTuple):
        id : autoinc
        stuff: str

    # when creating a new instance of FancyData, you need to pass an instance of
    # autoinc to FancyData. tinysql will filter out autoinc fields when
    # inserting data into the database. when loading data, you'll get a regular
    # integer back.
    my_data = FancyData(autoinc(), 'really amazing data!')



Contributing
------------
Contributions are welcome!
If you have suggestions, bug reports, or want to contribute code, please open an issue or submit a pull request on GitHub.


License
-------
`tinysql` is licensed under the MIT License.
See the `LICENSE <LICENSE>`_ file for details.

