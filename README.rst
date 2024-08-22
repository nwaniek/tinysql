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
While more powerful alternatives like SQLAlchemy or DataJoint are available, they often come with additional complexity.
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

..code-block:: sh

    $ pip install tinyql


Usage
-----

To use Tinysql, you define your tables and interact with your database using minimalistic ORM methods.
A brief example could look as follows:


..code-block:: python

    from typing import NamedTuple
    from tinysql import DatabaseContext, Equals, Or, select, db_table

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

    # Open the database. This will create the database if it does not exist. If
    # a second argument is given to DatabaseContext, then tinysql will assume
    # storage of BLOBs and ndarrays should happen outside the database, i.e. on
    # disk
    with DatabaseContext('database.db', '/path/to/external/storage') as context:
        # use some basic Conditionals (Equals, Or, ...) to restrict results
        results = select(context, SpikeData, Or(Equals('animal_name', 'Fievel'), Equals('animal_name', 'Tanya')))
        for result in results:
            print(result)


Contributing
------------
Contributions are welcome!
If you have suggestions, bug reports, or want to contribute code, please open an issue or submit a pull request on GitHub.


License
-------
`tinysql` is licensed under the MIT License.
See the `LICENSE <LICENSE>`_ file file for details.

