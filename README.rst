tinysql - A tiny object-relational mapper
=========================================

Intro: mostly tabular, in computational neuroscience, data science, analytics,
time series analysis, machine learning & AI, etc.
Goal: There was a need for a barebones layer that maps data onto a database that
does not hide the SQL below it
Why not more powerful alterantives (sqlalchemy, datajoint)? a lot of the time
spent during data analysis is transforming and actively working with the data
Meaning: write custom SQL scripts, glue together things from several structs,
etc. sometimes only to identify particular records. But in any case, retain the
(expressive) power of SQL.

Also: very often working locally, but with a need to exchange files with
colleagues. Don't want to send the entire database, but only the data containing
files. Hence, automate storing BLOBs and numpy ndarrays locally on disk, so that
I can take one, mail it around, etc.


Who should use this?
--------------------



How to use this?
----------------



