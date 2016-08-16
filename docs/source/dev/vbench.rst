:orphan:

.. _vbenchdoc:

vbench
======

`vbench`_ is a tool for benchmarking your code through time, for showing performance improvement or regressions.

WARNING: ``vbench`` is not yet compatible with python3.

New Dependencies
~~~~~~~~~~~~~~~~

* `vbench`_ (from github only)
* `sqlalchemy`_
* `gitpython`_
* `psutil`_
* `affinity`_ (As a fallback to psutil)

.. _vbench: https://github.com/pydata/vbench
.. _sqlalchemy: https://pypi.python.org/pypi/SQLAlchemy
.. _gitpython: https://pypi.python.org/pypi/GitPython/
.. _psutil: https://pypi.python.org/pypi/psutil
.. _affinity: https://pypi.python.org/pypi/affinity

Also note that you need to have sqlite3 working with python.

Writing a good vbench
~~~~~~~~~~~~~~~~~~~~~

A set of related benchmarks go together in a module (a ``.py`` file).
See ``vb_suite/discrete.py`` for an example.

There's typically some boilerplate common to all the tests, which can
be placed in a string ``common_setup``.

Now we can write our specific benchmark.

There are up to three items in a single benchmark:

* setup specific to that benchmark (typically a string concatenated to ``common_setup``)
* a statement to be executed, which is the first argument to the ``vbench.BenchmarkRunner`` class
* instantiation the ``vbench.Benchmark`` class

It's important to separate the setup from the statement we're interested in profiling.
The statement ought to be concise and should profile only one thing.
If you mix setup in with the statement to be profiled, then changes affecting the performance of the setup (which might even take place outside your library) will pollute the test.

Each module must be listed in the ``suite.py`` file in the modules list.

Not all tests can be run against the entire history of the project.
For newer features, each ``Benchmark`` object takes an optional ``start_date`` parameter.
For example:

.. code-block:: python

    start_date=datetime(2012, 1, 1)

If a ``start_date`` is not applied for a specific benchmark, the global setting from ``vb_suite.py`` is used.

Another reason that a benchmark can't be run against the entire project's history is that API's sometimes have to change in ways that are not backwards compatible.
For these cases, the easiest way to compare performance pre- to post-API change is probably the try-except idiom:

.. code-block:: python

    rng = date_range('1/1/2000', periods=N, freq='min')

Pre-PR
~~~~~~

Most contributors don't need to worry about writing a vbench or running the full suite against the project's entire history.
Use ``test_perf.py`` to see how the performance of your PR compares against a known-to-be-good benchmark.


Implementation
==============

There are two main uses for ``vbench.``
The first is most useful for someone submitting a pull request that might affect the performance of the library.
In this case, the submitter should run ``python vb_suite/test_perf.py -b base_commit -H``, where ``base_commit`` is the baseline commit hash you want to compare to.
The ``-H`` argument says to compare the HEAD of your branch against the baseline commit.

The second use-case is for measuring the long-term performance of the project.
For this case the file of interest is ``run_suite.py``.
Using the parameters specified in that file, the suite of benchmarks is run against the history of the project.
The results are stored in a sqlite database.

suite.py
~~~~~~~~

This is the main configuration file.
It pulls in the benchmarks from the various modules in ``vb_suite``, reads in the user configuration, and handles the setup and tear-down of the performance tests.

run_suite.py
~~~~~~~~~~~~

Useful for the maintainers of the project to track performance over time.
Runs with no arguments from the command line.
Persists the results in a database so that the the full suite needn't be rerun each time.

User config file
~~~~~~~~~~~~~~~~

Only necessary if you're running ``run_suite.py``.
Should look something like:

repo_path: /Home/Envs/statsmodels/lib/python2.7/site~packages/statsmodels/
repo_url: https://github.com/statsmodels/statsmodels.git
db_path: /Homevbench/statsmodels/vb_suite/benchmarks.db
tmp_dir: /Home/tmp


test_perf.py
~~~~~~~~~~~~

Use before commit to check for performance regressions.
CLT, use ``python test_perf.py -h`` for help.

Most of the time you'll be giving it one or two arguments:

* ``-b BASE_COMMIT``: the commit you're comparing your commit against
* ``-t TARGET_COMMIT``: or use -H to set the target to the ``HEAD`` of your branch.


generate_rst_files.py
~~~~~~~~~~~~~~~~~~~~~

Once you've run `run_suite.py` and generated a benchmark database, you can use ``generate_rst_files.py`` to graph performance over time.


References:
~~~~~~~~~~~

`http://wesmckinney.com/blog/?p=373 <http://wesmckinney.com/blog/?p=373>`_

`https://github.com/pydata/vbench <https://github.com/pydata/vbench>`_

`https://github.com/pydata/pandas/tree/master/vb_suite <https://github.com/pydata/pandas/tree/master/vb_suite>`_

`https://github.com/yarikoptic/numpy-vbench <https://github.com/yarikoptic/numpy-vbench>`_
