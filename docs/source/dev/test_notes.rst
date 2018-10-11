.. _testing:

Testing
=======

Setting up development environment locally
------------------------------------------
Follow our :ref:`installation instructions <install>` and set up a suitable
environment to build statsmodels from source. We recommend that you develop
using a development install of statsmodels::

    python setup.py develop

This will compile the C code and add statsmodels to your activate python
environment by creating links from your python environemnt's libraries
to the statsmodels source code. Therefore, changes to pure python code will
be immediately available to the user without a re-install.

Test Driven Development
-----------------------
We strive to follow a `Test Driven Development (TDD) <https://en.wikipedia.org/wiki/Test-driven_development>`_ pattern.
All models or statistical functions that are added to the main code base are to have
tests versus an existing statistical package, if possible.

Introduction to pytest
----------------------
Like many packages, statsmodels uses the `pytest testing system <https://docs.pytest.org/en/latest/contents.html>`__ and the convenient extensions in `numpy.testing <http://docs.scipy.org/doc/numpy/reference/routines.testing.html>`__.  Pytest will find any file, directory, function, or class name that starts with ``test`` or ``Test`` (classes only). Test function should start with ``test``, test classes should start with ``Test``. These functions and classes should be placed in files with names beginning with ``test`` in a directory called ``tests``.

.. _run-tests:

Running the Test Suite
----------------------

You can run all the tests by::

    >>> import statsmodels.api as sm
    >>> sm.test()

You can test submodules by::

    >>> sm.discrete.test()


Running Tests using the command line
------------------------------------
Test can also be run from the command line by calling ``pytest``.  Tests can be run
at different levels:

* Project level, which runs all tests.  Running the entire test suite is slow
  and normally this would only be needed if making deep changes to statsmodels.

.. code-block:: bash

    pytest statsmodels

* Folder level, which runs all tests below a folder

.. code-block:: bash

    pytest statsmodels/regression/tests

* File level, which runs all tests in a file

.. code-block:: bash

    pytest statsmodels/regression/tests/test_regression.py

* Class level, which runs all tests in a class

.. code-block:: bash

    pytest statsmodels/regression/tests/test_regression.py::TestOLS

* Test level, which runs a single test.  The first example runs a test in a
  class.  The second runs a stand alone test.

.. code-block:: bash

    pytest statsmodels/regression/tests/test_regression.py::TestOLS::test_missing
    pytest statsmodels/regression/tests/test_regression.py::test_ridge

How To Write A Test
-------------------
NumPy provides a good introduction to unit testing with pytest and NumPy extensions `here <https://github.com/numpy/numpy/blob/master/doc/TESTS.rst.txt>`__. It is worth a read for some more details.
Here, we will document a few conventions we follow that are worth mentioning. Often we want to test
a whole model at once rather than just one function, for example. The following is a pared down
version test_discrete.py. In this case, several different models with different options need to be
tested. The tests look something like

.. code-block:: python

    from numpy.testing import assert_almost_equal
    import statsmodels.api as sm
    from results.results_discrete import Spector

    class CheckDiscreteResults(object):
        """
        res2 are the results. res1 are the values from statsmodels
        """

        def test_params(self):
            assert_almost_equal(self.res1.params, self.res2.params, 4)

        decimal_tvalues = 4
        def test_tvalues(self):
            assert_almost_equal(self.res1.params, self.res2.params, self.decimal_tvalues)

        # ... as many more tests as there are common results

    class TestProbitNewton(CheckDiscreteResults):
        """
        Tests the Probit model using Newton's method for fitting.
        """

        @classmethod
        def setup_class(cls):
            # set up model
            data = sm.datasets.spector.load()
            data.exog = sm.add_constant(data.exog)
            cls.res1 = sm.Probit(data.endog, data.exog).fit(method='newton', disp=0)

            # set up results
            res2 = Spector()
            res2.probit()
            cls.res2 = res2

            # set up precision
            cls.decimal_tvalues = 3

        def test_model_specifc(self):
            assert_almost_equal(self.res1.foo, self.res2.foo, 4)

The main workhorse is the `CheckDiscreteResults` class. Notice that we can set the level of precision
for `tvalues` to be different than the default in the subclass  `TestProbitNewton`. All of the test
classes have a ``@classmethod`` called ``setup_class``. Otherwise, pytest would reinstantiate the class
before every single test method. If the fitting of the model is time consuming, then this is clearly
undesirable. Finally, we have a script at the bottom so that we can run the tests should be running
the Python file.

Test Results
------------
The test results are the final piece of the above example. For many tests, especially those for the
models, there are many results against which you would like to test. It makes sense then to separate
the hard-coded results from the actual tests to make the tests more readable. If there are only a few
results it's not necessary to separate the results. We often take results from some other statistical
package. It is important to document where you got the results from and why they might differ from
the results that we get. Each tests folder has a results subdirectory. Consider the folder structure
for the discrete models::

    tests/
        __init__.py
        test_discrete.py
        results/
            __init__.py
            results_discrete.py
            nbinom_resids.csv

It is up to you how best to structure the results. In the discrete model example, you will notice
that there are result classes based around particular datasets with a method for loading different
model results for that dataset. You can also include text files that hold results to be loaded by
results classes if it is easier than putting them in the class itself.
