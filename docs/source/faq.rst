:orphan:

.. currentmodule:: statsmodels

.. _faq:

Frequently Asked Question
=========================

What is statsmodels?
--------------------

statsmodels is a Python package that provides a collection of widely-used
statistical models. While statsmodels historically has an econometrics-heavy
user base, the package is designed to be useful to a large variety of
statistical use cases. In comparison with other Python-based modelling
tools, statsmodels focuses more heavily on the statistics and diagnostics
underlying the models than having the most cutting-edge or predictive models.

.. _endog-exog-faq:

What do endog and exog mean?
----------------------------

These are shorthand for endogenous and exogenous variables. You might be more
comfortable with the common ``y`` and ``X`` notation in linear models.
Sometimes the endogenous variable ``y`` is called a dependent variable.
Likewise, sometimes the exogenous variables ``X`` are called the independent
variables. You can read about this in greater detail at :ref:`endog_exog`

.. _missing-faq:

How does statsmodels handle missing data?
-----------------------------------------

Missing data can be handled via the ``missing`` keyword argument. Every model
takes this keyword. You can find more information in the docstring of
:class:`statsmodels.base.Model <statsmodels.base.model.Model>`.

.. _build-faq:

Why will not statsmodels build?
-------------------------------

Remember that to build, you must have:

- The appropriate dependencies (numpy, pandas, scipy, Cython) installed
- A suitable C compiler
- A working python installation

Please review our :ref:`installation instructions <install>` for details.

You might also try cleaning up your source directory by running:

.. code-block:: bash

    pip uninstall statsmodels
    python setup.py clean

And then attempting to re-compile. If you want to be more aggressive, you
could also reset git to a prior version by:

.. code-block:: bash

    git reset --hard
    git clean -xdf
    git checkout master
    python setup.py clean

I'd like to contribute. Where do I start?
-----------------------------------------

Check out our :doc:`development pages <dev/index>` for a guide on how to
get involved. We accept Pull Requests on our GitHub page for bugfixes and
topics germane to statistics and statistical modeling. In addition, usability
and quality of life enhancements are greatly appreciated as well.

What if my question is not answered here?
-----------------------------------------

You may find answers for questions that have not yet been added here on GitHub
under the `FAQ issues tag <https://github.com/statsmodels/statsmodels/labels/FAQ>`_.
If not, please ask your question on stackoverflow using the
`statsmodels tag <https://stackoverflow.com/questions/tagged/statsmodels>`_ or
on the `mailing list <https://groups.google.com/forum/#!forum/pystatsmodels>`_.
