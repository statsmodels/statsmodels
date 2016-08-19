:orphan:

.. currentmodule:: statsmodels

.. _faq:

Frequently Asked Question
-------------------------

.. _endog-exog-faq:

What do endog and exog mean?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

These are shorthand for endogenous and exogenous variables. You might be more comfortable with the common ``y`` and ``X`` notation in linear models. Sometimes the endogenous variable ``y`` is called a dependent variable. Likewise, sometimes the exogenous variables ``X`` are called the independent variables. You can read about this in greater detail at :ref:`endog_exog` 


.. _missing-faq:

How does statsmodels handle missing data?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Missing data can be handled via the ``missing`` keyword argument. Every model takes this keyword. You can find more information in the docstring of :class:`statsmodels.base.Model <base.model.Model>`. 

.. `Model class <http://www.statsmodels.org/devel/dev/generated/statsmodels.base.model.Model.html#statsmodels.base.model.Model>`_.

.. _build-faq:

Why won't statsmodels build?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

If you're on Python 3.4, you *must* use Cython 0.20.1. If you're still having problems, try running

.. code-block:: bash

    python setup.py clean

What if my question isn't answered here?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

You may find answers for questions that have not yet been added here on GitHub under the `FAQ issues tag <https://github.com/statsmodels/statsmodels/issues?labels=FAQ&page=1&state=open>`_. If not, please ask your question on stackoverflow using the `statsmodels tag <https://stackoverflow.com/questions/tagged/statsmodels>`_ or on the `mailing list <https://groups.google.com/forum/#!forum/pystatsmodels>`_.
