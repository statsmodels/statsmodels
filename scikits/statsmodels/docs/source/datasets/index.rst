.. _datasets:

.. ipython:: python
   :suppress:

   import numpy as np
   np.set_printoptions(precision=4)

The Datasets Package
====================

Original Proposal
~~~~~~~~~~~~~~~~~

The idea for a datasets package was originally proposed by David Cournapeau and
can be found :ref:`here <dataset_proposal>` with updates by me (Skipper
Seabold).

Available Datasets
~~~~~~~~~~~~~~~~~~

.. toctree::
   :maxdepth: 1
   :glob:

   generated/*

Main Usage
~~~~~~~~~~

To load a dataset do the following

.. ipython:: python

   import scikits.statsmodels.api as sm
   data = sm.datasets.longley.load()

The `Dataset` object follows the bunch pattern as explain in the
:ref:`proposal <dataset_proposal>`.

Most datasets have two attributes of particular interest to users for examples

.. ipython:: python

   data.endog
   data.exog

Univariate datasets, however, do not have an `exog` attribute. You can find
out the variable names by doing

.. ipython:: python

   data.endog_name
   data.exog_name

If the dataset does not have a clear interpretation of what should be an
`endog` and `exog`, then you can always access the `data` or `raw_data`
attributes. This is the case for the `macrodata` dataset, which is a collection
of US macroeconomic data rather than a dataset with a specific example in mind.
The `data` attribute contains a record array of the full dataset and the
`raw_data` attribute contains an ndarray with the names of the columns given
by the `names` attribute.

.. ipython:: python

   type(data.data)
   type(data.raw_data)
   data.names

Loading data as pandas objects
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

For many users it may be preferable to get the datasets as a pandas DataFrame or
Series object. Each of the dataset modules is equipped with a ``load_pandas``
method which returns a ``Dataset`` instance with the data as pandas objects:

.. ipython:: python

   data = sm.datasets.longley.load_pandas()
   data.exog
   data.endog

With pandas integration in the estimation classes, the metadata will be attached
to model results:

.. ipython:: python

   y, x = data.endog, data.exog
   res = sm.OLS(y, x).fit()
   res.params
   res.summary()

Extra Information
~~~~~~~~~~~~~~~~~

If you want to know more about the dataset itself, you can access the
following, again using the Longley dataset as an example ::

    >>> dir(sm.datasets.longley)[:6]
    ['COPYRIGHT', 'DESCRLONG', 'DESCRSHORT', 'NOTE', 'SOURCE', 'TITLE']

How to Add a Dataset
~~~~~~~~~~~~~~~~~~~~

See the :ref:`notes on adding a dataset <add_data>`.
