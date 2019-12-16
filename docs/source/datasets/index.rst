.. _datasets:

.. currentmodule:: statsmodels.datasets

.. ipython:: python
   :suppress:

   import numpy as np
   np.set_printoptions(suppress=True)

The Datasets Package
====================

``statsmodels`` provides data sets (i.e. data *and* meta-data) for use in
examples, tutorials, model testing, etc.

Using Datasets from Stata
-------------------------

.. autosummary::
   :toctree: ./

   webuse

Using Datasets from R
---------------------

The `Rdatasets project <https://vincentarelbundock.github.io/Rdatasets/>`__ gives access to the datasets available in R's core datasets package and many other common R packages. All of these datasets are available to statsmodels by using the :func:`get_rdataset` function. The actual data is accessible by the ``data`` attribute. For example:

.. ipython:: python

   import statsmodels.api as sm
   duncan_prestige = sm.datasets.get_rdataset("Duncan", "carData")
   print(duncan_prestige.__doc__)
   duncan_prestige.data.head(5)


R Datasets Function Reference
-----------------------------


.. autosummary::
   :toctree: ./

   get_rdataset
   get_data_home
   clear_data_home


Available Datasets
------------------

.. toctree::
   :maxdepth: 1
   :glob:

   generated/*

Usage
-----

Load a dataset:

.. ipython:: python

   import statsmodels.api as sm
   data = sm.datasets.longley.load_pandas()

The `Dataset` object follows the bunch pattern. The full dataset is available
in the ``data`` attribute.

.. ipython:: python

   data.data

Most datasets hold convenient representations of the data in the attributes `endog` and `exog`:

.. ipython:: python

   data.endog.iloc[:5]
   data.exog.iloc[:5,:]

Univariate datasets, however, do not have an `exog` attribute.

Variable names can be obtained by typing:

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
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

For many users it may be preferable to get the datasets as a pandas DataFrame or
Series object. Each of the dataset modules is equipped with a ``load_pandas``
method which returns a ``Dataset`` instance with the data readily available as pandas objects:

.. ipython:: python

   data = sm.datasets.longley.load_pandas()
   data.exog
   data.endog

The full DataFrame is available in the ``data`` attribute of the Dataset object

.. ipython:: python

   data.data


With pandas integration in the estimation classes, the metadata will be attached
to model results:

.. ipython:: python
   :okwarning:

   y, x = data.endog, data.exog
   res = sm.OLS(y, x).fit()
   res.params
   res.summary()

Extra Information
^^^^^^^^^^^^^^^^^

If you want to know more about the dataset itself, you can access the
following, again using the Longley dataset as an example ::

    >>> dir(sm.datasets.longley)[:6]
    ['COPYRIGHT', 'DESCRLONG', 'DESCRSHORT', 'NOTE', 'SOURCE', 'TITLE']

Additional information
----------------------

* The idea for a datasets package was originally proposed by David Cournapeau.
* To add datasets, see the :ref:`notes on adding a dataset <add_data>`.
