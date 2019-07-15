.. module:: statsmodels.base.distributed_estimation
.. currentmodule:: statsmodels.base.distributed_estimation

Working with Large Data Sets
============================

Big data is something of a buzzword in the modern world. While statsmodels
works well with small and moderately-sized data sets that can be loaded in
memory--perhaps tens of thousands of observations--use cases exist with
millions of observations or more. Depending your use case, statsmodels may or
may not be a sufficient tool.

statsmodels and most of the software stack it is written on operates in
memory. Resultantly, building models on larger data sets can be challenging
or even impractical. With that said, there are 2 general strategies for
building models on larger data sets with statsmodels.

Divide and Conquer - Distributing Jobs
--------------------------------------

If your system is capable of loading all the data, but the analysis you are
attempting to perform is slow, you might be able to build models on horizontal
slices of the data and then aggregate the individual models once fit.

A current limitation of this approach is that it generally does not support
`patsy <https://patsy.readthedocs.io/en/latest/>`_ so constructing your
design matrix (known as `exog`) in statsmodels, is a little challenging.

A detailed example is available
`here <examples/notebooks/generated/distributed_estimation.html>`_.

.. autosummary::
   :toctree: generated/

   DistributedModel
   DistributedResults

Subsetting your data
--------------------

If your entire data set is too large to store in memory, you might try storing
it in a columnar container like `Apache Parquet <https://parquet.apache.org/>`_
or `bcolz <http://bcolz.blosc.org/en/latest/>`_. Using the patsy formula
interface, statsmodels will use the `__getitem__` function (i.e. data['Item'])
to pull only the specified columns.

.. code-block:: python

    import pyarrow as pa
    import pyarrow.parquet as pq
    import statsmodels.formula.api as smf

    class DataSet(dict):
        def __init__(self, path):
            self.parquet = pq.ParquetFile(path)

        def __getitem__(self, key):
            try:
                return self.parquet.read([key]).to_pandas()[key]
            except:
                raise KeyError

    LargeData = DataSet('LargeData.parquet')

    res = smf.ols('Profit ~ Sugar + Power + Women', data=LargeData).fit()

Additionally, you can add code to this example `DataSet` object to return only
a subset of the rows until you have built a good model. Then, you can refit
your final model on more data.
