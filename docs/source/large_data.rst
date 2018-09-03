.. module:: statsmodels.base.distributed_estimation
.. currentmodule:: statsmodels.base.distributed_estimation

Working with "large"-ish data
=============================

    Big data is like teenage sex: everyone talks about it, nobody really knows
    how to do it, everyone thinks everyone else is doing it, so everyone claims
    they are doing it.

statsmodels and pandas generally work on data that is stored in-memory. That
means that larger datasets can be difficult to work with effectively. To
counter this, there are 2 main strategies for dealing with larger datasets.

Divide and Conquer - Distributing Jobs
--------------------------------------

If your system is capable of loading all the data, but the analysis you are
attempting to perform is slow, you might be able to perform your estimations
on horizontal slices of the data and then aggregate the models once fit.

A detailed example is available
`here <examples/notebooks/generated/distributed_estimation.html>`_.

.. autosummary::
   :toctree: generated/

   DistributedModel
   DistributedResults

Subsetting your data
--------------------

If your entire dataset is too large to store into memory, you might try storing
it in a columnar container like `Apache Paruqet <https://parquet.apache.org/>`_
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

Additionally, you can add code to the dataset object to return only a subset
of the rows until you have built a good model. Then, you can refit your
final model on more data.
