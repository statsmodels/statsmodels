:orphan:

.. _dataset_proposal:

Dataset for statmodels: design proposal
===============================================

One of the thing numpy/scipy is missing now is a set of datasets, available for
demo, courses, etc. For example, R has a set of dataset available at the core.

The expected usage of the datasets are the following:

        - examples, tutorials for model usage
        - testing of model usage vs. other statistical packages

That is, a dataset is not only data, but also some meta-data. The goal of this
proposal is to propose common practices for organizing the data, in a way which
is both straightforward, and does not prevent specific usage of the data.


Background
----------

This proposal was adapted from David Cournapeau's original proposal for a
datasets package for scipy and the learn scikit.  It has been adapted for use
in the statsmodels scikit.  The structure of the datasets itself, while
specific to statsmodels, should be general enough such that it might be used
for other types of data (e.g., in the learn scikit or scipy itself).

Organization
------------

Each dataset is a directory in the `datasets` directory and defines a python
package (e.g. has the __init__.py file). Each package is expected to define the
function load, returning the corresponding data. For example, to access datasets
data1, you should be able to do::

  >>> from statsmodels.datasets.data1 import load
  >>> d = load() # -> d is a Dataset object, see below

The `load` function is expected to return the `Dataset` object, which has certain
common attributes that make it readily usable in tests and examples. Load can do
whatever it wants: fetching data from a file (python script, csv file, etc...),
from the internet, etc.  However, it is strongly recommended that each dataset
directory contain a csv file with the dataset and its variables in the same form
as returned by load so that the dataset can easily be loaded into other
statistical packages.  In addition, an optional (though recommended) sub-directory
src should contain the dataset in its original form if it was "cleaned" (ie.,
variable transformations) in order to put it into the format needed for statsmodels.
Some special variables must be defined for each package, containing a Python string:

    - COPYRIGHT: copyright informations
    - SOURCE: where the data are coming from
    - DESCHOSRT: short description
    - DESCLONG: long description
    - NOTE: some notes on the datasets.

See `datasets/data_template.py` for more information.

Format of the data
------------------

This is strongly suggested a practice for the `Dataset` object returned by the
load function.  Instead of using classes to provide meta-data, the Bunch pattern
is used.

::

  class Bunch(dict):
    def __init__(self,**kw):
        dict.__init__(self,kw)
        self.__dict__ = self

See this `Reference <http://code.activestate.com/recipes/52308-the-simple-but-handy-collector-of-a-bunch-of-named/>`_

In practice, you can use ::

  >>> from statsmodels.datasets import Dataset

as the default collector as in `datasets/data_template.py`.

The advantage of the Bunch pattern is that it preserves look-up by attribute.
The key goals are:

    - For people who just want the data, there is no extra burden
    - For people who need more, they can easily extract what they need from
      the returned values. Higher level abstractions can be built easily
      from this model.
    - All possible datasets should fit into this model.

For the datasets to be useful in statsmodels the Dataset object
returned by load has the following conventions and attributes:

    - Calling the object itself returns the plain ndarray of the full dataset.
    - `data`: A recarray containing the actual data.  It is assumed
      that all of the data can safely be cast to a float at this point.
    - `raw_data`: This is the plain ndarray version of 'data'.
    - `names`: this returns data.dtype.names so that name[i] is the i-th
      column in 'raw_data'.
    - `endog`: this value is provided for convenience in tests and examples
    - `exog`: this value is provided for convenience in tests and examples
    - `endog_name`: the name of the endog attribute
    - `exog_name`: the names of the exog attribute


This contains enough information to get all useful information through
introspection and simple functions. Further, attributes are easily added that
may be useful for other packages.


Adding a dataset
----------------

See the :ref:`notes on adding a dataset <add_data>`.


Example Usage
-------------

::

  >>> from statsmodels import datasets
  >>> data = datasets.longley.load()


Remaining problems:
-------------------


    - If the dataset is big and cannot fit into memory, what kind of API do
      we want to avoid loading all the data in memory ? Can we use memory
      mapped arrays ?
    - Missing data: I thought about subclassing both record arrays and
      masked arrays classes, but I don't know if this is feasable, or even
      makes sense. I have the feeling that some Data mining software use
      Nan (for example, weka seems to use float internally), but this
      prevents them from representing integer data.
    - What to do with non-float data, i.e., strings or categorical variables?


Current implementation
----------------------

An implementation following the above design is available in `statsmodels`.


Note
----

Although the datasets package emerged from the learn package, we try to keep it
independant from everything else, that is once we agree on the remaining
problems and where the package should go, it can easily be put elsewhere
without too much trouble. If there is interest in re-using the datasets package,
please contact the developers on the `mailing list <http://groups.google.com/group/pystatsmodels?hl=en>`_.
