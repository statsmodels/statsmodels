.. currentmodule:: statsmodels.stats.contingency_tables


.. _contingency_tables:

Contingency tables
==================

Support for analysis of contingency tables includes methods for
assessing symmetry, homogeneity, association, and methods for working
with sets of stratified tables.

A contingency table is a multi-way table that describes a data set in
which each observation belongs to one category for each of several
variables.  For example, if there are two variables, one with `r`
levels and one with `c` levels, then we have a `r x c` contingency
table.  The table can be described in terms of the number of
observations that fall into a given 'cell', e.g. `T[i, j]` is the
number of observations that have level `i` for the first variable and
level `j` for the second variable.  Note that each variable must have
a finite number of levels (or categories), which can be either ordered
or unordered.

The underlying population for a contingency table is described by a
'distribution table' `P[i, j]`, where the sum of all elements in `P`
is 1.  Methods for analyzing contingency tables use the data in `T` to
learn about properties of `P`.

Symmetry and homogeneity
========================

Symmetry is the property that `P[i, j] = P[j, i]` for every `i` and
`j`.  Note that for this to make sense `P` (and `T`) must be square,
and the row and column categories must be equivalent and occur in the
same order.

Homogeneity is the property that the marginal distribution of the row
 factor and the column factor are identical:

.. math::

\sum_j P_{ij} = \sum_j P_{ji} \forall i

This property also only makes sense for square tables with equivalent
row and column categories.  To illustrate, we load a data set, create
a contingency table, and calculate the row and column margins.

.. ipython::

    data = sm.datasets.vision_ordnance.load()
    df = data.data
    tab = df.set_index(['left', 'right'])
    tab = tab.unstack()
    print(tab)
    print(tab.mean(0))
    print(tab.mean(1))

One way to obtain the homogeneity and symmetry test results is to pass
the contingency table into the `TableSymmetry` class directly.

.. ipython::

    st = sm.stats.TableSymmetry(tab)
    print(st.summary())

If we have the individual case records in a DataFrame called `data`,
then we can perform the same analysis by passing the raw data using
the ``TableSymmetry.from_data`` classmethod.  We also need to pass the
names of the columns of `data` that contain the row and column
factors.

    st = sm.stats.TableSymmetry.from_data('left', 'right', data)
    print(st.summary())

Note that the data used in the above examples have quite similar row
and column margins, and the joint table appears quite symmetric.  Due
to the large sample size, we have power to detect small deviations
from perfect symmetry and homogeneity.


Independence
============

Independence is the property that the row and column factors occur
independently. "Association" is the lack of independence.  If the
joint distribution is independent, it can be written as the outer
product of the row and column marginal distributions:

.. math::

P_{ij} = \sum_k P_{ij} \cdot \sum_k P_{kj} \forall i, j

This property can hold for either square or rectangular tables, and
the categories do not need to be related in any way.

If the rows and columns of a table are unordered (i.e.\ are nominal
factors), then the most common approach to assessing association is
using Pearson's chi^2 statistic.  For tables with ordered row and
column factors, we can us the "linear by linear association test" to
obtain more power against alternative hypotheses that respect the
ordering.

The test statistic for the linear by linear association test is

.. math::

\sum_k r_i c_j T_{ij}

where :math:`r_i` and :math:`c_j` are row and column 'scores'.
Usually these scores are set to the sequences 0, 1, ....  This gives
the 'Cochran-Armitage trend test'.

A famous dataset originally reported by Mack et al. (NEJM 1976)
records estrogen exposure within matched case/control pairs, where the
cases have endometrial cancer.  There are four ordinal levels of
estrogen exposure.  The following contingency table contains the
number of case/control pairs with each possible combination of
estrogen exposure levels (the rows are the cases and the columns are
the controls).

.. ipython::

    table = [[6, 2, 3, 1], [9, 4, 2, 1], [9, 2, 3, 1], [12, 1, 2, 1]]
    table = np.asarray(table)

If we want to conduct a formal test of independence treating the row
and column categories as nominal, we can use the Pearson chi^2 test:

.. ipython::
    lbl = sm.stats.TableAssociation(table, method='chi2')
    print(lbl.pvalue)

If instead we want to utilize the ordinal information in the row and
column factors (the estrogen levels), we can use the linear-by-linear
association test.  By default, the scores are equally spaced.

.. ipython::
    lbl = sm.stats.TableAssociation(table, method='lbl')
    print(lbl.pvalue)

A mosaic plot is a graphical approach to assessing dependence in
two-way tables.

    from statsmodels.graphics.mosaicplot import mosaic
    mosaic(data)

Stratified tables
=================

Stratification refers to a collection of contingency tables with the
same row and column factors.  For example, if we are interested in the
relationship between smoking and lung cancer, we may have a collection
of 2x2 tables reflecting the joint distribution of smoking and lung
cancer in each of several regions.  It is possible to test whether the
tables have a common odds ratio, whether the common odds ratio differs
from 1, and to estimate the common odds ratio and the common risk
ratio.

To illustrate, we load a dataset containing data on smoking and lung
cancer incidence in eight cities in China.

.. ipython::

    data = sm.datasets.china_smoking.load()

    # Create a list of tables
    mat = np.asarray(data.data)
    tables = [np.reshape(x, (2, 2)) for x in mat]

    st = sm.stats.StratifiedTables(tables)
    print(st.summary())


Module Reference
----------------

.. currentmodule:: statsmodels.stats.contingency_tables

.. autosummary::
   :toctree: generated/

   homogeneity
   symmetry
   ordinal_association
   StratifiedTables
   mcnemar
   cochrans_q

See also
--------

Scipy_ has several functions for analyzing contingency tables,
including Fisher's exact test which is not currently in Statsmodels.

.. _Scipy http://docs.scipy.org/doc/scipy-0.14.0/reference/stats.html#contingency-table-functions
