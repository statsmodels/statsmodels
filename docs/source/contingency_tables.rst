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

Symmetry
========

Symmetry is the property that `P[i, j] = P[j, i]` for every `i` and
`j`.  Note that for this to make sense `P` (and `T`) must be square,
and the row and column categories must be equivalent and occur in the
same order.


Homogeneity
===========

Homogeneity is the property that the marginal distribution of the row
 factor and the column factor are identical:

.. math::

\sum_j P_{ij} = \sum_j P_{ji} \forall i

This property also only makes sense for square tables with equivalent
row and column categories.

The following code example illustrates how to obtain the homogeneity
and symmetry test results.

.. ipython::

    data = sm.datasets.vision_ordnance.load()
    df = data.data
    tab = df.set_index(['left', 'right'])
    tab = tab.unstack()

    st = sm.stats.TableSymmetry(tab)
    print(st.summary())


Note that the data in the above example have quite similar row and
column margins, and the joint table appears quite symmetric.  Due to
the large sample size, we have power to detect small deviations from
perfect symmetry and homogeneity.


Independence
============

Independence is the property that the row and column factors occur
independently.  *Association* is the lack of independence.  If the
joint distribution is independent, it can be written as the outer
product of the row and column marginal distributions:

.. math::

P_{ij} = \sum_k P_{ij} \cdot \sum_k P_{kj} \forall i, j

This property can hold for either square or rectangular tables, and
the categories do not need to be related in any way.

Stratified tables
=================

Stratification refers to a collection of contingency tables, usually
with the same row and column factors.  For example, if we are
interested in the relationship between smoking and lung cancer, we may
have a collection of 2x2 tables reflecting the joint distribution of
smoking and lung cancer in each of several regions.  It is possible to
test whether the tables have a common odds ratio, whether the common
odds ratio differs from 1, and to estimate the common odds ratio and
the common risk ratio.

..

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
