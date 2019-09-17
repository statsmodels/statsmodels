.. currentmodule:: statsmodels.stats.contingency_tables

.. _contingency_tables:


Contingency tables
==================

statsmodels supports a variety of approaches for analyzing contingency
tables, including methods for assessing independence, symmetry,
homogeneity, and methods for working with collections of tables from a
stratified population.

The methods described here are mainly for two-way tables.  Multi-way
tables can be analyzed using log-linear models.  statsmodels does not
currently have a dedicated API for loglinear modeling, but Poisson
regression in :class:`statsmodels.genmod.GLM` can be used for this
purpose.

A contingency table is a multi-way table that describes a data set in
which each observation belongs to one category for each of several
variables.  For example, if there are two variables, one with
:math:`r` levels and one with :math:`c` levels, then we have a
:math:`r \times c` contingency table.  The table can be described in
terms of the number of observations that fall into a given cell of the
table, e.g. :math:`T_{ij}` is the number of observations that have
level :math:`i` for the first variable and level :math:`j` for the
second variable.  Note that each variable must have a finite number of
levels (or categories), which can be either ordered or unordered.  In
different contexts, the variables defining the axes of a contingency
table may be called **categorical variables** or **factor variables**.
They may be either **nominal** (if their levels are unordered) or
**ordinal** (if their levels are ordered).

The underlying population for a contingency table is described by a
**distribution table** :math:`P_{i, j}`.  The elements of :math:`P`
are probabilities, and the sum of all elements in :math:`P` is 1.
Methods for analyzing contingency tables use the data in :math:`T` to
learn about properties of :math:`P`.

The :class:`statsmodels.stats.Table` is the most basic class for
working with contingency tables.  We can create a ``Table`` object
directly from any rectangular array-like object containing the
contingency table cell counts:

.. ipython:: python

    import numpy as np
    import pandas as pd
    import statsmodels.api as sm

    df = sm.datasets.get_rdataset("Arthritis", "vcd").data

    tab = pd.crosstab(df['Treatment'], df['Improved'])
    tab = tab.loc[:, ["None", "Some", "Marked"]]
    table = sm.stats.Table(tab)

Alternatively, we can pass the raw data and let the Table class
construct the array of cell counts for us:

.. ipython:: python

    data = df[["Treatment", "Improved"]]
    table = sm.stats.Table.from_data(data)


Independence
------------

**Independence** is the property that the row and column factors occur
independently. **Association** is the lack of independence.  If the
joint distribution is independent, it can be written as the outer
product of the row and column marginal distributions:

.. math::

    P_{ij} = \sum_k P_{ij} \cdot \sum_k P_{kj} \quad \text{for all} \quad  i, j

We can obtain the best-fitting independent distribution for our
observed data, and then view residuals which identify particular cells
that most strongly violate independence:

.. ipython:: python

    print(table.table_orig)
    print(table.fittedvalues)
    print(table.resid_pearson)

In this example, compared to a sample from a population in which the
rows and columns are independent, we have too many observations in the
placebo/no improvement and treatment/marked improvement cells, and too
few observations in the placebo/marked improvement and treated/no
improvement cells.  This reflects the apparent benefits of the
treatment.

If the rows and columns of a table are unordered (i.e. are nominal
factors), then the most common approach for formally assessing
independence is using Pearson's :math:`\chi^2` statistic.  It's often
useful to look at the cell-wise contributions to the :math:`\chi^2`
statistic to see where the evidence for dependence is coming from.

.. ipython:: python

    rslt = table.test_nominal_association()
    print(rslt.pvalue)
    print(table.chi2_contribs)

For tables with ordered row and column factors, we can us the **linear
by linear** association test to obtain more power against alternative
hypotheses that respect the ordering.  The test statistic for the
linear by linear association test is

.. math::

    \sum_k r_i c_j T_{ij}

where :math:`r_i` and :math:`c_j` are row and column scores.  Often
these scores are set to the sequences 0, 1, ....  This gives the
'Cochran-Armitage trend test'.

.. ipython:: python

    rslt = table.test_ordinal_association()
    print(rslt.pvalue)

We can assess the association in a :math:`r\times x` table by
constructing a series of :math:`2\times 2` tables and calculating
their odds ratios.  There are two ways to do this.  The **local odds
ratios** construct :math:`2\times 2` tables from adjacent row and
column categories.

.. ipython:: python

    print(table.local_oddsratios)
    taloc = sm.stats.Table2x2(np.asarray([[7, 29], [21, 13]]))
    print(taloc.oddsratio)
    taloc = sm.stats.Table2x2(np.asarray([[29, 7], [13, 7]]))
    print(taloc.oddsratio)

The **cumulative odds ratios** construct :math:`2\times 2` tables by
dichotomizing the row and column factors at each possible point.

.. ipython:: python

    print(table.cumulative_oddsratios)
    tab1 = np.asarray([[7, 29 + 7], [21, 13 + 7]])
    tacum = sm.stats.Table2x2(tab1)
    print(tacum.oddsratio)
    tab1 = np.asarray([[7 + 29, 7], [21 + 13, 7]])
    tacum = sm.stats.Table2x2(tab1)
    print(tacum.oddsratio)

A mosaic plot is a graphical approach to informally assessing
dependence in two-way tables.

.. ipython:: python

    from statsmodels.graphics.mosaicplot import mosaic
    fig, _ = mosaic(data, index=["Treatment", "Improved"])


Symmetry and homogeneity
------------------------

**Symmetry** is the property that :math:`P_{i, j} = P_{j, i}` for
every :math:`i` and :math:`j`.  **Homogeneity** is the property that
the marginal distribution of the row factor and the column factor are
identical, meaning that

.. math::

    \sum_j P_{ij} = \sum_j P_{ji} \forall i

Note that for these properties to be applicable the table :math:`P`
(and :math:`T`) must be square, and the row and column categories must
be identical and must occur in the same order.

To illustrate, we load a data set, create a contingency table, and
calculate the row and column margins.  The :class:`Table` class
contains methods for analyzing :math:`r \times c` contingency tables.
The data set loaded below contains assessments of visual acuity in
people's left and right eyes.  We first load the data and create a
contingency table.

.. ipython:: python

    df = sm.datasets.get_rdataset("VisualAcuity", "vcd").data
    df = df.loc[df.gender == "female", :]
    tab = df.set_index(['left', 'right'])
    del tab["gender"]
    tab = tab.unstack()
    tab.columns = tab.columns.get_level_values(1)
    print(tab)

Next we create a :class:`SquareTable` object from the contingency
table.

.. ipython:: python

    sqtab = sm.stats.SquareTable(tab)
    row, col = sqtab.marginal_probabilities
    print(row)
    print(col)


The ``summary`` method prints results for the symmetry and homogeneity
testing procedures.

.. ipython:: python

    print(sqtab.summary())

If we had the individual case records in a dataframe called ``data``,
we could also perform the same analysis by passing the raw data using
the ``SquareTable.from_data`` class method.

::

    sqtab = sm.stats.SquareTable.from_data(data[['left', 'right']])
    print(sqtab.summary())


A single 2x2 table
------------------

Several methods for working with individual 2x2 tables are provided in
the :class:`sm.stats.Table2x2` class.  The ``summary`` method displays
several measures of association between the rows and columns of the
table.

.. ipython:: python

    table = np.asarray([[35, 21], [25, 58]])
    t22 = sm.stats.Table2x2(table)
    print(t22.summary())

Note that the risk ratio is not symmetric so different results will be
obtained if the transposed table is analyzed.

.. ipython:: python

    table = np.asarray([[35, 21], [25, 58]])
    t22 = sm.stats.Table2x2(table.T)
    print(t22.summary())


Stratified 2x2 tables
---------------------

Stratification occurs when we have a collection of contingency tables
defined by the same row and column factors.  In the example below, we
have a collection of 2x2 tables reflecting the joint distribution of
smoking and lung cancer in each of several regions of China.  It is
possible that the tables all have a common odds ratio, even while the
marginal probabilities vary among the strata.  The 'Breslow-Day'
procedure tests whether the data are consistent with a common odds
ratio.  It appears below as the `Test of constant OR`.  The
Mantel-Haenszel procedure tests whether this common odds ratio is
equal to one.  It appears below as the `Test of OR=1`.  It is also
possible to estimate the common odds and risk ratios and obtain
confidence intervals for them.  The ``summary`` method displays all of
these results.  Individual results can be obtained from the class
methods and attributes.

.. ipython:: python

    data = sm.datasets.china_smoking.load_pandas()

    mat = np.asarray(data.data)
    tables = [np.reshape(x.tolist(), (2, 2)) for x in mat]

    st = sm.stats.StratifiedTable(tables)
    print(st.summary())


Module Reference
----------------

.. module:: statsmodels.stats.contingency_tables
   :synopsis: Contingency table analysis

.. currentmodule:: statsmodels.stats.contingency_tables

.. autosummary::
   :toctree: generated/

   Table
   Table2x2
   SquareTable
   StratifiedTable
   mcnemar
   cochrans_q

See also
--------

Scipy_ has several functions for analyzing contingency tables,
including Fisher's exact test which is not currently in statsmodels.

.. _Scipy: https://docs.scipy.org/doc/scipy-0.18.0/reference/stats.html#contingency-table-functions
