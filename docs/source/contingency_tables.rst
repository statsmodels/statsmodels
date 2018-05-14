.. currentmodule:: statsmodels.stats.contingency_tables

.. _contingency_tables:


Contingency tables
==================

Statsmodels supports a variety of approaches for analyzing contingency
tables, including methods for assessing independence, symmetry,
homogeneity, and methods for working with collections of tables from a
stratified population.

The methods described here are mainly for two-way tables.  Multi-way
tables can be analyzed using log-linear models.  Statsmodels does not
currently have a dedicated API for loglinear modeling, but Poisson
regression in :class:`statsmodels.genmod.GLM` can be used for this
purpose.

A contingency table is a multi-way table that describes a data set in
which each observation belongs to one category for each of several
variables. For example, if there are two variables, one with
:math:`r` levels and one with :math:`c` levels, then we have a
:math:`r \times c` contingency table.

(If your data includes observations that may belong to *more*
than one category at once, e.g. a 'select all that apply' survey question,
see the section :ref:`contingency_table_mrcv_section`).

A contingency table can be described in terms of the number of observations
that fall into a given cell of the table, e.g. :math:`T_{ij}` is the
number of observations that have level :math:`i` for the first variable and level :math:`j` for the second variable.

Note that each variable must have a finite number of levels
(or categories), which can be either ordered or unordered. In
different contexts, the variables defining the axes of a contingency
table may be called **categorical variables** or **factor variables**.
They may be either **nominal** (if their levels are unordered) or
**ordinal** (if their levels are ordered).

The underlying population for a contingency table is described by a
**distribution table** :math:`P_{i, j}`.  The elements of :math:`P`
are probabilities, and the sum of all elements in :math:`P` is 1.
Most methods for analyzing contingency tables use the data in :math:`T` to
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

    table = sm.stats.Table.from_data(df[["Treatment", "Improved"]])

If your table includes "multiple response" categorical variables,
e.g. a "select all that apply" question on a survey, make sure to
use the :class:`MultipleResponseTable` class instead
of the regular :class:`Table` because the tests used by
:class:`Table` assume that any given observation
appears in exactly one table cell whereas
:class:`MultipleResponseTable` uses different statistical tests that can accommodate multiple (or zero) responses per observation.

.. _contingency_table_independence_section:

Independence
------------

**Independence** is the property that the row and column factors occur
independently. **Association** is the lack of independence.  If the
joint distribution is independent, it can be written as the outer
product of the row and column marginal distributions:

.. math:: P_{ij} = \sum_k P_{ij} \cdot \sum_k P_{kj} \forall i, j

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

.. math:: \sum_k r_i c_j T_{ij}

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

    @savefig contingency_mosaic.png width=4in
    mosaic(table.table)[0]

Symmetry and homogeneity
------------------------

**Symmetry** is the property that :math:`P_{i, j} = P_{j, i}` for
every :math:`i` and :math:`j`.  **Homogeneity** is the property that
the marginal distribution of the row factor and the column factor are
identical, meaning that

.. math:: \sum_j P_{ij} = \sum_j P_{ji} \forall i

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

# TODO fix broken doc test
    data = sm.datasets.china_smoking.load()

    mat = np.asarray(data.data)
    tables = [np.reshape(x.tolist()[1:], (2, 2)) for x in mat]

    st = sm.stats.StratifiedTable(tables)
    print(st.summary())

.. _contingency_table_mrcv_section:

Multiple Response Tables
------------------------

All of the above methods for analyzing contingency tables assume
that each observation in your data set falls into exactly one category (e.g.
"treatment" or "placebo" but not both). But sometimes you may want to
analyze a factor with non-exclusive categories (e.g. "statins" AND "stent" AND
"exercise" but NOT "diet"). A factor with non-exclusive categories is often
called a "Multiple Response Categorical Variable", i.e. "MRCV".

If either or both of the factors in your contingency table allow multiple
responses, make sure to only use the MultipleResponseTable class (and not
the other :class:`Table` types
in this module). :class:`MultipleResponseTable` implements
tests that allow you to test for independence/association among
multiple response categorical variables.

Imagine a study that asked hypertensive patients which, if any, of the following
interventions they'd received in the last year: ("statins", "stent",
"exercise plan") and then recorded whether each patient was still hypertensive
5 years later. Using :class:`MultipleResponseTable`, you could assess whether any of the listed interventions is associated
with a reduction in hypertension (vs. the null hypothesis that
treated and untreated patients will have the same rates of
future hypertension).

Most of the tests in :class:`MultipleResponseTable` work as follows:

#. Treat each level of each MRCV as if it were its own separate factor with two levels (`True` if selected and `False` if not).

#. Build a separate contingency table for each combination of row-variable levels with column-variable levels.

#. Within each sub-table, use a traditional Pearson's chi-squared test to assess the probability that, for example, statins specifically are associated with reduced hypertension.

#. Intelligently combine the independence tests from each sub-table to generate a "full table" p-value that tells you how likely it is that the factors in the table are independent.

Unlike :class:`Table`, :class:`MultipleResponseTable`
needs access to the underlying data that's going to be tabulated. So you cannot simply pass in a precompiled table of counts. Instead you must use the :class:`Factor` class to wrap your data and then pass your factors into the constructor for :class:`MultipleResponseTable`.

For example:

.. ipython:: python

    import statsmodels.api as sm
    from statsmodels.datasets import presidential2016

    data = sm.datasets.presidential2016.load_pandas()
    rows_factor = sm.stats.Factor(data.data.iloc[:, :6],
                                  "expected_choice", orientation="wide")
    columns_factor = sm.stats.Factor(data.data.iloc[:, 6:11],
                                     "believe_true", orientation="wide")
    multiple_response_table = sm.stats.MultipleResponseTable([rows_factor,],
                                                             [columns_factor])

Once you have an instance of the :class:`MultipleResponseTable` class,
you can see the compiled contingency table by printing out your table:

.. ipython:: python

    print(multiple_response_table)

Once you have built your table, you can test for independence by using the
:meth:`MultipleResponseTable.test_for_independence` method:

.. ipython:: python

    result = multiple_response_table.test_for_independence()
    print(result)

You can also construct a table directly from your data (without using
:class:`Factor`) by calling :meth:`MultipleResponseTable.from_data` and
passing in a dataframe and explicitly specifying how many
columns of the dataframe are levels of the row variable and how many are
levels of the column variable:

.. ipython:: python

    multiple_response_questions = data.data.iloc[:, 6:]
    construct = sm.stats.MultipleResponseTable.from_data
    table = construct(multiple_response_questions,
                      num_cols_1st_var=5,
                      num_cols_2nd_var=5)

.. _contingency_table_MRCV_under_the_hood:

Multiple Response Under-The-Hood
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

As mentioned in the :ref:`contingency_table_independence_section` section above, traditional chi-squared tests work by:

#. assuming independence between the factors
#. calculating an "expected value" for each cell in a contingency table by multiplying the marginal probabilities (e.g. :math:`P_{treatment,cured} = P_{treatment} * P_{cured}`)
#. seeing how much the observed values in the table differ from the expected values
#. testing how likely it would be to see a deviation as large as the one we observe if the factors actually were unrelated (and thus the deviation arose from randomness in the sampling process rather than an underlying relationship between the factors).

The last step, figuring out how likely the observed deviation is, requires
assuming a probability distribution for the deviations. As the name suggests,
the chi-squared test uses the chi-squared
distribution. But the chi-squared distribution has a parameter (the "degrees of
freedom") that determines its shape. When the categories in a contingency table
are mutually exclusive (e.g. the survey questions are both 'single-select') we can
use the the total number of cells in the table as the degrees of freedom parameter.

But when the categories are not mutually exclusive (i.e. because they're
"multiple response") then the key assumptions that allowed us to use the
chi-squared distribution break down and it is no longer straight-forward
to evaluate how unlikely a set of observed deviations are.

To move forward, we need to slightly change our perspective. Instead of
thinking about a multiple response question as "one question with multiple
selections" we can think of it as "multiple questions, each with a single
selection". I.e. the question "select all that apply: (A, B, C, D)" is
equivalent to "does A apply (yes/no)? Does B apply (yes/no)? Does C apply
(yes/no)? (etc.)"

Then instead of asking "is a relief of hypertension independent of the
the interventions received?", we can ask "is relief of hypertension simultaneously independent of
whether she received a statin **and** whether she exercised **and** whether she received a stent
". This new question is called *marginal mutual independence*
(MMI), i.e. whether the answer to a single response question is simultaneously
independent of whether a respondent selected or did not select each possible
answer to a multiple response question.

Similarly, if we're comparing two (or more) multiple response questions against
each other, we ask "is whether a respondent selected option A on question
1 independent of whether she selected option A on question 2 **and** whether she
selected option B on question 2" **and** "is whether she selected option B on
question 1 independent of whether she selected option A on question 2 (etc.)"
This concept is called *simultaneous pairwise mutual independence* (SPMI),
i.e. whether each possible choice on one multiple response question is
simultaneously independent of each possible choice on a different multiple
response question.

The :class:`MultipleResponseTable` class provides functionality for evaluating marginal
mutual independence (MMI) and simultaneous pairwise mutual independence (SPMI).
Based on the factors that you pass in the constructor it will
determine which test is warranted.

The tests for MMI and SPMI all start with the same step:
build a big table comparing each combination of answers from the two questions:

.. ipython:: python

    multiple_response_table._item_response_table_for_MMI(rows_factor,
                                                         columns_factor)

Using this full item response table we can easily look one at a time at
the relationship between the single response question and the individual
choice options for the multiple response question. So for example we could
look at the column for "Trump_is_a_successful_businessman" and compare whether
or not the respondent said 'yes' to that statement versus which candidate she is
most likely to support. We can then calculate a chi-squared statistic for just
that sub-table. Then we repeat that process for each sub-table:

.. ipython:: python

    multiple_response_table._chi2s_for_MMI_item_response_table(rows_factor,
                                                               columns_factor)

And now the hard part: each individual sub-table does satisfy the requirements
for a chi-square test so we could perform a chi-square test to get a p value
for each sub-table. The p value for each sub-table reflects the likelihood that a
deviation as large as we see in the specific sub-table could appear by chance.
But what we actually care about is the likelihood that the deviations we see in
each of the sub-tables could appear all at the same time.

So we need a correction for the fact that we are considering multiple tables at
once.

The simplest correction to apply is the Bonferroni correction, where we find
the sub-table with the lowest p value and then multiply that p value by
the number of sub-tables. The result is our overall p value for the relationship
between the questions.

.. ipython:: python

    bonferroni_test = multiple_response_table._test_MMI_using_bonferroni
    results = bonferroni_test(rows_factor, columns_factor)
    table_p_value_bonferroni_corrected, pairwise_bonferroni_corrected_p_values = results
    print("Overall table p value: {}\n\n".format(table_p_value_bonferroni_corrected))
    print("Pairwise p values (likelihood of independence between single select variable and specific multi-select option):")
    pairwise_bonferroni_corrected_p_values

A well-known problem with Bonferroni correction is that it it can be too conservative (i.e. it can require an inefficiently large amount of evidence in order to reject the null hypothesis).

Another test we have available is the "Rao Scott Second Order Correction". It gives us a p value which uses fewer assumptions and is hopefully less conservative than the Bonferroni corrected p value.

.. ipython:: python

    rao_scott_test = multiple_response_table._test_MMI_using_rao_scott_2
    table_p_value_rao_scott_corrected = rao_scott_test(rows_factor, columns_factor)
    print("Overall table p value: {}\n\n".format(table_p_value_rao_scott_corrected))

In this case the Rao Scott adjusted test is much less conservative, even compared to the naive chi-squared test.

For a discussion of the relative merits of the Bonferroni vs Rao Scott
Correction, please see [2]_, [3]_, and [5]_.

Factors
^^^^^^^

The testing methods used by multiple response table tests need the underlying
data/observations because they need to be able to
form the "item response" sub-tables that compare each individual answer to
each multiple response question vs. the other factor. See :ref:`contingency_table_MRCV_under_the_hood`.

The :class:`Factor` class allows you to provide your data format that
:class:`MultipleResponseTable` can understand.

A *Factor* is one of the variables that will be on one of the axes of your
contingency table. So treatment/placebo might be one factor. Eye color might be
another factor (if you're doing an unusual experiment!)

You should also explicitly specify whether or not your data is a multiple response categorical variable.

For example:

.. ipython:: python

    columns_factor = sm.stats.Factor(dataframe=data.data.iloc[:, 6:11],
                                 name="believe_true",
                                 orientation="wide",
                                 multiple_response=True)

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
   MultipleResponseTable
   Factor
   MRCVTableNominalIndependenceResult

See also
--------

Scipy_ has several functions for analyzing contingency tables,
including Fisher's exact test which is not currently in Statsmodels.

.. _Scipy: https://docs.scipy.org/doc/scipy-0.18.0/reference/stats.html#contingency-table-functions

The :class:`MultipleResponseTable` functionality in this package is for the most part a direct translation of the functionality in the R MRCV_ library.

.. _MRCV: https://cran.r-project.org/web/packages/MRCV/index.html

References
----------

.. [1] Natalie A. Koziol and Christopher R. Bilder.
       MRCV: A package for analyzing categorical variables with multiple
       response options.
       The R Journal, 6(1):144–150, June 2014.
       CODEN ???? ISSN 2073-4859.
       URL https://cran.r-project.org/web/packages/MRCV/MRCV.pdf
.. [2] C. Bilder and T. Loughin.
       Testing for marginal independence between two categorical
       variables with multiple responses.
       Biometrics, 60(1):241–248, 2004. [p144, 146]
       The R Journal Vol. 6/1, June ISSN 2073-4859
       CONTRIBUTED RESEARCH ARTICLE 150
.. [3] C. Bilder and T. Loughin.
       Modeling association between two or more categorical variables
       that allow for multiple category choices.
       Communications in Statistics–Theory and Methods, 36(2):433–451,
       2007. [p144, 146, 149]
.. [4] C. Bilder and T. Loughin.
       Modeling multiple-response categorical data from complex surveys.
       The Canadian Journal of Statistics, 37(4):553–570, 2009. [p149]
.. [5] C. Bilder, T. Loughin, and D. Nettleton.
       Multiple marginal independence testing for pick any/c variables.
       Communications in Statistics–Simulation and Computation,
       29(4):1285–1316, 2000. [p149]