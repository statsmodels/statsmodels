:orphan:

===========
0.7 Release
===========

Release 0.7.0
=============

Release summary

The following major new features appear in this version.

Principal Component Analysis
----------------------------

A new class-based Principal Component Analysis has been added.  This
class replaces the function-based PCA that previously existed in the
sandbox.  This change bring a number of new features, including:

* Options to control the standardization (demeaning/studentizing)
* Scree plotting
* Information criteria for selecting the number of factors
* R-squared plots to assess component fit
* NIPALS implementation when only a small number of components are required and the dataset is large
* Missing-value filling using the EM algorithm

.. code-block:: python

   import statsmodels.api as sm
   from statsmodels.tools.pca import PCA

   data = sm.datasets.fertility.load_pandas().data

   columns = map(str, range(1960, 2012))
   data.set_index('Country Name', inplace=True)
   dta = data[columns]
   dta = dta.dropna()

   pca_model = PCA(dta.T, standardize=False, demean=True)
   pca_model.plot_scree()

*Note* : A function version is also available which is compatible with the
call in the sandbox.  The function version is just a thin wrapper around the
class-based PCA implementation.

Regression graphics for GLM/GEE
-------------------------------

Added variable plots, partial residual plots, and CERES residual plots
are available for GLM and GEE models by calling the methods
`plot_added_variable`, `plot_partial_residuals`, and
`plot_ceres_residuals` that are attached to the results classes.

Other important new features
----------------------------

* Bullet
* List
* of
* new
* features

Major Bugs fixed
----------------

* Bullet
* list
* use :ghissue:`XXX` to link to issue.

Backwards incompatible changes and deprecations
-----------------------------------------------

* List backwards incompatible changes

Development summary and credits
-------------------------------

A blurb about the number of changes and the contributors list.

.. note::

   Obtained by running ``git log v0.6.0..HEAD --format='* %aN <%aE>' | sed 's/@/\-at\-/' | sed 's/<>//' | sort -u``.

