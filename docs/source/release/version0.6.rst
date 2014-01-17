:orphan:

===========
0.6 Release
===========

Release 0.6.0
=============

Release summary.

Major changes:

Addition of Generalized Estimating Equations GEE



Generalized Estimating Equations
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Generalized Estimating Equations (GEE) provide and approach to
handling dependent data in a regression analysis.  Dependent data
arise commonly in practice, such as in a longitudinal study where
repeated observations are collected on subjects. GEE can be viewed as
an extension of the generalized linear modeling (GLM) framework to the
dependent data setting.  The familiar GLM families such as the
Gaussian, Poisson, and logistic families can be used to accommodate
dependent variables with various distributions.

Here is an example of GEE Poisson regression in a data set with four
count-type repeated measures per subject, and three explanatory
covariates.

.. code-block:: python

import numpy as np
import pandas as pd
from statsmodels.genmod.generalized_estimating_equations import GEE
from statsmodels.genmod.dependence_structures import Exchangeable,\
    Independence,Autoregressive
from statsmodels.genmod.families import Poisson

data_url = "http://vincentarelbundock.github.io/Rdatasets/csv/MASS/epil.csv"
data = pd.read_csv(data_url)

fam = Poisson()
ind = Independence()
md1 = GEE.from_formula("y ~ age + trt + base", data, groups=data["subject"],\
                       covstruct=ind, family=fam)
mdf1 = md1.fit()
mdf1.summary()


The dependence structure in a GEE is treated as a nuiscance parameter
and is modeled in terms of a "working dependence structure".  The
statsmodels GEE implementation currently includes five working
dependence structures (independent, exchangeable, autoregressive,
nested, and a global odds ratio for working with categorical data).
Since the GEE estimates are not maximum likelihood estimates,
alternative approaches to some common inference procedures have been
developed.  The statsmodels GEE implementation currently provides
Wald-type standard errors and allows score tests for arbitrary
parameter contrasts.



Other important new features
----------------------------

* Other new changes can go
* In a
* Bullet list

Major Bugs fixed
----------------

* Bullet list of major bugs
* With a link to its github issue.
* Use the syntax ``:ghissue:`###```.

Backwards incompatible changes and deprecations
-----------------------------------------------

* RegressionResults.norm_resid is now a readonly property, rather than a function.

Development summary and credits
-------------------------------

A blurb about the number of changes and the contributors list.

.. note::

   Obtained by running ``git log v0.5.0..HEAD --format='* %aN <%aE>' | sed 's/@/\-at\-/' | sed 's/<>//' | sort -u``.

