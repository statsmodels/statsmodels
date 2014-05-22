:orphan:

===========
0.6 Release
===========

Release 0.6.0
=============

Release summary.

Major changes:

Addition of Generalized Estimating Equations GEE

Addition of Linear Mixed Effects Models (MixedLM)

Linear Mixed Effects Models
~~~~~~~~~~~~~~~~~~~~~~~~~~~

Linear Mixed Effects models are used for regression analyses involving
dependent data.  Such data arise when working with longitudinal and
other study designs in which multiple observations are made on each
subject.  Two specific mixed effects models are "random intercepts
models", where all responses in a single group are additively shifted
by a value that is specific to the group, and "random slopes models",
where the values follow a mean trajectory that is linear in observed
covariates, with both the slopes and intercept being specific to the
group.  The Statsmodels MixedLM implementation allows arbitrary random
effects design matrices to be specified for the groups, so these and
other types of random effects models can all be fit.

Here is an example of fitting a random intercepts model to data from a
longitudinal study:

data = pd.read_csv("http://vincentarelbundock.github.io/Rdatasets/csv/geepack/dietox.csv")
md = MixedLM.from_formula("Weight ~ Time", data, groups=data["Pig"])
mdf = md.fit()
print mdf.summary()

To extend this to a random slopes model, we would add the statement
`md.set_random("Time", data)` before calling the `fit` method.

The Statsmodels LME framework currently supports post-estimation
inference via Wald tests and confidence intervals on the coefficients,
profile likelihood analysis, likelihood ratio testing, and AIC.  Some
limitations of the current implementation are that it does not support
structure more complex on the residual errors (they are always
homoscedastic), and it does not support crossed random effects.  We
hope to implement these features for the next release.

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

