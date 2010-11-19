.. currentmodule:: scikits.statsmodels.sandbox.regression.gmm


.. _gmm_techn1:

Technical Documentation
=======================

Introduction
------------

Generalized Method of Moments is an extension of the Method of Moments
if there are more moment conditions than parameters that are estimated.

simple example


General Structure and Implementation
------------------------------------

The main class for GMM estimation, makes little assumptions about the
moment conditions. It is designed for the general case when moment
conditions are given as function by the user.

::

 def momcond(params)

which should return a two dimensional array with observation in rows
and moment conditions in columns. Denote this function by `$g(\theta)$`. Then
the GMM estimator is given as the solution to the maximization problem:

..math: max_{\theta) g(theta)' W g(theta)  (1)

The weighting matrix can be estimated in several different ways. The
basic method `fitgmm` takes the weighting matrix as argument or if it is
not given takes the identity matrix and maximizes (1)
taking W as given. Since the optimizing functions solve minimization problems,
we usually minimizes the negative of the objective function.
`fit_iterative` calculates the optimal weighting matrix and maximizes the
criterion function in alternating steps. The number of iterations can
be given as an argument to this fit method. The optimal weighting matrix,
which is the covariance matrix of the moment conditions, can be estimated
in different ways. Kernel and shrinkage estimators are planned but not yet
implemented. TODO

The GMM class itself does not define any moment conditions. To get an
estimator for given moment conditions, GMM needs to be subclassed.
The basic structure of writing new models based on
the generic MLE or GMM framework and subclassing is described in
`extending.rst` (TODO: link)

As an example
