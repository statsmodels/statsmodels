.. _regularization:


Regularized estimation
======================

Statsmodels supports estimation using regularized (penalized) fitting
procedures through the `fit_regularized` method.  Currently, two
approaches are supported.  In the discrete models, L1 penalized
maximum likelihood (Lasso) estimation is obtained using either cvxopt
(an optional dependency), or using sequential least squares (SLSQP).

For most other statsmodels models, `fit_regularized` uses coordinate
descent to provide the "elastic net" penalized log-likelihood
estimator.  The elastic net maxizes

L(params) + (1 - L1_wt)*\|params\|_2^2 + L1_wt*\|params\|_1,

where \|...\|_2 and \|...\|_1 are the 2 and 1 norms, respectively.
