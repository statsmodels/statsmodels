

Pitfalls and Caution required
=============================

The following lists some possible problems in the use of statsmodels. Some are
the results of the current design, some are statistical problems, that the user
needs to be aware of since statsmodels does by default not check whether a
model is appropriate for a given dataset, for example edge cases. The
implementation and treatment of this will likely change in future versions.
Also for the cases below where the model or estimation procedure might not be
appropriate for the data, we will add more warnings and helper function to
allow users to do the diagnostic checking. For example, the summary method
of some results, checks some known potential problems and informs the user.

Please report corner cases for which the models might not work, so we can add
them.

Repeated calls to fit with different parameters
-----------------------------------------------

In general, statsmodels is designed that only one result instance is associated
with a given model instance. The result instance needs to be able to access the
data and other attributes of the model instance.

However, when fit is called repeatedly with different arguments, then a new
result instance is created that refers to the same model instance. Some
attributes of the model instance are changed during the fit. This implies that
the model attributes are correct for the result instance from last call to fit,
but previous result instances might not access the correct attributes anymore.

As an example,

this works without problems ::

  mod = AR(endog)
  aic = []
  for lag in range(1,11):
      res = mod.fit(lag=lag)
      aic.append(res.aic)


However, when we want to hold on to two different estimation results, then it
is recommended to create two separate model instances.

  mod1 = RLM(endog, exog)
  res1 = mod1.fit(scale='mad')
  mod2 = RLM(endog, exog)
  res2 = mod1.fit(scale_est='stand_mad')

TODO: maybe use ARMA with and without trend as example ?


Unidentified Parameters
-----------------------

Rank deficient exog, perfect multicollinearity
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Models based on linear models, GLS, RLM, GLM and similar, use a generalized
inverse and therefore do not raise an error if the design matrix does not have
full rank. Also in the case of almost perfect multicollinearity or of
ill-conditioned design matrices, the estimation in statsmodels might produce
results that are numerically not stable. If this is not the desired behavior
the user needs to check that the rank or condition number of the design matrix.

Incomplete convergence in maximum likelihood estimation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

As often with numerical optimization routines, it is possible that for some
dataset, finding the solution might have problems. One example for this is
complete (quasi-) separation in models with binary endogenous variable, for
example, discrete Logit and Probit. In these cases, the maximum likelihood
estimator might not exist, parameters might be infinite or not unique. In some
cases the optimization routine will stop without convergence, which will be
printed to the screen in the default setting. In general a user needs to
verify convergence, however it is possible to have convergence criteria that
indicate convergence, e.g. in the value of the objective function, even if
parameters do not have converged.


Other Problems
--------------

Insufficient variation in the data
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

It is possible that there is insufficient variation in the data for small
datasets, or for data with small groups in categorical variables. In these
cases the results might not be identified or some hidden problems might occur.

The only known case currently is a perfect fit in robust linear model estimation.
For RLM having a zero residual does not cause an exception, but having a
perfect fit can produce NaNs in some results (scale=0 and 0/0 division)
(issue #55).
