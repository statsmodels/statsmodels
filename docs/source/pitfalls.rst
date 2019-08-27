Pitfalls
========

This page lists issues which may arise while using statsmodels. These 
can be the result of data-related or statistical problems, software design,
"non-standard" use of models, or edge cases. 

statsmodels provides several warnings and helper functions for diagnostic
checking (see this `blog article
<http://jpktd.blogspot.ca/2012/01/anscombe-and-diagnostic-statistics.html>`_
for an example of misspecification checks in linear regression). The coverage
is of course not comprehensive, but more warnings and diagnostic functions will
be added over time.

While the underlying statistical problems are the same for all statistical
packages, software implementations differ in the way extreme or corner cases
are handled. Please report corner cases for which the models might not work, so
we can treat them appropriately.

Repeated calls to fit with different parameters
-----------------------------------------------

Result instances often need to access attributes from the corresponding model
instance. Fitting a model multiple times with different arguments can change
model attributes. This means that the result instance may no longer point to
the correct model attributes after the model has been re-fit. 

It is therefore best practice to create separate model instances when we want
to fit a model using different fit function arguments. 

For example, this works without problem because we are not keeping the results
instance for further use ::

  mod = AR(endog)
  aic = []
  for lag in range(1,11):
      res = mod.fit(maxlag=lag)
      aic.append(res.aic)


However, when we want to hold on to two different estimation results, then it
is recommended to create two separate model instances. ::

  mod1 = RLM(endog, exog)
  res1 = mod1.fit(scale_est='mad')
  mod2 = RLM(endog, exog)
  res2 = mod2.fit(scale_est=sm.robust.scale.HuberScale())


Unidentified Parameters
-----------------------

Rank deficient exog, perfect multicollinearity
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Models based on linear models, GLS, RLM, GLM and similar, use a generalized
inverse. This means that: 

+ Rank deficient matrices will not raise an error
+ Cases of almost perfect multicollinearity or ill-conditioned design matrices might produce numerically unstable results. Users need to manually check the rank or condition number of the matrix if this is not the desired behavior
  
Note: statsmodels currently fails on the NIST benchmark case for Filip if the
data is not rescaled, see `this blog <http://jpktd.blogspot.ca/2012/03/numerical-accuracy-in-linear-least.html>`_

Incomplete convergence in maximum likelihood estimation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

In some cases, the maximum likelihood estimator might not exist, parameters
might be infinite or not unique (e.g. (quasi-)separation in models with binary
endogenous variable). Under the default settings, statsmodels will print
a warning if the optimization algorithm stops without reaching convergence.
However, it is important to know that the convergence criteria may sometimes
falsely indicate convergence (e.g. if the value of the objective function
converged but not the parameters). In general, a user needs to verify
convergence.

For binary Logit and Probit models, statsmodels raises an exception if perfect
prediction is detected. There is, however, no check for quasi-perfect
prediction.

Other Problems
--------------

Insufficient variation in the data
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

It is possible that there is insufficient variation in the data for small
datasets or for data with small groups in categorical variables. In these
cases, the results might not be identified or some hidden problems might occur.

The only currently known case is a perfect fit in robust linear model estimation.
For RLM, if residuals are equal to zero, then it does not cause an exception,
but having this perfect fit can produce NaNs in some results (scale=0 and 0/0
division) (issue #55).
