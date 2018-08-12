if __name__ == "__main__":
    import statsmodels.api as sm
    data = sm.datasets.longley.load()
    # data.exog = add_constant(data.exog)
    GLMmod = GLM(data.endog, data.exog).fit()
    GLMT = GLMmod.summary(returns='tables')
    # GLMT[0].extend_right(GLMT[1])
    # print(GLMT[0])
    # print(GLMT[2])
    GLMTp = GLMmod.summary(title='Test GLM')

    """
From Stata
. webuse beetle
. glm r i.beetle ldose, family(binomial n) link(cloglog)

Iteration 0:   log likelihood = -79.012269
Iteration 1:   log likelihood =  -76.94951
Iteration 2:   log likelihood = -76.945645
Iteration 3:   log likelihood = -76.945645

Generalized linear models                          No. of obs      =        24
Optimization     : ML                              Residual df     =        20
                                                   Scale parameter =         1
Deviance         =  73.76505595                    (1/df) Deviance =  3.688253
Pearson          =   71.8901173                    (1/df) Pearson  =  3.594506

Variance function: V(u) = u*(1-u/n)                [Binomial]
Link function    : g(u) = ln(-ln(1-u/n))           [Complementary log-log]

                                                   AIC             =   6.74547
Log likelihood   = -76.94564525                    BIC             =  10.20398

------------------------------------------------------------------------------
             |                 OIM
           r |      Coef.   Std. Err.      z    P>|z|     [95% Conf. Interval]
-------------+----------------------------------------------------------------
      beetle |
          2  |  -.0910396   .1076132    -0.85   0.398    -.3019576    .1198783
          3  |  -1.836058   .1307125   -14.05   0.000     -2.09225   -1.579867
             |
       ldose |   19.41558   .9954265    19.50   0.000     17.46458    21.36658
       _cons |  -34.84602    1.79333   -19.43   0.000    -38.36089   -31.33116
------------------------------------------------------------------------------
"""

    # NOTE: wfs dataset has been removed due to a licensing issue
    # example of using offset
    # data = sm.datasets.wfs.load()
    # get offset
    # offset = np.log(data.exog[:,-1])
    # exog = data.exog[:,:-1]

    # convert dur to dummy
    # exog = sm.tools.categorical(exog, col=0, drop=True)
    # drop reference category
    # convert res to dummy
    # exog = sm.tools.categorical(exog, col=0, drop=True)
    # convert edu to dummy
    # exog = sm.tools.categorical(exog, col=0, drop=True)
    # drop reference categories and add intercept
    # exog = sm.add_constant(exog[:,[1,2,3,4,5,7,8,10,11,12]])

    # endog = np.round(data.endog)
    # mod = sm.GLM(endog, exog, family=sm.families.Poisson()).fit()

    # res1 = GLM(endog, exog, family=sm.families.Poisson(),
    #                         offset=offset).fit(tol=1e-12, maxiter=250)
    # exposuremod = GLM(endog, exog, family=sm.families.Poisson(),
    #                   exposure = data.exog[:,-1]).fit(tol=1e-12,
    #                                                   maxiter=250)
    # assert(np.all(res1.params == exposuremod.params))
if __name__ == "__main__":
    import statsmodels.api as sm
    data = sm.datasets.longley.load()
    # data.exog = add_constant(data.exog)
    GLMmod = GLM(data.endog, data.exog).fit()
    GLMT = GLMmod.summary(returns='tables')
    # GLMT[0].extend_right(GLMT[1])
    # print(GLMT[0])
    # print(GLMT[2])
    GLMTp = GLMmod.summary(title='Test GLM')

    """
From Stata
. webuse beetle
. glm r i.beetle ldose, family(binomial n) link(cloglog)

Iteration 0:   log likelihood = -79.012269
Iteration 1:   log likelihood =  -76.94951
Iteration 2:   log likelihood = -76.945645
Iteration 3:   log likelihood = -76.945645

Generalized linear models                          No. of obs      =        24
Optimization     : ML                              Residual df     =        20
                                                   Scale parameter =         1
Deviance         =  73.76505595                    (1/df) Deviance =  3.688253
Pearson          =   71.8901173                    (1/df) Pearson  =  3.594506

Variance function: V(u) = u*(1-u/n)                [Binomial]
Link function    : g(u) = ln(-ln(1-u/n))           [Complementary log-log]

                                                   AIC             =   6.74547
Log likelihood   = -76.94564525                    BIC             =  10.20398

------------------------------------------------------------------------------
             |                 OIM
           r |      Coef.   Std. Err.      z    P>|z|     [95% Conf. Interval]
-------------+----------------------------------------------------------------
      beetle |
          2  |  -.0910396   .1076132    -0.85   0.398    -.3019576    .1198783
          3  |  -1.836058   .1307125   -14.05   0.000     -2.09225   -1.579867
             |
       ldose |   19.41558   .9954265    19.50   0.000     17.46458    21.36658
       _cons |  -34.84602    1.79333   -19.43   0.000    -38.36089   -31.33116
------------------------------------------------------------------------------
"""

    # NOTE: wfs dataset has been removed due to a licensing issue
    # example of using offset
    # data = sm.datasets.wfs.load()
    # get offset
    # offset = np.log(data.exog[:,-1])
    # exog = data.exog[:,:-1]

    # convert dur to dummy
    # exog = sm.tools.categorical(exog, col=0, drop=True)
    # drop reference category
    # convert res to dummy
    # exog = sm.tools.categorical(exog, col=0, drop=True)
    # convert edu to dummy
    # exog = sm.tools.categorical(exog, col=0, drop=True)
    # drop reference categories and add intercept
    # exog = sm.add_constant(exog[:,[1,2,3,4,5,7,8,10,11,12]])

    # endog = np.round(data.endog)
    # mod = sm.GLM(endog, exog, family=sm.families.Poisson()).fit()

    # res1 = GLM(endog, exog, family=sm.families.Poisson(),
    #                         offset=offset).fit(tol=1e-12, maxiter=250)
    # exposuremod = GLM(endog, exog, family=sm.families.Poisson(),
    #                   exposure = data.exog[:,-1]).fit(tol=1e-12,
    #                                                   maxiter=250)
    # assert(np.all(res1.params == exposuremod.params))
