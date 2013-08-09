# -*- coding: utf-8 -*-
"""
Conditional logit

Sources: sandbox-statsmodels:runmnl.py

General References
--------------------

Greene, W. `Econometric Analysis`. Prentice Hall, 5th. edition. 2003.
Train, K. `Discrete Choice Methods with Simulation`.
    Cambridge University Press. 2003
--------------------

# TODO:
    adapt it to the structure of others discrete models
        (source:discrete_model.py)
    add dataset Mode choice
        (http://statsmodels.sourceforge.net/devel/datasets/
            dataset_proposal.html#dataset-proposal)
    add example
        (http://statsmodels.sourceforge.net/devel/dev/examples.html)
    add test
    send patsy proposal for data handle (see Issue #941)

"""
import numpy as np
from statsmodels.base.model import GenericLikelihoodModel

class CLogit(GenericLikelihoodModel):

    '''
    Conditional Logit

    Parameters
    ----------
    endog : array (nobs,nchoices)
        dummy encoding of realized choices
    exog_bychoices : list of arrays
        explanatory variables, one array of exog for each choice. Variables
        with common coefficients have to be first in each array
    ncommon : int
        number of explanatory variables with common coefficients

    Notes
    -----

    Utility for choice j is given by

        $V_j = X_j * beta + Z * gamma_j$

    where X_j contains generic variables (terminology Hess) that have the same
    coefficient across choices, and Z are variables, like individual-specific
    variables that have different coefficients across variables.

    If there are choice specific constants, then they should be contained in Z.
    For identification, the constant of one choice should be dropped.
    '''

    def __init__(self, endog, exog_bychoices, ncommon, **kwds):
        super(CLogit, self).__init__(endog, **kwds)
        self.endog = endog
        self.exog_bychoices = exog_bychoices
        self.ncommon = ncommon
        self.nobs, self.nchoices = endog.shape

        # TODO: rename beta to params
        betaind = [exog_bychoices[ii].shape[1]-ncommon for ii in range(self.nchoices)]
        zi = np.r_[[ncommon], ncommon + np.array(betaind).cumsum()]
        self.zi = zi
        z = np.arange(len(zi)+ncommon)

        beta_indices = [np.r_[np.arange(ncommon), z[zi[ii]:zi[ii+1]]]
                       for ii in range(len(zi)-1)]
        # beta_indices = [array([3, 0, 1, 2]), array([4, 0, 1]), array([5, 0, 1]), array([1])]
        self.beta_indices = beta_indices
        # print (beta_indices)

        params_num = []                            # num de params to estimate
        for sublist in beta_indices:
            for item in sublist:
                if item not in params_num:
                    params_num.append(item)

        self.params_num = params_num
        self.df_model = len(params_num)
        self.df_resid = int(self.nobs - len(params_num))
	# print self.params_num

        # TODO exog_names. See at the end

    def xbetas(self, params):
        '''these are the V_i
        '''
        res = np.empty((self.nobs, self.nchoices))
        for choiceind in range(self.nchoices):
            res[:, choiceind] = np.dot(self.exog_bychoices[choiceind],
                                      params[self.beta_indices[choiceind]])

        return res

    def loglike(self, params):
        # normalization ?
        xb = self.xbetas(params)
        expxb = np.exp(xb)

        probs = expxb/expxb.sum(1)[:, None]  # we don't really need this for all
        loglike = (self.endog * np.log(probs)).sum(1)
        # is this the same: YES
        # self.logliketest = (self.endog * xb).sum(1) - np.log(sumexpxb)
        # if self.endog where index then xb[self.endog]

        # we wanto to maximize the log-likelihood so we use positeve log-likelihood
        # if we want to use SciPy's optimize.fmin to find the mle. minimizing we use
        # negative log-likelihood
        return loglike.sum()   # return sum for now not for each observation

    def fit(self, start_params=None, maxiter=10000, maxfun=5000, method="newton",
            full_output=1, disp=1, callback=None,**kwds):
        """
        Fits CLogit() model using using maximum likelihood.
        In a model linear the log-likelihood function of the sample, is
        global concave for β parameters, which facilitates its numerical
        maximization (McFadden, 1973).
        Fixed Method = Newton, because it'll find the maximum in a few iterations.

        Returns
        -------
        Fit object for likelihood based models
        See: GenericLikelihoodModelResults

        """
        if start_params is None:
            start_params = np.zeros(len(self.params_num))
        else:
            start_params = np.asarray(start_params)

        # TODO: check number of  iterations. Seems too high.
        return super(CLogit, self).fit(start_params=start_params,
                                    maxiter=maxiter, maxfun=maxfun,**kwds)

if __name__=="__main__":

    import pandas as pandas
    from patsy import dmatrices

    u"""
    Examples
    --------
    See Greene, Econometric Analysis (5th Edition - 2003: Page 729)
    21.7.8. APPLICATION: CONDITIONAL LOGIT MODEL FOR TRAVEL MODE CHOICE

        *four alternative-specific constants (αair, αtrain, αbus, αcar)
            αcar dropped for identification
        *two alternative specific variables (gc, ttme)
            with a generic coefficient (βG, βT)
        *one alternative specific variable (hinc_air)
            with an alternative specific coefficient (γH*di,air)

    Ui j = αair*di,air + αtrain*di,train + αbus*di,bus + βG*GCij
            + βT*TTMEij + (γH*di,air)*HINCi + εij

    Note: There's a typo on TABLE 21.11. βT isn't -0.19612 is -0.09612
        see TABLE 21.13 to check
    """
    # TODO: use datasets instead
    url = "http://vincentarelbundock.github.io/Rdatasets/csv/Ecdat/ModeChoice.csv"
    file_ = "ModeChoice.csv"
    import os
    if not os.path.exists(file_):
        import urllib
        urllib.urlretrieve(url, "ModeChoice.csv")
    df = pandas.read_csv(file_)
    pandas.set_printoptions(max_rows=1000, max_columns=20)
    df.describe()

    nchoices = 4
    nobs = 210
    choice_index = np.arange(nchoices*nobs) % nchoices
    df['hinc_air'] = df['hinc']*(choice_index==0)

    f = 'mode  ~ ttme+invc+invt+gc+hinc+psize+hinc_air'
    y, X = dmatrices(f, df, return_type='dataframe')
    y.head()
    X.head()

    endog = y.to_records()
    endog = endog['mode'].reshape(-1, nchoices)

    dta = X.to_records()
    dta1 = np.array(dta)

    xivar = [['gc', 'ttme', 'Intercept','hinc_air'],
             ['gc', 'ttme', 'Intercept'],
             ['gc', 'ttme', 'Intercept'],
             ['gc', 'ttme' ]]

    xi = []

    for ii in range(nchoices):
        xi.append(dta1[xivar[ii]][choice_index==ii])

    # xifloat = [xx.view(float).reshape(nobs,-1) for xx in xi]
    # xifloat = [X[xi_names][choice_index==ii].values for ii, xi_names in enumerate(xivar)]
    xifloat = [X.ix[choice_index == ii, xi_names].values for ii, xi_names in enumerate(xivar)]

    clogit_mod  = CLogit(endog, xifloat, 2)
    # Iterations:  ¿ 957 ?
    clogit_res =  clogit_mod.fit()

    exog_names = u'     βG         βT        αair          γH          αtrain       αbus'.split()
    print u'     βG         βT        αair          γH          αtrain       αbus'
    print clogit_res.params

    # TODO: why are df_resid and df_model nan
    # clogit_res.df_resid = clogit_res.model.endog.shape[0] - len(clogit_res.params)
    # clogit_res.df_model = len(clogit_res.params)

    exog_names = u'G T const_air H const_train const_bus'.split()
    print clogit_res.summary(yname='Travel Mode', xname=exog_names)
    # TODO: it looks like R reports p-value based on t-distribution
    # TODO on summary: frequencies of alternatives, McFadden R^2, Likelihood
    #   ratio test, method, iterations.
    hessian = clogit_mod.hessian(clogit_res.params)
    s = 'The value of hessian hessian is '+ '\r' + str(hessian)
    print s

    print u"""

    Example 1. Replicate Greene (2003) results.
    TABLE 21.11 Parameter Estimates. Unweighted Sample
        βG       βT      αair        γH         αtrain       αbus
    [-0.015501  -0.09612   5.2074  0.01328757  3.86905293  3.16319074]

    """

    """
    # R code for example 1

    library("mlogit", "TravelMode")
    names(TravelMode)<- c("individual", "mode", "choice", "ttme", "invc",
                             "invt", "gc", "hinc", "psize")
    TravelMode$hinc_air <- with(TravelMode, hinc * (mode == "air"))
    res <- mlogit(choice ~ gc + ttme + hinc_air, data = TravelMode,
                shape = "long", alt.var = "mode", reflevel = "car")
    summary(res)
    model$hessian       #the hessian of the log-likelihood at convergence

    # R results for example 1

    Call:
    mlogit(formula = choice ~ gc + ttme + hinc_air, data = TravelMode,
        reflevel = "car", shape = "long", alt.var = "mode", method = "nr",
        print.level = 0)

    Frequencies of alternatives:
        car     air   train     bus
    0.28095 0.27619 0.30000 0.14286

    nr method
    5 iterations, 0h:0m:0s
    g'(-H)^-1g = 0.000234
    successive function values within tolerance limits

    Coefficients :
                       Estimate Std. Error t-value  Pr(>|t|)
    air:(intercept)    5.207433   0.779055  6.6843 2.320e-11 ***
    train:(intercept)  3.869036   0.443127  8.7312 < 2.2e-16 ***
    bus:(intercept)    3.163190   0.450266  7.0252 2.138e-12 ***
    gc                -0.015501   0.004408 -3.5167  0.000437 ***
    ttme              -0.096125   0.010440 -9.2075 < 2.2e-16 ***
    hinc_air           0.013287   0.010262  1.2947  0.195414
    ---
    Signif. codes:  0 ‘***’ 0.001 ‘**’ 0.01 ‘*’ 0.05 ‘.’ 0.1 ‘ ’ 1

    Log-Likelihood: -199.13
    McFadden R^2:  0.29825
    Likelihood ratio test : chisq = 169.26 (p.value = < 2.22e-16)
    """
    print  u"""
    Summary R results for example 1

    air:(intercept) train:(intercept)   bus:(intercept)   gc
    5.20743293        3.86903570        3.16319033       -0.01550151
    ttme          hinc_air
    -0.09612462    0.01328701

    model$hessian       #the hessian of the log-likelihood at convergence,
                      air:(intercept) train:(intercept) bus:(intercept)           gc        ttme    hinc_air
    air:(intercept)        -25.613627          7.715062        3.883696    192.37152  -1109.9784   -993.2641
    train:(intercept)        7.715062        -28.707527        6.766574   -776.60445   -313.7511    284.4266
    bus:(intercept)          3.883696          6.766574      -17.978427    -21.70683   -159.8403    144.5267
    gc                     192.371522       -776.604449      -21.706830 -75474.20527 -16841.6889   7780.6315
    ttme                 -1109.978447       -313.751079     -159.840260 -16841.68892 -91446.9712 -43448.0365
    hinc_air              -993.264146        284.426623      144.526736   7780.63148 -43448.0365 -48054.1196
    """

    print u"""
    Example 2

        *four alternative-specific constants (αair, αtrain, αbus, αcar)
            αcar dropped for identification
        *one alternative specific variables (invc)
            with a generic coefficient (βinvc)

    """
    xivar2 = [['invc', 'Intercept'],
              ['invc', 'Intercept'],
              ['invc', 'Intercept'],
              ['invc' ]]

    xi2 = []

    for ii in range(nchoices):
        xi2.append(dta1[xivar2[ii]][choice_index==ii])

    xifloat2 = [X.ix[choice_index == ii, xi2_names].values for ii, xi2_names in enumerate(xivar2)]

    clogit_mod2 = CLogit(endog, xifloat2, 1)
    clogit_res2 = clogit_mod2.fit()

    print clogit_res2.params
    exog_names = u'invc const_air const_train const_bus'.split()
    print clogit_res2.summary(yname='Travel Mode', xname=exog_names)

    """
        Call:
    mlogit(formula = choice ~ invc, data = TravelMode, reflevel = "car",
        shape = "long", alt.var = "mode", print.level = 2, method = "nr")

    Frequencies of alternatives:
        car     air   train     bus
    0.28095 0.27619 0.30000 0.14286

    nr method
    4 iterations, 0h:0m:0s
    g'(-H)^-1g = 0.000482
    successive function values within tolerance limits

    Coefficients :
                        Estimate Std. Error t-value Pr(>|t|)
    air:(intercept)    0.8711172  0.3979705  2.1889  0.02860 *
    train:(intercept)  0.4825992  0.2455787  1.9652  0.04940 *
    bus:(intercept)   -0.5000892  0.2356369 -2.1223  0.03381 *
    invc              -0.0138883  0.0055318 -2.5106  0.01205 *
    ---
    Signif. codes:  0 ‘***’ 0.001 ‘**’ 0.01 ‘*’ 0.05 ‘.’ 0.1 ‘ ’ 1

    Log-Likelihood: -280.54
    McFadden R^2:  0.011351
    Likelihood ratio test : chisq = 6.4418 (p.value = 0.011147)
    """

    hessian2 = clogit_mod2.hessian(clogit_res2.params)

    print hessian2

    print u"""
    R results

        model2$coefficient
    air:(intercept) train:(intercept)   bus:(intercept)              invc
       0.87111722        0.48259924       -0.50008925       -0.01388828

        model2$hessian
                      air:(intercept) train:(intercept) bus:(intercept)          invc
    air:(intercept)        -41.485888         17.171385        8.218602   -2022.67713
    train:(intercept)       17.171385        -43.402569        8.885814     -81.87671
    bus:(intercept)          8.218602          8.885814      -25.618418     455.92294
    invc                 -2022.677132        -81.876710      455.922944 -157872.76175
    """
    print u"""
    Example 3

        *one alternative specific variables (gc)
            with a generic coefficient (βG)
    """


    xivar3 = [['gc'],
             ['gc'],
             ['gc'],
             ['gc']]

    xi = []

    for ii in range(nchoices):
        xi.append(dta1[xivar3[ii]][choice_index==ii])

    xifloat3 = [X.ix[choice_index == ii, xi_names].values for ii, xi_names in enumerate(xivar3)]

    clogit_mod3  = CLogit(endog, xifloat3, 1)
    # Iterations:  ¿ 957 ?
    clogit_res3 =  clogit_mod3.fit()

    exog_names = u'βT        αai        αtrain       αbus'.split()
    print u'βT        αai        αtrain       αbus'
    print clogit_res3.params


    exog_names = u'gc'.split()
    print clogit_res3.summary(yname='Travel Mode', xname=exog_names)
    # TODO: it looks like R reports p-value based on t-distribution

    hessian3 = clogit_mod3.hessian(clogit_res3.params)
    s = 'The value of hessian hessian is '+ '\r' + str(hessian3)
    print s

    ###
    beta_indices = [np.array([0, 1, 2, 3]), np.array([0, 1, 4]), np.array([0, 1, 5]), np.array([0, 1])]
    name = []
    ind = []

    for sublist in xivar:
        for item in sublist:
            name.append(item)

    for sublist in beta_indices:
        for item in sublist:
            ind.append(item)

    print name, ind
    print len(name), len(ind)

    exog_vrbles = []

    for ii in range(0, len(name)):
        exog_vrbles.append(name[ii] + '_' + map(str, ind)[ii])

    print exog_vrbles, len (exog_vrbles)

    exog_num = []

    for item in exog_vrbles:
        if item not in exog_num:
            exog_num.append(item)

#    exog_num.sort()
    print exog_num , len(exog_num)
