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

TODO:
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

        # TODO: rename beta to params and include inclusive values for nested CL
        betaind = [exog_bychoices[ii].shape[1]-ncommon for ii in range(self.nchoices)]
        zi = np.r_[[ncommon], ncommon + np.array(betaind).cumsum()]
        z=np.arange(len(zi)+ncommon)
        beta_indices = [np.r_[np.array([0, 1]),z[zi[ii]:zi[ii+1]]]
                       for ii in range(len(zi)-1)]
        #beta_indices = [array([3, 0, 1, 2]), array([4, 0, 1]), array([5, 0, 1]), array([1])]
        self.beta_indices = beta_indices


    def xbetas(self, params):
        '''these are the V_i
        '''
        res = np.empty((self.nobs, self.nchoices))
        for choiceind in range(self.nchoices):
            res[:,choiceind] = np.dot(self.exog_bychoices[choiceind],
                                      params[self.beta_indices[choiceind]])
        return res


    def loglike(self, params):
        # normalization ?
        xb = self.xbetas(params)
        expxb = np.exp(xb)

        probs = expxb/expxb.sum(1)[:,None]  # we don't really need this for all
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
        if start_params is None:
            start_params = np.zeros(6)  # need better np.zeros(6)
        else:
            start_params = np.asarray(start_params)

        return super(CLogit, self).fit( start_params=start_params,
                                    maxiter=maxiter, maxfun=maxfun,**kwds)


if __name__=="__main__":

    u"""
    Example
    See Greene, Econometric Analysis (5th Edition - 2003: Page 729)
    21.7.8. APPLICATION: CONDITIONAL LOGIT MODELFOR TRAVEL MODE CHOICE

        four alternative-specific constants (αair, αtrain, αbus, αcar)
            αcar dropped for identification
        two alternative specific variables (GC, TTME)
            with a generic coefficient (βG, βT)
        one alternative specific variable (HINC)
            with an alternative specific coefficient (γH*di,air)

    Ui j = αair*di,air + αtrain*di,train + αbus*di,bus + βG*GCij
            + βT*TTMEij + (γH*di,air)*HINCi + εij
    """

    import pandas as pandas
    import patsy
    import numpy.lib.recfunctions as recf

    #TODO: use datasets instead
    url = "http://vincentarelbundock.github.io/Rdatasets/csv/Ecdat/ModeChoice.csv"
    file_ = "ModeChoice.csv"
    import os
    if not os.path.exists(file_):
        import urllib
        urllib.urlretrieve(url, "ModeChoice.csv")

    df = pandas.read_csv(file_)
    pandas.set_printoptions(max_rows=1000, max_columns=20)
    df.describe()


    nchoices=4
    nobs=210
    choice_index = np.arange(nchoices*nobs) % nchoices
    df['hinc_air'] = df['hinc']*(choice_index==0)
    df.head()[:]

    f = 'mode  ~ ttme+invc+invt+gc+hinc+psize+hinc_air'
    y,X = patsy.dmatrices(f, df, return_type='dataframe')
    y.head()
    X.head()

    endog = y.to_records()
    endog = endog['mode'].reshape(-1,nchoices)

    dta= X.to_records()
    dta1=np.array(dta)
    xivar = [['gc', 'ttme', 'Intercept','hinc_air'],
             ['gc', 'ttme', 'Intercept'],
             ['gc', 'ttme', 'Intercept'],
             ['gc', 'ttme' ]]
    xi = []
    for ii in range(nchoices):
        xi.append(dta1[xivar[ii]][choice_index==ii])
        # this doesn't change sequence of columns, bug report by Skipper I think
    xifloat = [xx.view(float).reshape(nobs,-1) for xx in xi]
    xifloat = [X[xi_names][choice_index==ii].values for ii, xi_names in enumerate(xivar)]
    #xifloat[-1] = xifloat[-1][:,1:]

    clogit = CLogit(endog, xifloat, 2)
    # Iterations:  ¿ 957 ?

    resclogit=clogit.fit()


    print u'     βG         βT        αair          γH          αtrain       αbus'
    print resclogit.params


    print u"""
    Greene TABLE 21.11 Parameter Estimates. Unweighted Sample
        βG       βT      αair        γH         αtrain       αbus
    [-0.015501  -0.09612   5.2074  0.01328757  3.86905293  3.16319074]

    There's a typo on TABLE 21.11. βT isn't -0.19612 is -0.09612
        see TABLE 21.13 to check
    """

    """
    # R code:

    library("mlogit", "TravelMode")
    names(TravelMode)<- c("individual", "mode", "choice", "ttme", "invc",
                             "invt", "gc", "hinc", "psize")
    TravelMode$hinc_air <- with(TravelMode, hinc * (mode == "air"))
    res <- mlogit(choice ~ gc + ttme + hinc_air, data = TravelMode,
                shape = "long", alt.var = "mode", reflevel = "car")
    summary(res)

    # R results:

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

    print  """
    R results
    air:(intercept) train:(intercept)   bus:(intercept)   gc
    5.20743293        3.86903570        3.16319033       -0.01550151
    ttme          hinc_air
    -0.09612462    0.01328701
    """
