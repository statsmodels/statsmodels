# -*- coding: utf-8 -*-
"""
Mixed logit (or random-coefficients model)

Copyright (c) 2013 Ana Martinez Pardo <anamartinezpardo@gmail.com>
License: BSD-3 [see LICENSE.txt]

General References
--------------------
Garrow, L. A. 'Discrete Choice Modelling and Air Travel Demand: Theory and
    Applications'. Ashgate Publishing, Ltd. 2010.
Hensher, D. and Greene, W. (2003) 'The Mixed Logit model: the state of practice'
    in Transportation, 30, pp133-176.
Orro Arcay, A. `Modelos de elección discreta en transportes con coeficientes
    aleatorios. 2006
Train, K. `Discrete Choice Methods with Simulation`.
    Cambridge University Press. 2003

Notes
--------------------
 * specification random Coefficients.
 * simulated maximum likelihood.
 * fixed and/or random coefficients.

Estimation methodology design
--------------------
 * give a initial hypothesis about the distribution of the random parameters
 * draw R random numbers of this distribution.
    Halton sequence numbers are used for the draws
 * for each draw :math:`\beta^r' compute the probability:
  .. math:: L^r_{ni} (\beta^r_{n}) = \frac{e^{\beta^r_{n}X_{ni}}} {\sum_{j} e^{\beta^r_{n}X_{nj}}}
 * compute the average of these probabilities
  .. math:: \bar{P}_{ni} = \frac{1}{R} \sum_{r=0}^{R} L_{ni} (\beta^r)
 * compute the log-likelihood for these probabilities
 * iterate until the maximun.

TODO
--------------------
 * notes about distribution(s) to use.
     Orro (2006, chapter 3); Train (2003, chapter 6)
 * draws: see [1] and Modified Latin Hypercube Sampling (Hess et al., 2006) [2]
 * panel data : see Train (2003, chapter 6: 6.7)
 * Bayesian approach (Train, Chapter 12)
 * specification on error-components logit. (do note about specification)

 [1] Train, “Halton sequences for mixed logit,” working paper, 1999.
 [2] http://elsa.berkeley.edu/~train/hesstrainpolak.pdf

"""

import numpy as np
import pandas as pd
from statsmodels.base.model import (LikelihoodModel,
                                    LikelihoodModelResults, ResultMixin)
import statsmodels.api as sm
import time
from collections import OrderedDict
from scipy import stats
from statsmodels.tools.decorators import (resettable_cache,
        cache_readonly)
from halton_sequence import halton

class MXLogit(LikelihoodModel):
    __doc__ = """
    Mixed Logit

    Parameters
    ----------
    endog_data : array
        dummy encoding of realized choices.
    exog_data : array (nobs, k*)
        array with explanatory variables.Variables for the model are select
        by V, so, k* can be >= than k. An intercept is not included by
        default and should be added by the user.
    V: dict
        a dictionary with the names of the explanatory variables for the
        utility function for each alternative.
        Alternative specific variables (common coefficients) have to be first.
        For specific variables (various coefficients), choose an alternative and
        drop all specific variables on it.
    ncommon : int
        number of explanatory variables with common coefficients.
    ref_level : str
        Name of the key for the alternative of reference.
    name_intercept : str
        name of the column with the intercept. 'None' if an intercept is not
        included.
    random_params : dict
        indefify random parameters and distribution(s) to use.
    draws : int
        number of draws used for simulation

    Attributes
    ----------
    endog : array (nobs*J, )
        the endogenous response variable
    endog_bychoices: array (nobs,J)
        the endogenous response variable by choices
    exog_matrix: array   (nobs*J,K)
        the enxogenous response variables
    exog_bychoices: list of arrays J * (nobs,K)
        the enxogenous response variables by choices. one array of exog
        for each choice.
    nobs : float
        number of observations.
    J  : float
        The number of choices for the endogenous variable. Note that this
        is zero-indexed.
    K : float
        The actual number of parameters for the exogenous design.  Includes
        the constant if the design has one and excludes the constant of one
        choice which should be dropped for identification.
    summary : Summary instance
        summarize the results inside MXLogitResults class.

    Notes
    -----
    This model allows :math:'\beta_n' to be random.
    When utility is linear in :math:'beta', the utility of person 'n' choosing
    alternative ''j'' can be written as:

    .. math:: U_{nj} = \beta_n x_{nj} + \gamma_nj z_{n}+ \varepsilon_{nj}

    where :
    :math:' \varepsilon_{nj}' ~ iid extreme value
    .. math:: \quad \beta_n x_{nj} \sim f(\beta_n | \theta)

    and :math:'\theta' is the distribution parameters over the population.
        x_j contains generic variables (terminology Hess) that have the same
            coefficient across choices
        z are variables, like individual-specific, that have different
          coefficients across variables.

    If there are choice (or alternative) specific constants, then they should
    be contained in Z. For identification, the constant of one choice should be
    dropped.
    """

    def __init__(self, endog_data, exog_data,  V, random_params, draws, ncommon,
                 ref_level, name_intercept = None, **kwds):

        self.endog_data = endog_data
        self.exog_data = exog_data

        self.V = V
        self.ncommon = ncommon

        self.ref_level = ref_level

        if name_intercept == None:
            self.exog_data['Intercept'] = 1
            self.name_intercept = 'Intercept'
        else:
            self.name_intercept = name_intercept

        self.draws = draws
        self.random_params = random_params
        self.num_rparam = len(random_params)

        self._initialize()
        super(MXLogit, self).__init__(endog = endog_data,
                exog = self.exog_matrix, **kwds)

    def _initialize(self):
        """
        Preprocesses the data discrete choice models

        The data for MXLogit is almost equal as required for CLogit
        """

        self.J = len(self.V)
        self.nobs = self.endog_data.shape[0] / self.J

        # Endog_bychoices
        self.endog_bychoices = self.endog_data.values.reshape(-1, self.J)

        # Exog_bychoices
        exog_bychoices = []
        exog_bychoices_names = []
        choice_index = np.array(self.V.keys() * self.nobs)

        for key in iter(self.V):
            (exog_bychoices.append(self.exog_data[self.V[key]]
                                    [choice_index == key].values))

        for key in self.V:
            exog_bychoices_names.append(self.V[key])

        self.exog_bychoices = exog_bychoices

        # Betas
        beta_not_common = ([len(exog_bychoices_names[ii]) - self.ncommon
                            for ii in range(self.J)])
        exog_names_prueba = []

        for ii, key in enumerate(self.V):
            exog_names_prueba.append(key * beta_not_common[ii])

        zi = np.r_[[self.ncommon], self.ncommon + np.array(beta_not_common)\
                    .cumsum()]
        z = np.arange(max(zi))
        beta_ind = [np.r_[np.arange(self.ncommon), z[zi[ii]:zi[ii + 1]]]
                               for ii in range(len(zi) - 1)]  # index of betas
        self.beta_ind = beta_ind

        beta_ind_str = ([map(str, beta_ind[ii]) for ii in range(self.J)])
        beta_ind_J = ([map(str, beta_ind[ii]) for ii in range(self.J)])

        for ii in range(self.J):
            for jj, item in enumerate(beta_ind[ii]):
                if item in np.arange(self.ncommon):
                    beta_ind_J[ii][jj] = ''
                else:
                    beta_ind_J[ii][jj] = ' (' + self.V.keys()[ii] + ')'

        self.betas = OrderedDict()

        for sublist in range(self.J):
            aa = []
            for ii in range(len(exog_bychoices_names[sublist])):
                aa.append(
                beta_ind_str[sublist][ii] + ' ' +
                exog_bychoices_names[sublist][ii]
                + beta_ind_J[sublist][ii])
            self.betas[sublist] = aa

        # Exog

        pieces = []
        for ii in range(self.J):
            pieces.append(pd.DataFrame(exog_bychoices[ii], columns=self.betas[ii]))

        self.exog_matrix_all = (pd.concat(pieces, axis = 0, keys = self.V.keys(),
                                     names = ['choice', 'nobs'])
                           .fillna(value = 0).sortlevel(1).reset_index())

        self.exog_matrix = self.exog_matrix_all.iloc[:, 2:]

        self.K = len(self.exog_matrix.columns) + len(self.random_params)

        self.df_model = self.K
        self.df_resid = int(self.nobs - self.K)

    def xbetas(self, params):
        """the Utilities V_i

        """
        res = np.empty((self.nobs, self.J))
        for ii in range(self.J):
            res[:, ii] = np.dot(self.exog_bychoices[ii],
                                      params[self.beta_ind[ii]])
        return res

    def drawnvalues(self):
        """

        """
        # TODO : implement test
        # from math import sqrt
        # sqrt(210)/2 # the ratio should be small (See Train, 2001)

        haltonsequence = halton(self.num_rparam, self.draws)

        # TODO: set user-provided dist
        # TODO: estimate loc and scale as part of the estimation procedure
        # Initial values of:
        #    means -> coeficients from conditional logit
        #    standart deviations -> 0.1
        # see fit

        # haltonsequence = halton(1, 5)
        # drawnvalues = stats.norm.ppf(haltonsequence,-0.016, 0.1 )
        if DEBUG:
            print "___working in haltonsequence"

        haltonsequence = halton(1, self.draws)
        self.dv = stats.norm.ppf(haltonsequence, self.mean, self.std)

        return self.dv

    def cdf(self, item):
        """
        Mixed Logit cumulative distribution function.

        Parameters
        ----------
        X : array (nobs, J)
            the linear predictor of the model.

        Returns
        --------
        cdf : ndarray
            the cdf evaluated at `X`.

        Notes
        -----
        .. math:: P_{ni} = \int L_{ni} (\beta) f(\beta) d\beta = \int \frac{e^{\beta x_{ni}}} {\sum_J e^{\beta x_{nj}}} f(\beta) d \beta

        if :math:` \phi(\beta | b,W)' is the normal distribution density with
        mean 'b' and covariance 'W', it can be re-written:

        .. math:: P_{ni} = \int \frac{e^{\beta x_{ni}}} {\sum_J e^{\beta x_{nj}}} \phi(\beta | b,W) d \beta

        However, it can be applied with different distribution types, such as
        lognormal or uniform distributions.

       """
        eXB = np.exp(item)
        return  eXB / eXB.sum(1)[:, None]

    def cdf_average(self, params):
        """
        """
        if DEBUG:
            print "___working in average"

        prob_average = []
        self.drawnvalues()
        for ii in self.dv:
            params[0] = ii  # numpy.ndarray
            xb = self.xbetas(params)
            prob_average.append(self.cdf(xb))

        if DEBUG:
            print "___returning to average"

        return sum(prob_average) / self.draws

    def loglike(self, params):

        """
        Log-likelihood of the mixed logit model.

        Parameters
        ----------
        params : array
            the parameters of the model.

        Returns
        -------
        loglike : float
            the log-likelihood function of the model evaluated at `params`.

        Notes
        ------
        Since mixed logit probability is not a close form, it needs to be
        approximated through simulation.

        Assume :math:`\beta^r;' is generated from the 'r'-th random draw.
        There are totally R times of draws in the simulation.
        The average simulated probability can be expressed as:

        .. math:: \bar{P}_{ni} = \frac{1}{R} \sum_{r=0}^{R} L_{ni} (\beta^r)

        so, the simulated log-likelihood:

        .. math:: \LL = \sum_{n=1}^{N} \sum_{j=0}^{J} d_{ij} \ln \bar{P}_{nj} </math>

        where :math:`d_{ij}=1` if individual `i` chose alternative `j` and 0
        if not.
        """
        if DEBUG:
            print "___working in loglike"

        loglike = (self.endog_bychoices *
                    np.log(self.cdf_average(params))).sum(1)
        if DEBUG:
            print "___returning to loglike"

        return loglike.sum()

    def score(self, params):
        """
        """
        from statsmodels.tools.numdiff import approx_fprime
        return approx_fprime(params, self.loglike, epsilon=1e-8)

    def hessian(self, params):
        """
        """
        from statsmodels.tools.numdiff import approx_hess
        return approx_hess(params, self.loglike)

    def fit(self, start_params=None, maxiter=10000, maxfun=5000,
            method='bfgs', full_output=1, disp=None, callback=None, **kwds):
        """
        Fits MXLogit() model using maximum likelihood.

        Returns
        -------
        Fit object for likelihood based models
        See: GenericLikelihoodModelResults

        """

        if DEBUG:
            print "_working on fit"

        if start_params is None:

            if DEBUG:
                print "__working on start params"

        # TODO: func_params
            self.mean = -0.01550155  # set to check with R results
            self.std = 0.00027105    # set to check with R results
#            self.mean = -0.016 # loc
#            self.std = 1.0  # scale
            Logit_res = sm.Logit(self.endog, self.exog_matrix).fit(disp=0)
            # func_params = np.array([self.mean, self.std])
            # start_params = np.r_[Logit_res.params.values,
            #                      func_params]
            start_params =  Logit_res.params

            if DEBUG:
                print "start_params", start_params

            self.satpar = start_params

        else:
            start_params = np.asarray(start_params)

        if DEBUG:
            print "___working on fit"

        start_time = time.time()
        model_fit = super(MXLogit, self).fit(disp = disp,
                                            start_params = start_params,
                                            method=method, maxiter=maxiter,
                                            maxfun=maxfun, **kwds)

        self.params = model_fit.params

        if DEBUG:
            print self.params
            print "___returning to fit"

        end_time = time.time()
        self.elapsed_time = end_time - start_time

        return model_fit

    def summary(self):
        # TODO add __doc__ = MXLogitResults.__doc__
        return MXLogitResults(self).summary()


### Results Class ###
class MXLogitResults(LikelihoodModelResults, ResultMixin):
    __doc__ = """
    Parameters
    ----------
    model : A Discrete Choice Model instance.

    Returns
    -------
    aic : float
        Akaike information criterion.  -2*(`llf` - p) where p is the number
        of regressors including the intercept.
    bic : float
        Bayesian information criterion. -2*`llf` + ln(`nobs`)*p where p is the
        number of regressors including the intercept.
    bse : array
        The standard errors of the coefficients.
    df_resid : float
        Residual degrees-of-freedom of model.
    df_model : float
        Params.
    llf : float
        Value of the loglikelihood
    llnull : float
        Value of the constant-only loglikelihood
    llr : float
        Likelihood ratio chi-squared statistic; -2*(`llnull` - `llf`)
    llrt: float
        Likelihood ratio test
    llr_pvalue : float
        The chi-squared probability of getting a log-likelihood ratio
        statistic greater than llr.  llr has a chi-squared distribution
        with degrees of freedom `df_model`.
    prsquared : float
        McFadden's pseudo-R-squared. 1 - (`llf`/`llnull`)

    """

    def __init__(self, model):

        self.model = model
        self.mlefit = model.fit()
        self.nobs_bychoice = model.nobs
        self.nobs = model.endog.shape[0]
        self.alt = model.V.keys()
        self.freq_alt = model.endog_bychoices[:, ].sum(0).tolist()
        self.perc_alt = (model.endog_bychoices[:, ].sum(0) / model.nobs)\
                        .tolist()
        self.__dict__.update(self.mlefit.__dict__)
        self._cache = resettable_cache()

    def __getstate__(self):
        try:
            #remove unpicklable callback
            self.mle_settings['callback'] = None
        except (AttributeError, KeyError):
            pass
        return self.__dict__

    @cache_readonly
    def llnull(self):
        # TODO: Use Clogit
        # loglike model without predictors
        model = self.model
        V = model.V

        V_null = OrderedDict()

        for ii in range(len(V.keys())):
            if V.keys()[ii] == model.ref_level:
                V_null[V.keys()[ii]] = []
            else:
                V_null[V.keys()[ii]] = [model.name_intercept]

        mxlogit_null_mod = model.__class__(endog_data = model.endog_data,
                                           exog_data = model.exog_data,
                                           V = V_null,
                                           random_params = random_params,
                                           draws = draws,
                                           ncommon = 0,
                                           ref_level = model.ref_level,
                                           name_intercep = model.name_intercept)

        mxlogit_null_mod.mean = None
        mxlogit_null_mod.std = None
        mxlogit_null_res = mxlogit_null_mod.fit(start_params = np.zeros(\
                                                len(V.keys()) - 1), disp = 0)

    # Fit model
        return mxlogit_null_res.llf

    @cache_readonly
    def llr(self):
        return -2 * (self.llnull - self.llf)

    @cache_readonly
    def llrt(self):
        return 2 * (self.llf - self.llnull)

    @cache_readonly
    def llr_pvalue(self):
        return stats.chisqprob(self.llr, self.model.df_model)

    @cache_readonly
    def prsquared(self):
        """
        McFadden's Pseudo R-Squared: comparing two models on the same data, would
        be higher for the model with the greater likelihood.
        """
        return (1 - self.llf / self.llnull)

    def summary(self, title = None, alpha = .05):
        """Summarize the MXLogit Results

        Parameters
        -----------
        title : string, optional
            Title for the top table. If not None, then this replaces the
            default title
        alpha : float
            significance level for the confidence intervals

        Returns
        -------
        smry : Summary instance
            this holds the summary tables and text, which can be printed or
            converted to various output formats.

        See Also
        --------
        statsmodels.iolib.summary.Summary : class to hold summary
            results

        """

        top_left = [('Dep. Variable:', None),
                    ('Model:', [self.model.__class__.__name__]),
                    ('Method:', [self.mle_settings['optimizer']]),
                    ('Date:', None),
                    ('Time:', None),
                    ('Converged:', ["%s" % self.mle_retvals['converged']]),
#                    ('Iterations:', ["%s" % self.mle_retvals['iterations']]),
                    ('Elapsed time (seg.):',
                                    ["%10.4f" % self.model.elapsed_time]),
                    ('Num. alternatives:', [self.model.J]),
                    ('Draws:', [self.model.draws]),
                    ('Method:', ["Halton sequence"]),
                      ]

        top_right = [
                     ('No. Cases:', [self.nobs]),
                     ('No. Observations:', [self.nobs_bychoice]),
                     ('Df Residuals:', [self.model.df_resid]),
                     ('Df Model:', [self.model.df_model]),
                     ('Log-Likelihood:', None),
                     ('LL-Null (constant only):', ["%#8.5g" % self.llnull]),
                     ('Pseudo R-squ.:', ["%#6.4g" % self.prsquared]),
                     ('LLR p-value:', ["%#6.4g" % self.llr_pvalue]),
                     ('Likelihood ratio test:', ["%#8.5g" %self.llrt]),
                     ('AIC:', ["%#8.5g" %self.aic])

                                     ]
        if title is None:
            title = self.model.__class__.__name__ + ' ' + \
            "results"

        #boiler plate
        from statsmodels.iolib.summary import Summary, SimpleTable

        smry = Summary()
        # for top of table
        smry.add_table_2cols(self, gleft=top_left, gright=top_right,
                             title=title)

        # Frequencies of alternatives
        mydata = [self.freq_alt, self.perc_alt]
        myheaders = self.alt
        mytitle = ("")
        mystubs = ["Frequencies of alternatives: ", "Percentage:"]
        tbl = SimpleTable(mydata, myheaders, mystubs, title = mytitle,
                          data_fmts = ["%5.2f"])
        smry.tables.append(tbl)

#        # for parameters
#        smry.add_table_params(self, alpha=alpha, use_t=True)

        # TODO
        # for ramdom parameters
        mydata = [self.model.satpar, self.params]
        mystubs = self.model.exog_names
        myheaders = ["Initial Params:", "Params: "]

        mytitle = ("Params MXLogit")
        tb2 = SimpleTable(mydata, mystubs, myheaders, title = mytitle,
                          data_fmts = ["%5.4f"])
        smry.tables.append(tb2)
        return smry


if __name__ == "__main__":

    DEBUG = 0

    print 'Example:'

    # Load data
    from patsy import dmatrices

    url = "http://vincentarelbundock.github.io/Rdatasets/csv/Ecdat/ModeChoice.csv"
    file_ = "ModeChoice.csv"
    import os
    if not os.path.exists(file_):
        import urllib
        urllib.urlretrieve(url, "ModeChoice.csv")
    df = pd.read_csv(file_)
    df.describe()

    f = 'mode  ~ ttme+invc+invt+gc+hinc+psize'
    y, X = dmatrices(f, df, return_type='dataframe')

    # Set up model

    # Names of the variables for the utility function for each alternative
    # variables with common coefficients have to be first in each array
    # the order should be the same as sequence in data
    # ie: row1 -- air data, row2 -- train data, row3 -- bus data, ...

    V = OrderedDict((
        ('air',   ['gc', 'ttme', 'Intercept', 'hinc']),
        ('train', ['gc', 'ttme', 'Intercept']),
        ('bus',   ['gc', 'ttme', 'Intercept']),
        ('car',   ['gc', 'ttme']))
        )

    # Number of common coefficients
    ncommon = 2
    # Random coefficients and distributions
    random_params = OrderedDict((
        ('gc', stats.norm),
        ))

    # Number of draws used for simulation
    # eg: 50 for draf model, 1000 for final model
    draws = 50

    # Describe model
    mxlogit_mod = MXLogit(endog_data = y, exog_data = X,  V = V,
                          random_params = random_params, draws = draws,
                          ncommon = ncommon, ref_level = 'car',
                          name_intercept = 'Intercept')
    # Fit model
    mxlogit_res = mxlogit_mod.fit(method = "bfgs", disp = 1)

    # Summarize model
    # TODO means and standard errors for mixed logit parameters
    print mxlogit_mod.summary()

    # R results

    #mlogit(formula = choice ~ gc + ttme + hinc_air, data = TravelMode,
    #    reflevel = "car", rpar = c(gc = "n"), R = 50, shape = "long",
    #    alt.var = "mode")
    #
    #Frequencies of alternatives:
    #    car     air   train     bus
    #0.28095 0.27619 0.30000 0.14286
    #
    #bfgs method
    #16 iterations, 0h:0m:4s
    #g'(-H)^-1g = 1.33E-07
    #gradient close to zero
    #
    #Coefficients :
    #                     Estimate  Std. Error  t-value  Pr(>|t|)
    #air:(intercept)    5.20736267  0.77769684   6.6959 2.144e-11 ***
    #train:(intercept)  3.86902862  0.44728704   8.6500 < 2.2e-16 ***
    #bus:(intercept)    3.16319599  0.43712507   7.2364 4.610e-13 ***
    #gc                -0.01550155  0.00410559  -3.7757 0.0001595 ***
    #ttme              -0.09612401  0.00809057 -11.8810 < 2.2e-16 ***
    #hinc_air           0.01328820  0.01209832   1.0984 0.2720515
    #sd.gc              0.00027105  0.13136215   0.0021 0.9983537
    #---
    #Signif. codes:  0 ‘***’ 0.001 ‘**’ 0.01 ‘*’ 0.05 ‘.’ 0.1 ‘ ’ 1
    #
    #Log-Likelihood: -199.13
    #McFadden R^2:  0.29825
    #Likelihood ratio test : chisq = 169.26 (p.value = < 2.22e-16)
    #
    #random coefficients
    #   Min.     1st Qu.      Median        Mean     3rd Qu. Max.
    #gc -Inf -0.01568437 -0.01550155 -0.01550155 -0.01531873  Inf