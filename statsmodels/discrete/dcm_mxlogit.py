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

import time
import re
from statsmodels.compat.collections import OrderedDict

import numpy as np
from scipy import stats

from .discrete_model import Logit
from .halton_sequence import halton
from .dcm_base import (DiscreteChoiceModel, DiscreteChoiceModelResults)


### Public Model Classes ####


class MXLogit(DiscreteChoiceModel):
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

    If there are choice (or alternative)LikelihoodModel specific constants,
    then they should be contained in Z. For identification, the constant
    of one choice should be dropped.
    """

    def __init__(self, endog_data, exog_data,  V, draws, ncommon,
                 NORMAL,ref_level, name_intercept = None, **kwds):

        super(MXLogit, self).__init__(endog_data, exog_data,  V, ncommon,
                 ref_level, name_intercept = None, **kwds)

        self._initialize()

    def _initialize(self):
        """
        Preprocesses the data for MXLogit
        """
        # Need a private Model Class for preprocesses the data for this model?

        super(MXLogit, self)._initialize()

        self.NORMAL = NORMAL
        if  NORMAL is not None:
            self.draws = draws
            self.num_rparam = len(NORMAL)

        self.values_ramdon = []
        self.n_ramdon = []

        for name in self.NORMAL:
            random = re.compile("[ ]\b*" + name)
            for ii, jj in enumerate(self.paramsnames):
                if random.search(jj) is not None:
                    self.values_ramdon.append(ii)
                    self.n_ramdon.append(jj)

        if DEBUG:
            print self.values_ramdon
            print self.n_ramdon

        ms_params = []

        for param in self.n_ramdon:
            ms_params.append('mean_%s' % param)
            ms_params.append('sd_%s' % param)

        self.ms_params = ms_params
        self.paramsnames = np.r_[self.paramsnames, ms_params]
        self.nparams = len(self.paramsnames)

        #mapping coefficient names to indices to unique/parameter array
        self.paramsidx = OrderedDict((name, idx) for (idx, name) in
                              enumerate(self.paramsnames))

        if DEBUG:
            print self.paramsnames
            print self.paramsidx
            print self.nparams

        # TODO : implement test
        # from math import sqrt
        # sqrt(210)/2 # the ratio should be small (See Train, 2001)
        self.haltonsequence = halton(len(self.n_ramdon), self.draws)

    def drawndistri(self, params):
        """
        """
        dv = np.empty((len(self.haltonsequence), len(self.n_ramdon)))

        for ii, name in enumerate(self.n_ramdon):
            mean = params[self.paramsidx['mean_' + name]]
            std = params[self.paramsidx['sd_' + name]]
            hs = self.haltonsequence[:, ii]
            dv[:, ii] = stats.norm.ppf(hs, mean, std)

        self.dv = dv

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

        self.drawndistri(params)

        for jj in range(self.dv.shape[0]):
                if DEBUG:
                    print self.values_ramdon
                    print self.dv[jj]

                params[self.values_ramdon] = self.dv[jj]  # numpy.ndarray

                if DEBUG:
                    print params
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

        .. math:: \LL = \sum_{n=1}^{N} \sum_{j=0}^{J} d_{ij} \ln \bar{P}_{nj}

        where :math:`d_{ij}=1` if individual `i` chose alternative `j` and 0
        if not.
        """
        if DEBUG:
            print "___working in loglike"
            print params

        for name in self.n_ramdon:
            std = params[self.paramsidx['sd_' + name]]
            std = 1e-8 if std < 1e-8 else std
            params[self.paramsidx['sd_' + name]] = std

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

    def fit(self, start_params=None, maxiter=5000, maxfun=5000,
            method='bfgs', full_output=1, disp=None, callback=None, **kwds):
        """
        Fits MXLogit() model using maximum likelihood.

        Returns
        -------
        Fit object for likelihood based models
        See: GenericLikelihoodModelResults

        """
        start_time = time.time()

        if start_params is None:

            if DEBUG:
                print "__working on start params"

            Logit_res = Logit(self.endog, self.exog_matrix).fit(disp=0)

            # Initial values of:
            # means -> coeficients from conditional logit
            # standart deviations -> 0.1
            func_params = []

            for rand in self.values_ramdon:
                mean = Logit_res.params[rand]  # loc
                func_params.append(mean)
                sd = 0.1   # loc
                func_params.append(sd)

            start_params = np.r_[Logit_res.params, func_params]

            if DEBUG:
                print "start_params", start_params

            self.satpar = start_params

        else:
            start_params = np.asarray(start_params)

        if DEBUG:
            print "___working on fit"

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
class MXLogitResults(DiscreteChoiceModelResults):

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
#                     ('LL-Null (constant only):', ["%#8.5g" % self.llnull]),
#                     ('Pseudo R-squ.:', ["%#6.4g" % self.prsquared]),
#                     ('LLR p-value:', ["%#6.4g" % self.llr_pvalue]),
#                     ('Likelihood ratio test:', ["%#8.5g" %self.llrt]),
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
        paramlist = self.model.exog_names + self.model.ms_params
        mystubs = paramlist
        myheaders = ["Initial Params:", "Params: "]

        mytitle = ("Params MXLogit")
        tb2 = SimpleTable(mydata, mystubs, myheaders, title = mytitle,
                          data_fmts = ["%5.4f"])
        smry.tables.append(tb2)
        return smry


if __name__ == "__main__":

    DEBUG = 0

    print('Example:')
    import statsmodels.api as sm
    # Loading data as pandas object
    data = sm.datasets.modechoice.load_pandas()
    data.endog[:5]
    data.exog[:5]
    data.exog['Intercept'] = 1  # include an intercept
    y, X = data.endog, data.exog

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

    # Random coefficients and distributions (By now, only Normal)
    # TODO: add Uniform, Triangular and Log-nomal

    NORMAL = ['gc']

    # Number of draws used for simulation
    # eg: 50 for draf model, 1000 for final model
    draws = 50

    # Describe model
    ref_level = 'car'
    name_intercept = 'Intercept'
    mxlogit_mod = MXLogit(endog_data = y, exog_data = X,  V = V,
                          NORMAL = NORMAL, draws = draws,
                          ncommon = ncommon, ref_level = ref_level,
                          name_intercept = name_intercept)

    # Fit model
#    mxlogit_res = mxlogit_mod.fit(method = "bfgs", disp = 1)
#    print mxlogit_res.params
#    print mxlogit_res.llf

    # Summarize model
    # TODO means and standard errors for mixed logit parameters
    print(mxlogit_mod.summary())
