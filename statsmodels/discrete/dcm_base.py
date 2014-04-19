# -*- coding: utf-8 -*-
"""
Discrete Choice Models.

Common code for:
 * Conditional logit (dcm_clogit.py)
 * Mixed logit or random-coefficients model (dcm_mxlogit.py)

Copyright (c) 2013 Ana Martinez Pardo <anamartinezpardo@gmail.com>
License: BSD-3 [see LICENSE.txt]

General References
--------------------
Garrow, L. A. 'Discrete Choice Modelling and Air Travel Demand: Theory and
    Applications'. Ashgate Publishing, Ltd. 2010.
Greene, W. `Econometric Analysis`. Prentice Hall, 5th. edition. 2003.
Orro Arcay, A. `Modelos de elección discreta en transportes con coeficientes
    aleatorios. 2006
Train, K. `Discrete Choice Methods with Simulation`.
    Cambridge University Press. 2003
"""

from statsmodels.compat.collections import OrderedDict
from statsmodels.compat import range

import numpy as np
from scipy import stats
import pandas as pd

from statsmodels.base.model import (LikelihoodModel,
                                    LikelihoodModelResults, ResultMixin)

from statsmodels.tools.decorators import (resettable_cache,
        cache_readonly)

# TODO: work on base/specifics docs

_DiscreteChoiceModel_docs = """
"""

_DiscreteChoiceModelResults_docs = """
    %(one-line_description)s

"""

#### Private Model Classes ####


class DiscreteChoiceModel(LikelihoodModel):
    __doc__ = """

    """

    def __init__(self, endog_data, exog_data,  V, ncommon,
                 ref_level, name_intercept = None,
                 **kwds):

        self.endog_data = endog_data
        self.exog_data = exog_data

        self.V = V
        self.ncommon = ncommon

        self.ref_level = ref_level
        self.name_intercept = name_intercept

        self._initialize()

        super(DiscreteChoiceModel, self).__init__(endog = endog_data,
                exog = self.exog_matrix, **kwds)

    def _initialize(self):
        """
        Preprocesses the data discrete choice models
        """
        if self.name_intercept == None:
            #if model hasn't one, create a Intercept for null model
            self.exog_data['Intercept'] = 1
            self.name_intercept = 'Intercept'

        self.J = len(self.V)
        self.nobs = self.endog_data.shape[0] / self.J

        # Endog_bychoices
        self.endog_bychoices = self.endog_data.values.reshape(-1, self.J)

        # Exog_bychoices
        exog_bychoices = []
        exog_bychoices_names = []
        choice_index = np.array(list(self.V.keys()) * int(self.nobs))

        for key in iter(self.V):
            (exog_bychoices.append(self.exog_data[self.V[key]]
                                    [choice_index == key].values))

        for key in self.V:
            exog_bychoices_names.append(self.V[key])

        self.exog_bychoices = exog_bychoices

        # Betas
        beta_not_common = ([len(exog_bychoices_names[ii]) - self.ncommon
                            for ii in range(self.J)])

        zi = np.r_[[self.ncommon], self.ncommon + np.array(beta_not_common)\
                    .cumsum()]
        z = np.arange(max(zi))
        beta_ind = [np.r_[np.arange(self.ncommon), z[zi[ii]:zi[ii + 1]]]
                               for ii in range(len(zi) - 1)]  # index of betas
        self.beta_ind = beta_ind

        beta_ind_str = ([list(map(str, beta_ind[ii])) for ii in range(self.J)])
        beta_ind_J = ([list(map(str, beta_ind[ii])) for ii in range(self.J)])

        for ii in range(self.J):
            for jj, item in enumerate(beta_ind[ii]):
                if item in np.arange(self.ncommon):
                    beta_ind_J[ii][jj] = ''
                else:
                    beta_ind_J[ii][jj] = ' (' + list(self.V.keys())[ii] + ')'

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

        self.exog_matrix_all = (pd.concat(pieces, axis = 0, keys = list(self.V.keys()),
                                     names = ['choice', 'nobs'])
                           .fillna(value = 0).sortlevel(1).reset_index())

        self.exog_matrix = self.exog_matrix_all.iloc[:, 2:]

        self.K = len(self.exog_matrix.columns)

        self.df_model = self.K
        self.df_resid = int(self.nobs - self.K)

        self.paramsnames = sorted(set([i for j in self.betas.values()
                                       for i in j]))

    def xbetas(self, params):
        """the Utilities V_i

        """
        res = np.empty((self.nobs, self.J))
        for ii in range(self.J):
            res[:, ii] = np.dot(self.exog_bychoices[ii],
                                      params[self.beta_ind[ii]])
        return res


### Results Class ###

class DiscreteChoiceModelResults(LikelihoodModelResults, ResultMixin):
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
        # loglike model without predictors
        model = self.model
        V = model.V

        V_null = OrderedDict()

        for ii in range(len(V.keys())):
            if V.keys()[ii] == model.ref_level:
                V_null[V.keys()[ii]] = []
            else:
                V_null[V.keys()[ii]] = [model.name_intercept]

        # TODO: work for mixed logit
        null_mod = model.__class__(endog_data = model.endog_data,
                                          exog_data = model.exog_data,
                                          V = V_null,
                                          ncommon = 0,
                                          ref_level = model.ref_level,
                                          name_intercep = model.name_intercept)
        null_mod_res = null_mod.fit(start_params = np.zeros(\
                                                len(V.keys()) - 1), disp = 0)

        return null_mod_res.llf

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


if __name__ == "__main__":

    # Example for text preprocessing data for discrete choice models
    import statsmodels.api as sm
    # Loading data as pandas object
    data = sm.datasets.modechoice.load_pandas()
    data.endog[:5]
    data.exog[:5]
    data.exog['Intercept'] = 1  # include an intercept
    y, X = data.endog, data.exog

    ncommon = 2
    ref_level = 'car'
    name_intercept = 'Intercept'
    V = OrderedDict((
        ('air',   ['gc', 'ttme', 'Intercept', 'hinc']),
        ('train', ['gc', 'ttme', 'Intercept']),
        ('bus',   ['gc', 'ttme', 'Intercept']),
        ('car',   ['gc', 'ttme']))
        )

    base_mod = DiscreteChoiceModel(endog_data = y, exog_data = X,  V = V,
                          ncommon = ncommon, ref_level = ref_level,
                          name_intercept = name_intercept)

    # dir(base_mod)
#    print base_mod.endog_bychoices
#    print base_mod.exog_bychoices
    print(base_mod.paramsnames)
