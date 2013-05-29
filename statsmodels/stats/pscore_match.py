'''Propensity Score Matchig


Author: 
License: BSD-3

'''

import statsmodels.api as sm


class PropensityScoreMatch(object):
    def __init__(self, assigment_index, covariates, treatment_objective_variable):
        self.assigment_index = assigment_index
        self.covariates = covariates
        self.tratment_objective_variable = tratment_objective_variable
        
    def fit(self):
        result = PScoreMatchResult()
        p_mod = sm.Logit(self.assigment_index, self.covariates)
        p_res = mod.fit()
        result.propensity_estimation(p_res)
        self.scores = p_res.predict(self.covariates)
        self.stratify()
        
    def check_balance_for(strata):
        endog, exog = self.assigment_index[strata], sm.add_constant(self.covariates[strata], prepend = True)
        om = sm.OM(endog, exog)
        res= om.fit()
        return (res.f_pvalue < 0.05) and (result.pvalues < 0.05).all() 
        
        
class PScoreMatchResult(object):
    def propensity_estimation(self, result):
        self.propensity_estimation = result