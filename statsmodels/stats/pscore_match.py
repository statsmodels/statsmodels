'''Propensity Score Matchig


Author: 
License: BSD-3

'''

import statsmodels.api as sm


class PropensityScoreMatch(object):
    def __init__(self, assigment_index, covariates, treatment_objective_variable):
        self.assigment_index = assigment_index
        self.covariates = covariates
        self.treatment_objective_variable = treatment_objective_variable
        
    def fit(self):
        self.result = PScoreMatchResult()
        self.compute_pscore()
        self.stratify()
        
    def compute_pscore(self):
        p_mod = sm.Logit(self.assigment_index, self.covariates)
        p_res = p_mod.fit()
        self.scores = p_res.predict(self.covariates)
        self.result.propensity_estimation = p_res
        
    def compute_stratas(self, strata = None):
        if not(strata is None) and self.check_balance_for(strata):
            return [strata,]
        half = (self.scores.max() + self.scores.min())/2
        left, right = self.scores < half, self.scores > half
        return self.compute_stratas(left) + self.compute_stratas(right)
            
        
        
    def check_balance_for(self, strata):
        endog, exog = self.assigment_index[strata], sm.add_constant(self.covariates[strata], prepend = True)
        om = sm.OLS(endog, exog)
        res= om.fit()
        return (res.f_pvalue > 0.05) and (result.pvalues > 0.05).all() 
        
        
class PScoreMatchResult(object):
    pass