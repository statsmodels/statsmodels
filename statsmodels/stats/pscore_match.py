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
        
    def compute_stratas(self, strata = None, rec= 0):
        if not(strata is None) and self.check_balance_for(strata):
            return [strata,]
        if strata is None:
            strata = self.common_support()
  #      if strata is None:
   #         min, max = self.scores.min(), self.scores.max()
    #    else:
        min, max = self.scores[strata].min(), self.scores[strata].max()
        half = (min+ max)/2
        print min, max, half
        left, right = ((self.scores >= min) & (self.scores < half)) , ((self.scores >= half) & (self.scores < max))
        print left
        print right
        return self.compute_stratas(left, rec=rec +1) + self.compute_stratas(right, rec = rec +1)
            
        
        
    def check_balance_for(self, strata):
        endog, exog = self.assigment_index[strata], sm.add_constant(self.covariates[strata], prepend = True)
        om = sm.OLS(endog, exog)
        res= om.fit()
        print res.pvalues > 0.05
        if (res.pvalues > 0.05).all():
            print 'eh!'
            print res.f_pvalue
        return (res.f_pvalue > 0.05) and (res.pvalues > 0.05).all()
        
    def common_support(self):
        min_treated, max_treated = self.scores[self.assigment_index ==1].min(), self.scores[self.assigment_index ==1].max()
        min_control, max_control = self.scores[self.assigment_index ==0].min(), self.scores[self.assigment_index ==0].max()
        common_min, common_max = max(min_treated, min_control), min(max_treated, max_control)
        return (self.scores >= common_min) & (self.scores <= common_max)
        
        
class PScoreMatchResult(object):
    pass