'''Propensity Score Matchig


Author: 
License: BSD-3

'''

import statsmodels.api as sm
import numpy as np


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
        
    def compute_strata(self, strat):
        if self.assigment_index[strat].where(self.assigment_index[strat] == 1).count() <= 1:
            return []
        if self.check_balance_for(strat):
            return [strat,]
        else:
            return self.divide_strat(strat)
        
    def basic_strata(self):
        common = self.common_support()
        percentiles = np.percentile(self.scores[common], [0, 20, 40, 60, 80, 100])
        strata = []
        min = percentiles[0]
        for max in percentiles[1:]:
            strat = (self.scores >= min) & (self.scores <max)
            strata += self.compute_strata(strat)
        self.strata = strata
        return strata
    
    def divide_strat(self, strat):
        min, max = np.percentile(self.scores[strat], [0, 100])
        half = (min+ max)/2
        print min, max, half
        left, right = ((self.scores >= min) & (self.scores < half)) , ((self.scores >= half) & (self.scores < max))
        return self.compute_strata(left) + self.compute_strata(right)
        
        
    def strat_effect(self, strat):
        assigments = self.assigment_index[strat]
        objective = self.treatment_objective_variable[strat]
        diff = objective.where(assigments ==1).mean() - objective.where(assigments ==0).mean()
        return diff
    
    def weights(self):
        return [self.assigment_index[strat].where(self.assigment_index[strat] == 1).count() for strat in self.strata]
        
    def treatment_effect(self):
        weights = self.weights()
        return np.sum((w*self.strat_effect(strat) for w, strat in zip(weights, self.strata)))/np.sum(weights)
        
        
    def check_balance_for(self, strat):
        endog, exog = self.assigment_index[strat], sm.add_constant(self.covariates[strat], prepend = True)
        print self.covariates[strat].count()
        print endog.count()
        om = sm.OLS(endog, exog)
        res= om.fit()
        #print res.pvalues > 0.05
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