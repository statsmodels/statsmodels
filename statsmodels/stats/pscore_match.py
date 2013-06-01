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
        self.matching_algo = StrataMatchingAlgorithm(self)
        
    def fit(self):
        self.result = PScoreMatchResult()
        self.compute_pscore()
        self.matching_algo.fit()
        
    def compute_pscore(self):
        p_mod = sm.Logit(self.assigment_index, self.covariates)
        p_res = p_mod.fit()
        self.scores = p_res.predict(self.covariates)
        self.result.propensity_estimation = p_res
        
    def treatment_effect(self):
        return self.matching_algo.treatment_effect()
        
    def treated(self):
        return self.assigment_index == 1
    
    def control(self):
        return self.assigment_index == 0
        
 
        
    def common_support(self):
        treated, control = self.treated(), self.control()
        min_treated, max_treated = self.scores[treated].min(), self.scores[treated].max()
        min_control, max_control = self.scores[control].min(), self.scores[control].max()
        common_min, common_max = max(min_treated, min_control), min(max_treated, max_control)
        return (self.scores >= common_min) & (self.scores <= common_max)
        
        
class PScoreMatchResult(object):
    pass

class StrataMatchingAlgorithm(object):
    def __init__(self, psmatch):
        self.ps = psmatch
        
    def compute_strata(self, strat):
        assigment = self.ps.assigment_index[strat]
        if assigment.where(assigment == 1).count() <= 1:
            return []
        if self.check_balance_for(strat):
            return [strat,]
        else:
            return self.divide_strat(strat)
        
    def fit(self):
        common = self.ps.common_support()
        scores = self.ps.scores
        percentiles = np.percentile(scores[common], [0, 20, 40, 60, 80, 100])
        strata = []
        min = percentiles[0]
        for max in percentiles[1:]:
            strat = (scores >= min) & (scores <max)
            strata += self.compute_strata(strat)
        self.strata = strata
        return strata
    
    def divide_strat(self, strat):
        scores = self.ps.scores
        min, max = np.percentile(scores[strat], [0, 100])
        half = (min+ max)/2
        print min, max, half
        left, right = ((scores >= min) & (scores < half)) , ((scores >= half) & (scores < max))
        return self.compute_strata(left) + self.compute_strata(right)
        
        
    def strat_effect(self, strat):
        treated, control = self.ps.treated()[strat], self.ps.control()[strat]
        objective = self.ps.treatment_objective_variable[strat]
        diff = objective.where(treated).mean() - objective.where(control).mean()
        return diff
    
    def weights(self):
        treated = self.ps.treated()
        return [self.ps.assigment_index[strat].where(treated).count() for strat in self.strata]
        
    def treatment_effect(self):
        weights = self.weights()
        return np.sum((w*self.strat_effect(strat) for w, strat in zip(weights, self.strata)))/np.sum(weights)
        
        
    def check_balance_for(self, strat):
        endog, exog = self.ps.assigment_index[strat], sm.add_constant(self.ps.covariates[strat], prepend = True)
        #print self.ps.covariates[strat].count()
        #print endog.count()
        om = sm.OLS(endog, exog)
        res= om.fit()
        #print res.pvalues > 0.05
        if (res.pvalues > 0.05).all():
            print 'eh!'
            print res.f_pvalue
        return (res.f_pvalue > 0.05) and (res.pvalues > 0.05).all()
    