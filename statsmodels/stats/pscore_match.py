'''Propensity Score Matchig


Author: 
License: BSD-3

'''

import statsmodels.api as sm
import numpy as np
from statsmodels.stats.weightstats import CompareMeans, DescrStatsW


class PropensityScoreMatch(object):
    def __init__(self, assigment_index, covariates, treatment_objective_variable, caliper = 0.01):
        self.assigment_index = assigment_index
        self.covariates = covariates
        self.treatment_objective_variable = treatment_objective_variable
        #self.matching_algo = StrataMatchingAlgorithm(self)
        self.matching_algo = CaliperMatchingAlgorithm(self, caliper)
        
    def fit(self):
        self.result = PScoreMatchResult()
        self.compute_pscore()
        self.matching_algo.fit()
        
    def compute_pscore(self):
        p_mod = sm.Logit(self.assigment_index, self.covariates)
        p_res = p_mod.fit()
        print p_res.summary()
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
            strat = (scores >= min) & (scores < max)
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
        treated = self.ps.covariates[strat][self.ps.assigment_index[strat] == 1]
        control = self.ps.covariates[strat][self.ps.assigment_index[strat] == 0]
        cm = CompareMeans(DescrStatsW(treated), DescrStatsW(control))
        alpha = 0.2
        test = cm.ttest_ind()
        pvalues = test[1]
        print test[0]
        print pvalues
        print pvalues > alpha
        print np.all(pvalues > alpha)

        if np.all(pvalues > alpha):
            print 'eh!'
            print test[0]
            print pvalues
        return np.all(pvalues > alpha)
        
        
        
class CaliperMatchingAlgorithm(object):
    def __init__(self, psmatch, caliper):
        self.psmatch = psmatch
        self.caliper = caliper
        
    def neighbors_of(self, index):
        scores = self.psmatch.scores
        score = scores[index]
        common = self.psmatch.common_support()
        answer = []
        return common & self.psmatch.control() & (np.abs(scores - score) < self.caliper)
        
    def matches(self):
        self.matched = {}
        commonT = self.psmatch.common_support() & self.psmatch.treated()
        for idx, value in enumerate(commonT):
            if value:
                neighbors = self.neighbors_of(idx)
                if np.any(neighbors):
                    self.matched[idx] = neighbors
                    
    def treat_effect_per_treat(self, treated_id, nb):
        return self.psmatch.treatment_objective_variable[treated_id] - np.mean(self.psmatch.treatment_objective_variable[nb])
        
    def treatment_effect(self):
        return np.mean([self.treat_effect_per_treat(key, value) for key, value in self.matched.items()])
        
    def fit(self):
        self.matches()
        
        
        
    
    