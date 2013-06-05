'''Propensity Score Matchig


Author: 
License: BSD-3

'''

import statsmodels.api as sm
import numpy as np
from statsmodels.stats.weightstats import CompareMeans, DescrStatsW
import pandas as pd


class PropensityScoreMatch(object):
    def __init__(self, assigment_index, covariates, treatment_objective_variable, caliper = 0.01):
        self.assigment_index = assigment_index
        self.covariates = covariates
        self.treatment_objective_variable = treatment_objective_variable
        self.matching_algo = StrataMatchingAlgorithm(self)
        #self.matching_algo = CaliperMatchingAlgorithm(self, caliper)
        self.use_comm_sup = True
        
    def fit(self):
        self.result = PScoreMatchResult()
        self.compute_pscore()
        self.matching_algo.fit()
        
    def compute_pscore(self):
        p_mod = sm.Logit(self.assigment_index, self.covariates)
        p_res = p_mod.fit()
        print p_res.summary()
        self.scores = pd.Series(p_res.predict(self.covariates))
        self.result.propensity_estimation = p_res
        if self.use_comm_sup:
            comm_sup = self.common_support()
            scores = self.scores[comm_sup]
            print 'Using common support: %s' % str([scores.min(), scores.max()]) 
        else:
            scores = self.scores
        print 'propensity scores description'
        print scores.describe()
        
        
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
    class Stratum(object):
        def __init__(self, ps, index):
            self.ps = ps
            self.index = index
            
            
        def scores(self):
            return self.ps.scores[self.index]
            
        def bisect(self):
            scores = self.scores()
            min, max = scores.min(), scores.max()
            half = (min + max)/2
            all_scores = self.ps.scores
            left, right = ((all_scores >= min) & (all_scores < half)) , ((all_scores >= half) & (all_scores < max))
            return StrataMatchingAlgorithm.Stratum(self.ps, self.index & left), StrataMatchingAlgorithm.Stratum(self.ps, self.index & right)
            
        def describe(self):
            scores = self.scores()
            min, max = scores.min(), scores.max()
            half = (min+ max)/2
            print 'stratum min, max, half: ' + str([min, max, half])
            
        def treated(self):
            return self.index & (self.ps.assigment_index == 1)
            
        def control(self):
            return self.index & (self.ps.assigment_index == 0)
            
        def check_balance(self):
            treated = self.ps.covariates[self.treated()]
            control = self.ps.covariates[self.control()]
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
            
        def strat_effect(self):
            objective = self.ps.treatment_objective_variable
            diff = objective[self.treated()].mean() - objective[self.control()].mean()
            return diff
        
        def weight(self):
            return self.ps.assigment_index[self.treated()].count()
            
        def has_treated(self):
            return self.weight() > 0
                
    def __init__(self, psmatch):
        self.ps = psmatch
        
    def compute_strata(self, strat):
        if not strat.has_treated():
            return []
        if strat.check_balance():
            return [strat,]
        else:
            left, right = strat.bisect()
            return self.compute_strata(left) + self.compute_strata(right)
        
    def fit(self):
        common = self.ps.common_support()
        scores = self.ps.scores
        limits = np.linspace(scores[common].min(), scores[common].max(), 5)
        strata = []
        min = limits[0]
        for max in limits[1:]:
            strat = StrataMatchingAlgorithm.Stratum(self.ps, (scores >= min) & (scores < max))
            strata += self.compute_strata(strat)
            min = max
        self.strata = strata
        return strata

        
    def treatment_effect(self):
        weights = self.weights()
        return np.sum((w*self.strat_effect(strat) for w, strat in zip(weights, self.strata)))/np.sum(weights)
        
        
        
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
        
        
        
    
    