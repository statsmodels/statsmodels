'''Propensity Score Matchig


Author: 
License: BSD-3

'''

import statsmodels.api as sm
import numpy as np
from statsmodels.stats.weightstats import CompareMeans, DescrStatsW
import pandas as pd
from scipy import stats


class PropensityScoreMatch(object):
    def __init__(self, assigment_index, covariates, treatment_objective_variable,\
                 algo = 'strata', radius = 0.01, regression = 'logit'):
        self.assigment_index = assigment_index
        self.covariates = covariates
        self.treatment_objective_variable = treatment_objective_variable
        if regression == 'logit':
            self.regression_class = sm.Logit
        elif regression == 'probit':
            self.regression_class = sm.Probit
        else:
            raise
        if algo == 'strata':
            self.matching_algo = StrataMatchingAlgorithm(self)
        elif algo == 'radius':
            self.matching_algo = RadiusMatchingAlgorithm(self, radius)
        else:
            raise
        self.use_comm_sup = True
        
    def fit(self):
        self.result = PScoreMatchResult()
        self.compute_pscore()
        self.matching_algo.fit()
        
    def compute_pscore(self):
        covariates = sm.add_constant(self.covariates, prepend=True)
        p_mod = self.regression_class(self.assigment_index, covariates)
        p_res = p_mod.fit(disp=False)
        #print p_res.summary()
        self.scores = pd.Series(p_res.predict(covariates), \
                                index=self.assigment_index.index)
        self.result.propensity_estimation = p_res
        if self.use_comm_sup:
            comm_sup = self.common_support()
            scores = self.scores[comm_sup]
            #print 'Using common support: %s' % str([scores.min(), scores.max()]) 
        else:
            scores = self.scores
        #print 'propensity scores description'
        #print scores.describe()
        
        
    def treatment_effect(self):
        return self.matching_algo.treatment_effect()
        
    def treatment_variance(self):
        return self.matching_algo.treatment_variance()
        
    def treatment_std(self):
        return np.sqrt(self.treatment_variance())
        
    def confidence_interval(self, alpha=0.05):
        #assuming normality (don't know df)
        mean, std = self.treatment_effect(), self.treatment_std()
        return stats.norm.interval(1-alpha, loc=mean, scale=std )
        
        
    def _basic_treated(self):
        return self.assigment_index == 1
    
    def _basic_control(self):
        return self.assigment_index == 0
    
    def treated(self):
        return self._basic_treated() & self.common_support() 
    
    def control(self):
        return self._basic_control() & self.common_support()
        
 
        
    def common_support(self):
        if not self.use_comm_sup:
            return True
        treated, control = self._basic_treated(), self._basic_control()
        min_treated, max_treated = self.scores[treated].min(), self.scores[treated].max()
        min_control, max_control = self.scores[control].min(), self.scores[control].max()
        common_min, common_max = max(min_treated, min_control), min(max_treated, max_control)
        return (self.scores >= common_min) & (self.scores <= common_max)
        
        
class PScoreMatchResult(object):
    pass

class Stratum(object):
    def __init__(self, ps, min, max):
        self.ps = ps
        self.index = ps.common_support() & (ps.scores >= min) & (ps.scores < max)
        self.min, self.max = min, max
        
    def bisect(self):
        half = (self.min + self.max)/2
        return Stratum(self.ps, self.min, half), Stratum(self.ps, half, self.max)
        
    def describe(self):
        scores = self.ps.scores[self.index]
        min, max = scores.min(), scores.max()
        half = (min+ max)/2
        print 'stratum min, half, max: ' + str([min, half, max])
        print 'Treated, Control:' + str([self.ps.assigment_index[x].count() for x in (self.treated(), self.control())])
        
    def treated(self):
        return self.ps.treated() & self.index 
        
    def control(self):
        return self.ps.control() & self.index
        
    def check_balance(self):
        return self.check_balance_for(self.ps.scores) and self.check_balance_for(self.ps.covariates)
        
    def check_balance_for(self, data):
        treated, control = data[self.treated()], data[self.control()]
        cm = CompareMeans(DescrStatsW(treated), DescrStatsW(control))
        test = cm.ttest_ind()
        pvalues = test[1]
        alpha = 0.005
        return np.all(pvalues > alpha)
        
    def treatment_effect(self):
        objective = self.ps.treatment_objective_variable
        diff = objective[self.treated()].mean() - objective[self.control()].mean()
        return diff
    
    def weighted_treatment_effeect(self):
        return self.treatment_effect() * self.weight()
    
    def weight(self):
        return self.ps.assigment_index[self.treated()].count()
        
    def empty(self):
        return not (np.any(self.treated()) & np.any(self.control()))

class StrataMatchingAlgorithm(object):
    def __init__(self, psmatch):
        self.ps = psmatch
        
    def compute_strata(self, strat):
        if strat.empty():
            return []
        if strat.check_balance():
            return [strat,]
        else:
            left, right = strat.bisect()
            return self.compute_strata(left) + self.compute_strata(right)
            
            
    def basic_strata(self):
        #limits = np.linspace(scores[common].min(), scores[common].max(), 5+1)
        limits = np.linspace(0, 1, 5+1)
        strata = []
        min = limits[0]
        for max in limits[1:]:
            strat = Stratum(self.ps, min, max)
            strata.append(strat)
            min = max
        return strata
        
    def fit(self):
        self.strata = []
        for strat in self.basic_strata():
            self.strata += self.compute_strata(strat)
        print 'found: %d strata' % len(self.strata)
        [strat.describe() for strat in self.strata]
        return self.strata

        
    def treatment_effect(self):
        weights = self.weights()
        return np.sum([strat.weighted_treatment_effeect() for strat in self.strata])/np.sum(weights)
        
    def weights(self):
        return (strat.weight() for strat in self.strata)
        
        
        
class RadiusMatchingAlgorithm(object):
    def __init__(self, psmatch, radius):
        self.psmatch = psmatch
        self.radius = radius
        
    def neighbors_of(self, index):
        scores = self.psmatch.scores
        score = scores[index]
        return self.psmatch.control() & (np.abs(scores - score) < self.radius)
        
    def matches(self):
        self.matched = {}
        for idx in self.psmatch.scores[self.psmatch.treated()].index:
            neighbors = self.neighbors_of(idx)
            if np.any(neighbors):
                self.matched[idx] = neighbors
                    
    def treat_effect_per_treat(self, treated_id, nb):
        return (self.psmatch.treatment_objective_variable[treated_id]) - (self.psmatch.treatment_objective_variable[nb].mean())
        
    def treatment_effect(self):
        return np.mean([self.treat_effect_per_treat(key, value) for key, value in self.matched.items()])
        
    def treatment_variance(self):
        treated = self.matched.keys()
        controls = self.matched.values()
        cvar = 0
        for matched_control in controls:
            count = float(self.psmatch.treatment_objective_variable[matched_control].count())
            if count > 1:
                cvar  +=  self.psmatch.treatment_objective_variable[matched_control].var()/(count**2)
        tvar = len(treated) *(self.psmatch.treatment_objective_variable[treated].var())
        return 1/float(len(treated)**2)*(tvar+cvar)
        
        
    def matched_control(self):
        return np.any(self.matched.values(), axis=0).sum()
        
    def matched_treatment(self):
        return len(self.matched)
        
    def describe(self):
        print 'radius: %f\t treated :%d control: %d \t treatment effect: %f' % (self.radius, self.matched_treatment(), self.matched_control(), self.treatment_effect())
        
    def fit(self):
        self.matches()
        
        
        
    
    