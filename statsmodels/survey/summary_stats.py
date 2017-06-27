import numpy as np 
import pandas as pd 

class SurveyDesign(object):

    def __init__(self, strata=None, cluster=None, weights=None, nest=True):

        strata, cluster, self.weights = self._check_args(strata, cluster, weights)
        # if any of them are none, we should make them an array of ones
        # how to know the len each array should be efficiently?
        self.weights = weights
        # Recode strata and clusters as integer values 0, 1, ...                                                                                             
        _, self.strat = np.unique(strata, return_inverse=True)
        _, clust = np.unique(cluster, return_inverse=True)

        # the number of distinct strata
        self.nstrat = max(self.strat) + 1

        # If requested, recode the PSUs to be sure that the same PSU number in                                                                               
        # different strata are treated as distinct PSUs.  This is the same as                                                                                
        # the nest option in R.                                                                                                                              
        if nest:
            m = max(clust) + 1
            sclust = clust + m*self.strat
            _, self.sclust = np.unique(sclust, return_inverse=True)
        else:
            self.sclust = clust.copy()
            
        # The number of clusters per stratum                                                                                                                 
        _, ii = np.unique(self.sclust, return_index=True)
        self.ncs = np.bincount(self.strat[ii])

        # The stratum for each cluster                                                                                                                       
        _, ii = np.unique(self.sclust, return_index=True)
        self.sfclust = self.strat[ii]

    def summary(self):
        print("Number of observations:", len(self.strat))
        print("Sum of weights:", self.weights.sum())
        print("Number of strata:", self.nstrat)
        print("The number of clusters per stratum:", self.ncs)

    def _check_args(self, strata, cluster, weights):
        if all([x is None for x in (strata, cluster, weights)]):
            raise ValueError("At least one of strata, cluster, and weights musts not be None")
        v = [len(x) for x in (strata, cluster, weights) if x is not None]
        if len(set(v)) != 1:
            raise ValueError("lengths of strata, cluster, and weights are not compatible")
        n = v[0]
        vals = []
        for x in (strata, cluster, weights):
            if x is None:
                vals.append(np.ones(n))
            else:
                vals.append(np.asarray(x))

        return vals[0], vals[1], vals[2] 


class SurveyStat(SurveyDesign):

    def __init__(self, design):
        self.design = design

    def bootstrap(self, stat, replicates=1000):
        jdata = []
        for r in range(replicates):
            w = self.design.weights.copy()
            bin = np.zeros(max(self.design.sclust) + 1)
            for s in range(nstrat):
                # how to handle strata w/ only one cluster?
                w[self.design.strat == s] *= self.design.ncs[s] / float(self.design.ncs[s] - 1)
                # if there is only one or two clusters then weights will stay the same
                if (self.design.ncs[s] == 1 or self.design.ncs[s] == 2):
                    continue
                ## array of clusters to resample from
                ii = np.flatnonzero(self.design.sfclust == s)
                ## resample them
                ii_resample = np.random.choice(ii, size = (self.design.ncs[s] - 1), replace=True)
                ## accumulate number of times cluster i was resampled
                bin += np.bincount(ii_resample, minlength=max(self.design.sclust)+1)
            ## augment weights
            w *= bin[self.design.sclust]
            # call the stat w/ the new weights
            jdata.append(stat._stat(weights=w))
        jdata = np.asarray(jdata)
        # nh = self.design.ncs[self.design.sfclust].astype(np.float64)
        # pseudo = jdata + nh[:, None] * (np.dot(w, stat.data) - jdata)

        boot_mean = jdata.mean(0)
        var = ((jdata - boot_mean)**2).sum(0) / (replicates - 1)
        est = stat._stat(self.design.weights)
        return est, var

    def jack(self, stat):
        """                                                                                                                                              
        Jackknife variance estimation for survey data.                                                                                                   
                                                                                                                                                         
        Returns                                                                                                                                          
        -------                                                                                                                                          
        est : ndarray                                                                                                                                    
            The point estimates of the statistic, calculated on the columns                                                                              
            of data.                                                                                                                                     
        vc : square ndarray                                                                                                                              
            The variance-covariance matrix of the estimates, obtained using                                                                              
            the (drop 1) jackknife procedure.                                                                                                            
        pseudo : ndarray                                                                                                                                 
            The jackknife pseudo-values.                                                                                                                 
        """

        ngrp = max(self.design.sclust) + 1
        jdata = []
        for c in range(ngrp):
            s = self.design.sfclust[c]
            nh = self.design.ncs[s]
            self.w = self.design.weights.copy()
            # all weights within the strat are modified
            self.w[self.design.strat == s] *= nh / float(nh - 1)
            # but if you're within the cluster to be removed, set as 0
            self.w[self.design.sclust == c] = 0
            jdata.append(stat._stat(self.w))
        jdata = np.asarray(jdata)

        nh = self.design.ncs[self.design.sfclust].astype(np.float64)
        pseudo = jdata + nh[:, None] * (np.dot(self.w, stat.data) - jdata)

        for s in range(self.design.nstrat):
            ii = np.flatnonzero(self.design.sfclust == s)
            jdata[ii, :] -= jdata[ii, :].mean(0)

        u = np.sqrt((nh - 1) / nh)
        jdata = u[:, None] * jdata
        vc = np.dot(jdata.T, jdata)
        est = stat._stat(self.design.weights)

        return est, vc, pseudo



class SurveyMean(SurveyStat):
    
    def __init__(self, design, data, method):
        self.data = np.asarray(data)
        self.design = design
        super().__init__(design)
        if method == "jack":
            self.est, self.vc, self.pseudo = super().jack(self)
            self.vc = np.sqrt(np.diag(self.vc))

        elif method == "boot":
            self.est, self.vc = super().bootstrap(self)
            self.vc = np.sqrt(self.vc)
    # default is the original design weights, overridden when 
    # weights are recalculated via jk, boot, etc
    def _stat(self, weights):
        """                                                                                                                                              
        Calculate a statistic with possible cluster deletion.                                                                                            
                                                                                                                                                         
        Parameters                                                                                                                                       
        ----------                                                                                                                                       
        weights : np.array                                                                                                                                  
            The weights used to calculate the mean, will either be
            original design weights or recalculated weights via jk,
            boot, etc                                                                                                           
                                                                                                                                                         
        Returns                                                                                                                                          
        -------                                                                                                                                          
        An array containing the statistic calculated on the columns                                                                                      
        of the dataset.                                                                                                                                  
        """

        weights /= weights.sum()

        return np.dot(weights, self.data) / np.sum(weights)

class SurveyTotal(SurveyStat):
    def __init__(self, design, data, method):
        super().__init__(design)
        self.design = design
        self.data = np.asarray(data)
        if method == "jack":
            self.est, self.vc, self.pseudo = super().jack(self)
            self.vc = np.sqrt(np.diag(self.vc))
        elif method == "boot":
            self.est, self.vc = super().bootstrap(self)
            self.vc = np.sqrt(self.vc)
    def _stat(self,weights):
        """                                                                                                                                              
        Calculate a statistic with possible cluster deletion.                                                                                            
                                                                                                                                                         
        Parameters                                                                                                                                       
        ----------                                                                                                                                       
        c : int or None                                                                                                                                  
            If an integer, return the statistic calculated with                                                                                          
            cluster c deleted.  If c is None, return the statistic                                                                                       
            calculated for the whole data set.                                                                                                           
                                                                                                                                                         
        Returns                                                                                                                                          
        -------                                                                                                                                          
        An array containing the statistic calculated on the columns                                                                                      
        of the dataset.                                                                                                                                  
        """
        return np.dot(weights, self.data)

class SurveyQuantile(SurveyStat):

    def __init__(self, design, data, quantile):
        self.data = np.asarray(data)
        self.design = design
        self.quantile = np.asarray(quantile)
        large_q = np.asarray([x > 1 for x in self.quantile])
        if large_q.sum() > 0:
            print("warning:", large_q.sum(), "inputed quantile > 1")
        self.cumsum_weights = np.cumsum(self.design.weights)
        self.est = [self._stat(index) for index in range(self.data.shape[1])]

    def _stat(self, col_index):
        perc_list = []
        for q in self.quantile:
            if q >= 1:
                perc_list.append(sorted_data[-1])
                continue
            q *= (self.cumsum_weights[-1])
            sorted_data = np.sort(self.data[:, col_index])
            pos = np.searchsorted(self.cumsum_weights, q)
            if pos in np.array([len(self.cumsum_weights), len(self.cumsum_weights) -1]):
                perc_list.append(sorted_data[-1])
                continue
            if (sorted_data[pos] == q):
                perc_list.append((sorted_data[pos] + sorted_data[pos+1]) / 2)
            else:
                perc_list.append(sorted_data[pos])
        return perc_list

class SurveyMedian(SurveyQuantile):

    def __init__(self, SurveyDesign, data):
        # sp = super(SurveyMedian, self).__init__(SurveyDesign, data, [50])
        sp = SurveyQuantile(SurveyDesign, data, [.50])
        self.est = sp.est
