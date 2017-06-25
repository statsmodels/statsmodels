import numpy as np 
import pandas as pd 

class SurveyDesign(object):

    def __init__(self, strata=None, cluster=None, weights=None, nest=True):

    
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


    def show(self):
        print(self.sfclust)
        print(self.ncs)
        print(self.sclust)
        print(self.nstrat)

    def summary(self):
        print("Number of observations:", len(self.strat))
        print("Sum of weights:", self.weights.sum())
        print("Number of strata:", self.nstrat)
        print("The number of clusters per stratum:", self.ncs)

class SurveyStat(object):

    def __init__(self):
        pass

    def jack(self,stat, SurveyDesign):
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

        ngrp = max(SurveyDesign.sclust) + 1
        jdata = [stat._stat(SurveyDesign,j) for j in range(ngrp)]
        jdata = np.asarray(jdata)

        nh = SurveyDesign.ncs[SurveyDesign.sfclust].astype(np.float64)
        pseudo = jdata + nh[:, None] * (np.dot(stat.w, stat.data) - jdata)

        for s in range(SurveyDesign.nstrat):
            ii = np.flatnonzero(SurveyDesign.sfclust == s)
            jdata[ii, :] -= jdata[ii, :].mean(0)

        u = np.sqrt((nh - 1) / nh)
        jdata = u[:, None] * jdata
        vc = np.dot(jdata.T, jdata)
        est = stat._stat(SurveyDesign)

        return est, vc, pseudo



class SurveyMean(SurveyDesign):
    
    def __init__(self, SurveyDesign, data):
        self.data = np.asarray(data)
        ss = SurveyStat()
        self.est, self.vc, self.pseudo = ss.jack(self, SurveyDesign)
        self.vc = np.sqrt(np.diag(self.vc))

    def _stat(self, SurveyDesign,c=None):
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

        if c is None:
            return np.dot(SurveyDesign.weights, self.data) / np.sum(SurveyDesign.weights)

        s = SurveyDesign.sfclust[c]
        nh = SurveyDesign.ncs[s]
        self.w = SurveyDesign.weights.copy()
        self.w[SurveyDesign.strat == s] *= nh / float(nh - 1)
        self.w[SurveyDesign.sclust == c] = 0
        self.w /= self.w.sum()
        return np.dot(self.w, self.data)


class SurveyTotal(SurveyDesign):
    def __init__(self, SurveyDesign, data):
        self.data = np.asarray(data)
        ss = SurveyStat()
        self.est, self.vc, self.pseudo = ss.jack(self, SurveyDesign)
        self.vc = np.sqrt(np.diag(self.vc))

    def _stat(self, SurveyDesign,c=None):
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

        if c is None:
            return np.dot(SurveyDesign.weights, self.data)

        s = SurveyDesign.sfclust[c]
        nh = SurveyDesign.ncs[s]
        self.w = SurveyDesign.weights.copy()
        self.w[SurveyDesign.strat == s] *= nh / float(nh - 1)
        self.w[SurveyDesign.sclust == c] = 0
        return np.dot(self.w, self.data)

class SurveyPercentile(SurveyDesign):

    def __init__(self, SurveyDesign, data, percentile):
        self.data = data
        self.cumsum_weights = np.cumsum(SurveyDesign.weights)
        self.est = [[self._stat(perc, index) for index in range(self.data.shape[1])] for perc in percentile]

    def _stat(self, percentile, col_index):

        percentile *= (self.cumsum_weights[-1] / 100)
        sorted_data = np.sort(self.data[:, col_index])
        pos = np.searchsorted(self.cumsum_weights, percentile)
        if (sorted_data[pos] == percentile):
            percentile = (sorted_data[pos] + sorted_data[pos+1]) / 2
        else:
            percentile = sorted_data[pos]
        return percentile

class SurveyMedian(SurveyPercentile):

    def __init__(self, SurveyDesign, data):
        # sp = super(SurveyMedian, self).__init__(SurveyDesign, data, [50])
        sp = SurveyPercentile(SurveyDesign, data, [50])
        self.est = sp.est
