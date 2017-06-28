"""
Methods for creating summary statistics and their SE for survey data.

The main classes are:

  * SurveyDesign : Parent class that creates attributes for easy 
  implementation of other methods. Attributes include relabeled 
  clusters, number of clusters per strata, etc.

  * SurveyStat : implements methods to calculate the standard 
  error of each statistic via either the bootstrap or jackknife

  * SurveyMean : Calculates the mean of each column

  * SurveyTotal : Calculates the total of each column

  * SurveyQuantile: Calculates the specified quantile[s] of each column
"""

import numpy as np 
import pandas as pd 

class SurveyDesign(object):
    """                                                                                                                                              
    Creates attributes that allow easy implementation of survey methods.                                                                                                   
                                                                                                                                                     
    Parameters                                                                                                                                          
    -------                                                                                                                                          
    strata : array-like or None                                                                                                                                 
        Strata for each observation. If none, an array 
        of ones is constructed                                                                                                                                 
    cluster : array-like or None                                                                                                                 
        Cluster for each observation. If none, an array 
        of ones is constructed                                                                                                         
    weights : array-like or None                                                                                                                               
        The weight for each observation. If none, an array 
        of ones is constructed
    nest : boolean
        allows user to specify if PSU's with the same
        PSU number in different strata are treated as distinct PSUs.

    Attributes
    ----------
    self.weights : (n, ) array
        The weight for each observation
    self.nstrat : integer
        The number of district strata
    self.sclust : (n, ) array
        The relabeled cluster array from 0, 1, ..
    self.strat : (n, ) array
        The related strata array from 0, 1, ...
    self.ncs : (self.nstrat, ) array
        Holds the number of clusters in each stratum
    self.sfclust : ndarray
        The stratum for each cluster
    self.nclust : integer
        The total number of clusters across strata
    """
    
    def __init__(self, strata=None, cluster=None, weights=None, nest=True):
        strata, cluster, self.weights = self._check_args(strata, cluster, weights)
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

        # The total number of clusters over all stratum
        self.nclust = np.sum(self.ncs)

    def summary(self):
        """
        The __str__ method for our data
        """
        print("Number of observations:", len(self.strat))
        print("Sum of weights:", self.weights.sum())
        print("Number of strata:", self.nstrat)
        print("The number of clusters per stratum:", self.ncs)


    def _check_args(self, strata, cluster, weights):
        """
        Minor error checking to make sure user supplied any of 
        strata, cluster, or weights. For unspecified subgroup labels
        an array of ones is created

        Parameters
        ----------
        strata : array-like or None                                                                                                                                 
            Strata for each observation. If none, an array 
            of ones is constructed                                                                                                                                 
        cluster : array-like or None                                                                                                                 
            Cluster for each observation. If none, an array 
            of ones is constructed                                                                                                         
        weights : array-like or None                                                                                                                               
            The weight for each observation. If none, an array 
            of ones is constructed

        Returns
        -------
        vals : list
            List of strata, cluster, and weights
        """
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
    """                                                                                                                                              
    Holds methods to calculate standard error of statistics.                                                                                                 
                                                                                                                                                     
    Parameters                                                                                                                                          
    -------                                                                                                                                          
    design : SurveyDesign object
    
    Attributes
    ----------
    self.est : ndarray                                                                                                                                    
        The point estimates of the statistic, calculated on the columns                                                               
        of data.                                                                                                                                     
    self.vc : ndarray                                                                                                                              
        The variance-covariance of the estimates.                                                                                                            
    self.pseudo : ndarray                                                                                                                                 
        The jackknife pseudo-values.
    """

    def __init__(self, design):
        """
        Inherits from SurveyDesign object

        Parameters
        ----------
        design : SurveyDesign object

        """
        self.design = design

    def bootstrap(self, stat, replicates=1000):
        """
        Calculates bootstrap standard errors

        Parameters
        ----------
        stat : object 
            Object of class SurveyMean, SurveyTotal, SurveyPercentile, etc
        replicates : integer
            The number of replicates that the user wishes to specify

        Returns                                                                                                                                          
        -------                                                                                                                                          
        est : ndarray                                                                                                                                    
            The point estimates of the statistic, calculated on the columns                                                               
            of data.                                                                                                                                     
        vc : ndarray                                                                                                                              
            The variance-covariance of the estimates.                                                                                                            
        pseudo : ndarray                                                                                                                                 
            The jackknife pseudo-values.
        """
        jdata = []
        for r in range(replicates):
            w = self.design.weights.copy()
            bin = np.zeros(self.design.nclust)
            for s in range(nstrat):
                # how to handle strata w/ only one cluster?
                w[self.design.strat == s] *= self.design.ncs[s] / float(self.design.ncs[s] - 1)
                
                # if there is only one or two clusters then weights will stay the same
                if (self.design.ncs[s] == 1 or self.design.ncs[s] == 2):
                    continue              
                # array of clusters to resample from
                ii = np.flatnonzero(self.design.sfclust == s)
                # resample them
                ii_resample = np.random.choice(ii, size = (self.design.ncs[s] - 1), replace=True)
                # accumulate number of times cluster i was resampled
                bin += np.bincount(ii_resample, minlength=max(self.design.sclust)+1)
            # augment weights
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

        Parameters
        ----------
        stat : object
            Object of class SurveyMean, SurveyTotal, SurveyPercentile, etc

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

        jdata = []
        # for each cluster
        for c in range(self.design.nclust):
            # get stratum that the cluster belongs in
            s = self.design.sfclust[c]
            nh = self.design.ncs[s]
            self.w = self.design.weights.copy()
            # all weights within the stratum are modified
            self.w[self.design.strat == s] *= nh / float(nh - 1)
            # but if you're within the cluster to be removed, set as 0
            self.w[self.design.sclust == c] = 0
            jdata.append(stat._stat(self.w))
        jdata = np.asarray(jdata)

        nh = self.design.ncs[self.design.sfclust].astype(np.float64)
        pseudo = jdata + nh[:, None] * (np.dot(self.w, stat.data) - jdata)

        for s in range(self.design.nstrat):
            # get indices of all clusters within a stratum
            ii = np.flatnonzero(self.design.sfclust == s)
            # center the 'delete 1' statistic
            jdata[ii, :] -= jdata[ii, :].mean(0)

        u = np.sqrt((nh - 1) / nh)
        jdata = u[:, None] * jdata
        vc = np.dot(jdata.T, jdata)
        est = stat._stat(self.design.weights)

        return est, vc, pseudo



class SurveyMean(SurveyStat):
    """                                                                                                                                              
    Calculates the mean for each column.                                                                                                 
                                                                                                                                                     
    Parameters                                                                                                                                          
    -------                                                                                                                                          
    design : SurveyDesign object
    data : ndarray
        nxp array of the data to calculate the mean on
    method: string
        User inputs whether to get bootstrap or jackknife SE
    
    Attributes
    ----------
    self.data : ndarray
        The data which to calculate the mean on
    self.design :
        Points to the SurveyDesign object
    self.est : ndarray                                                                                                                                    
        The point estimates of the statistic, calculated on the columns                                                               
        of data.                                                                                                                                     
    self.vc : ndarray                                                                                                                              
        The variance-covariance of the estimates.                                                                                                            
    self.pseudo : ndarray                                                                                                                                 
        The jackknife pseudo-values.
    """
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

    def _stat(self, weights):
        """                                                                                                                                              
        Returns calculation of mean.                                                                                            
                                                                                                                                                         
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

        # weights /= weights.sum()

        return np.dot(weights, self.data) / np.sum(weights)

class SurveyTotal(SurveyStat):
    """                                                                                                                                              
    Calculates the total for each column.                                                                                                 
                                                                                                                                                     
    Parameters                                                                                                                                          
    -------                                                                                                                                          
    design : SurveyDesign object
    data : ndarray
        nxp array of the data to calculate the mean on
    method: string
        User inputs whether to get bootstrap or jackknife SE
    
    Attributes
    ----------
    self.data : ndarray
        The data which to calculate the mean on
    self.design :
        Points to the SurveyDesign object
    self.est : ndarray                                                                                                                                    
        The point estimates of the statistic, calculated on the columns                                                               
        of data.                                                                                                                                     
    self.vc : ndarray                                                                                                                              
        The variance-covariance of the estimates.                                                                                                            
    self.pseudo : ndarray                                                                                                                                 
        The jackknife pseudo-values.
    """
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
        Returns calculation of mean.                                                                                            
                                                                                                                                                         
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
        return np.dot(weights, self.data)

class SurveyQuantile(SurveyStat):
    """                                                                                                                                              
    Calculates the qualtile[s] for each column.                                                                                                 
                                                                                                                                                     
    Parameters                                                                                                                                          
    -------                                                                                                                                          
    design : SurveyDesign object
    data : ndarray
        nxp array of the data to calculate the mean on
    parameter: array-like
        array of quantiles to calculate for each column
    
    Attributes
    ----------
    self.data : ndarray
        The data which to calculate the quantiles on
    self.design :
        Points to the SurveyDesign object
    self.est : ndarray                                                                                                                                    
        The point estimates of the statistic, calculated on the columns
        of data.
    self.quantile : ndarray
        The quantile[s] to calculate for each column
    self.cumsum_weights : ndarray
        The cumulative sum of self.design.weights
    self.vc : ndarray
        The variance-covariance of the estimates.
    self.pseudo : ndarray
        The jackknife pseudo-values.
    """

    def __init__(self, design, data, quantile):
        self.data = np.asarray(data)
        self.design = design
        self.quantile = np.asarray(quantile)

        # give warning if user entered in quantile bigger than one
        large_q = np.asarray([x > 1 for x in self.quantile])
        n = large_q.sum()
        if n > 0:
            print("warning:", n, "inputed quantile[s] > 1")
        self.cumsum_weights = np.cumsum(self.design.weights)
        self.n_cw = len(self.cumsum_weights)

        # get quantile[s] for each column
        self.est = [self._stat(index) for index in range(self.data.shape[1])]

    def _stat(self, col_index):
        quant_list = []
        sorted_data = np.sort(self.data[:, col_index])
        q = self.quantile.copy() * self.cumsum_weights[-1]
        # find index i such that self.cumsum_weights[i] >= q
        ind = np.searchsorted(self.cumsum_weights, q)

        for i, pos in enumerate(ind):
            # if searchsorted returns length of list
            # return last observation
            if pos in np.array([self.n_cw - 1, self.n_cw]):
                quant_list.append(sorted_data[-1])
                continue
            if (self.cumsum_weights[pos] == q[i]):
                quant_list.append((sorted_data[pos] + sorted_data[pos+1]) / 2)
            else:
                quant_list.append(sorted_data[pos])
        return quant_list

class SurveyMedian(SurveyQuantile):
    """
    Wrapper function that calls SurveyQuantile
    with quantile = [.50]
    """
    def __init__(self, SurveyDesign, data):
        # sp = super(SurveyMedian, self).__init__(SurveyDesign, data, [50])
        sp = SurveyQuantile(SurveyDesign, data, [.50])
        self.est = sp.est


strata = np.r_[1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0]
cluster = np.r_[0, 0, 2, 2, 3, 3, 4, 4, 4, 6, 6, 6]
weights = np.r_[1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2].astype(np.float64)
data = np.asarray([[1, 3, 2, 5, 4, 1, 2, 3, 4, 6, 9, 5],
                   [5, 3, 2, 1, 4, 7, 8, 9, 5, 4, 3, 5]], dtype=np.float64).T

design = SurveyDesign(strata, cluster, weights)
avg = SurveyMean(design,data, 'jack')
tot = SurveyTotal(design, data, 'jack')
# print(avg.est)
# print(avg.vc)
print(tot.est)
print(tot.vc)
quant = SurveyQuantile(design, data, [.1, .25, .33, .5, .75, .99])
print(quant.est)