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
# import pandas as pd


class SurveyDesign(object):
    """
    Description of a survey design, used by most methods
    implemented in this module.

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
    weights : (n, ) array
        The weight for each observation
    nstrat : integer
        The number of district strata
    sclust : (n, ) array
        The relabeled cluster array from 0, 1, ..
    strat : (n, ) array
        The related strata array from 0, 1, ...
    ncs : (self.nstrat, ) array
        Holds the number of clusters in each stratum
    sfclust : ndarray
        The stratum for each cluster
    nclust : integer
        The total number of clusters across strata
    """

    def __init__(self, strata=None, cluster=None, weights=None, rep_weights=None, nest=True):
        strata, cluster, self.weights = self._check_args(strata, cluster, weights)
        self.rep_weights = rep_weights
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

    def __str__(self):
        """
        The __str__ method for our data
        """
        summary_list = ["Number of observations: ", str(len(self.strat)),
                        "Sum of weights: ", str(self.weights.sum()),
                        "Number of strata: ", str(self.nstrat),
                        "Number of clusters per stratum:", str(self.ncs)]

        return "\n".join(summary_list)
        # print("Number of observations: ", len(self.strat))
        # print("Sum of weight: ", self.weights.sum())
        # print("Number of strata: ", self.nstrat)
        # print("The number of clusters per stratum: ", self.ncs)

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
        vals[0] : ndarray
            array of the strata labels
        vals[1] : ndarray
            array of the cluster labels
        vals[2] : ndarray
            array of the observation weights
        """
        if all([x is None for x in (strata, cluster, weights)]):
            raise ValueError("""At least one of strata, cluster, and weights
                             musts not be None""")
        v = [len(x) for x in (strata, cluster, weights) if x is not None]
        if len(set(v)) != 1:
            raise ValueError("""lengths of strata, cluster, and weights
                             are not compatible""")
        n = v[0]
        vals = []
        for x in (strata, cluster, weights):
            if x is None:
                vals.append(np.ones(n))
            else:
                vals.append(np.asarray(x))

        return vals[0], vals[1], vals[2]


class SurveyStat(object):
    """
    Estimation and inference for summary statistics in complex surveys.

    Parameters
    -------
    design : SurveyDesign object

    Attributes
    ----------
    est : ndarray
        The point estimates of the statistic, calculated on the columns
        of data.
    vc : ndarray
        The variance-covariance of the estimates.
    pseudo : ndarray
        The jackknife pseudo-values.
    """

    def __init__(self, design, mse):
        self.design = design
        self.mse = mse

    def bootstrap(self, replicates, index=None):
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
        """
        if index is not None:
            est = self._stat(self.design.weights, index)
        else:
            est = self._stat(self.design.weights)

        jdata = []
        has_reps = self.design.rep_weights is not None
        for r in range(replicates):
            # if rep_weights were supplied
            while has_reps:
                if index is None:
                    jdata.append(self._stat(weights=self.design.rep_weights))
                else:
                    jdata.append(self._stat(self.design.rep_weights, index))
                continue
            w = self.design.weights.copy()
            bin = np.zeros(self.design.nclust)
            for s in range(self.design.nstrat):
                # how to handle strata w/ only one cluster?
                w[self.design.strat == s] *= self.design.ncs[s] \
                                             / float(self.design.ncs[s] - 1)

                # If there is only one or two clusters then weights wont change
                if (self.design.ncs[s] == 1 or self.design.ncs[s] == 2):
                    continue
                # array of clusters to resample from
                ii = np.flatnonzero(self.design.sfclust == s)
                # resample them
                ii_resample = np.random.choice(ii, size=(self.design.ncs[s]-1))
                # accumulate number of times cluster i was resampled
                bin += np.bincount(ii_resample,
                                   minlength=max(self.design.sclust)+1)
            # augment weights
            w *= bin[self.design.sclust]
            # call the stat w/ the new weights
            if index is None:
                jdata.append(self._stat(weights=w))
            else:
                jdata.append(self._stat(w, index))
        jdata = np.asarray(jdata)
        # nh = self.design.ncs[self.design.sfclust].astype(np.float64)
        # pseudo = jdata + nh[:, None] * (np.dot(w, stat.data) - jdata)

        boot_mean = jdata.mean(0)
        var = ((jdata - boot_mean)**2).sum(0) / (replicates - 1)

        return est, var

    def jack(self, index=None):
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
            # if dealing w/ survey quantile...
            if index is not None:
                # 3d array, nclust x col x len(quantiles)
                jdata.append(self._stat(self.w, index))
            else:
                jdata.append(self._stat(self.w))
        jdata = np.asarray(jdata)
        nh = self.design.ncs[self.design.sfclust].astype(np.float64)
        # pseudo = jdata + nh[:, None] * (np.dot(self.w, stat.data) - jdata)

        if index is not None:
            est = self._stat(self.design.weights, index)
        else:
            est = self._stat(self.design.weights)

        if self.mse:
            print('mse method')
            jdata -= est
        else:
            for s in range(self.design.nstrat):
                # get indices of all clusters within a stratum
                ii = np.flatnonzero(self.design.sfclust == s)
                # center the 'delete 1' statistic
                jdata[ii, :] -= jdata[ii, :].mean(0)
        u = np.sqrt((nh - 1) / nh)
        jdata = u[:, None] * jdata
        vc = np.dot(jdata.T, jdata)


        # if not working w/ percentile and est is []

        return est, np.sqrt(np.diag(vc))


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
    data : ndarray
        The data which to calculate the mean on
    design :
        Points to the SurveyDesign object
    est : ndarray
        The point estimates of the statistic, calculated on the columns
        of data.
    vc : ndarray
        The variance-covariance of the estimates.
    pseudo : ndarray
        The jackknife pseudo-values.
    """
    def __init__(self, design, data, se_method, mse=False, replicates=None):
        super().__init__(design, mse)
        self.data = np.asarray(data)
        if se_method == "jack":
            self.est, self.vc = self.jack()
            # self.vc = np.sqrt(np.diag(self.vc))
        elif se_method == "boot":
            self.est, self.vc = self.bootstrap(replicates)
            self.vc = np.sqrt(self.vc)
        else:
            raise ValueError("Method %s not supported" % se_method)

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
        nxp array of the data to calculate the total on
    method: string
        User inputs whether to get bootstrap or jackknife SE

    Attributes
    ----------
    data : ndarray
        The data which to calculate the mean on
    design :
        Points to the SurveyDesign object
    est : ndarray
        The point estimates of the statistic, calculated on the columns
        of data.
    vc : ndarray
        The variance-covariance of the estimates.
    pseudo : ndarray
        The jackknife pseudo-values.
    """
    def __init__(self, design, data, se_method,replicates=None, mse=False):
        super().__init__(design, mse)
        self.data = np.asarray(data)

        if se_method == "jack":
            self.est, self.vc = self.jack()
            # self.vc = np.sqrt(np.diag(self.vc))
        elif se_method == "boot":
            self.est, self.vc = self.bootstrap(replicates)
            self.vc = np.sqrt(self.vc)
        else:
            raise ValueError("Method %s not supported" % se_method)

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
        return np.dot(weights, self.data)


class SurveyQuantile(SurveyStat):
    """
    Calculates the quantiles[s] for each column.

    Parameters
    -------
    design : SurveyDesign object
    data : ndarray
        nxp array of the data to calculate the mean on
    parameter: array-like
        array of quantiles to calculate for each column

    Attributes
    ----------
    data : ndarray
        The data which to calculate the quantiles on
    design :
        Points to the SurveyDesign object
    est : ndarray
        The point estimates of the statistic, calculated on the columns
        of data.
    quantile : ndarray
        The quantile[s] to calculate for each column
    vc : ndarray
        The variance-covariance of the estimates.
    pseudo : ndarray
        The jackknife pseudo-values.
    """

    def __init__(self, design, data, quantile, se_method, mse=False):
        self.data = np.asarray(data)
        super().__init__(design, mse)
        self.quantile = np.asarray(quantile)

        # give warning if user entered in quantile bigger than one
        large_q = np.asarray([x > 1 for x in self.quantile])
        n = large_q.sum()
        if n > 0:
            print("warning:", n, "inputed quantile[s] > 1")
        self.n_cw = len(self.design.weights)

        # get quantile[s] for each column
        self.est = [0] * self.data.shape[1]
        # need to call this several times
        self.vc = [0] * self.data.shape[1]
        if se_method == "jack":
            for index in range(self.data.shape[1]):
                self.est[index], self.vc[index] = self.jack(index)
        elif se_method == "boot":
            for index in range(self.data.shape[1]):
                self.est[index], self.vc[index] = self.bootstrap(index)
        else:
            raise ValueError("method %s not supported" % se_method)
    def _stat(self, weights, col_index):
        quant_list = []
        cw = np.cumsum(weights)
        sorted_data = np.sort(self.data[:, col_index])
        q = self.quantile.copy() * cw[-1]
        # find index i such that self.cumsum_weights[i] >= q
        ind = np.searchsorted(cw, q)

        for i, pos in enumerate(ind):
            # if searchsorted returns length of list
            # return last observation
            if pos in np.array([self.n_cw - 1, self.n_cw]):
                quant_list.append(sorted_data[-1])
                continue
            if (cw[pos] == q[i]):
                quant_list.append((sorted_data[pos] + sorted_data[pos+1]) / 2)
            else:
                quant_list.append(sorted_data[pos])
        return quant_list


class SurveyMedian(SurveyQuantile):
    """
    Wrapper function that calls SurveyQuantile
    with quantile = [.50]
    """
    def __init__(self, SurveyDesign, data, se_method, mse=False):
        # sp = super(SurveyMedian, self).__init__(SurveyDesign, data, [50])
        sp = SurveyQuantile(SurveyDesign, data, [.50], se_method, mse)
        self.est = sp.est
        self.vc = sp.vc
