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

    def __init__(self, strata=None, cluster=None, weights=None,
                 rep_weights=None, fpc=None, se_method='jack', nest=True):
        # Ensure method for SE is supported
        if (se_method not in ["boot", 'mean_boot', 'jack']):
            raise ValueError("Method %s not supported" % se_method)
        else:
            self.se_method = se_method

        self.rep_weights = rep_weights

        if self.rep_weights is None:
            strata, cluster, self.weights,
            self.fpc = self._check_args(strata, cluster, weights, fpc)

            # Recode strata and clusters as integer values 0, 1, ...
            _, self.strat = np.unique(strata, return_inverse=True)
            _, clust = np.unique(cluster, return_inverse=True)

            # the number of distinct strata
            self.nstrat = max(self.strat) + 1

            # If requested, recode the PSUs to be sure that the same PSU # in
            # different strata are treated as distinct PSUs. This is the same
            # as the nest option in R.
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
            self.sfclust = self.strat[ii]

            # The fpc for each cluster
            self.fpc = self.fpc[ii]

            # The total number of clusters over all stratum
            self.nclust = np.sum(self.ncs)

            # get indices of all clusters within a stratum
            self.ii = []
            for s in range(self.nstrat):
                self.ii.append(np.flatnonzero(self.sfclust == s))

        else:
            if strata is not None or cluster is not None:
                raise ValueError("If providing rep_weights, do not provide \
                                 cluster or strata")
            if weights is None:
                self.weights = np.ones(self.rep_weights[0])
            else:
                self.weights = weights

    def __str__(self):
        """
        The __str__ method for our data
        """
        summary_list = ["Number of observations: ", str(len(self.strat)),
                        "Sum of weights: ", str(self.weights.sum()),
                        "Number of strata: ", str(self.nstrat),
                        "Number of clusters per stratum: ", str(self.ncs),
                        "Method to compute SE: ", self.se_method]

        return "\n".join(summary_list)

    def _check_args(self, strata, cluster, weights, fpc):
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
            raise ValueError("""At least one of strata, cluster, rep_weights, and weights
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

        if fpc is None:
            vals.append(np.zeros(n))
        else:
            vals.append(np.asarray(fpc))

        return vals[0], vals[1], vals[2], vals[3]

    def get_rep_weights(self, c=None, bsn=None):
        """
        Returns replicate weights if provided, else computes rep weights
        and returns them.

        Parameters
        ----------
        c : integer or None
            Represents which cluster to leave out when computing
            'delete 1' jackknife replicate weights
        bsn : integer or None
            bootstrap mean-weight adjustment. Value of bsn is the # of
            bootstrap replicate-weight variables were used to
            generate each bootstrap

        Returns
        -------
        rep_weights : ndarray
            Either the provided rep_weights when a design object
            was created, or calculated rep_weights from jackknife,
            bootstrap, or mean bootstrap
        """
        if self.rep_weights is not None:
            return self.rep_weights[c]
        if self.se_method == 'jack':
            return self._jackknife_rep_weights(c)
        elif self.se_method == 'boot':
            return self._bootstrap_weights()
        else:
            return self._mean_bootstrap_weight(bsn)

    def _jackknife_rep_weights(self, c):
        """
        Computes 'delete 1' jackknife replicate weights

        Parameters
        ----------
        c : integer or None
            Represents which cluster to leave out when computing
            'delete 1' jackknife replicate weights

        Returns
        -------
        w : ndarray
            Augmented weight
        """
        # get stratum that the cluster belongs in
        s = self.sfclust[c]
        nh = self.ncs[s]
        w = self.weights.copy()
        # all weights within the stratum are modified
        w[self.strat == s] *= nh / float(nh - 1)
        # but if you're within the cluster to be removed, set as 0
        w[self.sclust == c] = 0
        return w

    def _bootstrap_weights(self):
        """
        Computes bootstrap replicate weight

        Returns
        -------
        w : ndarray
            Augmented weight
        """
        w = self.weights.copy()
        clust_count = np.zeros(self.nclust)
        for s in range(self.nstrat):
            # how to handle strata w/ only one cluster?
            w[self.strat == s] *= float(self.ncs[s] - 1) \
                                         / self.ncs[s]
            # If there is only one cluster then weights wont change
            if self.ncs[s] == 1:
                continue

            # resample array of clusters
            ii_resample = np.random.choice(self.ii[s], size=(self.ncs[s]-1))
            # accumulate number of times cluster i was resampled
            clust_count += np.bincount(ii_resample,
                                       minlength=max(self.sclust)+1)

        w *= clust_count[self.sclust]
        return w

    def _mean_bootstrap_weight(self, bsn):
        """
        Computes mean bootstrap replicate weight

        Parameters
        ----------
        bsn : integer
            Mean bootstrap averages the number of resampled clusters over bsn
        Returns
        -------
        w : ndarray
            Augmented weight
        """
        clust_count = np.zeros(self.design.nclust)
        # for each replicate, I accumulate bsn number of times?
        for b in range(bsn):
            for s in range(self.nstrat):
                w[self.strat == s] *= ((float(self.ncs[s] - 1) /
                                        self.ncs[s])**(1/bsn))
                # If there is only one or two clusters then weights wont change
                if (self.ncs[s] == 1 or self.ncs[s] == 2):
                    continue
                # resample array of clusters in strata s
                ii_resample = np.random.choice(self.ii[s], size=(self.ncs[s]-1))
                # accumulate number of times cluster i was resampled
                clust_count += np.bincount(ii_resample,
                                           minlength=max(self.sclust)+1)
        # avg number of times cluster i was resampled
        clust_count /= bsn
        # augment weights
        w *= clust_count[self.sclust]
        return w


class SurveyStat(SurveyDesign):
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

    def _bootstrap(self, replicates=None, bsn=None):
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
        est = self._stat(self.design.weights)

        jdata = []
        for i in range(replicates):
            w = self.design.get_rep_weights(i, bsn=bsn)
            jdata.append(self._stat(w))
        jdata = np.asarray(jdata)
        if self.mse:
            print("mse specified")
            var = np.dot((jdata-est).T, (jdata-est)) / replicates
            var = np.sqrt(np.diag(var))
        else:
            boot_mean = jdata.mean(0)
            var = ((jdata - boot_mean)**2).sum(0) / replicates

        return est, var

    def _jack(self):
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
        est = self._stat(self.design.weights)

        jdata = []
        # for each cluster
        for c in range(self.design.nclust):
            # get jackknife weights
            w = self.design.get_rep_weights(c)
            jdata.append(self._stat(w))
        jdata = np.asarray(jdata)

        nh = self.design.ncs[self.design.sfclust].astype(np.float64)
        _pseudo = jdata + nh[:, None] * (np.dot(self.design.weights,
                                                self.data) - jdata)

        if self.mse:
            print('mse specified')
            jdata -= est
        else:
            if self.design.rep_weights is None:
                for s in range(self.design.nstrat):
                    # center the 'delete 1' statistic
                    jdata[self.design.ii[s], :] -= jdata[self.design.ii[s],
                                                         :].mean(0)
            else:
                jdata -= jdata.mean(0)

        if self.design.rep_weights is None:
            mh = np.sqrt((nh - 1) / nh)
            fh = np.sqrt(1 - self.design.fpc)
            jdata = fh[:, None] * mh[:, None] * jdata
        vc = np.dot(jdata.T, jdata)

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
    def __init__(self, design, data, mse=False, replicates=None, bsn=None):
        super().__init__(design, mse)
        self.data = np.asarray(data)
        if self.design.se_method == "jack":
            self.est, self.vc = self._jack()
            # self.vc = np.sqrt(np.diag(self.vc))
        elif self.design.se_method == "boot":
            self.est, self.vc = self._bootstrap(replicates, bsn)
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
    def __init__(self, design, data, replicates=None, mse=False, bsn=None):
        super().__init__(design, mse)
        self.data = np.asarray(data)

        if self.design.se_method == "jack":
            self.est, self.vc = self._jack()
            # self.vc = np.sqrt(np.diag(self.vc))
        elif self.design.se_method == "boot":
            self.est, self.vc = self._bootstrap(replicates, bsn)
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
    def __init__(self, design, data, quantile, replicates=None, mse=False,
                 bsn=None):
        self.data = np.asarray(data)
        super().__init__(design, mse)
        self.quantile = np.asarray(quantile)

        # give warning if user entered in quantile bigger than one
        if (self.quantile.min() < 0 or self.quantile.max > 1):
            raise ValueError("quantile[s] should be within [0, 1]")
        self.n_cw = len(self.design.weights)

        # get quantile[s] for each column
        self.est = [0] * self.data.shape[1]
        # need to call this several times
        self.vc = [0] * self.data.shape[1]
        if se_method == "jack":
            for index in range(self.data.shape[1]):
                self.est[index], self.vc[index] = self._jackknife()
        elif se_method == "boot":
            for index in range(self.data.shape[1]):
                self.est[index], self.vc[index] = self._bootstrap()
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
    Derived class from SurveyQuantile with quantile = [.50]
    """
    def __init__(self, SurveyDesign, data, se_method, mse=False):
        # sp = super(SurveyMedian, self).__init__(SurveyDesign, data, [50])
        sp = SurveyQuantile(SurveyDesign, data, [.50], se_method, mse)
        self.est = sp.est
        self.vc = sp.vc
