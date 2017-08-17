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
# from __future__ import division
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
    n_strat : integer
        The number of district strata
    sclust : (n, ) array
        The relabeled cluster array from 0, 1, ..
    strat : (n, ) array
        The related strata array from 0, 1, ...
    clust_per_strat : (self.n_strat, ) array
        Holds the number of clusters in each stratum
    strat_for_clust : ndarray
        The stratum for each cluster
    n_clust : integer
        The total number of clusters across strata
    """

    def __init__(self, strata=None, cluster=None, weights=None,
                 rep_weights=None, fpc=None, nest=True):

        self.rep_weights = rep_weights
        if self.rep_weights is not None:
            if isinstance(self.rep_weights, list):
                self.rep_weights = np.asarray(rep_weights).T
            elif isinstance(rep_weights, np.ndarray):
                self.rep_weights = rep_weights
            else:
                return ValueError("rep_weights should be array-like")

            if strata is not None or cluster is not None:
                raise ValueError("If providing rep_weights, do not provide \
                             cluster or strata")
            if weights is None:
                self.weights = np.ones(self.rep_weights.shape[0])
            else:
                self.weights = weights
            self.n_clust = self.rep_weights.shape[1]
            return

        strata, cluster, self.weights, \
            self.fpc = self._check_args(strata, cluster, weights, fpc)

        # Recode strata and clusters as integer values 0, 1, ...
        _, self.strat = np.unique(strata, return_inverse=True)
        _, clust = np.unique(cluster, return_inverse=True)

        # the number of distinct strata
        self.n_strat = max(self.strat) + 1

        # If requested, recode the PSUs to be sure that the same PSU # in
        # different strata are treated as distinct PSUs. This is the same
        # as the nest option in R.
        if nest:
            m = max(clust) + 1
            clust = clust + m*self.strat
            _, self.clust = np.unique(clust, return_inverse=True)
        else:
            self.clust = clust.copy()

        # The number of clusters per stratum
        _, ii = np.unique(self.clust, return_index=True)
        self.clust_per_strat = np.bincount(self.strat[ii])

        # The stratum for each cluster
        self.strat_for_clust = self.strat[ii]

        # The fpc for each cluster
        self.fpc = self.fpc[ii]

        # The total number of clusters over all stratum
        self.n_clust = np.sum(self.clust_per_strat)

        # get indices of all clusters within a stratum
        self.ii = []
        for s in range(self.n_strat):
            self.ii.append(np.flatnonzero(self.strat_for_clust == s))

    def __str__(self):
        """
        The __str__ method for our data
        """
        summary_list = ["Number of observations: ", str(len(self.strat)),
                        "Sum of weights: ", str(self.weights.sum()),
                        "Number of strata: ", str(self.n_strat),
                        "Clusters per stratum: ", str(self.clust_per_strat)]

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

    def get_rep_weights(self, cov_method, c=None, bsn=None):
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
        # Ensure method for SE is supported
        if cov_method not in ["boot", 'mean_boot', 'jack']:
            raise ValueError("Method %s not supported" % cov_method)

        if self.rep_weights is not None:
            # should rep_weights be a list of arrays or a ndarray
            return self.rep_weights[:, c]
        if cov_method == 'jack':
            return self._jackknife_rep_weights(c)
        elif cov_method == 'boot':
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
        s = self.strat_for_clust[c]
        nh = self.clust_per_strat[s]
        w = self.weights.copy()
        # all weights within the stratum are modified
        w[self.strat == s] *= nh / float(nh - 1)
        # but if you're within the cluster to be removed, set as 0
        w[self.clust == c] = 0
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
        clust_count = np.zeros(self.n_clust)
        for s in range(self.n_strat):
            # how to handle strata w/ only one cluster?
            w[self.strat == s] *= float(self.clust_per_strat[s] - 1) \
                                         / self.clust_per_strat[s]
            # If there is only one cluster then weights wont change
            if self.clust_per_strat[s] == 1:
                continue

            # resample array of clusters
            ii_resample = np.random.choice(self.ii[s],
                                           size=(self.clust_per_strat[s] - 1))
            # accumulate number of times cluster i was resampled
            clust_count += np.bincount(ii_resample,
                                       minlength=max(self.clust)+1)

        w *= clust_count[self.clust]
        return w

    # not sure whether this should be kept or not
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
        clust_count = np.zeros(self.design.n_clust)
        w = self.weights.copy()
        # for each replicate, I accumulate bsn number of times?
        for b in range(bsn):
            for s in range(self.n_strat):
                w[self.strat == s] *= ((float(self.clust_per_strat[s] - 1) /
                                        self.clust_per_strat[s])**(1/bsn))
                # If there is only one or two clusters then weights wont change
                if self.clust_per_strat[s] in (1, 2):
                    continue
                # resample array of clusters in strata s
                ii_resample = np.random.choice(self.ii[s],
                                               size=(self.clust_per_strat[s] -
                                                     1))
                # accumulate number of times cluster i was resampled
                clust_count += np.bincount(ii_resample,
                                           minlength=max(self.clust)+1)
        # avg number of times cluster i was resampled
        clust_count /= bsn
        # augment weights
        w *= clust_count[self.clust]
        return w


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
    vcov : ndarray
        The variance-covariance of the estimates.
    pseudo : ndarray
        The jackknife pseudo-values.
    """

    def __init__(self, design, data, center_by):
        self.design = design
        if center_by not in ['global', 'stratum', 'est']:
            return ValueError("center_by = %s not supported" % center_by)
        self.center_by = center_by
        self.data = np.asarray(data)
        if self.data.ndim == 1:
            self.data = self.data[:, None]

    def _bootstrap(self, n_reps=None, bsn=None):
        """
        Calculates bootstrap standard errors

        Parameters
        ----------
        stat : object
            Object of class SurveyMean, SurveyTotal, SurveyPercentile, etc
        n_reps : integer
            The number of replicates that the user wishes to specify

        Returns
        -------
        est : ndarray
            The point estimates of the statistic, calculated on the columns
            of data.
        vcov : ndarray
            The variance-covariance of the estimates.
        """
        self.est = self._stat(self.design.weights)

        jdata = []
        if n_reps is None:
            n_reps = self.design.rep_weights.shape[1]
        for i in range(n_reps):
            # does not support mean_boot But not sure if ppl even use mean_boot
            w = self.design.get_rep_weights(c=i, cov_method='boot', bsn=bsn)
            jdata.append(self._stat(w))
        jdata = np.asarray(jdata)
        if jdata.ndim == 1:
            jdata = jdata[:, None]
        if self.center_by == 'est':
            print("centering by est")
            jdata -= self.est
        elif self.center_by == "global":
            print("centering by global mean")
            jdata -= jdata.mean(0)
        else:
            raise ValueError("For bootstrap, center by 'est' or 'global'")
        self.vcov = np.dot(jdata.T, jdata) / n_reps
        if self.vcov.ndim == 2:
            self.stderr = np.sqrt(np.diag(self.vcov))
        else:
            self.stderr = np.sqrt(self.vcov)

    def _jackknife(self):
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
        vcov : square ndarray
            The variance-covariance matrix of the estimates, obtained using
            the (drop 1) jackknife procedure.
        pseudo : ndarray
            The jackknife pseudo-values.
        """
        if self.center_by != 'est' and self.design.rep_weights is not None:
            raise ValueError("If providing replicate weights, center by est")

        self.est = self._stat(self.design.weights)

        jdata = []
        # for each cluster
        for c in range(self.design.n_clust):
            # get jackknife weights
            w = self.design.get_rep_weights(c=c, cov_method='jack')
            jdata.append(self._stat(w))
        jdata = np.asarray(jdata)
        if jdata.ndim == 1:
            jdata = jdata[:, None]
        if self.center_by == 'est':
            print('centering by estimate')
            jdata -= self.est
        elif self.center_by == 'global':
            print("centering by global mean")
            jdata -= jdata.mean(0)
        else:
            print("centering by stratum")
            if self.design.rep_weights is None:
                for s in range(self.design.n_strat):
                    # center the 'delete 1' statistic
                    jdata[self.design.ii[s], :] -= jdata[self.design.ii[s],
                                                         :].mean(0)
            else:
                raise ValueError("Can't center by stratum with rep_weights")
        if self.design.rep_weights is None:
            nh = self.design.clust_per_strat[self.design.strat_for_clust].astype(np.float64)
            _pseudo = jdata + nh[:, None] * (np.dot(self.design.weights,
                                                    self.data) - jdata)
            mh = np.sqrt((nh - 1) / nh)
            fh = np.sqrt(1 - self.design.fpc)
            jdata = fh[:, None] * mh[:, None] * jdata
        else:
            nh = self.design.rep_weights.shape[1]
            mh = (nh - 1) / nh
            jdata *= np.sqrt(mh)
        self.vcov = np.dot(jdata.T, jdata)
        if self.vcov.ndim == 2:
            self.stderr = np.sqrt(np.diag(self.vcov))
        else:
            self.stderr = np.sqrt(self.vcov)

    def _linearized(self):
        if self.center_by != 'stratum':
            raise ValueError("Must center by stratum with linearized variance")
        self.est = self._stat(self.design.weights)
        jdata = []
        # for each cluster
        for c in range(self.design.n_clust):
            w = self.design.weights.copy()
            # but if you're not in that cluster, set as 0
            w[self.design.clust != c] = 0
            jdata.append(self._weighted_score(w))
        jdata = np.asarray(jdata)
        # we usually deal w/ jdata as nxp
        # unless w/ ratio, in which 2 columns
        if jdata.ndim == 1:
            jdata = jdata[:, None]
        for s in range(self.design.n_strat):
            # center the 'delete 1' statistic
            jdata[self.design.ii[s], :] -= jdata[self.design.ii[s],
                                                 :].mean(0)
        nh = self.design.clust_per_strat[self.design.strat_for_clust].astype(np.float64)
        mh = np.sqrt(nh / (nh-1))
        fh = np.sqrt(1 - self.design.fpc)
        jdata = fh[:, None] * mh[:, None] * jdata
        self.vcov = np.dot(jdata.T, jdata)

        if self.vcov.ndim == 2:
            self.stderr = np.sqrt(np.diag(self.vcov))
        else:
            self.stderr = np.sqrt(self.vcov)


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
    vcov : ndarray
        The variance-covariance of the estimates.
    pseudo : ndarray
        The jackknife pseudo-values.
    """
    def __init__(self, design, data, cov_method='jack', center_by='global',
                 n_reps=None, bsn=None):
        super().__init__(design, data, center_by)
        if cov_method == "jack":
            self._jackknife()
        elif cov_method == 'boot':
            self._bootstrap(n_reps, bsn)
        elif cov_method == 'linearized':
            self._linearized()
        else:
            raise ValueError("cov_method %s is not supported" % cov_method)

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

        return np.dot(weights, self.data) / np.sum(weights)

    def _weighted_score(self, weights):
        """
        Returns weighted sum of the score variable for linearized variance.


        Parameters
        ----------
        weights : np.array
            The weights used to calculate the total, will either be
            original design weights or recalculated weights via jk,
            boot, etc

        Returns
        -------
        An array containing the statistic calculated on the columns
        of the dataset.
        """

        # using try/except to prevent self._z from being calculated
        # with each call
        try:
            return np.dot(weights, self._z)
        except AttributeError:
            self._z = (self.data - self.est) / np.sum(self.design.weights)
        return np.dot(weights, self._z)


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
    vcov : ndarray
        The variance-covariance of the estimates.
    pseudo : ndarray
        The jackknife pseudo-values.
    """
    def __init__(self, design, data, cov_method='jack', n_reps=None,
                 center_by='global', bsn=None):
        super().__init__(design, data, center_by)

        if cov_method == "jack":
            self._jackknife()
        elif cov_method == 'boot':
            self._bootstrap(n_reps, bsn)
        elif cov_method == 'linearized':
            self._linearized()
        else:
            raise ValueError("cov_method %s is not supported" % cov_method)

    def _stat(self, weights):
        """
        Returns calculation of total.

        Parameters
        ----------
        weights : np.array
            The weights used to calculate the total, will either be
            original design weights or recalculated weights via jk,
            boot, etc

        Returns
        -------
        An array containing the statistic calculated on the columns
        of the dataset.
        """
        return np.dot(weights, self.data)

    def _weighted_score(self, weights):
        """
        Returns weighted sum of the score variable for linearized variance.
        For SurveyTotal, this is just the data

        Parameters
        ----------
        weights : np.array
            The weights used to calculate the total, will either be
            original design weights or recalculated weights via jk,
            boot, etc

        Returns
        -------
        An array containing the statistic calculated on the columns
        of the dataset.
        """
        return np.dot(weights, self.data)


class SurveyRatio(SurveyStat):
    def __init__(self, design, data, cov_method='jack', n_reps=None,
                 center_by='global', bsn=None):

        super().__init__(design, data, center_by)

        if cov_method == "jack":
            self._jackknife()
        elif cov_method == 'boot':
            self._bootstrap(n_reps, bsn)
        elif cov_method == 'linearized':
            self._linearized()
        else:
            raise ValueError("cov_method %s is not supported" % cov_method)

    def _stat(self, weights):
        """
        Returns calculation of total.

        Parameters
        ----------
        weights : np.array
            The weights used to calculate the total, will either be
            original design weights or recalculated weights via jk,
            boot, etc

        Returns
        -------
        An array containing the statistic calculated on the columns
        of the dataset.
        """
        X = self.data[:, 0]
        Y = self.data[:, 1]
        return np.dot(weights, X) / np.dot(weights, Y)

    def _weighted_score(self, weights):
        """
        Returns weighted sum of the score variable for linearized variance.
        For SurveyTotal, this is just the data

        Parameters
        ----------
        weights : np.array
            The weights used to calculate the total, will either be
            original design weights or recalculated weights via jk,
            boot, etc

        Returns
        -------
        An array containing the statistic calculated on the columns
        of the dataset.
        """
        # using try/except to prevent self._z from being calculated
        # with each call
        try:
            return np.dot(weights, self._z)
        except AttributeError:
            self._z = (self.data[:, 0] - (self.est * self.data[:, 1]))
            self._z = self._z / np.dot(self.design.weights, self.data[:, 1])
        return np.dot(weights, self._z)


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
    vcov : ndarray
        The variance-covariance of the estimates.
    pseudo : ndarray
        The jackknife pseudo-values.
    """
    def __init__(self, design, data, quantile, cov_method='jack', n_reps=None,
                 center_by='global', bsn=None):
        super().__init__(design, data, center_by)
        self.quantile = quantile

        # give warning if user entered in quantile bigger than one
        if self.quantile < 0 or self.quantile > 1:
            raise ValueError("quantile should be within [0, 1]")
        self._n_cw = len(self.design.weights)

        # get quantile[s] for each column
        # self.est = np.empty(self.data.shape[1])
        # # need to call this several times
        # self.stderr = np.empty(self.data.shape[1])
        if cov_method == "jack":
            self._jackknife()
        elif cov_method == 'boot':
            self._bootstrap()
        else:
            raise ValueError("cov_method %s is not supported" % cov_method)

    def _stat(self, weights):
        quant_list = []
        q = self.quantile * np.sum(weights)
        for col_index in range(self.data.shape[1]):
            # get weights based on sorted data
            sorted_weights = [x for y, x in sorted(zip(self.data[:, col_index],
                                                       weights))]
            sorted_weights = np.asarray(sorted_weights)
            cw = np.cumsum(sorted_weights)
            sorted_data = np.sort(self.data[:, col_index])

            # find index i such that self.cumsum_weights[i] >= q
            ind = np.searchsorted(cw, q)
            if ind in np.array([self._n_cw - 1, self._n_cw]):
                quant_list.append(sorted_data[-1])
                continue
            # this is true if q is equal to cw[ind], but we want the first
            # index st cw[ind] > q. So we use the STATA formula w/ ind and
            # ind + 1
            if cw[ind] == q:
                quant_list.append((sorted_data[ind] + sorted_data[ind+1]) / 2)
            else:
                quant_list.append(sorted_data[ind])
        return quant_list


class SurveyMedian(SurveyQuantile):
    """
    Derived class from SurveyQuantile with quantile = [.50]
    """
    def __init__(self, design, data, quantile=.5, cov_method='jack',
                 n_reps=None, center_by='global', bsn=None):
        # initialize SurveyQuantile
        super().__init__(design, data, quantile, cov_method, n_reps,
                         center_by, bsn)
