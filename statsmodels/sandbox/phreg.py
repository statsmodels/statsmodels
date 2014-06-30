import numpy as np
from statsmodels.base import model
import statsmodels.base.model as base
from statsmodels.tools.decorators import cache_readonly

"""
Implementation of proportional hazards regression models for duration
data that may be censored ("Cox models").

References
----------
T Therneau (1996).  Extending the Cox model.  Technical report.
http://www.mayo.edu/research/documents/biostat-58pdf/DOC-10027288

G Rodriguez (2005).  Non-parametric estimation in survival models.
http://data.princeton.edu/pop509/NonParametricSurvival.pdf

B Gillespie (2006).  Checking the assumptions in the Cox proportional
hazards model.
http://www.mwsug.org/proceedings/2006/stats/MWSUG-2006-SD08.pdf
"""


class PH_SurvivalTime(object):

    def __init__(self, time, status, exog, strata=None, entry=None,
                 offset=None):
        """
        Represent a collection of survival times with possible
        stratification and left truncation.

        Parameters
        ----------
        time : array_like
            The times at which either the event (failure) occurs or
            the observation is censored.
        status : array_like
            Indicates whether the event (failure) occurs at `time`
            (`status` is 1), or if `time` is a censoring time (`status`
            is 0).
        exog : array_like
            The exogeneous (covariate) data matrix, cases are rows and
            variables are columns.
        strata : array_like
            Grouping variable defining the strata.  If None, all
            observations are in a single stratum.
        entry : array_like
            Entry (left truncation) times.  The observation is not
            part of the risk set for times before the entry time.  If
            None, the entry time is treated as being zero, which
            gives no left truncation.  The entry time must be less
            than or equal to `time`.
        offset : array-like
            An optional array of offsets
        """

        # Default strata
        if strata is None:
            strata = np.zeros(len(time), dtype=np.int32)

        # Default entry times
        if entry is None:
            entry = np.zeros(len(time))

        # Parameter validity checks.
        n1, n2, n3, n4 = len(time), len(status), len(strata),\
            len(entry)
        nv = [n1, n2, n3, n4]
        if max(nv) != min(nv):
            raise ValueError("endog, status, strata, and " +
                             "entry must all have the same length")
        if min(time) < 0:
            raise ValueError("endog must be non-negative")
        if min(entry) < 0:
            raise ValueError("entry time must be non-negative")

        # In Stata, this is entry >= time, in R it is >.
        if np.any(entry > time):
            raise ValueError("entry times may not occur " +
                             "after event or censoring times")

        # Get the row indices for the cases in each stratum
        if strata is not None:
            stu = np.unique(strata)
            #sth = {x: [] for x in stu} # needs >=2.7
            sth = dict([(x, []) for x in stu])
            for i,k in enumerate(strata):
                sth[k].append(i)
            stratum_rows = [np.asarray(sth[k], dtype=np.int32) for k in stu]
            stratum_names = stu
        else:
            stratum_rows = [np.arange(len(time)),]
            stratum_names = [0,]

        # Remove strata with no events
        ix = [i for i,ix in enumerate(stratum_rows) if status[ix].sum() > 0]
        stratum_rows = [stratum_rows[i] for i in ix]
        stratum_names = [stratum_names[i] for i in ix]

        # The number of strata
        nstrat = len(stratum_rows)
        self.nstrat = nstrat

        # Remove subjects whose entry time occurs after the last event
        # in their stratum.
        for stx,ix in enumerate(stratum_rows):
            last_failure = max(time[ix][status[ix] == 1])

            # Stata uses < here, R uses <=
            ii = [i for i,t in enumerate(entry[ix]) if
                  t <= last_failure]
            stratum_rows[stx] = stratum_rows[stx][ii]

        # Remove subjects who are censored before the first event in
        # their stratum.
        for stx,ix in enumerate(stratum_rows):
            first_failure = min(time[ix][status[ix] == 1])

            ii = [i for i,t in enumerate(time[ix]) if
                  t >= first_failure]
            stratum_rows[stx] = stratum_rows[stx][ii]

        # Order by time within each stratum
        for stx,ix in enumerate(stratum_rows):
            ii = np.argsort(time[ix])
            stratum_rows[stx] = stratum_rows[stx][ii]

        if offset is not None:
            self.offset_s = []
            for stx in range(nstrat):
                self.offset_s.append(offset[stratum_rows[stx]])
        else:
            self.offset_s = None

        # Number of informative subjects
        self.n_obs = sum([len(ix) for ix in stratum_rows])

        # Split everything by stratum
        self.time_s = []
        self.exog_s = []
        self.status_s = []
        self.entry_s = []
        for ix in stratum_rows:
            self.time_s.append(time[ix])
            self.exog_s.append(exog[ix,:])
            self.status_s.append(status[ix])
            self.entry_s.append(entry[ix])

        self.stratum_rows = stratum_rows
        self.stratum_names = stratum_names

        # Precalculate some indices needed to fit Cox models.
        # Distinct failure times within a stratum are always taken to
        # be sorted in ascending order.
        #
        # ufailt_ix[stx][k] is a list of indices for subjects who fail
        # at the k^th sorted unique failure time in stratum stx
        #
        # risk_enter[stx][k] is a list of indices for subjects who
        # enter the risk set at the k^th sorted unique failure time in
        # stratum stx
        #
        # risk_exit[stx][k] is a list of indices for subjects who exit
        # the risk set at the k^th sorted unique failure time in
        # stratum stx
        self.ufailt_ix, self.risk_enter, self.risk_exit, self.ufailt =\
            [], [], [], []

        for stx in range(self.nstrat):

            # All failure times
            ift = np.flatnonzero(self.status_s[stx] == 1)
            ft = self.time_s[stx][ift]

            # Unique failure times
            uft = np.unique(ft)
            nuft = len(uft)

            # Indices of cases that fail at each unique failure time
            #uft_map = {x:i for i,x in enumerate(uft)} # requires >=2.7
            uft_map = dict([(x, i) for i,x in enumerate(uft)]) # 2.6
            uft_ix = [[] for k in range(nuft)]
            for ix,ti in zip(ift,ft):
                uft_ix[uft_map[ti]].append(ix)

            # Indices of cases (failed or censored) that enter the
            # risk set at each unique failure time.
            risk_enter1 = [[] for k in range(nuft)]
            for i,t in enumerate(self.time_s[stx]):
                ix = np.searchsorted(uft, t, "right") - 1
                if ix >= 0:
                    risk_enter1[ix].append(i)

            # Indices of cases (failed or censored) that exit the
            # risk set at each unique failure time.
            risk_exit1 = [[] for k in range(nuft)]
            for i,t in enumerate(self.entry_s[stx]):
                ix = np.searchsorted(uft, t)
                risk_exit1[ix].append(i)

            self.ufailt.append(uft)
            self.ufailt_ix.append([np.asarray(x, dtype=np.int32) for x in uft_ix])
            self.risk_enter.append([np.asarray(x, dtype=np.int32) for x in risk_enter1])
            self.risk_exit.append([np.asarray(x, dtype=np.int32) for x in risk_exit1])



class PHreg(model.LikelihoodModel):
    """
    Fit the Cox proportional hazards regression model for right
    censored data.

    Arguments
    ---------
    endog : array-like
        The observed times (event or censoring)
    exog : 2D array-like
        The covariates or exogeneous variables
    status : array-like
        The censoring status values; status=1 indicates that an
        event occured (e.g. failure or death), status=0 indicates
        that the observation was right censored. If None, defaults
        to status=1 for all cases.
    entry : array-like
        The entry times, if left truncation occurs
    strata : array-like
        Stratum labels.  If None, all observations are taken to be
        in a single stratum.
    ties : string
        The method used to handle tied times, must be either 'breslow'
        or 'efron'.
    offset : array-like
        Array of offset values
    missing : string
        The method used to handle missing data

    Notes
    -----
    Proportional hazards regression models should not include an
    explicit or implicit intercept.  The effect of an intercept is
    not identified using the partial likelihood approach.

    `endog`, `event`, `strata`, `entry`, and the first dimension
    of `exog` all must have the same length
    """

    def __init__(self, endog, exog, status=None, entry=None,
                 strata=None, offset=None, ties='breslow',
                 missing='drop'):

        # Default is no censoring
        if status is None:
            status = np.ones(len(endog))

        super(PHreg, self).__init__(endog, exog, status=status,
                                    entry=entry, strata=strata,
                                    offset=offset, missing=missing)

        # endog and exog are automatically converted, but these are
        # not
        if self.status is not None:
            self.status = np.asarray(self.status)
        if self.entry is not None:
            self.entry = np.asarray(self.entry)
        if self.strata is not None:
            self.strata = np.asarray(self.strata)
        if self.offset is not None:
            self.offset = np.asarray(self.offset)

        self.surv = PH_SurvivalTime(self.endog, self.status,
                                    self.exog, self.strata,
                                    self.entry, self.offset)

        ties = ties.lower()
        if ties not in ("efron", "breslow"):
            raise ValueError("`ties` must be either `efron` or " +
                             "`breslow`")

        self.ties = ties

    def fit(self, groups=None, **args):
        """
        Fit a proportional hazards regression model.

        Parameters
        ----------
        groups : array-like
            Labels indicating groups of observations that may be
            dependent.  If present, the standard errors account for
            this dependence. Does not affect fitted values.

        Returns a PHregResults instance.
        """

        # TODO process for missing values
        if groups is not None:
            self.groups = np.asarray(groups)
        else:
            self.groups = None

        if 'disp' not in args:
            args['disp'] = False
        fit_rslts = super(PHreg, self).fit(**args)

        if self.groups is None:
            cov_params = fit_rslts.cov_params()
        else:
            cov_params = self.robust_covariance(fit_rslts.params)

        results = PHregResults(self, fit_rslts.params, cov_params)

        return results

    def loglike(self, params):
        """
        Returns the log partial likelihood function evaluated at
        `params`.
        """

        if self.ties == "breslow":
            return self.breslow_loglike(params)
        elif self.ties == "efron":
            return self.efron_loglike(params)

    def score(self, params):
        """
        Returns the score function evaluated at `params`.
        """

        if self.ties == "breslow":
            return self.breslow_gradient(params)
        elif self.ties == "efron":
            return self.efron_gradient(params)

    def hessian(self, params):
        """
        Returns the Hessian matrix of the log partial likelihood
        function evaluated at `params`.
        """

        if self.ties == "breslow":
            return self.breslow_hessian(params)
        else:
            return self.efron_hessian(params)

    def breslow_loglike(self, params):
        """
        Returns the value of the log partial likelihood function
        evaluated at `params`, using the Breslow method to handle tied
        times.
        """

        surv = self.surv

        like = 0.

        # Loop over strata
        for stx in range(surv.nstrat):

            uft_ix = surv.ufailt_ix[stx]
            exog_s = surv.exog_s[stx]
            nuft = len(uft_ix)

            linpred = np.dot(exog_s, params)
            if surv.offset_s is not None:
                linpred += surv.offset_s[stx]
            linpred -= linpred.max()
            e_linpred = np.exp(linpred)

            xp0 = 0.

            # Iterate backward through the unique failure times.
            for i in range(nuft)[::-1]:

                # Update for new cases entering the risk set.
                ix = surv.risk_enter[stx][i]
                xp0 += e_linpred[ix].sum()

                # Account for all cases that fail at this point.
                ix = uft_ix[i]
                like += (linpred[ix] - np.log(xp0)).sum()

                # Update for cases leaving the risk set.
                ix = surv.risk_exit[stx][i]
                xp0 -= e_linpred[ix].sum()

        return like

    def efron_loglike(self, params):
        """
        Returns the value of the log partial likelihood function
        evaluated at `params`, using the Efron method to handle tied
        times.
        """

        surv = self.surv

        like = 0.

        # Loop over strata
        for stx in range(surv.nstrat):

            # exog and linear predictor for this stratum
            exog_s = surv.exog_s[stx]
            linpred = np.dot(exog_s, params)
            if surv.offset_s is not None:
                linpred += surv.offset_s[stx]
            linpred -= linpred.max()
            e_linpred = np.exp(linpred)

            xp0 = 0.

            # Iterate backward through the unique failure times.
            uft_ix = surv.ufailt_ix[stx]
            nuft = len(uft_ix)
            for i in range(nuft)[::-1]:

                # Update for new cases entering the risk set.
                ix = surv.risk_enter[stx][i]
                xp0 += e_linpred[ix].sum()
                xp0f = e_linpred[uft_ix[i]].sum()

                # Account for all cases that fail at this point.
                ix = uft_ix[i]
                like += linpred[ix].sum()

                m = len(ix)
                J = np.arange(m, dtype=np.float64) / m
                like -= np.log(xp0 - J*xp0f).sum()

                # Update for cases leaving the risk set.
                ix = surv.risk_exit[stx][i]
                xp0 -= e_linpred[ix].sum()

        return like

    def breslow_gradient(self, params, return_grad_obs=False):
        """
        Returns the gradient of the log partial likelihood, using the
        Breslow method to handle tied times.
        """

        surv = self.surv

        grad = 0.
        if return_grad_obs:
            grad_obs = np.zeros(self.exog.shape, dtype=np.float64)

        # Loop over strata
        for stx in range(surv.nstrat):

            # Indices of subjects in the stratum
            strat_ix = surv.stratum_rows[stx]

            # Unique failure times in the stratum
            uft_ix = surv.ufailt_ix[stx]
            nuft = len(uft_ix)

            # exog and linear predictor for the stratum
            exog_s = surv.exog_s[stx]
            linpred = np.dot(exog_s, params)
            if surv.offset_s is not None:
                linpred += surv.offset_s[stx]
            linpred -= linpred.max()
            e_linpred = np.exp(linpred)

            xp0, xp1 = 0., 0.

            # Iterate backward through the unique failure times.
            for i in range(nuft)[::-1]:

                # Update for new cases entering the risk set.
                ix = surv.risk_enter[stx][i]
                if len(ix) > 0:
                    v = exog_s[ix,:]
                    xp0 += e_linpred[ix].sum()
                    xp1 += (e_linpred[ix][:,None] * v).sum(0)

                # Account for all cases that fail at this point.
                ix = uft_ix[i]
                grad += (exog_s[ix,:] - xp1 / xp0).sum(0)
                if return_grad_obs:
                    ii = strat_ix[ix]
                    grad_obs[ii, :] = exog_s[ix, :] - xp1 / xp0

                # Update for cases leaving the risk set.
                ix = surv.risk_exit[stx][i]
                if len(ix) > 0:
                    v = exog_s[ix,:]
                    xp0 -= e_linpred[ix].sum()
                    xp1 -= (e_linpred[ix][:,None] * v).sum(0)

        if return_grad_obs:
            return grad, grad_obs
        else:
            return grad

    def efron_gradient(self, params, return_grad_obs=False):
        """
        Returns the gradient of the log partial likelihood evaluated
        at `params`, using the Efron method to handle tied times.
        """

        surv = self.surv
        if return_grad_obs:
            grad_obs = np.zeros(self.exog.shape, dtype=np.float64)

        grad = 0.

        # Loop over strata
        for stx in range(surv.nstrat):

            # Indices of cases in the stratum
            strat_ix = surv.stratum_rows[stx]

            # exog and linear predictor of the stratum
            exog_s = surv.exog_s[stx]
            linpred = np.dot(exog_s, params)
            if surv.offset_s is not None:
                linpred += surv.offset_s[stx]
            linpred -= linpred.max()
            e_linpred = np.exp(linpred)

            xp0, xp1 = 0., 0.

            # Iterate backward through the unique failure times.
            uft_ix = surv.ufailt_ix[stx]
            nuft = len(uft_ix)
            for i in range(nuft)[::-1]:

                # Update for new cases entering the risk set.
                ix = surv.risk_enter[stx][i]
                if len(ix) > 0:
                    v = exog_s[ix,:]
                    xp0 += e_linpred[ix].sum()
                    xp1 += (e_linpred[ix][:,None] * v).sum(0)
                ixf = uft_ix[i]
                if len(ixf) > 0:
                    v = exog_s[ixf,:]
                    xp0f = e_linpred[ixf].sum()
                    xp1f = (e_linpred[ixf][:,None] * v).sum(0)

                    # Consider all cases that fail at this point.
                    grad += v.sum(0)
                    if return_grad_obs:
                        ii = strat_ix[ixf]
                        grad_obs[ii, :] += v

                    m = len(ixf)
                    J = np.arange(m, dtype=np.float64) / m
                    numer = xp1 - np.outer(J, xp1f)
                    denom = xp0 - np.outer(J, xp0f)
                    ratio = numer / denom
                    rsum = ratio.sum(0)
                    grad -= rsum
                    if return_grad_obs:
                        ii = strat_ix[ixf]
                        grad_obs[ii, :] -= rsum / m

                # Update for cases leaving the risk set.
                ix = surv.risk_exit[stx][i]
                if len(ix) > 0:
                    v = exog_s[ix,:]
                    xp0 -= e_linpred[ix].sum()
                    xp1 -= (e_linpred[ix][:,None] * v).sum(0)

        if return_grad_obs:
            return grad, grad_obs
        else:
            return grad

    def breslow_hessian(self, params):
        """
        Returns the Hessian of the log partial likelihood evaluated at
        `params`, using the Breslow method to handle tied times.
        """

        surv = self.surv

        hess = 0.

        # Loop over strata
        for stx in range(surv.nstrat):

            uft_ix = surv.ufailt_ix[stx]
            nuft = len(uft_ix)

            exog_s = surv.exog_s[stx]

            linpred = np.dot(exog_s, params)
            if surv.offset_s is not None:
                linpred += surv.offset_s[stx]
            linpred -= linpred.max()
            e_linpred = np.exp(linpred)

            xp0, xp1, xp2 = 0., 0., 0.

            # Iterate backward through the unique failure times.
            for i in range(nuft)[::-1]:

                # Update for new cases entering the risk set.
                ix = surv.risk_enter[stx][i]
                if len(ix) > 0:
                    xp0 += e_linpred[ix].sum()
                    v = exog_s[ix,:]
                    xp1 += (e_linpred[ix][:,None] * v).sum(0)
                    mat = v[None,:,:]
                    elx = e_linpred[ix]
                    xp2 += (mat.T * mat * elx[None,:,None]).sum(1)

                # Account for all cases that fail at this point.
                m = len(uft_ix[i])
                hess += m*(xp2 / xp0  - np.outer(xp1, xp1) / xp0**2)

                # Update for new cases entering the risk set.
                ix = surv.risk_exit[stx][i]
                if len(ix) > 0:
                    xp0 -= e_linpred[ix].sum()
                    v = exog_s[ix,:]
                    xp1 -= (e_linpred[ix][:,None] * v).sum(0)
                    mat = v[None,:,:]
                    elx = e_linpred[ix]
                    xp2 -= (mat.T * mat * elx[None,:,None]).sum(1)

        return -hess

    def efron_hessian(self, params):
        """
        Returns the Hessian matrix of the partial log-likelihood
        evaluated at `params`, using the Efron method to handle tied
        times.
        """

        surv = self.surv

        hess = 0.

        # Loop over strata
        for stx in range(surv.nstrat):

            exog_s = surv.exog_s[stx]

            linpred = np.dot(exog_s, params)
            if surv.offset_s is not None:
                linpred += surv.offset_s[stx]
            linpred -= linpred.max()
            e_linpred = np.exp(linpred)

            xp0, xp1, xp2 = 0., 0., 0.

            # Iterate backward through the unique failure times.
            uft_ix = surv.ufailt_ix[stx]
            nuft = len(uft_ix)
            for i in range(nuft)[::-1]:

                # Update for new cases entering the risk set.
                ix = surv.risk_enter[stx][i]
                if len(ix) > 0:
                    xp0 += e_linpred[ix].sum()
                    v = exog_s[ix,:]
                    xp1 += (e_linpred[ix][:,None] * v).sum(0)
                    mat = v[None,:,:]
                    elx = e_linpred[ix]
                    xp2 += (mat.T * mat * elx[None,:,None]).sum(1)

                ixf = uft_ix[i]
                if len(ixf) > 0:
                    v = exog_s[ixf,:]
                    xp0f = e_linpred[ixf].sum()
                    xp1f = (e_linpred[ixf][:,None] * v).sum(0)
                    mat = v[None,:,:]
                    elx = e_linpred[ixf]
                    xp2f = (mat.T * mat * elx[None,:,None]).sum(1)

                # Account for all cases that fail at this point.
                m = len(uft_ix[i])
                J = np.arange(m, dtype=np.float64) / m
                c0 = xp0 - J*xp0f
                mat = (xp2[None,:,:] - J[:,None,None]*xp2f) / c0[:,None,None]
                hess += mat.sum(0)
                mat = (xp1[None, :] - np.outer(J, xp1f)) / c0[:, None]
                mat = mat[:, :, None] * mat[:, None, :]
                hess -= mat.sum(0)

                # Update for new cases entering the risk set.
                ix = surv.risk_exit[stx][i]
                if len(ix) > 0:
                    xp0 -= e_linpred[ix].sum()
                    v = exog_s[ix,:]
                    xp1 -= (e_linpred[ix][:,None] * v).sum(0)
                    mat = v[None,:,:]
                    elx = e_linpred[ix]
                    xp2 -= (mat.T * mat * elx[None,:,None]).sum(1)

        return -hess

    def robust_covariance(self, params):
        """
        Returns a covariance matrix for the proportional hazards model
        regresion coefficient estimates that is robust to certain
        forms of model misspecification.

        Parameters
        ----------
        params : ndarray
            The parameter vector at which the covariance matrix is
            calculated.

        Returns
        -------
        The robust covariance matrix as a square ndarray.

        Notes
        -----
        This function uses the `groups` argument to determine groups
        within which observations may be dependent.  The covariance
        matrix is calculated using the Huber-White "sandwich" approach.
        """

        if self.groups is None:
            raise ValueError("`groups` must be specified to calculate the robust covariance matrix")

        hess = self.hessian(params)

        score_obs = self.score_residuals(params)

        # Collapse
        grads = {}
        for i,g in enumerate(self.groups):
            if g not in grads:
                grads[g] = 0.
            grads[g] += score_obs[i, :]
        grads = np.asarray(grads.values())

        mat = grads[None, :, :]
        mat = mat.T * mat
        mat = mat.sum(1)

        hess_inv = np.linalg.inv(hess)
        cmat = np.dot(hess_inv, np.dot(mat, hess_inv))

        return cmat

    def score_residuals(self, params):
        """
        Returns the score residuals calculated at a given vector of
        parameters.

        Parameters
        ----------
        params : ndarray
            The parameter vector at which the score residuals are
            calculated.

        Returns
        -------
        The score residuals, returned as a ndarray having the same
        shape as `exog`.

        Notes
        -----
        Observations in a stratum with no observed events have undefined
        score residuals, and contain NaN in the returned matrix.
        """

        surv = self.surv

        score_resid = np.zeros(self.exog.shape, dtype=np.float64)

        # Use to set undefined values to NaN.
        mask = np.zeros(self.exog.shape[0], dtype=np.int32)

        w_avg = self.weighted_covariate_averages(params)

        # Loop over strata
        for stx in range(surv.nstrat):

            uft_ix = surv.ufailt_ix[stx]
            exog_s = surv.exog_s[stx]
            nuft = len(uft_ix)
            strat_ix = surv.stratum_rows[stx]

            xp0 = 0.

            linpred = np.dot(exog_s, params)
            if surv.offset_s is not None:
                linpred += surv.offset_s[stx]
            linpred -= linpred.max()
            e_linpred = np.exp(linpred)

            at_risk_ix = set([])

            # Iterate backward through the unique failure times.
            for i in range(nuft)[::-1]:

                # Update for new cases entering the risk set.
                ix = surv.risk_enter[stx][i]
                at_risk_ix |= set(ix)
                xp0 += e_linpred[ix].sum()

                atr_ix = list(at_risk_ix)
                leverage = exog_s[atr_ix, :] - w_avg[stx][i, :]

                # Event indicators
                d = np.zeros(exog_s.shape[0])
                d[uft_ix[i]] = 1

                # The increment in the cumulative hazard
                dchaz = len(uft_ix[i]) / xp0

                # Piece of the martingale residual
                mrp = d[atr_ix] - e_linpred[atr_ix] * dchaz

                # Update the score residuals
                ii = strat_ix[atr_ix]
                score_resid[ii,:] += leverage * mrp[:, None]
                mask[ii] = 1

                # Update for cases leaving the risk set.
                ix = surv.risk_exit[stx][i]
                at_risk_ix -= set(ix)
                xp0 -= e_linpred[ix].sum()

        jj = np.flatnonzero(mask == 0)
        if len(jj) > 0:
            score_resid[jj, :] = np.nan

        return score_resid

    def weighted_covariate_averages(self, params):
        """
        Returns the hazard-weighted average of covariate values for
        subjects who are at-risk at a particular time.

        Arguments
        ---------
        params : ndarray
            Parameter vector

        Returns
        -------
        averages : list of ndarrays
            averages[stx][i,:] is a row vector containing the weighted
            average values (for all the covariates) of at-risk
            subjects a the i^th largest observed failure time in
            stratum `stx`, using the hazard multipliers as weights.

        Notes
        -----
        Used to calculate leverages and score residuals.
        """

        surv = self.surv

        averages = []
        xp0, xp1 = 0., 0.

        # Loop over strata
        for stx in range(surv.nstrat):

            uft_ix = surv.ufailt_ix[stx]
            exog_s = surv.exog_s[stx]
            nuft = len(uft_ix)

            average_s = np.zeros((len(uft_ix), exog_s.shape[1]),
                                  dtype=np.float64)

            linpred = np.dot(exog_s, params)
            if surv.offset_s is not None:
                linpred += surv.offset_s[stx]
            linpred -= linpred.max()
            e_linpred = np.exp(linpred)

            # Iterate backward through the unique failure times.
            for i in range(nuft)[::-1]:

                # Update for new cases entering the risk set.
                ix = surv.risk_enter[stx][i]
                xp0 += e_linpred[ix].sum()
                xp1 += np.dot(e_linpred[ix], exog_s[ix, :])

                average_s[i, :] = xp1 / xp0

                # Update for cases leaving the risk set.
                ix = surv.risk_exit[stx][i]
                xp0 -= e_linpred[ix].sum()
                xp1 -= np.dot(e_linpred[ix], exog_s[ix, :])

            averages.append(average_s)

        return averages

    def baseline_cumulative_hazard(self, params):
        """
        Estimate the baseline cumulative hazard and survival
        functions.

        Parameters
        ----------
        params : ndarray
            The model parameters.

        Returns
        -------
        A list of triples (time, hazard, survival) containing the time
        values and corresponding cumulative hazard and survival
        function values for each stratum.

        Notes
        -----
        Uses the Nelson-Aalen estimator.
        """

        # TODO: some disagreements with R, not the same algorithm but
        # hard to deduce what R is doing.  Our results are reasonable.

        surv = self.surv
        rslt = []

        # Loop over strata
        for stx in range(surv.nstrat):

            uft = surv.ufailt[stx]
            uft_ix = surv.ufailt_ix[stx]
            exog_s = surv.exog_s[stx]
            nuft = len(uft_ix)

            linpred = np.dot(exog_s, params)
            if surv.offset_s is not None:
                linpred += surv.offset_s[stx]
            e_linpred = np.exp(linpred)

            xp0 = 0.
            h0 = np.zeros(nuft, dtype=np.float64)

            # Iterate backward through the unique failure times.
            for i in range(nuft)[::-1]:

                # Update for new cases entering the risk set.
                ix = surv.risk_enter[stx][i]
                xp0 += e_linpred[ix].sum()

                # Account for all cases that fail at this point.
                ix = uft_ix[i]
                h0[i] = len(ix) / xp0

                # Update for cases leaving the risk set.
                ix = surv.risk_exit[stx][i]
                xp0 -= e_linpred[ix].sum()

            cumhaz = np.cumsum(h0) - h0
            surv = np.exp(-cumhaz)
            rslt.append([uft, cumhaz, surv])

        return rslt

    def baseline_cumulative_hazard_function(self, params):
        """
        Returns a function that calculates the baseline cumulative
        hazard function for each stratum.

        Parameters
        ----------
        params : ndarray
            The model parameters.

        Returns
        -------
        A dict mapping stratum names to the estimated baseline
        cumulative hazard function.
        """

        from scipy.interpolate import interp1d
        surv = self.surv
        base = self.baseline_cumulative_hazard(params)

        cumhaz_f = {}
        for stx in range(surv.nstrat):
            time_h = base[stx][0]
            cumhaz = base[stx][1]
            time_h = np.r_[-np.inf, time_h, np.inf]
            cumhaz = np.r_[cumhaz[0], cumhaz, cumhaz[-1]]
            func = interp1d(time_h, cumhaz, kind='zero')
            cumhaz_f[self.surv.stratum_names[stx]] = func

        return cumhaz_f

    def predict(self, params, cov_params=None, endog=None, exog=None,
                strata=None, offset=None, pred_type="lhr"):
        """
        Returns predicted values from the proportional hazards
        regression model.

        Parameters:
        -----------
        params : array-like
            The proportional hazards model parameters.
        cov_params : array-like
            The covariance matrix of the estimated `params` vector,
            used to obtain prediction errors if pred_type='lhr',
            otherwise optional.
        endog : array-like
            Duration (time) values at which the predictions are made.
            Only used if pred_type is either 'cumhaz' or 'surv'.  If
            using model `exog`, defaults to model `endog` (time), but
            may be provided explicitly to make predictions at
            alternative times.
        exog : array-like
            Data to use as `exog` in forming predictions.  If not
            provided, the `exog` values from the model used to fit the
            data are used.
        strata : array-like
            A vector of stratum values used to form the predictions.
            Not used (may be 'None') if pred_type is 'lhr' or 'hr'.
            If `exog` is None, the model stratum values are used.  If
            `exog` is not None and pred_type is 'surv' or 'cumhaz',
            stratum values must be provided (unless there is only one
            stratum).
        offset : array-like
            Offset values used to create the predicted values.
        pred_type : string
            If 'lhr', returns log hazard ratios, if 'hr' returns
            hazard ratios, if 'surv' returns the survival function, if
            'cumhaz' returns the cumulative hazard function.

        Returns
        -------
        A bunch containing two fields: `predicted_values` and
        `standard_errors`.

        Notes
        -----
        Standard errors are only returned when predicting the log
        hazard ratio (pred_type is 'lhr').

        Types `surv` and `cumhaz` require estimation of the cumulative
        hazard function.
        """

        pred_type = pred_type.lower()
        if pred_type not in ["lhr", "hr", "surv", "cumhaz"]:
            msg = "Type %s not allowed for prediction" % pred_type
            raise ValueError(msg)

        class bunch:
            predicted_values = None
            standard_errors = None
        ret_val = bunch()

        # Don't do anything with offset here because we want to allow
        # different offsets to be specified even if exog is the model
        # exog.
        exog_provided = True
        if exog is None:
            exog = self.exog
            exog_provided = False

        lhr = np.dot(exog, params)
        if offset is not None:
            lhr += offset
        # Never use self.offset unless we are also using self.exog
        elif self.offset is not None and not exog_provided:
            lhr += self.offset

        # Handle lhr and hr prediction first, since they don't make
        # use of the hazard function.

        if pred_type == "lhr":
            ret_val.predicted_values = lhr
            mat = np.dot(exog, cov_params)
            va = (mat * exog).sum(1)
            ret_val.standard_errors = np.sqrt(va)
            return ret_val

        hr = np.exp(lhr)

        if pred_type == "hr":
            ret_val.predicted_values = hr
            return ret_val

        # Makes sure endog is defined
        if endog is None and exog_provided:
            msg = "If `exog` is provided `endog` must be provided."
            raise ValueError(msg)
        # Use model endog if using model exog
        elif endog is None and not exog_provided:
            endog = self.endog

        # Make sure strata is defined
        if strata is None:
            if exog_provided and self.surv.nstrat > 1:
                raise ValueError("`strata` must be provided")
            if self.strata is None:
                strata = [self.surv.stratum_names[0],] * len(endog)
            else:
                strata = self.strata

        cumhaz = np.nan * np.ones(len(endog), dtype=np.float64)
        stv = np.unique(strata)
        bhaz = self.baseline_cumulative_hazard_function(params)
        for stx in stv:
            ix = np.flatnonzero(strata == stx)
            func = bhaz[stx]
            cumhaz[ix] = func(endog[ix]) * hr[ix]

        if pred_type == "cumhaz":
            ret_val.predicted_values = cumhaz

        elif pred_type == "surv":
            ret_val.predicted_values = np.exp(-cumhaz)

        return ret_val

    def get_distribution(self, params):
        """
        Returns a scipy distribution object corresponding to the
        distribution of uncensored endog (duration) values for each
        case.

        Parameters:
        -----------
        params : arrayh-like
            The model proportional hazards model parameters.

        Returns:
        --------
        A list of objects of type scipy.stats.distributions.rv_discrete

        Notes:
        ------
        The distributions are obtained from a simple discrete estimate
        of the survivor function that puts all mass on the observed
        failure times wihtin a stratum.
        """

        # TODO: this returns a Python list of rv_discrete objects, so
        # nothing can be vectorized.  It appears that rv_discrete does
        # not allow vectorization.

        from scipy.stats.distributions import rv_discrete

        surv = self.surv
        bhaz = self.baseline_cumulative_hazard(params)

        # The arguments to rv_discrete_float, first obtained by
        # stratum
        pk, xk = [], []

        for stx in range(self.surv.nstrat):

            exog_s = surv.exog_s[stx]

            linpred = np.dot(exog_s, params)
            if surv.offset_s is not None:
                linpred += surv.offset_s[stx]
            e_linpred = np.exp(linpred)

            # The unique failure times for this stratum (the support
            # of the distribution).
            pts = bhaz[stx][0]

            # The individual cumulative hazards for everyone in this
            # stratum.
            ichaz = np.outer(e_linpred, bhaz[stx][1])

            # The individual survival functions.
            usurv = np.exp(-ichaz)
            usurv = np.concatenate((usurv, np.zeros((usurv.shape[0], 1))),
                                   axis=1)

            # The individual survival probability masses.
            probs = -np.diff(usurv, 1)

            pk.append(probs)
            xk.append(np.outer(np.ones(probs.shape[0]), pts))

        # Pad to make all strata have the same shape
        mxc = max([x.shape[1] for x in xk])
        for k in range(self.surv.nstrat):
            if xk[k].shape[1] < mxc:
                xk1 = np.zeros((xk.shape[0], mxc))
                pk1 = np.zeros((pk.shape[0], mxc))
                xk1[:, -mxc:] = xk
                pk1[:, -mxc:] = pk
                xk[k], pk[k] = xk1, pk1

        xka = np.nan * np.zeros((len(self.endog), mxc), dtype=np.float64)
        pka = np.ones((len(self.endog), mxc), dtype=np.float64) / mxc

        for stx in range(self.surv.nstrat):
            ix = self.surv.stratum_rows[stx]
            xka[ix, :] = xk[stx]
            pka[ix, :] = pk[stx]

        dist = rv_discrete_float(xka, pka)

        return dist


class PHregResults(base.LikelihoodModelResults):
    '''
    Class to contain results of fitting a Cox proportional hazards
    survival model.

    PHregResults inherits from statsmodels.LikelihoodModelResults

    Parameters
    ----------
    See statsmodels.LikelihoodModelResults

    Returns
    -------
    **Attributes**

    model : class instance
        PHreg model instance that called fit.
    normalized_cov_params : array
        The sampling covariance matrix of the estimates
    params : array
        The coefficients of the fitted model.  Each coefficient is the
        log hazard ratio corresponding to a 1 unit difference in a
        single covariate while holding the other covariates fixed.
    bse : array
        The standard errors of the fitted parameters.

    See Also
    --------
    statsmodels.LikelihoodModelResults
    '''

    def __init__(self, model, params, cov_params, covariance_type="naive"):

        self.covariance_type = covariance_type

        super(PHregResults, self).__init__(model, params,
           normalized_cov_params=cov_params)

    @cache_readonly
    def standard_errors(self):
        """
        Returns the standard errors of the parameter estimates.
        """
        return np.sqrt(np.diag(self.cov_params()))

    @cache_readonly
    def bse(self):
        """
        Returns the standard errors of the parameter estimates.
        """
        return self.standard_errors

    def get_distribution(self):
        """
        Returns a scipy distribution object corresponding to the
        distribution of uncensored endog (duration) values for each
        case.

        Returns:
        --------
        A list of objects of type scipy.stats.distributions.rv_discrete

        Notes:
        ------
        The distributions are obtained from a simple discrete estimate
        of the survivor function that puts all mass on the observed
        failure times wihtin a stratum.
        """

        return self.model.get_distribution(self.params)


    def predict(self, endog=None, exog=None, strata=None,
                offset=None, pred_type="lhr"):
        """
        Returns predicted values from the fitted proportional hazards
        regression model.

        Parameters:
        -----------
        params : array-;like
            The proportional hazards model parameters.
        endog : array-like
            Duration (time) values at which the predictions are made.
            Only used if pred_type is either 'cumhaz' or 'surv'.  If
            using model `exog`, defaults to model `endog` (time), but
            may be provided explicitly to make predictions at
            alternative times.
        exog : array-like
            Data to use as `exog` in forming predictions.  If not
            provided, the `exog` values from the model used to fit the
            data are used.
        strata : array-like
            A vector of stratum values used to form the predictions.
            Not used (may be 'None') if pred_type is 'lhr' or 'hr'.
            If `exog` is None, the model stratum values are used.  If
            `exog` is not None and pred_type is 'surv' or 'cumhaz',
            stratum values must be provided (unless there is only one
            stratum).
        offset : array-like
            Offset values used to create the predicted values.
        pred_type : string
            If 'lhr', returns log hazard ratios, if 'hr' returns
            hazard ratios, if 'surv' returns the survival function, if
            'cumhaz' returns the cumulative hazard function.

        Returns
        -------
        A bunch containing two fields: `predicted_values` and
        `standard_errors`.

        Notes
        -----
        Standard errors are only returned when predicting the log
        hazard ratio (pred_type is 'lhr').

        Types `surv` and `cumhaz` require estimation of the cumulative
        hazard function.
        """

        return self.model.predict(self.params, self.cov_params(),
                                  endog, exog, strata, offset,
                                  pred_type)

    def _group_stats(self, groups):
        """
        Descriptive statistics of the groups.
        """
        gsize = {}
        for x in groups:
            if x not in gsize:
                gsize[x] = 0
            gsize[x] += 1
        gsize = np.asarray(gsize.values())
        return gsize.min(), gsize.max(), gsize.mean()

    @cache_readonly
    def weighted_covariate_averages(self):
        """
        The average covariate values within the at-risk set at each
        event time point, weighted by hazard.
        """
        return self.model.weighted_covariate_averages(self.params)

    @cache_readonly
    def score_residuals(self):
        """
        A matrix containing the score residuals.
        """
        return self.model.score_residuals(self.params)

    @cache_readonly
    def baseline_cumulative_hazard(self):
        """
        A list (corresponding to the strata) containing the baseline
        cumulative hazard function evaluated at the event points.
        """
        return self.model.baseline_cumulative_hazard(self.params)

    @cache_readonly
    def baseline_cumulative_hazard_function(self):
        """
        A list (corresponding to the strata) containing function
        objects that calculate the cumulative hazard function.
        """
        return self.model.baseline_cumulative_hazard_function(self.params)

    @cache_readonly
    def schoenfeld_residuals(self):
        """
        A matrix containing the Schoenfeld residuals.

        Notes
        -----
        Schoenfeld residuals for censored observations are set to zero.
        """

        surv = self.model.surv
        w_avg = self.weighted_covariate_averages

        # Initialize at NaN since rows that belong to strata with no
        # events have undefined residuals.
        sch_resid = np.nan*np.ones(self.model.exog.shape, dtype=np.float64)

        # Loop over strata
        for stx in range(surv.nstrat):

            uft = surv.ufailt[stx]
            exog_s = surv.exog_s[stx]
            time_s = surv.time_s[stx]
            strat_ix = surv.stratum_rows[stx]

            ii = np.searchsorted(uft, time_s)

            # These subjects are censored after the last event in
            # their stratum, so have empty risk sets and undefined
            # residuals.
            jj = np.flatnonzero(ii < len(uft))

            sch_resid[strat_ix[jj], :] = exog_s[jj, :] - w_avg[stx][ii[jj], :]

        jj = np.flatnonzero(self.model.status == 0)
        sch_resid[jj, :] = np.nan

        return sch_resid

    @cache_readonly
    def martingale_residuals(self):
        """
        The martingale residuals.
        """

        surv = self.model.surv

        # Initialize at NaN since rows that belong to strata with no
        # events have undefined residuals.
        mart_resid = np.nan*np.ones(len(self.model.endog), dtype=np.float64)

        cumhaz_f_list = self.baseline_cumulative_hazard_function

        # Loop over strata
        for stx in range(surv.nstrat):

            cumhaz_f = cumhaz_f_list[stx]

            exog_s = surv.exog_s[stx]
            time_s = surv.time_s[stx]

            linpred = np.dot(exog_s, self.params)
            if surv.offset_s is not None:
                linpred += surv.offset_s[stx]
            e_linpred = np.exp(linpred)

            ii = surv.stratum_rows[stx]
            chaz = cumhaz_f(time_s)
            mart_resid[ii] = self.model.status[ii] - e_linpred * chaz

        return mart_resid

    def summary(self, yname=None, xname=None, title=None, alpha=.05):
        """
        Summarize the proportional hazards regression results.

        Parameters
        -----------
        yname : string, optional
            Default is `y`
        xname : list of strings, optional
            Default is `x#` for ## in p the number of regressors
        title : string, optional
            Title for the top table. If not None, then this replaces
            the default title
        alpha : float
            significance level for the confidence intervals

        Returns
        -------
        smry : Summary instance
            this holds the summary tables and text, which can be
            printed or converted to various output formats.

        See Also
        --------
        statsmodels.iolib.summary.Summary : class to hold summary
            results
        """

        from statsmodels.iolib import summary2
        from statsmodels.compat.collections import OrderedDict
        smry = summary2.Summary()
        float_format = "%8.3f"

        info = OrderedDict()
        info["Model:"] = "PH Reg"
        if yname is None:
            yname = self.model.endog_names
        info["Dependent variable:"] = yname
        info["Ties:"] = self.model.ties.capitalize()
        info["Sample size:"] = str(self.model.surv.n_obs)
        info["Num. events:"] = str(int(sum(self.model.status)))

        if self.model.groups is not None:
            mn, mx, avg = self._group_stats(self.model.groups)
            info["Max. group size:"] = str(mx)
            info["Min. group size:"] = str(mn)
            info["Avg. group size:"] = str(avg)

        smry.add_dict(info, align='l', float_format=float_format)

        param = summary2.summary_params(self, alpha=alpha)
        param = param.rename(columns={"Coef.": "log HR",
                                      "Std.Err.": "log HR SE"})
        param.insert(2, "HR", np.exp(param["log HR"]))
        a = "[%.3f" % (alpha / 2)
        param.loc[:, a] = np.exp(param.loc[:, a])
        a = "%.3f]" % (1 - alpha / 2)
        param.loc[:, a] = np.exp(param.loc[:, a])
        if xname != None:
            param.index = xname
        smry.add_df(param, float_format=float_format)
        smry.add_title(title=title, results=self)
        smry.add_text("Confidence intervals are for the hazard ratios")

        if self.model.groups is not None:
            smry.add_text("Standard errors account for dependence within groups")

        return smry

class rv_discrete_float(object):
    """
    A class representing a collection of discrete distributions.

    Parameters:
    ----------
    xk : 2d array-like
        The support points, should be non-decreasing within each
        row.
    pk : 2d array-like
        The probabilities, should sum to one within each row.

    Notes
    -----
    Each row of `xk`, and the corresponding row of `pk` describe a
    discrete distribution.

    `xk` and `pk` should both be two-dimensional ndarrays.  Each row
    of `pk` should sum to 1.

    This class is used as a substitute for scipy.distributions.
    rv_discrete, since that class does not allow non-integer support
    points, or vectorized operations.

    Only a limited number of methods are implemented here compared to
    the other scipy distribution classes.
    """

    def __init__(self, xk, pk):

        self.xk = xk
        self.pk = pk
        self.cpk = np.cumsum(self.pk, axis=1)

    def rvs(self):
        """
        Returns a random sample from the discrete distribution.

        A vector is returned containing a single draw from each row of
        `xk`, using the probabilities of the corresponding row of `pk`
        """

        n = self.xk.shape[0]
        u = np.random.uniform(size=n)

        ix = (self.cpk < u[:, None]).sum(1)
        ii = np.arange(n, dtype=np.int32)
        return self.xk[(ii,ix)]

    def mean(self):
        """
        Returns a vector containing the mean values of the discrete
        distributions.

        A vector is returned containing the mean value of each row of
        `xk`, using the probabilities in the corresponding row of
        `pk`.
        """

        return (self.xk * self.pk).sum(1)

    def var(self):
        """
        Returns a vector containing the variances of the discrete
        distributions.

        A vector is returned containing the variance for each row of
        `xk`, using the probabilities in the corresponding row of
        `pk`.
        """

        mn = self.mean()
        xkc = self.xk - mn[:, None]

        return (self.pk * (self.xk - xkc)**2).sum(1)

    def std(self):
        """
        Returns a vector containing the standard deviations of the
        discrete distributions.

        A vector is returned containing the standard deviation for
        each row of `xk`, using the probabilities in the corresponding
        row of `pk`.
        """

        return np.sqrt(self.var())
