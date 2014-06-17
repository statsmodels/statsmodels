import numpy as np
from statsmodels.base import model
import statsmodels.base.model as base
from statsmodels.tools.decorators import cache_readonly
import warnings

"""
Implementation of proportional hazards regression models for data that
may be censored ("Cox models").

References
---------
T Therneau (1996).  Extending the Cox model.  Technical report.
http://www.mayo.edu/research/documents/biostat-58pdf/DOC-10027288

G Rodriguez (2005).  Non-parametric estimation in survival models.
http://data.princeton.edu/pop509/NonParametricSurvival.pdf
"""


class PH_SurvivalTime(object):

    def __init__(self, time, status, exog, strata=None, entry=None,
                 offset=None):
        """
        Represent a collection of survival times with possible
        stratification and left truncation.  Various indexes needed
        for fitting proportional hazards regression models are
        precalculated.

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

        Notes
        ------
        time, event, strata, entry, and the first dimension of exog
        all must have the same length
        """

        # Default strata
        if strata is None:
            strata = np.zeros(len(time))

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
        else:
            stratum_rows = [np.arange(len(time)),]

        # Remove strata with no events
        ix = [i for i,ix in enumerate(stratum_rows) if status[ix].sum() > 0]
        stratum_rows = [stratum_rows[i] for i in ix]

        # The number of strata
        nstrat = len(stratum_rows)
        self.nstrat = nstrat

        # Remove subjects whose entry time occurs after the last event
        # in their stratum.
        for stx,ix in enumerate(stratum_rows):
            last_failure = max(time[ix][status[ix] == 1])
            ii = [i for i,t in enumerate(entry[ix]) if
                  t < last_failure]
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
    groups : array-like
        Labels indicating groups of observations that may be
        dependent.  If present, the standard errors account for this
        dependence. Does not affect fitted values.
    offset : array-like
        Array of offset values
    missing : string
        The method used to handle missing data
    """

    def __init__(self, endog, exog, status=None, entry=None,
                 strata=None, groups=None, offset=None,
                 ties='breslow', missing='drop'):

        # Default is no censoring
        if status is None:
            status = np.ones(len(endog))

        super(PHreg, self).__init__(endog, exog, status=status,
                                    entry=entry, strata=strata,
                                    groups=groups, offset=offset,
                                    missing=missing)

        # endog and exog are automatically converted, but these are
        # not
        if self.status is not None:
            self.status = np.asarray(self.status)
        if self.entry is not None:
            self.entry = np.asarray(self.entry)
        if self.strata is not None:
            self.strata = np.asarray(self.strata)
        if self.groups is not None:
            self.groups = np.asarray(self.groups)
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

    def fit(self, **args):

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

        # TODO: some disagreements with R

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
        A list (corresponding to the strata) of functions returning
        the baseline cumulative hazard function.
        """

        from scipy.interpolate import interp1d
        surv = self.surv
        base = self.baseline_cumulative_hazard(params)

        # Get the cumulative hazard function for this stratum.
        cumhaz_f = []

        for stx in range(surv.nstrat):
            time_h = base[stx][0]
            cumhaz = base[stx][1]
            time_h = np.r_[-np.inf, time_h, np.inf]
            cumhaz = np.r_[cumhaz[0], cumhaz, cumhaz[-1]]
            func = interp1d(time_h, cumhaz, kind='zero')
            cumhaz_f.append(func)

        return cumhaz_f


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
    def martingale_residuals(self):
        """
        A matrix containing the martingale residuals.
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
        info["Sample size:"] = str(len(self.model.endog))
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
