"""
Linear mixed effects models

"""

import numpy as np
import statsmodels.base.model as base
from scipy.optimize import fmin_cg, fmin
from scipy.misc import derivative

class LME(base.Model):

    def __init__(self, endog, exog, exog_re, groups, missing='none'):

        groups = np.array(groups)

        # Calling super creates self.ndog, etc. as ndarrays and the
        # original exog, endog, etc. are self.data.endog, etc.
        super(LME, self).__init__(endog, exog, exog_re=exog_re,
                                  groups=groups, missing=missing)

        # Convert the data to the internal representation, which is a
        # list of arrays, corresponding to the groups.
        group_labels = list(set(groups))
        group_labels.sort()
        row_indices = dict((s, []) for s in group_labels)
        [row_indices[groups[i]].append(i) for i in range(len(self.endog))]
        self.row_indices = row_indices
        self.group_labels = group_labels
        self.ngroup = len(self.group_labels)

        # Split the data by groups
        self.endog_li = self.group_list(self.endog)
        self.exog_li = self.group_list(self.exog)
        self.exog_re_li = self.group_list(self.exog_re)

        # The total number of observations, summed over all groups
        self.ntot = sum([len(y) for y in self.endog_li])


    def group_list(self, array):
        """
        Returns `array` split into subarrays corresponding to the
        grouping structure.
        """

        if array.ndim == 1:
            return [np.array(array[self.row_indices[k]])
                    for k in self.group_labels]
        else:
            return [np.array(array[self.row_indices[k], :])
                    for k in self.group_labels]


    def _unpack(self, params):
        """
        Takes as input the packed parameter vector and returns three
        values:

        params : 1d ndarray
            The fixed effects coefficients
        revar : 2d ndarray
            The random effects covariance matrix
        sig2 : non-negative real scaoar
            The error variance
        """

        pf = self.exog.shape[1]
        pr = self.exog_re.shape[1]
        nr = pr * (pr + 1) / 2
        params_fe = params[0:pf]
        params_re = params[pf:pf+nr]
        params_rv = params[-1]

        # Unpack the covariance matrix of the random effects
        revar = np.zeros((pr, pr), dtype=np.float64)
        ix = np.tril_indices(pr)
        revar[ix] = params_re
        revar = (revar + revar.T) - np.diag(np.diag(revar))

        return params_fe, revar, params_rv

    def _pack(self, params_fe, revar, sig2):
        """
        Packs the model parameters into a single vector.

        Arguments
        ---------
        params_fe : 1d ndarray
            The fixed effects parameters
        revar : 2d ndarray
            The covariance matrix of the random effects
        sig2 : non-negative scalar
            The error variance
        """

        ix = np.tril_indices(revar.shape[0])
        return np.concatenate((params_fe, revar[ix], np.r_[sig2,]))

    def like(self, params):
        """
        Evaluate the log-likelihood of the model.

        Arguments
        ---------
        params : 1d ndarray
            The parameter values, packed into a single vector

        Returns
        -------
        likeval : scalar
            The log-likelihood valuoe at `params`.
        """

        params_fe, revar, sig2 = self._unpack(params)

        try:
            np.linalg.cholesky(revar)
        except np.linalg.LinAlgError:
            return -np.inf

        if sig2 <= 0:
            return -np.inf

        likeval = 0.
        for k in range(self.ngroup):

            # Get the residuals
            expval = np.dot(self.exog_li[k], params_fe)
            resid = self.endog_li[k] - expval

            # The marginal covariance matrix for this group
            ex_r = self.exog_re_li[k]
            vmat = np.dot(ex_r, np.dot(revar, ex_r.T))
            vmat += sig2*np.eye(vmat.shape[0])

            # Update the log-likelihood
            u = np.linalg.solve(vmat, resid)
            _,ld = np.linalg.slogdet(vmat)
            likeval -= 0.5*(ld + np.dot(resid, u))

        return likeval


    def score(self, params):
        """
        Calculates the score vector for the mixed effects model.

        Parameters
        ----------
        params : 1d ndarray
            All model parameters i packed form

        Returns
        -------
        scorevec : 1d ndarray
            The score vector, calculated at `params`.
        """

        params_fe, revar, sig2 = self._unpack(params)

        # Construct the score.
        score_fe = 0.
        pr = self.exog_re.shape[1]
        score_re = np.zeros(pr*(pr+1)/2, dtype=np.float64)
        score_rv = 0.
        for k in range(self.ngroup):

            # Get the residuals
            expval = np.dot(self.exog_li[k], params_fe)
            resid = self.endog_li[k] - expval

            # Contruct the marginal covariance matrix for this group
            ex_r = self.exog_re_li[k]
            vmat = np.dot(ex_r, np.dot(revar, ex_r.T))
            vmat += sig2*np.eye(vmat.shape[0])

            # Update the fixed effects score
            u = np.linalg.solve(vmat, resid)
            score_fe += 2*np.dot(self.exog_li[k].T, u)

            # Variance score with respect to the marginal covariance
            vmati = np.linalg.inv(vmat)
            score_dv = -0.5*vmati + 0.5*np.outer(u, u)

            # Pack the variance score.
            # TODO: try to reduce looping via broadcasting
            jx = 0
            for j1 in range(pr):
                for j2 in range(j1 + 1):
                    f = 1 if j1 ==j2 else 2
                    score_re[jx] += f*np.sum(score_dv *
                                  np.outer(ex_r[:, j1], ex_r[:, j2]))
                    jx += 1

            score_rv += np.sum(np.diag(score_dv))

        return np.concatenate((score_fe, score_re, np.r_[score_rv,]))


    def Estep(self, params_fe, revar, sig2):
        """
        The E-step of the EM algorithm.

        Parameters
        ----------
        params_fe : 1d ndarray
            The current value of the fixed effect coefficients
        revar : 2d ndarray
            The current value of the covariance matrix of random
            effects
        sig2 : positive scalar
            The current value of the error variance

        Returns
        -------
        m1x : 1d ndarray
            sum_groups X'*Z*E[gamma | Y], where X and Z are the fixed
            and random effects covariates, gamma is the random
            effects, and Y is the observed data
        m1y : scalar
            sum_groups Y'*E[gamma | Y]
        m2 : 2d ndarray
            sum_groups E[gamma * gamma' | Y]
        m2xx : 2d ndarray
            sum_groups Z'*Z * E[gamma * gamma' | Y]
        """

        m1x, m1y, m2, m2xx = 0., 0., 0., 0.

        for k in range(self.ngroup):

            # Get the residuals
            expval = np.dot(self.exog_li[k], params_fe)
            resid = self.endog_li[k] - expval

            # Contruct the marginal covariance matrix for this group
            ex_r = self.exog_re_li[k]
            vmat = np.dot(ex_r, np.dot(revar, ex_r.T))
            vmat += sig2*np.eye(vmat.shape[0])

            vr1 = np.linalg.solve(vmat, resid)
            vr1 = np.dot(ex_r.T, vr1)
            vr1 = np.dot(revar, vr1)

            vr2 = np.linalg.solve(vmat, self.exog_re_li[k])
            vr2 = np.dot(vr2, revar)
            vr2 = np.dot(ex_r.T, vr2)
            vr2 = np.dot(revar, vr2)

            rg = np.dot(ex_r, vr1)
            m1x += np.dot(self.exog_li[k].T, rg)
            m1y += np.dot(self.endog_li[k].T, rg)
            egg = revar - vr2 + np.outer(vr1, vr1)
            m2 += egg
            m2xx += np.dot(np.dot(ex_r.T, ex_r), egg)

        return m1x, m1y, m2, m2xx


    def EM(self, num_em=10, full_output=False):
        """
        Performs `num_em` steps of the EM algorithm.

        Returns
        -------
        params_fe : 1d ndarray
            The final value of the fixed effects coefficients
        revar : 2d ndarray
            The final value of the random effects covariance
            matrix
        sig2 : float
            The final value of the error variance
        hist : list
            The iteration history, only returned if `full_output`
            is True.
        """

        # The iteration history
        hist = []

        # Starting values
        pf = self.exog_li[0].shape[1]
        pr = self.exog_re_li[0].shape[1]
        params_fe = np.zeros(pf, dtype=np.float64)
        revar = np.eye(pr, dtype=np.float64)
        sig2 = 1

        xxtot = 0.
        for x in self.exog_li:
            xxtot += np.dot(x.T, x)

        xytot = 0.
        for x,y in zip(self.exog_li, self.endog_li):
            xytot += np.dot(x.T, y)

        pp = []
        for itr in range(num_em):

            m1x, m1y, m2, m2xx = self.Estep(params_fe, revar, sig2)

            params_fe = np.linalg.solve(xxtot, xytot - m1x)
            revar = m2 / self.ngroup

            sig2 = 0.
            for x,y in zip(self.exog_li, self.endog_li):
                sig2 += np.sum((y - np.dot(x, params_fe))**2)
            sig2 -= 2*m1y
            sig2 += 2*np.dot(params_fe, m1x)
            sig2 += np.trace(m2xx)
            sig2 /= self.ntot

            if full_output:
                hist.append([params_fe, revar, sig2])

        if full_output:
            return params_fe, revar, sig2, hist
        else:
            return params_fe, revar, sig2

    def fit(self, num_em=10):
        """
        Fit a linear mixed model to the data.

        Parameters
        ----------
        num_em : non-negative integer
            The number of EM steps to take before switching to
            gradient optimization

        """

        like = lambda x: -self.like(x)
        score = lambda x: -self.score(x)

        x0 = np.zeros(self.exog.shape[1])
        pr = self.exog_re.shape[1]
        x1 = np.eye(pr)
        ix = np.tril_indices(pr)
        x1 = x1[ix]
        x0 = np.concatenate((x0, x1, np.r_[1.,]))

        params_fe, revar, sig2 = self.EM(num_em)

        params_em = self._pack(params_fe, revar, sig2)

        params = fmin_cg(like, params_em, score)

        m = len(params)
        hess = np.zeros((m,m), dtype=np.float64)
        for j1 in range(m):
            for j2 in range(j1+1):
                params0 = params.copy()
                def g(x):
                    params0[j2] = x
                    return self.score(params0)[j1]
                hess[j1, j2] = derivative(g, params[j2], dx=1e-6)
        pcov = np.linalg.inv(-hess)

        results = LMEResults(self, params, pcov)
        return results


class LMEResults(base.LikelihoodModelResults):
    '''
    Class to contain results of fitting a linear mixed effects model.

    LMEResults inherits from statsmodels.LikelihoodModelResults

    Parameters
    ----------
    See statsmodels.LikelihoodModelResults

    Returns
    -------
    **Attributes**

    model : class instance
        Pointer to PHreg model instance that called fit.
    normalized_cov_params : array
        The sampling covariance matrix of the estimates
    params_fe : array
        The fitted fixed-effects coefficients
    params_re : array
        The fitted random-effects covariance matrix
    bse_fe : array
        The standard errors of the fitted fixed effects coefficients
    bse_re : array
        The standard errors of the fitted random effects covariance
        matrix

    See Also
    --------
    statsmodels.LikelihoodModelResults
    '''


    def __init__(self, model, params, cov_params):

        super(LMEResults, self).__init__(model, params,
           normalized_cov_params=cov_params)


    def summary(self, yname=None, xname=None, title=None, alpha=.05):
        """Summarize the Regression Results

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
        smry = summary2.Summary()
        float_format = "%.3f"
        smry.add_base(results=self, alpha=alpha,
                      float_format=float_format,
                      xname=xname, yname=yname, title=title)

        return smry
