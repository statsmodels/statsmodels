"""
Linear mixed effects models

"""


import numpy as np
import statsmodels.base.model as base
from scipy.optimize import fmin_cg, fmin

class LME(base.Model):


    def __init__(self, endog, exog, exog_re, groups, missing='none'):

        groups = np.array(groups)

        # Calling super creates self.ndog, etc. as ndarrays and the
        # original exog, endog, etc. are self.data.endog, etc.
        super(LME, self).__init__(endog, exog, exog_re=exog_re,
                                  groups=groups, missing=missing)

        # Convert the data to the internal representation, which is a
        # list of arrays, corresponding to the clusters.
        group_labels = list(set(groups))
        group_labels.sort()
        row_indices = dict((s, []) for s in group_labels)
        [row_indices[groups[i]].append(i) for i in range(len(self.endog))]
        self.row_indices = row_indices
        self.group_labels = group_labels
        self.ngroup = len(self.group_labels)

        self.endog_li = self.cluster_list(self.endog)
        self.exog_li = self.cluster_list(self.exog)
        self.exog_re_li = self.cluster_list(self.exog_re)


    def cluster_list(self, array):
        """
        Returns `array` split into subarrays corresponding to the
        cluster structure.
        """

        if array.ndim == 1:
            return [np.array(array[self.row_indices[k]])
                    for k in self.group_labels]
        else:
            return [np.array(array[self.row_indices[k], :])
                    for k in self.group_labels]


    def _unpack(self, params):
        """
        Takes as input the packed parameter vector and returns three values:

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


    def like(self, params):

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

        print likeval
        return likeval


    def score(self, params):

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

        if self.like(params) > 1e5: 1/0

        return np.concatenate((score_fe, 0*score_re, np.r_[score_rv,]))


    def fit(self):

        like = lambda x: -self.like(x)
        score = lambda x: -self.score(x)

        x0 = np.zeros(self.exog.shape[1])
        pr = self.exog_re.shape[1]
        x1 = np.eye(pr)
        ix = np.tril_indices(pr)
        x1 = x1[ix]
        x0 = np.concatenate((x0, x1, np.r_[1.,]))

        rslt = fmin_cg(like, x0, score)
        1/0


def test():

    XFE = []
    XRE = []
    Y = []
    G = []
    for k in range(300):
        xfe = np.random.normal(size=(5,3))
        xre = np.random.normal(size=(5,3))
        fe = xfe.sum(1)
        re = np.dot(xre, np.random.normal(size=3))
        y = fe + re + np.random.normal(size=5)
        XRE.append(xre)
        XFE.append(xfe)
        Y.append(y)
        G.append(k*np.ones(5))

    XFE = np.concatenate(XFE)
    XRE = np.concatenate(XRE)
    Y = np.concatenate(Y)
    G = np.concatenate(G)

    lme = LME(Y, XFE, XRE, G)
    lme.fit()
    1/0

test()
