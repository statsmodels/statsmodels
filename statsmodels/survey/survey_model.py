from __future__ import division
import numpy as np
class SurveyModel(object):

    def __init__(self, design, model_class, init_args={}, fit_args={}):
        self.design = design
        self.model = model_class
        self.init_args = dict(init_args)
        self.fit_args = dict(fit_args)

    def _centering(self, array=None, center_by=None):
        # can be used to overwrite center_by if necessasry
        if center_by is None:
            center_by = self.center_by
        if center_by == 'est':
            array -= self.params
        elif center_by == 'global':
            array -= array.mean(0)
        elif center_by == 'stratum':
            if self.design.rep_weights is None:
                for s in range(self.design.n_strat):
                    # center the 'delete 1' statistic
                    array[self.design.ii[s], :] -= array[self.design.ii[s],
                                                         :].mean(0)
        else:
            raise ValueError("Centering option not implemented")
        return array

    def _get_linearization_vcov(self, X, y, technique):
        model = self.model(y, X, **self.init_args)
        params = self.params.copy()
        d_hat = model.score_obs(params=params).T
        cond_mean = self.result.mu
        # let w = self.design.weights.copy() or self.design.rep_weights[:,c] if
        # user wants to specify jack-sandwhich or something
        w = self.design.weights.copy()
        e = []
        for c in range(self.design.n_clust):
            ind = (self.design.clust == c)
            cond_inv = np.linalg.inv(np.diag(cond_mean[ind]) - np.dot(cond_mean[ind], cond_mean.T[ind]))
            e.append(np.dot(w[ind] * d_hat[:,ind], np.dot((y - cond_mean)[ind], cond_inv)))
        e = np.asarray(e)
        e = self._centering(e, 'stratum')
        nh = self.design.clust_per_strat[self.design.strat_for_clust].astype(np.float64)

        mh = np.sqrt(nh / (nh - 1))
        fh = np.sqrt(1 - self.design.fpc)
        e = fh[:, None] * mh[:, None] * e
        g_hat = (len(y) - 1) / (len(y) - X.shape[1]) * np.dot(e.T, e)

        if technique=="newton":
            q_hat = model.hessian(self.params)
        else:
            cond_inv = np.linalg.inv(np.diag(cond_mean) - np.dot(cond_mean, cond_mean.T))
            q_hat = np.dot(w * d_hat, np.dot(cond_inv, d_hat.T))
        vcov = np.dot(np.linalg.inv(q_hat), np.dot(g_hat, np.linalg.inv(q_hat)))
        return vcov

    def fit(self, y, X, cov_method='jack', center_by='est', replicates=None, technique=None):
        self.center_by = center_by
        self.init_args["weights"] = self.design.weights
        self.params = self._get_params(y, X)


        # for now, just working with jackknife to see if it works
        if cov_method == 'jack':
            if replicates is None:
                k = self.design.n_clust
            else:
                k = replicates
            replicate_params = []
            for c in range(k):
                w = self.design.get_rep_weights(cov_method=cov_method, c=c)
                self.init_args["weights"] = w
                print('weights', self.init_args['weights'])
                replicate_params.append(self._get_params(y, X))

            replicate_params = np.asarray(replicate_params)
            print('new params', replicate_params)
            self.replicate_params = self._centering(replicate_params)

            try:
                nh = self.design.clust_per_strat[self.design.strat_for_clust].astype(np.float64)
                mh = np.sqrt((nh - 1) / nh)
                self.replicate_params = mh[:, None] * self.replicate_params
            except AttributeError:
                nh = self.design.rep_weights.shape[1]
                mh = np.sqrt((nh - 1) / nh)
                self.replicate_params *= mh
            self.vcov = np.dot(self.replicate_params.T, self.replicate_params)
        elif cov_method == 'linearized':
            self.vcov = self._get_linearization_vcov(X, y, technique)

        if self.vcov.ndim == 2:
            self.stderr = np.sqrt(np.diag(self.vcov))
        else:
            self.stderr = np.sqrt(self.vcov)

    def _get_params(self, y, X):
        model = self.model(y, X, **self.init_args)
        self.result = model.fit(**self.fit_args)
        return self.result.params



# questions
"""
# use glm for prototyping
weights is called freq_weights for now until later.

1. It seems like 'weights' is not a uniform keyword for the models?
Bc I dont see any mention of it anywhere except OLS and WLS

2. The linearized variance is just (see STATA equation and confirm)
But how do I extract those? I remember Josef mentioned "score factor/obs"
but can't find out exactly where

3. Who is your contact for the jackknife STATA stuff? I can't find out why
my results are different
pi_hij are just conditional means
"""