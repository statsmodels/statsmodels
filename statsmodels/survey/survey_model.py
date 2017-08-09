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

    def _stata_linearization_vcov(self, X, y):
        model = self.model(y, X, **self.init_args)
        # doing model.fit() to get the hessian
        model.fit(**self.fit_args)

        hessian = model.hessian(self.params)
        hess_inv = np.linalg.inv(hessian)

        lin_pred = np.dot(X, self.params)
        idl = model.family.link.inverse_deriv(lin_pred)
        # d_hat is the matrix of partial derivatives of the link function
        # w.r.t self.params.
        d_hat = (X * idl[:, None])

        jdata = []
        for c in range(self.design.n_clust):
            w = self.design.weights.copy()
            # but if you're not in that cluster, set as 0
            w[self.design.clust != c] = 0
            jdata.append(np.dot(w, d_hat))
        jdata = np.asarray(jdata)
        # we usually deal w/ jdata as nxp
        # unless w/ ratio, in which 2 columns
        if jdata.ndim == 1:
            jdata = jdata[:, None]
        jdata = self._centering(jdata, 'stratum')
        nh = self.design.clust_per_strat[self.design.strat_for_clust].astype(np.float64)
        mh = np.sqrt(nh / (nh-1))
        fh = np.sqrt(1 - self.design.fpc)
        jdata = fh[:, None] * mh[:, None] * jdata
        vcov = np.dot(jdata.T, jdata)

        return vcov

    def _sas_linearization_vcov(self, X, y, technique):
        model = self.model(y, X, **self.init_args)
        lin_pred = np.dot(X, self.params)
        idl = model.family.link.inverse_deriv(lin_pred)
        # d_hat is the matrix of partial derivatives of the link function
        # w.r.t self.params.
        d_hat = (X * idl[:, None]).T

        cond_mean = self.result.mu
        # TODO: let w = self.design.weights.copy() or self.design.rep_weights[:,c] if
        # user wants to specify jack-sandwich or something
        w = self.design.weights.copy()
        e = []
        # calculate e for each stratum
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
            # TODO: Figure out if hessian should be used when
            # 'jack-sandwich' or 'boot-sandwich' is requested
            model.fit(**self.fit_args)
            factor = model.hessian_factor(self.params)
            q_hat = -np.dot(X.T * factor, X)
        else:
            cond_inv = np.linalg.inv(np.diag(cond_mean) - np.dot(cond_mean, cond_mean.T))
            q_hat = np.dot(w * d_hat, np.dot(cond_inv, d_hat.T))
        q_hat_inv = np.linalg.inv(q_hat)
        vcov = np.dot(q_hat_inv, np.dot(g_hat, q_hat_inv))
        return vcov

    def _get_jackknife_vcov(self, X, y):
        replicate_params = []
        for c in range(self.design.n_clust):
            w = self.design.get_rep_weights(cov_method='jack', c=c)
            self.init_args["weights"] = w
            replicate_params.append(self._get_params(y, X))
        replicate_params = np.asarray(replicate_params)
        self.replicate_params = self._centering(replicate_params)
        try:
            nh = self.design.clust_per_strat[self.design.strat_for_clust].astype(np.float64)
            mh = np.sqrt((nh - 1) / nh)
            self.replicate_params = mh[:, None] * self.replicate_params
        except AttributeError:
            nh = self.design.rep_weights.shape[1]
            mh = np.sqrt((nh - 1) / nh)
            self.replicate_params *= mh
        vcov = np.dot(self.replicate_params.T, self.replicate_params)
        return vcov

    def fit(self, y, X, cov_method='jack', center_by='est', technique=None):
        self.center_by = center_by
        self.init_args["weights"] = self.design.weights
        self.params = self._get_params(y, X)


        # for now, just working with jackknife to see if it works
        if cov_method == 'jack':
            self.vcov = self._get_jackknife_vcov(X, y)
        elif cov_method == 'linearized_sas':
            self.vcov = self._sas_linearization_vcov(X, y, technique)
        elif cov_method == 'linearized_stata':
            self.vcov = self._stata_linearization_vcov(X, y)
        else:
            return ValueError('cov_method %s not supported' %cov_method)
        if self.vcov.ndim == 2:
            self.stderr = np.sqrt(np.diag(self.vcov))
        else:
            self.stderr = np.sqrt(self.vcov)

    def _get_params(self, y, X):
        model = self.model(y, X, **self.init_args)
        self.result = model.fit(**self.fit_args)
        return self.result.params
