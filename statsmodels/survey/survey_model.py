"""
Methods for creating models for survey data.

The main classes are:

  * SurveyModel : Creates and fits the specified model. Returns
  estimates and stderr

Note: This makes use of survey weights, which is supplied to
the 'freq_weights' parameter in StatsModel's GLM. SurveyModel can
only be utilized by models w/ a freq_weight or weight parameter
"""

from __future__ import division
import numpy as np
import statsmodels


class SurveyModel(object):
    """

    Parameters
    -------
    design : Instance of class SurveyDesign
    model_class : Instance of class GLM
    init_args : Dictionary of arguments
        when initializing the model
    fit_args : Dictionary of arguments
        when fitting the model

    Attributes
    ----------
    design : Instance of class SurveyDesign
    model : Instance of class GLM
    init_args : Dictionary of arguments
        when initializing the model
    fit_args : Dictionary of arguments
        when fitting the model
    params : (p, ) array
        Array of coefficients of model
    vcov : (p, p) array
        Covariance matrix
    stderr : (p, ) array
        Standard error of cofficients
    """
    def __init__(self, design, model_class, init_args={}, fit_args={}):
        self.design = design
        self.model = model_class
        self.init_args = dict(init_args)
        self.fit_args = dict(fit_args)
        if self.model == statsmodels.genmod.generalized_linear_model.GLM:
            self.glm_flag = True
        else:
            self.glm_flag = False

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
            raise ValueError("Centering option %s not implemented" % center_by)
        return array

    def _stata_linearization_vcov(self, X, y):
        """
        Get the linearized covariance matrix using STATA's methodology

        Parameters
        ----------
        X : array-like
            A n x p array where 'n' is the number of observations and 'p'
            is the number of regressors. An intercept is not included by
            deault and should be added by the user
        y : array-like
            1d array of the response variable

        Returns
        -------
        vcov : (p,p) array
            The covariance matrix

        Notes
        -----
        This uses a 'sandwich' method, where three matrices are multiplied
        together. The outer parts of the 'sandwich' is the inverse of the
        negative hessian. The inside is the design-based variance of the
        estimation of a total, ie np.dot(weights, observations) where
        'observations' is the derivative of the log pseudolikelihood w.r.t
        theta (X*Beta) multiplied by the data

        Reference
        ---------
        http://www.stata.com/manuals13/svyvarianceestimation.pdf
        """

        # model = self.model(y, X, **self.init_args)

        # doing model.fit() to get the hessian
        # self.initialized_model.fit(**self.fit_args)
        hessian = self.initialized_model.hessian(self.params, observed=True)
        hess_inv = np.linalg.inv(hessian)

        d_hat = self.initialized_model.score_obs(self.params)
        # d_hat /= self.design.weights[:, None]
        jdata = []
        # design-based variance estimator for a total
        # This is the same as getting linearized variance
        # of SurveyTotal
        for c in range(self.design.n_clust):
            # d_hat already incorporates weights
            w = np.ones(len(self.design.weights))
            # If you're not in that cluster, set to 0
            w[self.design.clust != c] = 0
            # calculate the total
            jdata.append(np.dot(w, d_hat))
        jdata = np.asarray(jdata)

        if jdata.ndim == 1:
            jdata = jdata[:, None]
        jdata = self._centering(jdata, 'stratum')
        nh = self.design.clust_per_strat[self.design.strat_for_clust].astype(np.float64)
        mh = np.sqrt(nh / (nh-1))
        fh = np.sqrt(1 - self.design.fpc)
        jdata = fh[:, None] * mh[:, None] * jdata
        vcov = np.dot(jdata.T, jdata)
        vcov = np.dot(hess_inv, vcov).dot(hess_inv.T)
        return vcov

    def _sas_linearization_vcov(self, X, y, lin_method):
        """
        Get the linearized covariance matrix using STATA's methodology

        Parameters
        ----------
        X : array-like
            A n x p array where 'n' is the number of observations and 'p'
            is the number of regressors. An intercept is not included by
            deault and should be added by the user
        y : array-like
            1d array of the response variable
        lin_method : str
            Requests the Newton-Raphson algorithm if "Newton"
            is specified


        Returns
        -------
        vcov : (p,p) array
            The covariance matrix

        Notes
        -----
        This uses a 'sandwich' method, where three matrices are multiplied
        together. The outer parts of the 'sandwich' is either the inverse of
        the negative expected hessian or __(not sure how to describe Q_hat)__.
        The inside is the design-based variance of the estimation of a total,
        ie np.dot(weights, observations) where 'observations' is a sort of
        residual. It is multiplied by an additional factor of (n-1)/(p-1) that
        is not in STATA's methodology to reduce the small sample bias
        associated with using the estimated function to calculate deviations

        Reference
        ---------
        https://support.sas.com/documentation/cdl/en/statug/63033/HTML/default/viewer.htm#statug_surveylogistic_a0000000364.htm
        """

        lin_pred = np.dot(X, self.params)
        idl = self.initialized_model.family.link.inverse_deriv(lin_pred)
        # d_hat is the matrix of partial derivatives of the link function
        # w.r.t self.params.
        d_hat = (X * idl[:, None]).T

        cond_mean = self.result.mu
        w = self.design.weights.copy()
        e = []
        # calculate e for each stratum
        for c in range(self.design.n_clust):
            ind = (self.design.clust == c)
            cond_inv = np.linalg.inv(np.diag(cond_mean[ind]) -
                                     np.dot(cond_mean[ind], cond_mean.T[ind]))
            e.append(np.dot(w[ind] * d_hat[:, ind],
                            np.dot((y - cond_mean)[ind], cond_inv)))
        e = np.asarray(e)
        e = self._centering(e, 'stratum')
        nh = self.design.clust_per_strat[self.design.strat_for_clust].astype(np.float64)

        mh = np.sqrt(nh / (nh - 1))
        fh = np.sqrt(1 - self.design.fpc)
        e = fh[:, None] * mh[:, None] * e
        g_hat = (len(y) - 1) / (len(y) - X.shape[1]) * np.dot(e.T, e)

        if lin_method == "newton":
            # TODO: Figure out if hessian should be used when
            # 'jack-sandwich' or 'boot-sandwich' is requested
            # self.initialized_model.fit(**self.fit_args)
            q_hat = self.initialized_model.hessian(self.params, observed=False)
        else:
            cond_inv = np.linalg.inv(np.diag(cond_mean) -
                                     np.dot(cond_mean, cond_mean.T))
            q_hat = np.dot(w * d_hat, np.dot(cond_inv, d_hat.T))

        q_hat_inv = np.linalg.inv(q_hat)
        vcov = np.dot(q_hat_inv, np.dot(g_hat, q_hat_inv))
        return vcov

    def _jackknife_vcov(self, X, y):
        replicate_params = []
        for c in range(self.design.n_clust):
            w = self.design.get_rep_weights(cov_method='jack', c=c)
            if self.glm_flag:
                self.init_args["freq_weights"] = w
            else:
                self.init_args["weights"] = w
            replicate_params.append(self._get_params(y, X))
        replicate_params = np.asarray(replicate_params)
        self.replicate_params = self._centering(replicate_params)

        if hasattr(self.design, 'clust_per_strat'):
            nh = self.design.clust_per_strat[self.design.strat_for_clust].astype(np.float64)
            mh = np.sqrt((nh - 1) / nh)
            self.replicate_params = mh[:, None] * self.replicate_params
        else:
            nh = self.design.rep_weights.shape[1]
            mh = np.sqrt((nh - 1) / nh)
            self.replicate_params *= mh

        vcov = np.dot(self.replicate_params.T, self.replicate_params)
        return vcov

    def fit(self, y, X, cov_method='jack', center_by='est', lin_method=None):
        self.center_by = center_by
        if self.glm_flag:
            self.init_args["freq_weights"] = self.design.weights
        else:
            self.init_args["weights"] = self.design.weights
        self.params = self._get_params(y, X)

        if cov_method == 'jack':
            self.vcov = self._jackknife_vcov(X, y)
        elif cov_method == 'linearized_sas':
            self.vcov = self._sas_linearization_vcov(X, y, lin_method)
        elif cov_method == 'linearized_stata':
            self.vcov = self._stata_linearization_vcov(X, y)
        else:
            return ValueError('cov_method %s not supported' % cov_method)
        if self.vcov.ndim == 2:
            self.stderr = np.sqrt(np.diag(self.vcov))
        else:
            self.stderr = np.sqrt(self.vcov)

    def _get_params(self, y, X):
        # note, can make 'model' into self.initialized_model
        # so that i dont have to worry about it getting called multiple times
        # when doing linearized variance
        # when doing jackknife, it'll get called mult times anyway
        self.initialized_model = self.model(y, X, **self.init_args)
        self.result = self.initialized_model.fit(**self.fit_args)
        return self.result.params
