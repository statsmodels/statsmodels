"""
Methods for creating models for survey data.

The main classes are:

  * SurveyModel : Creates and fits the specified model. Returns
  estimates and stderr

Note: This makes use of survey weights, which is supplied to
the 'freq_weights' parameter in StatsModel's GLM or 'weights' in
StatsModels WLS.
"""

from __future__ import division
import numpy as np
import statsmodels


class SurveyModel(object):
    """
    Generalized Linear Models class for survey data

    Parameters
    -------
    design : Object
        Instance of class SurveyDesign
    model_class : Class
        class GLM or WLS
    init_args : dict-like
        Keyword arguments passed on model initialization
    fit_args : dict-like
        Keyword arguments passed to model fit method.

    Attributes
    ----------
    design : Object
        Instance of class SurveyDesign
    model : Class
        Class GLM or WLS
    init_args : dict-like
        Keyword arguments passed on model initialization
    fit_args : dict-like
        Keyword arguments passed to model fit method.
    params : (k_params, ) array
        Array of coefficients of model
    vcov : (k_params, k_params) array
        Covariance matrix
    stderr : (k_params, ) array
        Standard error of cofficients
    """
    def __init__(self, design, model_class, init_args={}, fit_args={}):
        self.design = design
        self.model = model_class
        self.init_args = dict(init_args)
        self.fit_args = dict(fit_args)

        if self.model is statsmodels.genmod.generalized_linear_model.GLM:
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
                raise ValueError("Can't center by stratum with rep_weights")
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
        vcov : (k_params, k_params) array
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
        jdata *= (fh[:, None] * mh[:, None])
        v_hat = np.dot(jdata.T, jdata)
        vcov = np.dot(hess_inv, v_hat).dot(hess_inv.T)
        return vcov

# doesnt work whether lin_method is specified or not
# which means it must be the calculation of g_hat
    def _sas_linearization_vcov(self, X, y, lin_method):
        """
        Get the linearized covariance matrix using STATA's methodology

        Parameters
        ----------
        X : array-like
            A n x k_params array where 'n' is the number of observations and
            k_params is the number of regressors. An intercept is not included
            by default and should be added by the user
        y : array-like
            1d array of the response variable
        lin_method : str
            Requests the Newton-Raphson algorithm if "Newton"
            is specified


        Returns
        -------
        vcov : (k_params, k_params) array
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

        d_hat = self.initialized_model.score_obs(self.params)

        cond_mean = self.result.mu
        w = self.design.weights.copy()
        e = []

        # calculate e for each stratum
        for c in range(self.design.n_clust):
            ind = (self.design.clust == c)
            cond_inv = np.linalg.inv(np.diag(cond_mean[ind]) -
                                     np.outer(cond_mean[ind],
                                              cond_mean.T[ind]))
            e.append(np.dot((y - cond_mean)[ind], cond_inv).dot(w[ind, None] *
                                                                d_hat[ind, :]))
        e = np.asarray(e)
        e = self._centering(e, 'stratum')
        nh = self.design.clust_per_strat[self.design.strat_for_clust].astype(np.float64)

        mh = np.sqrt(nh / (nh - 1))
        fh = np.sqrt(1 - self.design.fpc)
        e = fh[:, None] * mh[:, None] * e
        n = len(y)
        p = X.shape[1]
        g_hat = (n - 1) / (n - p) * np.dot(e.T, e)

        if lin_method.lower() in ("newton", 'nt', 'newt'):
            print('using hessian')
            # TODO: Figure out if hessian should be used when
            # 'jack-sandwich' or 'boot-sandwich' is requested
            self.q_hat = self.initialized_model.hessian(self.params,
                                                        observed=False)
        else:
            cond_inv = np.linalg.inv(np.diag(cond_mean) -
                                     np.outer(cond_mean, cond_mean.T))
            self.q_hat = np.dot((w[:, None] * d_hat).T,
                                np.dot(cond_inv, d_hat))

        q_hat_inv = np.linalg.inv(self.q_hat)
        vcov = np.dot(q_hat_inv, g_hat).dot(q_hat_inv.T)
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
            fh = np.sqrt(1 - self.design.fpc)
            self.replicate_params *= (mh[:, None] * fh[:, None])
        else:
            nh = self.design.rep_weights.shape[1]
            mh = np.sqrt((nh - 1) / nh)
            self.replicate_params *= mh

        vcov = np.dot(self.replicate_params.T, self.replicate_params)
        return vcov

    def fit(self, y, X, cov_method='linearized', center_by='est',
            lin_method=''):
        y = np.asarray(y)
        X = np.asarray(X)

        self.center_by = center_by
        if self.glm_flag:
            self.init_args["freq_weights"] = self.design.weights
        else:
            self.init_args["weights"] = self.design.weights
        self.params = self._get_params(y, X)

        if cov_method.lower() in ('jack', 'jackknife', 'jk'):
            self.vcov = self._jackknife_vcov(X, y)
        elif cov_method.lower() in ('linearized_sas', 'sas'):
            self.vcov = self._sas_linearization_vcov(X, y, lin_method)
        elif cov_method.lower() in ('linearized_stata', 'stata'):
            self.vcov = self._stata_linearization_vcov(X, y)
        else:
            return ValueError('cov_method %s not supported' % cov_method)
        if self.vcov.ndim == 2:
            self.stderr = np.sqrt(np.diag(self.vcov))
        else:
            self.stderr = np.sqrt(self.vcov)

    def _get_params(self, y, X):
        self.initialized_model = self.model(y, X, **self.init_args)
        self.result = self.initialized_model.fit(**self.fit_args)
        return self.result.params