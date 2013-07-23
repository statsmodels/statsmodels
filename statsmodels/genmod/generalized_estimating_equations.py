"""
Procedures for fitting marginal regression models to dependent
data using Generalized Estimating Equations.

References
----------
KY Liang and S Zeger. "Longitudinal data analysis using
generalized linear models". Biometrika (1986) 73 (1): 13-22.

S Zeger and KY Liang. "Longitudinal Data Analysis for Discrete and
Continuous Outcomes". Biometrics Vol. 42, No. 1 (Mar., 1986),
pp. 121-130

Xu Guo and Wei Pan (2002). "Small sample performance of the score
test in GEE".
http://www.sph.umn.edu/faculty1/wp-content/uploads/2012/11/rr2002-013.pdf
"""


import numpy as np
from scipy import stats
from statsmodels.tools.decorators import cache_readonly, \
    resettable_cache
import statsmodels.base.model as base
from statsmodels.genmod import families
from statsmodels.genmod import dependence_structures
from statsmodels.genmod.dependence_structures import VarStruct
from scipy.stats import norm
import pandas



class ParameterConstraint(object):
    """
    A class for managing linear equality constraints for a parameter
    vector.
    """

    def __init__(self, lhs, rhs, exog):
        """
        Parameters:
        ----------
        lhs : ndarray
           A q x p matrix which is the left hand side of the
           constraint lhs * param = rhs.  The number of constraints is
           q >= 1 and p is the dimension of the parameter vector.
        rhs : ndarray
          A q-dimensional vector which is the right hand side of the
          constraint equation.
        exog : ndarray
          The n x p exognenous data for the full model.
        """

        if type(lhs) != np.ndarray:
            raise ValueError("The left hand side constraint matrix L "
                             "must be a NumPy array.")
        if len(rhs) != lhs.shape[0]:
            raise ValueError("The number of rows of the left hand "
                             "side constraint matrix L must equal "
                             "the length of the right hand side "
                             "constraint vector R.")

        self.lhs = lhs
        self.rhs = rhs

        # The columns of lhs0 are an orthogonal basis for the
        # orthogonal complement to row(lhs), the columns of lhs1 are
        # an orthogonal basis for row(lhs).  The columns of lhsf =
        # [lhs0,lhs1] are mutually orthogonal.
        lhs_u, lhs_s, lhs_vt = np.linalg.svd(lhs.T, full_matrices=1)
        self.lhs0 = lhs_u[:, len(lhs_s):]
        self.lhs1 = lhs_u[:, 0:len(lhs_s)]
        self.lhsf = np.hstack((self.lhs0, self.lhs1))

        # param0 is one solution to the underdetermined system
        # L * param = R.
        self.param0 = np.dot(self.lhs1, np.dot(lhs_vt, self.rhs) / lhs_s)

        self._offset_increment = np.dot(exog, self.param0)

        self.orig_exog = exog
        self.exog_fulltrans = np.dot(exog, self.lhsf)


    def offset_increment(self):
        """
        Returns a vector that should be added to the offset vector to
        accommodate the constraint.

        Parameters:
        -----------
        exog : array-like
           The exogeneous data for the model.
        """

        return self._offset_increment


    def reduced_exog(self):
        """
        Returns a linearly transformed exog matrix whose columns span
        the constrained model space.

        Parameters:
        -----------
        exog : array-like
           The exogeneous data for the model.
        """
        return self.exog_fulltrans[:, 0:self.lhs0.shape[1]]


    def restore_exog(self):
        """
        Returns the full exog matrix before it was reduced to
        satisfy the constraint.
        """
        return self.orig_exog


    def unpack_param(self, params):
        """
        Converts the parameter vector `params` from reduced to full
        coordinates.
        """

        return self.param0 + np.dot(self.lhs0, params)


    def unpack_cov(self, bcov):
        """
        Converts the covariance matrix `bcov` from reduced to full
        coordinates.
        """

        return np.dot(self.lhs0, np.dot(bcov, self.lhs0.T))





class GEE(base.Model):
    __doc__ = """
    Parameters
    ----------
    endog : array-like
        1d array of endogenous response values.
    exog : array-like
        A nobs x k array where `nobs` is the number of
        observations and `k` is the number of regressors. An
        intercept is not included by default and should be added
        by the user. See `statsmodels.tools.add_constant`.
    groups : array-like
        A 1d array of length `nobs` containing the cluster labels.
    time : array-like
        A 2d array of time (or other index) values, used by some
        dependence structures to define similarity relationships among
        observations within a cluster.
    family : family class instance
        The default is Gaussian.  To specify the binomial
        distribution family = sm.family.Binomial(). Each family can
        take a link instance as an argument.  See
        statsmodels.family.family for more information.
    varstruct : VarStruct class instance
        The default is Independence.  To specify an exchangeable
        structure varstruct = sm.varstruct.Exchangeable().  See
        statsmodels.varstruct.varstruct for more information.
    offset : array-like
        An offset to be included in the fit.  If provided, must be
        an array whose length is the number of rows in exog.
    constraint : (ndarray, ndarray)
       If provided, the constraint is a tuple (L, R) such that the
       model parameters are estimated under the constraint L *
       param = R, where L is a q x p matrix and R is a
       q-dimensional vector.  If constraint is provided, a score
       test is performed to compare the constrained model to the
       unconstrained model.
    %(extra_params)s

    See also
    --------
    statsmodels.families.*

    Notes
    -----
    Only the following combinations make sense for family and link ::

                   + ident log logit probit cloglog pow opow nbinom loglog logc
      Gaussian     |   x    x                        x
      inv Gaussian |   x    x                        x
      binomial     |   x    x    x     x       x     x    x           x      x
      Poission     |   x    x                        x
      neg binomial |   x    x                        x          x
      gamma        |   x    x                        x

    Not all of these link functions are currently available.

    Endog and exog are references so that if the data they refer
    to are already arrays and these arrays are changed, endog and
    exog will change.

    """ % {'extra_params' : base._missing_param_doc}

    fit_history = None
    cached_means = None


    def __init__(self, endog, exog, groups, time=None, family=None,
                       varstruct=None, missing='none', offset=None,
                       constraint=None):

        # Pass groups, time, and offset so they are processed for
        # missing data along with endog and exog.  Calling super
        # creates self.exog, self.endog, etc. as ndarrays and the
        # original exog, endog, etc. are self.data.endog, etc.
        super(GEE, self).__init__(endog, exog, groups=groups,
                                  time=time, offset=offset,
                                  missing=missing)

        # Handle the family argument
        if family is None:
            family = families.Gaussian()
        else:
            if not issubclass(family.__class__, families.Family):
                raise ValueError("GEE: `family` must be a genmod "
                                 "family instance")
        self.family = family

        # Handle the varstruct argument
        if varstruct is None:
            varstruct = dependence_structures.Independence()
        else:
            if not issubclass(varstruct.__class__, VarStruct):
                raise ValueError("GEE: `varstruct` must be a genmod "
                                 "varstruct instance")
        self.varstruct = varstruct

        if offset is None:
            self.offset = np.zeros(self.exog.shape[0],
                                   dtype=np.float64)
        else:
            self.offset = offset

        if self.time is None:
            self.time = np.zeros((self.exog.shape[0],1),
                                 dtype=np.float64)
        else:
            self.time = time
        if len(self.time.shape) == 1:
            self.time = np.reshape(self.time, (len(self.time),1))

        # Handle the constraint
        self.constraint = None
        if constraint is not None:
            if len(constraint) != 2:
                raise ValueError("GEE: `constraint` must be a 2-tuple.")
            if constraint[0].shape[1] != self.exog.shape[1]:
                raise ValueError("GEE: the left hand side of the "
                   "constraint must have the same number of columns "
                   "as the exog matrix.")
            self.constraint = ParameterConstraint(constraint[0],
                                                  constraint[1],
                                                  self.exog)

            self.offset += self.constraint.offset_increment()
            self.exog = self.constraint.reduced_exog()

        # Convert the data to the internal representation, which is a
        # list of arrays, corresponding to the clusters.
        group_labels = list(set(groups))
        group_labels.sort()
        row_indices = dict((s, []) for s in group_labels)
        for i in range(len(self.endog)):
            row_indices[groups[i]].append(i)
        self.row_indices = row_indices
        self.group_labels = group_labels

        self.endog_li = self._cluster_list(self.endog)
        self.exog_li = self._cluster_list(self.exog)
        self.time_li = self._cluster_list(self.time)
        self.offset_li = self._cluster_list(self.offset)
        if constraint is not None:
            self.constraint.exog_fulltrans_li = \
                self._cluster_list(self.constraint.exog_fulltrans)

        self.family = family

        self.varstruct.initialize(self)

        # Total sample size
        group_ns = [len(y) for y in self.endog_li]
        self.nobs = sum(group_ns)

        # mean_deriv is the derivative of E[endog|exog] with respect
        # to params
        try:
            # This custom mean_deriv is currently only used for the
            # multinomial logit model
            self.mean_deriv = self.family.link.mean_deriv
        except AttributeError:
            # Otherwise it can be obtained easily from inverse_deriv
            mean_deriv_lpr = self.family.link.inverse_deriv
            def mean_deriv(exog, lpr):
                dmat = exog * mean_deriv_lpr(lpr)[:, None]
                return dmat
            self.mean_deriv = mean_deriv

        # mean_deriv_exog is the derivative of E[endog|exog] with
        # respect to exog
        try:
            # This custom mean_deriv_exog is currently only used for
            # the multinomial logit model
            self.mean_deriv_exog = self.family.link.mean_deriv_exog
        except AttributeError:
            # Otherwise it can be obtained easily from inverse_deriv
            mean_deriv_lpr = self.family.link.inverse_deriv
            def mean_deriv_exog(exog, params):
                lpr = np.dot(exog, params)
                dmat = np.outer(mean_deriv_lpr(lpr), params)
                return dmat
            self.mean_deriv_exog = mean_deriv_exog


    def _cluster_list(self, array):
        """
        Returns `array` split into subarrays corresponding to the
        cluster structure.
        """

        if len(array.shape) == 0:
            return [np.array(array[self.row_indices[k]])
                    for k in self.group_labels]
        else:
            return [np.array(array[self.row_indices[k], :])
                    for k in self.group_labels]



    def estimate_scale(self):
        """
        Returns an estimate of the scale parameter `phi` at the
        current parameter value.
        """

        endog = self.endog_li
        exog = self.exog_li
        offset = self.offset_li

        cached_means = self.cached_means

        num_clust = len(endog)
        nobs = self.nobs
        exog_dim = exog[0].shape[1]

        varfunc = self.family.variance

        scale_inv = 0.
        for i in range(num_clust):

            if len(endog[i]) == 0:
                continue

            expval, _ = cached_means[i]

            sdev = np.sqrt(varfunc(expval))
            resid = (endog[i] - offset[i] - expval) / sdev

            scale_inv += np.sum(resid**2)

        scale_inv /= (nobs - exog_dim)
        scale = 1 / scale_inv
        return scale



    def _beta_update(self):
        """
        Returns two values (update, score).  The vector `update` is
        the update vector such that params + update is the next
        iterate when solving the score equations.  The vector `score`
        is the current state of the score equations.
        """

        endog = self.endog_li
        exog = self.exog_li

        # Number of clusters
        num_clust = len(endog)

        cached_means = self.cached_means

        varfunc = self.family.variance

        bmat, score = 0, 0
        for i in range(num_clust):

            if len(endog[i]) == 0:
                continue

            expval, lpr = cached_means[i]

            dmat = self.mean_deriv(exog[i], lpr)

            sdev = np.sqrt(varfunc(expval))
            vmat, is_cor = self.varstruct.variance_matrix(expval, i)
            if is_cor:
                vmat *= np.outer(sdev, sdev)

            vinv_d = np.linalg.solve(vmat, dmat)
            bmat += np.dot(dmat.T, vinv_d)

            resid = endog[i] - expval
            vinv_resid = np.linalg.solve(vmat, resid)
            score += np.dot(dmat.T, vinv_resid)

        update = np.linalg.solve(bmat, score)

        return update, score


    def update_cached_means(self, beta):
        """
        cached_means should always contain the most recent
        calculation of the cluster-wise mean vectors.  This function
        should be called every time the value of beta is changed, to
        keep the cached means up to date.
        """

        endog = self.endog_li
        exog = self.exog_li
        offset = self.offset_li
        num_clust = len(endog)

        mean = self.family.link.inverse

        self.cached_means = []

        for i in range(num_clust):

            if len(endog[i]) == 0:
                continue

            lpr = offset[i] + np.dot(exog[i], beta)
            expval = mean(lpr)

            self.cached_means.append((expval, lpr))



    def _covmat(self):
        """
        Returns the sampling covariance matrix of the regression
        parameters and related quantities.

        Returns
        -------
        robust_covariance : array-like
           The robust, or sandwich estimate of the covariance, which
           is meaningful even if the working covariance structure is
           incorrectly specified.
        naive_covariance : array-like
           The model-based estimate of the covariance, which is
           meaningful if the covariance structure is correctly
           specified.
        robust_covariance_bc : array-like
           The "bias corrected" robust covariance of Mancl and DeRouen.
        cmat : array-like
           The center matrix of the sandwich expression, used in
           obtaining score test results.
        """

        endog = self.endog_li
        exog = self.exog_li
        num_clust = len(endog)

        varfunc = self.family.variance
        cached_means = self.cached_means

        # Calculate the naive (model-based) and robust (sandwich)
        # covariances.
        bmat, cmat = 0, 0
        for i in range(num_clust):

            if len(endog[i]) == 0:
                continue

            expval, lpr = cached_means[i]

            dmat = self.mean_deriv(exog[i], lpr)

            sdev = np.sqrt(varfunc(expval))
            vmat, is_cor = self.varstruct.variance_matrix(expval, i)
            if is_cor:
                vmat *= np.outer(sdev, sdev)

            vinv_d = np.linalg.solve(vmat, dmat)
            bmat += np.dot(dmat.T, vinv_d)

            resid = endog[i] - expval
            vinv_resid = np.linalg.solve(vmat, resid)
            dvinv_resid = np.dot(dmat.T, vinv_resid)
            cmat += np.outer(dvinv_resid, dvinv_resid)

        scale = self.estimate_scale()

        naive_covariance = scale*np.linalg.inv(bmat)
        cmat /= scale**2
        robust_covariance = np.dot(naive_covariance,
                                   np.dot(cmat, naive_covariance))

        # Calculate the bias-corrected sandwich estimate of Mancl and
        # DeRouen (requires naive_covariance so cannot be calculated
        # in the previous loop).
        bcm = 0
        for i in range(num_clust):

            if len(endog[i]) == 0:
                continue

            expval, lpr = cached_means[i]

            dmat = self.mean_deriv(exog[i], lpr)

            sdev = np.sqrt(varfunc(expval))
            vmat, is_cor = self.varstruct.variance_matrix(expval, i)
            if is_cor:
                vmat *= np.outer(sdev, sdev)

            vinv_d = np.linalg.solve(vmat, dmat)
            hmat = np.dot(vinv_d, naive_covariance)
            hmat = np.dot(hmat, dmat.T).T

            resid = endog[i] - expval
            aresid = np.linalg.solve(np.eye(len(resid)) - hmat, resid)
            srt = np.dot(dmat.T, np.linalg.solve(vmat, aresid))
            bcm += np.outer(srt, srt)

        robust_covariance_bc = np.dot(naive_covariance,
                                      np.dot(bcm, naive_covariance))

        return robust_covariance, naive_covariance, \
            robust_covariance_bc, cmat


    def predict(self, params, exog=None, offset=None, linear=False):
        """
        Return predicted values for a design matrix

        Parameters
        ----------
        params : array-like
            Parameters / coefficients of a GLM.
        exog : array-like, optional
            Design / exogenous data. If exog is None, model exog is
            used.
        offset : array-like, optional
            Offset for exog if provided.  If offset is None, model
            offset is used.
        linear : bool
            If True, returns the linear predicted values.  If False,
            returns the value of the inverse of the model's link
            function at the linear predicted values.

        Returns
        -------
        An array of fitted values
        """

        if exog is None:
            fitted = self.offset + np.dot(self.exog, params)
        else:
            fitted = offset + np.dot(exog, params)

        if not linear:
            fitted = self.family.link(fitted)

        return fitted


    def _starting_beta(self, starting_beta):
        """
        Returns a starting value for beta and a list of variable
        names.

        Parameters:
        -----------
        starting_beta : array-like
            Starting values if available, otherwise None

        Returns:
        --------
        beta : array-like
           Starting values for params

        """

        if starting_beta is None:
            beta_dm = self.exog_li[0].shape[1]
            beta = np.zeros(beta_dm, dtype=np.float64)

        else:
            beta = starting_beta.copy()

        return beta




    def fit(self, maxit=100, ctol=1e-6, starting_beta=None):
        """
        Fits a GEE model.

        Parameters
        ----------
        maxit : integer
            The maximum number of iterations
        ctol : float
            The convergence criterion for stopping the Gauss-Seidel
            iterations
        starting_beta : array-like
            A vector of starting values for the regression
            coefficients.  If None, a default is chosen.

        Returns
        -------
        An instance of the GEEResults class

        """

        self.fit_history = {'params' : [],
                            'score_change' : []}

        beta = self._starting_beta(starting_beta)

        self.update_cached_means(beta)

        for _ in xrange(maxit):
            update, _ = self._beta_update()
            beta += update
            self.update_cached_means(beta)
            stepsize = np.sqrt(np.sum(update**2))
            self.fit_history['params'].append(beta)
            if stepsize < ctol:
                break
            self._update_assoc(beta)

        bcov, ncov, bc_cov, _ = self._covmat()

        if self.constraint is not None:
            beta, bcov = self._handle_constraint(beta, bcov)

        scale = self.estimate_scale()

        results = GEEResults(self, beta, bcov/scale, scale)

        results.fit_history = self.fit_history
        results.naive_covariance = ncov
        results.robust_covariance_bc = bc_cov

        return results



    def _handle_constraint(self, beta, bcov):
        """
        Expand the parameter estimate `beta` and covariance matrix
        `bcov` to the coordinate system of the unconstrained model.

        Parameters:
        -----------
        beta : array-like
            A parameter vector estimate for the reduced model.
        bcov : array-like
            The covariance matrix of beta.

        Returns:
        --------
        beta : array-like
            The input parameter vector beta, expanded to the
            coordinate system of the full model
        bcov : array-like
            The input covariance matrix bcov, expanded to the
            coordinate system of the full model
        """

        # The number of variables in the full model
        red_p = len(beta)
        full_p = self.constraint.lhs.shape[1]
        beta0 = np.r_[beta, np.zeros(full_p - red_p)]

        # Get the score vector under the full model.
        save_exog_li = self.exog_li
        self.exog_li = self.constraint.exog_fulltrans_li
        import copy
        save_cached_means = copy.deepcopy(self.cached_means)
        self.update_cached_means(beta0)
        _, score = self._beta_update()
        _, ncov1, _, cmat = self._covmat()
        scale = self.estimate_scale()
        score2 = score[len(beta):] / scale

        amat = np.linalg.inv(ncov1)

        bmat_11 = cmat[0:red_p, 0:red_p]
        bmat_22 = cmat[red_p:, red_p:]
        bmat_12 = cmat[0:red_p, red_p:]
        amat_11 = amat[0:red_p, 0:red_p]
        amat_12 = amat[0:red_p, red_p:]

        score_cov = bmat_22 - \
            np.dot(amat_12.T, np.linalg.solve(amat_11, bmat_12))
        score_cov -= np.dot(bmat_12.T,
                        np.linalg.solve(amat_11, amat_12))
        score_cov += np.dot(amat_12.T,
                            np.dot(np.linalg.solve(amat_11, bmat_11),
                                   np.linalg.solve(amat_11, amat_12)))

        from scipy.stats.distributions import chi2
        score_statistic = np.dot(score2,
                                 np.linalg.solve(score_cov, score2))
        score_df = len(score2)
        score_pvalue = 1 - \
                 chi2.cdf(score_statistic, score_df)
        self.score_test_results = {"statistic": score_statistic,
                                   "df": score_df,
                                   "p-value": score_pvalue}

        beta = self.constraint.unpack_param(beta)
        bcov = self.constraint.unpack_cov(bcov)

        self.exog_li = save_exog_li
        self.cached_means = save_cached_means
        self.exog = self.constraint.restore_exog()

        return beta, bcov


    def _update_assoc(self, beta):
        """
        Update the association parameters
        """

        self.varstruct.update(beta, self)


    def _derivative_exog(self, params, exog=None, transform='dydx',
            dummy_idx=None, count_idx=None):
        """
        For computing marginal effects returns dF(XB) / dX where F(.) is
        the predicted probabilities

        transform can be 'dydx', 'dyex', 'eydx', or 'eyex'.

        Not all of these make sense in the presence of discrete regressors,
        but checks are done in the results in get_margeff.
        """
        #note, this form should be appropriate for
        ## group 1 probit, logit, logistic, cloglog, heckprob, xtprobit
        if exog == None:
            exog = self.exog
        margeff = self.mean_deriv_exog(exog, params)
#        lpr = np.dot(exog, params)
#        margeff = (self.mean_deriv(exog, lpr) / exog) * params
#        margeff = np.dot(self.pdf(np.dot(exog, params))[:,None],
#                                                          params[None,:])

        if 'ex' in transform:
            margeff *= exog
        if 'ey' in transform:
            margeff /= self.predict(params, exog)[:,None]
        if count_idx is not None:
            from statsmodels.discrete.discrete_margins import (
                    _get_count_effects)
            margeff = _get_count_effects(margeff, exog, count_idx, transform,
                    self, params)
        if dummy_idx is not None:
            from statsmodels.discrete.discrete_margins import (
                    _get_dummy_effects)
            margeff = _get_dummy_effects(margeff, exog, dummy_idx, transform,
                    self, params)
        return margeff




class GEEResults(base.LikelihoodModelResults):

    def __init__(self, model, params, cov_params, scale):

        super(GEEResults, self).__init__(model, params,
                normalized_cov_params=cov_params, scale=scale)


    @cache_readonly
    def standard_errors(self):
        return np.sqrt(np.diag(self.cov_params()))

    @cache_readonly
    def resid(self):
        """
        Returns the residuals, the endogeneous data minus the fitted
        values from the model.
        """
        return self.model.endog - self.fittedvalues


    @cache_readonly
    def centered_resid(self):
        """
        Returns the residuals centered within each group.
        """
        resid = self.resid
        for v in self.model.group_labels:
            ii = self.model.row_indices[v]
            resid[ii] -= resid[ii].mean()
        return resid


    @cache_readonly
    def fittedvalues(self):
        """
        Returns the fitted values from the model.
        """
        return self.model.family.link.inverse(np.dot(self.model.exog,
                                                     self.params))


    def conf_int(self, alpha=.05, cols=None):
        """
        Returns the confidence interval of the fitted parameters.

        Parameters
        ----------
        alpha : float, optional
             The `alpha` level for the confidence interval.  i.e., The
             default `alpha` = .05 returns a 95% confidence interval.
        cols : array-like, optional
             `cols` specifies which confidence intervals to return

        Notes
        -----
        The confidence interval is based on the Gaussian distribution.
        """
        bse = self.standard_errors
        params = self.params
        dist = stats.norm
        q = dist.ppf(1 - alpha / 2)

        if cols is None:
            lower = self.params - q * bse
            upper = self.params + q * bse
        else:
            cols = np.asarray(cols)
            lower = params[cols] - q * bse[cols]
            upper = params[cols] + q * bse[cols]
        return np.asarray(zip(lower, upper))


    def summary(self, yname=None, xname=None, title=None, alpha=.05):
        """Summarize the Regression Results

        Parameters
        -----------
        yname : string, optional
            Default is `y`
        xname : list of strings, optional
            Default is `var_##` for ## in p the number of regressors
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

        top_left = [('Dep. Variable:', None),
                    ('Model:', None),
                    ('Method:', ['Generalized Estimating Equations']),
                    ('Family:', [self.model.family.__class__.__name__]),
                    ('Dependence structure:', [self.model.varstruct.__class__.__name__]),
                    ('Date:', None),
        ]

        NY = [len(y) for y in self.model.endog_li]

        top_right = [('No. Observations:', [sum(NY)]),
                     ('No. clusters:', [len(self.model.endog_li)]),
                     ('Min. cluster size', [min(NY)]),
                     ('Max. cluster size', [max(NY)]),
                     ('Mean cluster size', ["%.1f" % np.mean(NY)]),
                     ('No. iterations', ['%d' % len(self.model.fit_history)]),
                     ('Time:', None),
                 ]

        # The skew of the residuals
        skew1 = stats.skew(self.resid)
        kurt1 = stats.kurtosis(self.resid)
        skew2 = stats.skew(self.centered_resid)
        kurt2 = stats.kurtosis(self.centered_resid)

        diagn_left = [('Skew:', ["%12.4f" % skew1]),
                      ('Centered skew:', ["%12.4f" % skew2])]

        diagn_right = [('Kurtosis:', ["%12.4f" % kurt1]),
                       ('Centered kurtosis:', ["%12.4f" % kurt2])
                   ]

        if title is None:
            title = self.model.__class__.__name__ + ' ' +\
                    "Regression Results"

        #create summary table instance
        from statsmodels.iolib.summary import Summary
        smry = Summary()
        smry.add_table_2cols(self, gleft=top_left, gright=top_right,
                             yname=self.model.endog_names,
                             xname=xname, title=title)
        smry.add_table_params(self, yname=yname,
                              xname=self.model.exog_names,
                              alpha=alpha, use_t=False)

        smry.add_table_2cols(self, gleft=diagn_left,
                             gright=diagn_right, yname=yname,
                             xname=xname, title="")

        return smry





def gee_setup_multicategorical(endog, exog, groups, time, offset,
                               endog_type):
    """
    Restructure nominal or ordinal multicategorical data as binary
    indicators so that they can be analysed using Generalized
    Estimating Equations.

    For nominal data, each element of endog is recoded as the sequence
    of |S|-1 indicators I(endog = S[0]), ..., I(endog = S[-1]), where
    S is the sorted list of unique values of endog (excluding the
    maximum value).

    For ordinal data, each element y of endog is recoded as |S|-1
    cumulative indicators I(endog > S[0]), ..., I(endog > S[-1]) where
    S is the sorted list of unique values of endog (excluding the
    maximum value).

    In addition, exog is modified as follows:

    For ordinal data, intercepts are prepended to the covariates, so
    that when defining a new variable as I(endog > S[j]) the intercept
    is a vector of zeros, with a 1 in the j^th position.

    For nominal data, when constructing the new exog for I(endog =
    S[j]), the covariate vector is expanded |S| - 1 times, with the
    exog values replaced with zeros except for block j.

    Arguments
    ---------
    endog: array-like
        A list of 1-dimensional NumPy arrays, giving the response
        values for the clusters
    exog: array-like
        A list of 2-dimensional NumPy arrays, giving the covariate
        data for the clusters.  exog[i] should include an intercept
        for nominal data but no intercept should be included for
        ordinal data.
    groups : array-like
        The group label for each observation
    time : List
        A list of 1-dimensional NumPy arrays containing time
        information
    offset : List
        A list of 1-dimensional NumPy arrays containing the offset
        information
    endog_type: string
        Either "ordinal" or "nominal"

    The number of rows of exog[i] must equal the length of endog[i],
    and all the exog[i] arrays should have the same number of columns.

    Returns:
    --------
    endog_ex:   endog recoded as described above
    exog_ex:    exog recoded as described above
    groups_ex:  groups expanded to fit the recoded data
    offset_ex:  offset expanded to fit the recoded data
    time_ex:    time expanded to fit the recoded data

    Examples:
    ---------

    >>> family = Binomial()
    >>> endog_ex,exog_ex,groups_ex,time_ex,offset_ex,nlevel =\
        gee_setup_multicategorical(endog, exog, group_n, None, None, "ordinal")
    >>> v = GlobalOddsRatio(nlevel, "ordinal")
    >>> nx = exog.shape[1] - nlevel + 1
    >>> beta = gee_multicategorical_starting_values(endog, nlevel, nx, "ordinal")
    >>> md = GEE(endog_ex, exog_ex, groups_ex, None, family, v)
    >>> mdf = md.fit(starting_beta = beta)

    """

    if endog_type not in ("ordinal", "nominal"):
        raise ValueError("setup_multicategorical: `endog_type` must "
                         "be either 'nominal' or 'categorical'")

    # The unique outcomes, except the greatest one.
    endog_values = list(set(endog))
    endog_values.sort()
    endog_cuts = endog_values[0:-1]

    ncut = len(endog_cuts)

    # Default offset
    if offset is None:
        offset = np.zeros(len(endog), dtype=np.float64)

    # Default time
    if time is None:
        time = np.zeros(len(endog), dtype=np.float64)

    # nominal=1, ordinal=0
    endog_type_ordinal = (endog_type == "ordinal")

    endog_ex = []
    exog_ex = []
    groups_ex = []
    time_ex = []
    offset_ex = []

    for endog1, exog1, grp, off, tim in \
            zip(endog, exog, groups, offset, time):

        # Loop over thresholds for the indicators
        for thresh_ix, thresh in enumerate(endog_cuts):

            # Code as cumulative indicators
            if endog_type_ordinal:

                endog_ex.append(int(endog1 > thresh))
                offset_ex.append(off)
                groups_ex.append(grp)
                time_ex.append(tim)
                icepts = np.zeros(ncut, dtype=np.float64)
                icepts[thresh_ix] = 1
                exog2 = np.concatenate((icepts, exog1))
                exog_ex.append(exog2)

            # Code as indicators
            else:

                endog_ex.append(int(endog1 == thresh))
                offset_ex.append(off)
                groups_ex.append(grp)
                time_ex.append(tim)
                exog2 = np.zeros(ncut * len(exog1), dtype=np.float64)
                exog2[thresh_ix*len(exog1):(thresh_ix+1)*len(exog1)] \
                    = exog1
                exog_ex.append(exog2)

    endog_ex = np.array(endog_ex)
    exog_ex = np.array(exog_ex)
    groups_ex = np.array(groups_ex)
    time_ex = np.array(time_ex)
    offset_ex = np.array(offset_ex)

    return endog_ex, exog_ex, groups_ex, time_ex, offset_ex, len(endog_values)


def gee_ordinal_starting_values(endog, n_exog):
    """

    Parameters:
    -----------
    endog : array-like
       Endogeneous (response) data for the unmodified data.

    n_exog : integer
       The number of exogeneous (predictor) variables
    """

    endog_values = list(set(endog))
    endog_values.sort()
    endog_cuts = endog_values[0:-1]

    prob = np.array([np.mean(endog > s) for s in endog_cuts])
    prob_logit = np.log(prob/(1-prob))
    beta = np.concatenate((prob_logit, np.zeros(n_exog)))

    return beta


def gee_nominal_starting_values(endog, n_exog):
    """

    Parameters:
    -----------
    endog : array-like
       Endogeneous (response) data for the unmodified data.

    n_exog : integer
       The number of exogeneous (predictor) variables
    """

    endog_values = list(set(endog))
    endog_values.sort()
    ncuts = len(endog_values) - 1

    return np.zeros(n_exog * ncuts, dtype=np.float64)



import statsmodels.genmod.families.varfuncs as varfuncs
from statsmodels.genmod.families.links import Link
from statsmodels.genmod.families import Family


class MultinomialLogit(Link):
    """
    The multinomial logit transform, only for use with GEE.

    Notes
    -----
    The data are assumed coded as binary indicators, where each
    observed multinomial value y is coded as I(y == S[0]), ..., I(y ==
    S[-1]), where S is the set of possible response labels, excluding
    the largest one.  Thererefore functions in this class should only
    be called using vector argument whose length is a multiple of |S|
    = ncut, which is an argument to be provided when initializing the
    class.

    call and derivative use a private method _clean to make trim p by
    1e-10 so that p is in (0,1)
    """

    def __init__(self, ncut):
        self.ncut = ncut


    def inverse(self, lpr):
        """
        Inverse of the multinomial logit transform, which gives the
        expected values of the data as a function of the linear
        predictors.

        Parameters
        ----------
        lpr : array-like (length must be divisible by `ncut`)
            The linear predictors

        Returns
        -------
        prob : array
            Probabilities, or expected values
        """

        expval = np.exp(lpr)

        denom = 1 + np.reshape(expval, (len(expval) / self.ncut,
                                        self.ncut)).sum(1)
        denom = np.kron(denom, np.ones(self.ncut, dtype=np.float64))

        prob = expval / denom

        return prob



    def mean_deriv(self, exog, lpr):
        """
        Derivative of the expected endog with respect to param.

        Parameters
        ----------
        exog : array-like
           The exogeneous data at which the derivative is computed,
           number of rows must be a multiple of `ncut`.
        lpr : array-like
           The linear predictor values, length must be multiple of
           `ncut`.

        Returns
        -------
        The value of the derivative of the expected endog with respect
        to param
        """

        expval = np.exp(lpr)

        expval_m = np.reshape(expval, (len(expval) / self.ncut,
                                       self.ncut))

        denom = 1 + expval_m.sum(1)
        denom = np.kron(denom, np.ones(self.ncut, dtype=np.float64))

        dmat = expval[:, None] * exog / denom[:, None]

        from scipy.sparse import block_diag
        ones = np.ones(self.ncut, dtype=np.float64)
        cmat = block_diag([np.outer(ones, x) for x in expval_m], "csr")
        rmat = cmat.dot(exog)
        dmat -= expval[:,None] * rmat / denom[:, None]**2

        return dmat


    # Minimally tested
    def mean_deriv_exog(self, exog, params):
        """
        Derivative of the expected endog with respect to exog, used in
        analyzing marginal effects.

        Parameters
        ----------
        exog : array-like
           The exogeneous data at which the derivative is computed,
           number of rows must be a multiple of `ncut`.
        lpr : array-like
           The linear predictor values, length must be multiple of
           `ncut`.

        Returns
        -------
        The value of the derivative of the expected endog with respect
        to exog.
        """

        lpr = np.dot(exog, params)
        expval = np.exp(lpr)

        expval_m = np.reshape(expval, (len(expval) / self.ncut,
                                       self.ncut))

        denom = 1 + expval_m.sum(1)
        denom = np.kron(denom, np.ones(self.ncut, dtype=np.float64))

        bmat0 = np.outer(np.ones(exog.shape[0]), params)

        # Masking matrix
        qmat = []
        for j in range(self.ncut):
            ee = np.zeros(self.ncut, dtype=np.float64)
            ee[j] = 1
            qmat.append(np.kron(ee, np.ones(len(params) / self.ncut)))
        qmat = np.array(qmat)
        qmat = np.kron(np.ones((exog.shape[0]/self.ncut, 1)), qmat)
        bmat = bmat0 * qmat

        dmat = expval[:, None] * bmat / denom[:, None]

        expval_mb = np.kron(expval_m, np.ones((self.ncut, 1)))
        expval_mb = np.kron(expval_mb, np.ones((1, self.ncut)))

        dmat -= expval[:, None] * (bmat * expval_mb) / denom[:,None]**2

        return dmat





class Multinomial(Family):
    """
    Pseudo-link function for fitting nominal multinomial models with
    GEE.  Not for use outside the GEE class.
    """

    links = [MultinomialLogit,]
    variance = varfuncs.binary


    def __init__(self, ncut):

        self.ncut = ncut
        self.link = MultinomialLogit(ncut)






from statsmodels.discrete.discrete_margins import \
    _get_margeff_exog, _get_const_index, _check_margeff_args, \
    _effects_at, margeff_cov_with_se, _check_at_is_all, \
    _transform_names




class GEEMargins(object):
    """Estimate the marginal effects of a model fit using generalized
    estimating equations.

    Parameters
    ----------
    results : GEEResults instance
        The results instance of a fitted discrete choice model
    args : tuple
        Args are passed to `get_margeff`. This is the same as
        results.get_margeff. See there for more information.
    kwargs : dict
        Keyword args are passed to `get_margeff`. This is the same as
        results.get_margeff. See there for more information.
    """
    def __init__(self, results, args, kwargs={}):
        self._cache = resettable_cache()
        self.results = results
        self.get_margeff(*args, **kwargs)

    def _reset(self):
        self._cache = resettable_cache()

    @cache_readonly
    def tvalues(self):
        _check_at_is_all(self.margeff_options)
        return self.margeff / self.margeff_se

    def summary_frame(self, alpha=.05):
        """
        Returns a DataFrame summarizing the marginal effects.

        Parameters
        ----------
        alpha : float
            Number between 0 and 1. The confidence intervals have the
            probability 1-alpha.

        Returns
        -------
        frame : DataFrames
            A DataFrame summarizing the marginal effects.
        """
        _check_at_is_all(self.margeff_options)
        from pandas import DataFrame
        names = [_transform_names[self.margeff_options['method']],
                                  'Std. Err.', 'z', 'Pr(>|z|)',
                                  'Conf. Int. Low', 'Cont. Int. Hi.']
        ind = self.results.model.exog.var(0) != 0 # True if not a constant
        exog_names = self.results.model.exog_names
        var_names = [name for i,name in enumerate(exog_names) if ind[i]]
        table = np.column_stack((self.margeff, self.margeff_se, self.tvalues,
                                 self.pvalues, self.conf_int(alpha)))
        return DataFrame(table, columns=names, index=var_names)

    @cache_readonly
    def pvalues(self):
        _check_at_is_all(self.margeff_options)
        return norm.sf(np.abs(self.tvalues)) * 2

    def conf_int(self, alpha=.05):
        """
        Returns the confidence intervals of the marginal effects

        Parameters
        ----------
        alpha : float
            Number between 0 and 1. The confidence intervals have the
            probability 1-alpha.

        Returns
        -------
        conf_int : ndarray
            An array with lower, upper confidence intervals for the marginal
            effects.
        """
        _check_at_is_all(self.margeff_options)
        me_se = self.margeff_se
        q = norm.ppf(1 - alpha / 2)
        lower = self.margeff - q * me_se
        upper = self.margeff + q * me_se
        return np.asarray(zip(lower, upper))

    def summary(self, alpha=.05):
        """
        Returns a summary table for marginal effects

        Parameters
        ----------
        alpha : float
            Number between 0 and 1. The confidence intervals have the
            probability 1-alpha.

        Returns
        -------
        Summary : SummaryTable
            A SummaryTable instance
        """
        _check_at_is_all(self.margeff_options)
        results = self.results
        model = results.model
        title = model.__class__.__name__ + " Marginal Effects"
        method = self.margeff_options['method']
        top_left = [('Dep. Variable:', [model.endog_names]),
                ('Method:', [method]),
                ('At:', [self.margeff_options['at']]),]

        from statsmodels.iolib.summary import (Summary, summary_params,
                                                table_extend)
        exog_names = model.exog_names[:] # copy
        smry = Summary()

        # sigh, we really need to hold on to this in _data...
        _, const_idx = _get_const_index(model.exog)
        if const_idx is not None:
            exog_names.pop(const_idx)

        J = int(getattr(model, "J", 1))
        if J > 1:
            yname, yname_list = results._get_endog_name(model.endog_names,
                                                None, all=True)
        else:
            yname = model.endog_names
            yname_list = [yname]

        smry.add_table_2cols(self, gleft=top_left, gright=[],
                yname=yname, xname=exog_names, title=title)

        #NOTE: add_table_params is not general enough yet for margeff
        # could use a refactor with getattr instead of hard-coded params
        # tvalues etc.
        table = []
        conf_int = self.conf_int(alpha)
        margeff = self.margeff
        margeff_se = self.margeff_se
        tvalues = self.tvalues
        pvalues = self.pvalues
        if J > 1:
            for eq in range(J):
                restup = (results, margeff[:,eq], margeff_se[:,eq],
                          tvalues[:,eq], pvalues[:,eq], conf_int[:,:,eq])
                tble = summary_params(restup, yname=yname_list[eq],
                              xname=exog_names, alpha=alpha, use_t=False,
                              skip_header=True)
                tble.title = yname_list[eq]
                # overwrite coef with method name
                header = ['', _transform_names[method], 'std err', 'z',
                        'P>|z|', '[%3.1f%% Conf. Int.]' % (100-alpha*100)]
                tble.insert_header_row(0, header)
                #from IPython.core.debugger import Pdb; Pdb().set_trace()
                table.append(tble)

            table = table_extend(table, keep_headers=True)
        else:
            restup = (results, margeff, margeff_se, tvalues, pvalues, conf_int)
            table = summary_params(restup, yname=yname, xname=exog_names,
                    alpha=alpha, use_t=False, skip_header=True)
            header = ['', _transform_names[method], 'std err', 'z',
                        'P>|z|', '[%3.1f%% Conf. Int.]' % (100-alpha*100)]
            table.insert_header_row(0, header)

        smry.tables.append(table)
        return smry

    def get_margeff(self, at='overall', method='dydx', atexog=None,
                          dummy=False, count=False):
        """Get marginal effects of the fitted model.

        Parameters
        ----------
        at : str, optional
            Options are:

            - 'overall', The average of the marginal effects at each
              observation.
            - 'mean', The marginal effects at the mean of each regressor.
            - 'median', The marginal effects at the median of each regressor.
            - 'zero', The marginal effects at zero for each regressor.
            - 'all', The marginal effects at each observation. If `at` is all
              only margeff will be available.

            Note that if `exog` is specified, then marginal effects for all
            variables not specified by `exog` are calculated using the `at`
            option.
        method : str, optional
            Options are:

            - 'dydx' - dy/dx - No transformation is made and marginal effects
              are returned.  This is the default.
            - 'eyex' - estimate elasticities of variables in `exog` --
              d(lny)/d(lnx)
            - 'dyex' - estimate semielasticity -- dy/d(lnx)
            - 'eydx' - estimate semeilasticity -- d(lny)/dx

            Note that tranformations are done after each observation is
            calculated.  Semi-elasticities for binary variables are computed
            using the midpoint method. 'dyex' and 'eyex' do not make sense
            for discrete variables.
        atexog : array-like, optional
            Optionally, you can provide the exogenous variables over which to
            get the marginal effects.  This should be a dictionary with the key
            as the zero-indexed column number and the value of the dictionary.
            Default is None for all independent variables less the constant.
        dummy : bool, optional
            If False, treats binary variables (if present) as continuous.  This
            is the default.  Else if True, treats binary variables as
            changing from 0 to 1.  Note that any variable that is either 0 or 1
            is treated as binary.  Each binary variable is treated separately
            for now.
        count : bool, optional
            If False, treats count variables (if present) as continuous.  This
            is the default.  Else if True, the marginal effect is the
            change in probabilities when each observation is increased by one.

        Returns
        -------
        effects : ndarray
            the marginal effect corresponding to the input options

        Notes
        -----
        When using after Poisson, returns the expected number of events
        per period, assuming that the model is loglinear.
        """
        self._reset() # always reset the cache when this is called
        #TODO: if at is not all or overall, we can also put atexog values
        # in summary table head
        method = method.lower()
        at = at.lower()
        _check_margeff_args(at, method)
        self.margeff_options = dict(method=method, at=at)
        results = self.results
        model = results.model
        params = results.params
        exog = model.exog.copy() # copy because values are changed
        effects_idx, const_idx =  _get_const_index(exog)

        if dummy:
            _check_discrete_args(at, method)
            dummy_idx, dummy = _get_dummy_index(exog, const_idx)
        else:
            dummy_idx = None

        if count:
            _check_discrete_args(at, method)
            count_idx, count = _get_count_index(exog, const_idx)
        else:
            count_idx = None

        # get the exogenous variables
        exog = _get_margeff_exog(exog, at, atexog, effects_idx)

        # get base marginal effects, handled by sub-classes
        effects = model._derivative_exog(params, exog, method,
                                                    dummy_idx, count_idx)

        J = getattr(model, 'J', 1)
        effects_idx = np.tile(effects_idx, J) # adjust for multi-equation.

        effects = _effects_at(effects, at)

        if at == 'all':
            if J > 1:
                K = model.K - np.any(~effects_idx) # subtract constant
                self.margeff = effects[:, effects_idx].reshape(-1, K, J,
                                                                order='F')
            else:
                self.margeff = effects[:, effects_idx]
        else:
            # Set standard error of the marginal effects by Delta method.
            margeff_cov, margeff_se = margeff_cov_with_se(model, params, exog,
                                                results.cov_params(), at,
                                                model._derivative_exog,
                                                dummy_idx, count_idx,
                                                method, J)

            # reshape for multi-equation
            if J > 1:
                K = model.K - np.any(~effects_idx) # subtract constant
                self.margeff = effects[effects_idx].reshape(K, J, order='F')
                self.margeff_se = margeff_se[effects_idx].reshape(K, J,
                                                                  order='F')
                self.margeff_cov = margeff_cov[effects_idx][:, effects_idx]
            else:
                # don't care about at constant
                self.margeff_cov = margeff_cov[effects_idx][:, effects_idx]
                self.margeff_se = margeff_se[effects_idx]
                self.margeff = effects[effects_idx]
