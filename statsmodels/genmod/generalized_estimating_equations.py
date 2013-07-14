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
from statsmodels.tools.decorators import cache_readonly
import statsmodels.base.model as base
from statsmodels.genmod import families
from statsmodels.genmod import dependence_structures
from statsmodels.genmod.dependence_structures import VarStruct
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
          The exognenous data for the parent model.
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
        Returns the original exog matrix before it was reduced to
        satisfy the constraint.
        """
        return self.orig_exog


    def unpack_param(self, beta):
        """
        Returns the parameter vector `beta` in the original
        coordinates.
        """

        return self.param0 + np.dot(self.lhs0, beta)


    def unpack_cov(self, bcov):
        """
        Returns the covariance matrix `bcov` in the original
        coordinates.
        """

        return np.dot(self.lhs0, np.dot(bcov, self.lhs0.T))





#TODO multinomial responses
class GEE(base.Model):
    __doc__ = """
    Parameters
    ----------
    endog : array-like
        1d array of endogenous response variable.
    exog : array-like
        A nobs x k array where `nobs` is the number of
        observations and `k` is the number of regressors. An
        interecept is not included by default and should be added
        by the user. See `statsmodels.tools.add_constant`.
    groups : array-like
        A 1d array of length `nobs` containing the cluster labels.
    time : array-like
        A 1d array of time (or other index) values.  This is only
        used if the dependence structure is Autoregressive
    family : family class instance
        The default is Gaussian.  To specify the binomial
        distribution family = sm.family.Binomial() Each family can
        take a link instance as an argument.  See
        statsmodels.family.family for more information.
    varstruct : VarStruct class instance
        The default is Independence.  To specify an exchangeable
        structure varstruct = sm.varstruct.Exchangeable() See
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
    _cached_means = None


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

        if time is None:
            self.time = np.zeros(self.exog.shape[0], dtype=np.float64)
        else:
            self.time = time

        # Handle the constraint
        self.constraint = None
        if constraint is not None:
            if len(constraint) != 2:
                raise ValueError("GEE: `constraint` must be a 2-tuple.")
            self.constraint = ParameterConstraint(constraint[0],
                                                  constraint[1],
                                                  self.exog)

            self.offset += self.constraint.offset_increment()
            self.exog = self.constraint.reduced_exog()

        # Convert the data to the internal representation, which is a
        # list of arrays, corresponding to the clusters.
        group_labels = list(set(groups))
        group_labels.sort()
        row_indices = {s: [] for s in group_labels}
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


    def _cluster_list(self, array):
        """
        Returns the `array` split into subarrays corresponding to the
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

        cached_means = self._cached_means

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

        _cached_means = self._cached_means

        mean_deriv = self.family.link.inverse_deriv
        varfunc = self.family.variance

        bmat, score = 0, 0
        for i in range(num_clust):

            if len(endog[i]) == 0:
                continue

            expval, lpr = _cached_means[i]

            dmat_t = exog[i] * mean_deriv(lpr)[:, None]
            dmat = dmat_t.T

            sdev = np.sqrt(varfunc(expval))
            vmat, is_cor = self.varstruct.variance_matrix(expval, i)
            if is_cor:
                vmat *= np.outer(sdev, sdev)

            vinv_d = np.linalg.solve(vmat, dmat_t)
            bmat += np.dot(dmat, vinv_d)

            resid = endog[i] - expval
            vinv_resid = np.linalg.solve(vmat, resid)
            score += np.dot(dmat, vinv_resid)

        update = np.linalg.solve(bmat, score)

        return update, score


    def _update_cached_means(self, beta):
        """
        _cached_means should always contain the most recent
        calculation of the cluster-wise mean vectors.  This function
        should be called every time the value of beta is changed, to
        keep the cached means up to date.
        """

        endog = self.endog_li
        exog = self.exog_li
        offset = self.offset_li
        num_clust = len(endog)

        mean = self.family.link.inverse

        self._cached_means = []

        for i in range(num_clust):

            if len(endog[i]) == 0:
                continue

            lpr = offset[i] + np.dot(exog[i], beta)
            expval = mean(lpr)

            self._cached_means.append((expval, lpr))



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
           The model based estimate of the covariance, which is
           meaningful if the covariance structure is correctly
           specified.
         cmat : array-like
           The center matrix of the sandwich expression.
        """

        endog = self.endog_li
        exog = self.exog_li
        num_clust = len(endog)

        mean_deriv = self.family.link.inverse_deriv
        varfunc = self.family.variance
        _cached_means = self._cached_means

        bmat, cmat = 0, 0
        for i in range(num_clust):

            if len(endog[i]) == 0:
                continue

            expval, lpr = _cached_means[i]

            dmat_t = exog[i] * mean_deriv(lpr)[:, None]
            dmat = dmat_t.T

            sdev = np.sqrt(varfunc(expval))
            vmat, is_cor = self.varstruct.variance_matrix(expval, i)
            if is_cor:
                vmat *= np.outer(sdev, sdev)

            vinv_d = np.linalg.solve(vmat, dmat_t)
            bmat += np.dot(dmat, vinv_d)

            resid = endog[i] - expval
            vinv_resid = np.linalg.solve(vmat, resid)
            dvinv_resid = np.dot(dmat, vinv_resid)
            cmat += np.outer(dvinv_resid, dvinv_resid)

        scale = self.estimate_scale()

        naive_covariance = scale*np.linalg.inv(bmat)
        cmat /= scale**2
        robust_covariance = np.dot(naive_covariance,
                                   np.dot(cmat, naive_covariance))

        return robust_covariance, naive_covariance, cmat


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
        xnames : array-like
           A list of variable names

        """

        try:
            xnames = list(self.data.exog.columns)
        except:
            xnames = ["X%d" % k for k in
                      range(1, self.data.exog.shape[1]+1)]

        if starting_beta is None:
            beta_dm = self.exog_li[0].shape[1]
            beta = np.zeros(beta_dm, dtype=np.float64)

        else:
            beta = starting_beta.copy()

        return beta, xnames




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
            coefficients

        Returns
        -------
        An instance of the GEEResults class

        """

        self.fit_history = {'params' : [],
                            'score_change' : []}

        beta, xnames = self._starting_beta(starting_beta)

        self._update_cached_means(beta)

        for _ in xrange(maxit):
            update, _ = self._beta_update()
            beta += update
            self._update_cached_means(beta)
            stepsize = np.sqrt(np.sum(update**2))
            self.fit_history['params'].append(beta)
            if stepsize < ctol:
                break
            self._update_assoc(beta)

        bcov, _, _ = self._covmat()

        if self.constraint is not None:
            beta, bcov = self._handle_constraint(beta, bcov)

        beta = pandas.Series(beta, xnames)
        scale = self.estimate_scale()

        results = GEEResults(self, beta, bcov/scale, scale)

        results.fit_history = self.fit_history

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
        save_cached_means = copy.deepcopy(self._cached_means)
        self._update_cached_means(beta0)
        _, score = self._beta_update()
        _, ncov1, cmat = self._covmat()
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
        self.score_statistic = np.dot(score2,
                                  np.linalg.solve(score_cov, score2))
        self.score_df = len(score2)
        self.score_pvalue = 1 -\
                 chi2.cdf(self.score_statistic, self.score_df)

        beta = self.constraint.unpack_param(beta)
        bcov = self.constraint.unpack_cov(bcov)

        self.exog_li = save_exog_li
        self._cached_means = save_cached_means
        self.exog = self.constraint.restore_exog()

        return beta, bcov


    def _update_assoc(self, beta):
        """
        Update the association parameters
        """

        self.varstruct.update(beta)



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
        R = self.resid
        skew1 = stats.skew(R)
        kurt1 = stats.kurtosis(R)
        V = R.copy() - R.mean()
        skew2 = stats.skew(V)
        kurt2 = stats.kurtosis(V)

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
                              xname=self.params.index.tolist(),
                              alpha=alpha, use_t=True)

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

    For ordinal data, intercepts are constructed corresponding to the
    different levels of the outcome. For nominal data, the covariate
    vector is expanded so that different coefficients arise for each
    class.

    In both cases, the covariates in exog are copied over to all of
    the indicators derived from the same original value.

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
    exog_ex:   exog recoded as described above
    groups_ex: groups recoded as described above
    offset_ex: offset expanded to fit the recoded data
    time_ex:   time expanded to fit the recoded data

    Examples:
    ---------

    >>> family = Binomial()

    >>> endog_ex,exog_ex,groups_ex,time_ex,offset_ex,nlevel =\
        setup_gee_multicategorical(endog, exog, group_n, None, None, "ordinal")

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
    endog_type_i = [0, 1][endog_type == "nominal"]

    endog_ex = []
    exog_ex = []
    groups_ex = []
    time_ex = []
    offset_ex = []

    for endog1, exog1, grp, off, tim in zip(endog, exog, groups, offset, time):

        # Loop over thresholds for the indicators
        for thresh_ix, thresh in enumerate(endog_cuts):

            # Code as cumulative indicators
            if endog_type_i == 0:

                endog_ex.append(int(endog1 > thresh))
                offset_ex.append(off)
                groups_ex.append(grp)
                time_ex.append(tim)
                zero = np.zeros(ncut, dtype=np.float64)
                exog1_icepts = np.concatenate((zero, exog1))
                exog1_icepts[thresh_ix] = 1
                exog_ex.append(exog1_icepts)

            # Code as indicators
            else:
                pass
                #y1.append(int(y2 == s))
                #xx = np.zeros(ncut, dtype=np.float64)
                #xx[js] = 1
                #x3 = np.kronecker(xx, x3)

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
    endog_cuts = endog_values[0:-1]

    raise NotImplementedError
