"""
Procedures for fitting marginal regression models to dependent data
using Generalized Estimating Equations.

References
----------
KY Liang and S Zeger. "Longitudinal data analysis using
generalized linear models". Biometrika (1986) 73 (1): 13-22.

S Zeger and KY Liang. "Longitudinal Data Analysis for Discrete and
Continuous Outcomes". Biometrics Vol. 42, No. 1 (Mar., 1986),
pp. 121-130

A Rotnitzky and NP Jewell (1990). "Hypothesis testing of regression
parameters in semiparametric generalized linear models for cluster
correlated data", Biometrika, 77, 485-497.

Xu Guo and Wei Pan (2002). "Small sample performance of the score
test in GEE".
http://www.sph.umn.edu/faculty1/wp-content/uploads/2012/11/rr2002-013.pdf

LA Mancl LA, TA DeRouen (2001). A covariance estimator for GEE with
improved small-sample properties.  Biometrics. 2001 Mar;57(1):126-34.
"""
from __future__ import division
from statsmodels.compat.python import range, lzip, zip

import numpy as np
from scipy import stats
import pandas as pd

from statsmodels.tools.decorators import (cache_readonly,
                                          resettable_cache)
import statsmodels.base.model as base
# used for wrapper:
import statsmodels.regression.linear_model as lm
import statsmodels.base.wrapper as wrap

from statsmodels.genmod import families
from statsmodels.genmod import cov_struct as cov_structs

import statsmodels.genmod.families.varfuncs as varfuncs
from statsmodels.genmod.families.links import Link

from statsmodels.tools.sm_exceptions import (ConvergenceWarning,
                                             DomainWarning,
                                             IterationLimitWarning,
                                             ValueWarning)
import warnings

from statsmodels.graphics._regressionplots_doc import (
    _plot_added_variable_doc,
    _plot_partial_residuals_doc,
    _plot_ceres_residuals_doc)


class ParameterConstraint(object):
    """
    A class for managing linear equality constraints for a parameter
    vector.
    """

    def __init__(self, lhs, rhs, exog):
        """
        Parameters
        ----------
        lhs : ndarray
           A q x p matrix which is the left hand side of the
           constraint lhs * param = rhs.  The number of constraints is
           q >= 1 and p is the dimension of the parameter vector.
        rhs : ndarray
          A 1-dimensional vector of length q which is the right hand
          side of the constraint equation.
        exog : ndarray
          The n x p exognenous data for the full model.
        """

        # In case a row or column vector is passed (patsy linear
        # constraints passes a column vector).
        rhs = np.atleast_1d(rhs.squeeze())

        if rhs.ndim > 1:
            raise ValueError("The right hand side of the constraint "
                             "must be a vector.")

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
        # [lhs0, lhs1] are mutually orthogonal.
        lhs_u, lhs_s, lhs_vt = np.linalg.svd(lhs.T, full_matrices=1)
        self.lhs0 = lhs_u[:, len(lhs_s):]
        self.lhs1 = lhs_u[:, 0:len(lhs_s)]
        self.lhsf = np.hstack((self.lhs0, self.lhs1))

        # param0 is one solution to the underdetermined system
        # L * param = R.
        self.param0 = np.dot(self.lhs1, np.dot(lhs_vt, self.rhs) /
                             lhs_s)

        self._offset_increment = np.dot(exog, self.param0)

        self.orig_exog = exog
        self.exog_fulltrans = np.dot(exog, self.lhsf)

    def offset_increment(self):
        """
        Returns a vector that should be added to the offset vector to
        accommodate the constraint.

        Parameters
        ----------
        exog : array-like
           The exogeneous data for the model.
        """

        return self._offset_increment

    def reduced_exog(self):
        """
        Returns a linearly transformed exog matrix whose columns span
        the constrained model space.

        Parameters
        ----------
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


_gee_init_doc = """
    Marginal regression model fit using Generalized Estimating Equations.

    GEE can be used to fit Generalized Linear Models (GLMs) when the
    data have a grouped structure, and the observations are possibly
    correlated within groups but not between groups.

    Parameters
    ----------
    endog : array-like
        1d array of endogenous values (i.e. responses, outcomes,
        dependent variables, or 'Y' values).
    exog : array-like
        2d array of exogeneous values (i.e. covariates, predictors,
        independent variables, regressors, or 'X' values). A `nobs x
        k` array where `nobs` is the number of observations and `k` is
        the number of regressors. An intercept is not included by
        default and should be added by the user. See
        `statsmodels.tools.add_constant`.
    groups : array-like
        A 1d array of length `nobs` containing the group labels.
    time : array-like
        A 2d array of time (or other index) values, used by some
        dependence structures to define similarity relationships among
        observations within a cluster.
    family : family class instance
%(family_doc)s
    cov_struct : CovStruct class instance
        The default is Independence.  To specify an exchangeable
        structure use cov_struct = Exchangeable().  See
        statsmodels.genmod.cov_struct.CovStruct for more
        information.
    offset : array-like
        An offset to be included in the fit.  If provided, must be
        an array whose length is the number of rows in exog.
    dep_data : array-like
        Additional data passed to the dependence structure.
    constraint : (ndarray, ndarray)
        If provided, the constraint is a tuple (L, R) such that the
        model parameters are estimated under the constraint L *
        param = R, where L is a q x p matrix and R is a
        q-dimensional vector.  If constraint is provided, a score
        test is performed to compare the constrained model to the
        unconstrained model.
    update_dep : bool
        If true, the dependence parameters are optimized, otherwise
        they are held fixed at their starting values.
    weights : array-like
        An array of weights to use in the analysis.  The weights must
        be constant within each group.  These correspond to
        probability weights (pweights) in Stata.
    %(extra_params)s

    See Also
    --------
    statsmodels.genmod.families.family
    :ref:`families`
    :ref:`links`

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

    The "robust" covariance type is the standard "sandwich estimator"
    (e.g. Liang and Zeger (1986)).  It is the default here and in most
    other packages.  The "naive" estimator gives smaller standard
    errors, but is only correct if the working correlation structure
    is correctly specified.  The "bias reduced" estimator of Mancl and
    DeRouen (Biometrics, 2001) reduces the downard bias of the robust
    estimator.

    The robust covariance provided here follows Liang and Zeger (1986)
    and agrees with R's gee implementation.  To obtain the robust
    standard errors reported in Stata, multiply by sqrt(N / (N - g)),
    where N is the total sample size, and g is the average group size.

    Examples
    --------
    %(example)s
"""

_gee_family_doc = """\
        The default is Gaussian.  To specify the binomial
        distribution use `family=sm.family.Binomial()`. Each family
        can take a link instance as an argument.  See
        statsmodels.family.family for more information."""

_gee_ordinal_family_doc = """\
        The only family supported is `Binomial`.  The default `Logit`
        link may be replaced with `probit` if desired."""

_gee_nominal_family_doc = """\
        The default value `None` uses a multinomial logit family
        specifically designed for use with GEE.  Setting this
        argument to a non-default value is not currently supported."""

_gee_fit_doc = """
    Fits a marginal regression model using generalized estimating
    equations (GEE).

    Parameters
    ----------
    maxiter : integer
        The maximum number of iterations
    ctol : float
        The convergence criterion for stopping the Gauss-Seidel
        iterations
    start_params : array-like
        A vector of starting values for the regression
        coefficients.  If None, a default is chosen.
    params_niter : integer
        The number of Gauss-Seidel updates of the mean structure
        parameters that take place prior to each update of the
        dependence structure.
    first_dep_update : integer
        No dependence structure updates occur before this
        iteration number.
    cov_type : string
        One of "robust", "naive", or "bias_reduced".
    ddof_scale : scalar or None
        The scale parameter is estimated as the sum of squared
        Pearson residuals divided by `N - ddof_scale`, where N
        is the total sample size.  If `ddof_scale` is None, the
        number of covariates (including an intercept if present)
        is used.
    scaling_factor : scalar
        The estimated covariance of the parameter estimates is
        scaled by this value.  Default is 1, Stata uses N / (N - g),
        where N is the total sample size and g is the average group
        size.

    Returns
    -------
    An instance of the GEEResults class or subclass

    Notes
    -----
    If convergence difficulties occur, increase the values of
    `first_dep_update` and/or `params_niter`.  Setting
    `first_dep_update` to a greater value (e.g. ~10-20) causes the
    algorithm to move close to the GLM solution before attempting
    to identify the dependence structure.

    For the Gaussian family, there is no benefit to setting
    `params_niter` to a value greater than 1, since the mean
    structure parameters converge in one step.
"""

_gee_results_doc = """
    Returns
    -------
    **Attributes**

    cov_params_default : ndarray
        default covariance of the parameter estimates. Is chosen among one
        of the following three based on `cov_type`
    cov_robust : ndarray
        covariance of the parameter estimates that is robust
    cov_naive : ndarray
        covariance of the parameter estimates that is not robust to
        correlation or variance misspecification
    cov_robust_bc : ndarray
        covariance of the parameter estimates that is robust and bias
        reduced
    converged : bool
        indicator for convergence of the optimization.
        True if the norm of the score is smaller than a threshold
    cov_type : string
        string indicating whether a "robust", "naive" or "bias_reduced"
        covariance is used as default
    fit_history : dict
        Contains information about the iterations.
    fittedvalues : array
        Linear predicted values for the fitted model.
        dot(exog, params)
    model : class instance
        Pointer to GEE model instance that called `fit`.
    normalized_cov_params : array
        See GEE docstring
    params : array
        The coefficients of the fitted model.  Note that
        interpretation of the coefficients often depends on the
        distribution family and the data.
    scale : float
        The estimate of the scale / dispersion for the model fit.
        See GEE.fit for more information.
    score_norm : float
        norm of the score at the end of the iterative estimation.
    bse : array
        The standard errors of the fitted GEE parameters.
"""

_gee_example = """
    Logistic regression with autoregressive working dependence:

    >>> import statsmodels.api as sm
    >>> family = sm.families.Binomial()
    >>> va = sm.cov_struct.Autoregressive()
    >>> model = sm.GEE(endog, exog, group, family=family, cov_struct=va)
    >>> result = model.fit()
    >>> print(result.summary())

    Use formulas to fit a Poisson GLM with independent working
    dependence:

    >>> import statsmodels.api as sm
    >>> fam = sm.families.Poisson()
    >>> ind = sm.cov_struct.Independence()
    >>> model = sm.GEE.from_formula("y ~ age + trt + base", "subject", \
                                 data, cov_struct=ind, family=fam)
    >>> result = model.fit()
    >>> print(result.summary())

    Equivalent, using the formula API:

    >>> import statsmodels.api as sm
    >>> import statsmodels.formula.api as smf
    >>> fam = sm.families.Poisson()
    >>> ind = sm.cov_struct.Independence()
    >>> model = smf.gee("y ~ age + trt + base", "subject", \
                    data, cov_struct=ind, family=fam)
    >>> result = model.fit()
    >>> print(result.summary())
"""

_gee_ordinal_example = """
    Fit an ordinal regression model using GEE, with "global
    odds ratio" dependence:

    >>> import statsmodels.api as sm
    >>> gor = sm.cov_struct.GlobalOddsRatio("ordinal")
    >>> model = sm.OrdinalGEE(endog, exog, groups, cov_struct=gor)
    >>> result = model.fit()
    >>> print(result.summary())

    Using formulas:

    >>> import statsmodels.formula.api as smf
    >>> model = smf.ordinal_gee("y ~ x1 + x2", groups, data,
                                    cov_struct=gor)
    >>> result = model.fit()
    >>> print(result.summary())
"""

_gee_nominal_example = """
    Fit a nominal regression model using GEE:

    >>> import statsmodels.api as sm
    >>> import statsmodels.formula.api as smf
    >>> gor = sm.cov_struct.GlobalOddsRatio("nominal")
    >>> model = sm.NominalGEE(endog, exog, groups, cov_struct=gor)
    >>> result = model.fit()
    >>> print(result.summary())

    Using formulas:

    >>> import statsmodels.api as sm
    >>> model = sm.NominalGEE.from_formula("y ~ x1 + x2", groups,
                     data, cov_struct=gor)
    >>> result = model.fit()
    >>> print(result.summary())

    Using the formula API:

    >>> import statsmodels.formula.api as smf
    >>> model = smf.nominal_gee("y ~ x1 + x2", groups, data,
                                cov_struct=gor)
    >>> result = model.fit()
    >>> print(result.summary())
"""


class GEE(base.Model):

    __doc__ = (
        "    Estimation of marginal regression models using Generalized\n"
        "    Estimating Equations (GEE).\n" + _gee_init_doc %
        {'extra_params': base._missing_param_doc,
         'family_doc': _gee_family_doc,
         'example': _gee_example})

    cached_means = None

    def __init__(self, endog, exog, groups, time=None, family=None,
                 cov_struct=None, missing='none', offset=None,
                 exposure=None, dep_data=None, constraint=None,
                 update_dep=True, weights=None, **kwargs):

        if family is not None:
            if not isinstance(family.link, tuple(family.safe_links)):
                import warnings
                msg = ("The {0} link function does not respect the "
                       "domain of the {1} family.")
                warnings.warn(msg.format(family.link.__class__.__name__,
                                         family.__class__.__name__),
                              DomainWarning)

        self.missing = missing
        self.dep_data = dep_data
        self.constraint = constraint
        self.update_dep = update_dep

        groups = np.array(groups)  # in case groups is pandas
        # Pass groups, time, offset, and dep_data so they are
        # processed for missing data along with endog and exog.
        # Calling super creates self.exog, self.endog, etc. as
        # ndarrays and the original exog, endog, etc. are
        # self.data.endog, etc.
        super(GEE, self).__init__(endog, exog, groups=groups,
                                  time=time, offset=offset,
                                  exposure=exposure, weights=weights,
                                  dep_data=dep_data, missing=missing,
                                  **kwargs)

        self._init_keys.extend(["update_dep", "constraint", "family",
                                "cov_struct"])

        # Handle the family argument
        if family is None:
            family = families.Gaussian()
        else:
            if not issubclass(family.__class__, families.Family):
                raise ValueError("GEE: `family` must be a genmod "
                                 "family instance")
        self.family = family

        # Handle the cov_struct argument
        if cov_struct is None:
            cov_struct = cov_structs.Independence()
        else:
            if not issubclass(cov_struct.__class__, cov_structs.CovStruct):
                raise ValueError("GEE: `cov_struct` must be a genmod "
                                 "cov_struct instance")

        self.cov_struct = cov_struct

        # Handle the offset and exposure
        self._offset_exposure = None
        if offset is not None:
            self._offset_exposure = self.offset.copy()
            self.offset = offset
        if exposure is not None:
            if not isinstance(self.family.link, families.links.Log):
                raise ValueError(
                    "exposure can only be used with the log link function")
            if self._offset_exposure is not None:
                self._offset_exposure += np.log(exposure)
            else:
                self._offset_exposure = np.log(exposure)
            self.exposure = exposure

        # Handle the constraint
        self.constraint = None
        if constraint is not None:
            if len(constraint) != 2:
                raise ValueError("GEE: `constraint` must be a 2-tuple.")
            if constraint[0].shape[1] != self.exog.shape[1]:
                raise ValueError(
                    "GEE: the left hand side of the constraint must have "
                    "the same number of columns as the exog matrix.")
            self.constraint = ParameterConstraint(constraint[0],
                                                  constraint[1],
                                                  self.exog)

            if self._offset_exposure is not None:
                self._offset_exposure += self.constraint.offset_increment()
            else:
                self._offset_exposure = (
                    self.constraint.offset_increment().copy())
            self.exog = self.constraint.reduced_exog()

        # Create list of row indices for each group
        group_labels, ix = np.unique(self.groups, return_inverse=True)
        se = pd.Series(index=np.arange(len(ix)))
        gb = se.groupby(ix).groups
        dk = [(lb, np.asarray(gb[k])) for k, lb in enumerate(group_labels)]
        self.group_indices = dict(dk)
        self.group_labels = group_labels

        # Convert the data to the internal representation, which is a
        # list of arrays, corresponding to the groups.
        self.endog_li = self.cluster_list(self.endog)
        self.exog_li = self.cluster_list(self.exog)

        if self.weights is not None:
            self.weights_li = self.cluster_list(self.weights)
            self.weights_li = [x[0] for x in self.weights_li]
            self.weights_li = np.asarray(self.weights_li)

        self.num_group = len(self.endog_li)

        # Time defaults to a 1d grid with equal spacing
        if self.time is not None:
            self.time = np.asarray(self.time, np.float64)
            if self.time.ndim == 1:
                self.time = self.time[:, None]
            self.time_li = self.cluster_list(self.time)
        else:
            self.time_li = \
                [np.arange(len(y), dtype=np.float64)[:, None]
                 for y in self.endog_li]
            self.time = np.concatenate(self.time_li)

        if self._offset_exposure is not None:
            self.offset_li = self.cluster_list(self._offset_exposure)
        else:
            self.offset_li = None
        if constraint is not None:
            self.constraint.exog_fulltrans_li = \
                self.cluster_list(self.constraint.exog_fulltrans)

        self.family = family

        self.cov_struct.initialize(self)

        # Total sample size
        group_ns = [len(y) for y in self.endog_li]
        self.nobs = sum(group_ns)
        # The following are column based, not on rank see #1928
        self.df_model = self.exog.shape[1] - 1  # assumes constant
        self.df_resid = self.nobs - self.exog.shape[1]

        # Skip the covariance updates if all groups have a single
        # observation (reduces to fitting a GLM).
        maxgroup = max([len(x) for x in self.endog_li])
        if maxgroup == 1:
            self.update_dep = False

    # Override to allow groups and time to be passed as variable
    # names.
    @classmethod
    def from_formula(cls, formula, groups, data, subset=None,
                     time=None, offset=None, exposure=None,
                     *args, **kwargs):
        """
        Create a GEE model instance from a formula and dataframe.

        Parameters
        ----------
        formula : str or generic Formula object
            The formula specifying the model
        groups : array-like or string
            Array of grouping labels.  If a string, this is the name
            of a variable in `data` that contains the grouping labels.
        data : array-like
            The data for the model.
        subset : array-like
            An array-like object of booleans, integers, or index
            values that indicate the subset of the data to used when
            fitting the model.
        time : array-like or string
            The time values, used for dependence structures involving
            distances between observations.  If a string, this is the
            name of a variable in `data` that contains the time
            values.
        offset : array-like or string
            The offset values, added to the linear predictor.  If a
            string, this is the name of a variable in `data` that
            contains the offset values.
        exposure : array-like or string
            The exposure values, only used if the link function is the
            logarithm function, in which case the log of `exposure`
            is added to the offset (if any).  If a string, this is the
            name of a variable in `data` that contains the offset
            values.
        %(missing_param_doc)s
        args : extra arguments
            These are passed to the model
        kwargs : extra keyword arguments
            These are passed to the model with one exception. The
            ``eval_env`` keyword is passed to patsy. It can be either a
            :class:`patsy:patsy.EvalEnvironment` object or an integer
            indicating the depth of the namespace to use. For example, the
            default ``eval_env=0`` uses the calling namespace. If you wish
            to use a "clean" environment set ``eval_env=-1``.

        Returns
        -------
        model : GEE model instance

        Notes
        ------
        `data` must define __getitem__ with the keys in the formula
        terms args and kwargs are passed on to the model
        instantiation. E.g., a numpy structured or rec array, a
        dictionary, or a pandas DataFrame.

        This method currently does not correctly handle missing
        values, so missing values should be explicitly dropped from
        the DataFrame before calling this method.
        """ % {'missing_param_doc': base._missing_param_doc}

        if type(groups) == str:
            groups = data[groups]

        if type(time) == str:
            time = data[time]

        if type(offset) == str:
            offset = data[offset]

        if type(exposure) == str:
            exposure = data[exposure]

        model = super(GEE, cls).from_formula(formula, data=data, subset=subset,
                                             groups=groups, time=time,
                                             offset=offset,
                                             exposure=exposure,
                                             *args, **kwargs)

        return model

    def cluster_list(self, array):
        """
        Returns `array` split into subarrays corresponding to the
        cluster structure.
        """

        if array.ndim == 1:
            return [np.array(array[self.group_indices[k]])
                    for k in self.group_labels]
        else:
            return [np.array(array[self.group_indices[k], :])
                    for k in self.group_labels]

    def estimate_scale(self):
        """
        Returns an estimate of the scale parameter at the current
        parameter value.
        """

        if isinstance(self.family, (families.Binomial, families.Poisson,
                                    _Multinomial)):
            return 1.

        endog = self.endog_li
        cached_means = self.cached_means
        nobs = self.nobs
        varfunc = self.family.variance

        scale = 0.
        fsum = 0.
        for i in range(self.num_group):

            if len(endog[i]) == 0:
                continue

            expval, _ = cached_means[i]

            f = self.weights_li[i] if self.weights is not None else 1.

            sdev = np.sqrt(varfunc(expval))
            resid = (endog[i] - expval) / sdev

            scale += f * np.sum(resid ** 2)
            fsum += f * len(endog[i])

        scale /= (fsum * (nobs - self.ddof_scale) / float(nobs))

        return scale

    def mean_deriv(self, exog, lin_pred):
        """
        Derivative of the expected endog with respect to the parameters.

        Parameters
        ----------
        exog : array-like
           The exogeneous data at which the derivative is computed.
        lin_pred : array-like
           The values of the linear predictor.

        Returns
        -------
        The value of the derivative of the expected endog with respect
        to the parameter vector.

        Notes
        -----
        If there is an offset or exposure, it should be added to
        `lin_pred` prior to calling this function.
        """

        idl = self.family.link.inverse_deriv(lin_pred)
        dmat = exog * idl[:, None]
        return dmat

    def mean_deriv_exog(self, exog, params, offset_exposure=None):
        """
        Derivative of the expected endog with respect to exog.

        Parameters
        ----------
        exog : array-like
            Values of the independent variables at which the derivative
            is calculated.
        params : array-like
            Parameter values at which the derivative is calculated.
        offset_exposure : array-like, optional
            Combined offset and exposure.

        Returns
        -------
        The derivative of the expected endog with respect to exog.
        """

        lin_pred = np.dot(exog, params)
        if offset_exposure is not None:
            lin_pred += offset_exposure

        idl = self.family.link.inverse_deriv(lin_pred)
        dmat = np.outer(idl, params)
        return dmat

    def _update_mean_params(self):
        """
        Returns
        -------
        update : array-like
            The update vector such that params + update is the next
            iterate when solving the score equations.
        score : array-like
            The current value of the score equations, not
            incorporating the scale parameter.  If desired,
            multiply this vector by the scale parameter to
            incorporate the scale.
        """

        endog = self.endog_li
        exog = self.exog_li

        cached_means = self.cached_means

        varfunc = self.family.variance

        bmat, score = 0, 0
        for i in range(self.num_group):

            expval, lpr = cached_means[i]
            resid = endog[i] - expval
            dmat = self.mean_deriv(exog[i], lpr)
            sdev = np.sqrt(varfunc(expval))

            rslt = self.cov_struct.covariance_matrix_solve(expval, i,
                                                           sdev, (dmat, resid))
            if rslt is None:
                return None, None
            vinv_d, vinv_resid = tuple(rslt)

            f = self.weights_li[i] if self.weights is not None else 1.

            bmat += f * np.dot(dmat.T, vinv_d)
            score += f * np.dot(dmat.T, vinv_resid)

        update = np.linalg.solve(bmat, score)

        self._fit_history["cov_adjust"].append(
            self.cov_struct.cov_adjust)

        return update, score

    def update_cached_means(self, mean_params):
        """
        cached_means should always contain the most recent calculation
        of the group-wise mean vectors.  This function should be
        called every time the regression parameters are changed, to
        keep the cached means up to date.
        """

        endog = self.endog_li
        exog = self.exog_li
        offset = self.offset_li

        linkinv = self.family.link.inverse

        self.cached_means = []

        for i in range(self.num_group):

            if len(endog[i]) == 0:
                continue

            lpr = np.dot(exog[i], mean_params)
            if offset is not None:
                lpr += offset[i]
            expval = linkinv(lpr)

            self.cached_means.append((expval, lpr))

    def _covmat(self):
        """
        Returns the sampling covariance matrix of the regression
        parameters and related quantities.

        Returns
        -------
        cov_robust : array-like
           The robust, or sandwich estimate of the covariance, which
           is meaningful even if the working covariance structure is
           incorrectly specified.
        cov_naive : array-like
           The model-based estimate of the covariance, which is
           meaningful if the covariance structure is correctly
           specified.
        cmat : array-like
           The center matrix of the sandwich expression, used in
           obtaining score test results.
        """

        endog = self.endog_li
        exog = self.exog_li
        varfunc = self.family.variance
        cached_means = self.cached_means

        # Calculate the naive (model-based) and robust (sandwich)
        # covariances.
        bmat, cmat = 0, 0
        for i in range(self.num_group):

            expval, lpr = cached_means[i]
            resid = endog[i] - expval
            dmat = self.mean_deriv(exog[i], lpr)
            sdev = np.sqrt(varfunc(expval))

            rslt = self.cov_struct.covariance_matrix_solve(
                expval, i, sdev, (dmat, resid))
            if rslt is None:
                return None, None, None, None
            vinv_d, vinv_resid = tuple(rslt)

            f = self.weights_li[i] if self.weights is not None else 1.

            bmat += f * np.dot(dmat.T, vinv_d)
            dvinv_resid = f * np.dot(dmat.T, vinv_resid)
            cmat += np.outer(dvinv_resid, dvinv_resid)

        scale = self.estimate_scale()

        bmati = np.linalg.inv(bmat)
        cov_naive = bmati * scale
        cov_robust = np.dot(bmati, np.dot(cmat, bmati))

        cov_naive *= self.scaling_factor
        cov_robust *= self.scaling_factor
        return cov_robust, cov_naive, cmat

    # Calculate the bias-corrected sandwich estimate of Mancl and
    # DeRouen.
    def _bc_covmat(self, cov_naive):

        cov_naive = cov_naive / self.scaling_factor
        endog = self.endog_li
        exog = self.exog_li
        varfunc = self.family.variance
        cached_means = self.cached_means
        scale = self.estimate_scale()

        bcm = 0
        for i in range(self.num_group):

            expval, lpr = cached_means[i]
            resid = endog[i] - expval
            dmat = self.mean_deriv(exog[i], lpr)
            sdev = np.sqrt(varfunc(expval))

            rslt = self.cov_struct.covariance_matrix_solve(
                expval, i, sdev, (dmat,))
            if rslt is None:
                return None
            vinv_d = rslt[0]
            vinv_d /= scale

            hmat = np.dot(vinv_d, cov_naive)
            hmat = np.dot(hmat, dmat.T).T

            f = self.weights_li[i] if self.weights is not None else 1.

            aresid = np.linalg.solve(np.eye(len(resid)) - hmat, resid)
            rslt = self.cov_struct.covariance_matrix_solve(
                expval, i, sdev, (aresid,))
            if rslt is None:
                return None
            srt = rslt[0]
            srt = f * np.dot(dmat.T, srt) / scale
            bcm += np.outer(srt, srt)

        cov_robust_bc = np.dot(cov_naive, np.dot(bcm, cov_naive))
        cov_robust_bc *= self.scaling_factor

        return cov_robust_bc

    def predict(self, params, exog=None, offset=None,
                exposure=None, linear=False):
        """
        Return predicted values for a marginal regression model fit
        using GEE.

        Parameters
        ----------
        params : array-like
            Parameters / coefficients of a marginal regression model.
        exog : array-like, optional
            Design / exogenous data. If exog is None, model exog is
            used.
        offset : array-like, optional
            Offset for exog if provided.  If offset is None, model
            offset is used.
        exposure : array-like, optional
            Exposure for exog, if exposure is None, model exposure is
            used.  Only allowed if link function is the logarithm.
        linear : bool
            If True, returns the linear predicted values.  If False,
            returns the value of the inverse of the model's link
            function at the linear predicted values.

        Returns
        -------
        An array of fitted values

        Notes
        -----
        Using log(V) as the offset is equivalent to using V as the
        exposure.  If exposure U and offset V are both provided, then
        log(U) + V is added to the linear predictor.
        """

        # TODO: many paths through this, not well covered in tests

        if exposure is not None:
            if not isinstance(self.family.link, families.links.Log):
                raise ValueError(
                    "exposure can only be used with the log link function")

        # This is the combined offset and exposure
        _offset = 0.

        # Using model exog
        if exog is None:
            exog = self.exog

            if not isinstance(self.family.link, families.links.Log):
                # Don't need to worry about exposure
                if offset is None:
                    if self._offset_exposure is not None:
                        _offset = self._offset_exposure.copy()
                else:
                    _offset = offset

            else:
                if offset is None and exposure is None:
                    if self._offset_exposure is not None:
                        _offset = self._offset_exposure
                elif offset is None and exposure is not None:
                    _offset = np.log(exposure)
                    if hasattr(self, "offset"):
                        _offset = _offset + self.offset
                elif offset is not None and exposure is None:
                    _offset = offset
                    if hasattr(self, "exposure"):
                        _offset = offset + np.log(self.exposure)
                else:
                    _offset = offset + np.log(exposure)

        # exog is provided: this is simpler than above because we
        # never use model exog or exposure if exog is provided.
        else:
            if offset is not None:
                _offset = _offset + offset
            if exposure is not None:
                _offset += np.log(exposure)

        lin_pred = _offset + np.dot(exog, params)

        if not linear:
            return self.family.link.inverse(lin_pred)

        return lin_pred

    def _starting_params(self):

        # TODO: use GLM to get Poisson starting values
        return np.zeros(self.exog.shape[1])

    def fit(self, maxiter=60, ctol=1e-6, start_params=None,
            params_niter=1, first_dep_update=0,
            cov_type='robust', ddof_scale=None, scaling_factor=1.):
        # Docstring attached below

        # Subtract this number from the total sample size when
        # normalizing the scale parameter estimate.
        if ddof_scale is None:
            self.ddof_scale = self.exog.shape[1]
        else:
            if not ddof_scale >= 0:
                raise ValueError(
                    "ddof_scale must be a non-negative number or None")
            self.ddof_scale = ddof_scale

        self.scaling_factor = scaling_factor

        self._fit_history = {'params': [],
                             'score': [],
                             'dep_params': [],
                             'cov_adjust': []}

        if self.weights is not None and cov_type == 'naive':
            raise ValueError("when using weights, cov_type may not be naive")

        if start_params is None:
            mean_params = self._starting_params()
        else:
            start_params = np.asarray(start_params)
            mean_params = start_params.copy()

        self.update_cached_means(mean_params)

        del_params = -1.
        num_assoc_updates = 0
        for itr in range(maxiter):

            update, score = self._update_mean_params()
            if update is None:
                warnings.warn("Singular matrix encountered in GEE update",
                              ConvergenceWarning)
                break
            mean_params += update
            self.update_cached_means(mean_params)

            # L2 norm of the change in mean structure parameters at
            # this iteration.
            del_params = np.sqrt(np.sum(score ** 2))

            self._fit_history['params'].append(mean_params.copy())
            self._fit_history['score'].append(score)
            self._fit_history['dep_params'].append(
                self.cov_struct.dep_params)

            # Don't exit until the association parameters have been
            # updated at least once.
            if (del_params < ctol and
                    (num_assoc_updates > 0 or self.update_dep is False)):
                break

            # Update the dependence structure
            if (self.update_dep and (itr % params_niter) == 0
                    and (itr >= first_dep_update)):
                self._update_assoc(mean_params)
                num_assoc_updates += 1

        if del_params >= ctol:
            warnings.warn("Iteration limit reached prior to convergence",
                          IterationLimitWarning)

        if mean_params is None:
            warnings.warn("Unable to estimate GEE parameters.",
                          ConvergenceWarning)
            return None

        bcov, ncov, _ = self._covmat()
        if bcov is None:
            warnings.warn("Estimated covariance structure for GEE "
                          "estimates is singular", ConvergenceWarning)
            return None
        bc_cov = None
        if cov_type == "bias_reduced":
            bc_cov = self._bc_covmat(ncov)

        if self.constraint is not None:
            x = mean_params.copy()
            mean_params, bcov = self._handle_constraint(mean_params, bcov)
            if mean_params is None:
                warnings.warn("Unable to estimate constrained GEE "
                              "parameters.", ConvergenceWarning)
                return None

            y, ncov = self._handle_constraint(x, ncov)
            if y is None:
                warnings.warn("Unable to estimate constrained GEE "
                              "parameters.", ConvergenceWarning)
                return None

            if bc_cov is not None:
                y, bc_cov = self._handle_constraint(x, bc_cov)
                if x is None:
                    warnings.warn("Unable to estimate constrained GEE "
                                  "parameters.", ConvergenceWarning)
                    return None

        scale = self.estimate_scale()

        # kwargs to add to results instance, need to be available in __init__
        res_kwds = dict(cov_type=cov_type,
                        cov_robust=bcov,
                        cov_naive=ncov,
                        cov_robust_bc=bc_cov)

        # The superclass constructor will multiply the covariance
        # matrix argument bcov by scale, which we don't want, so we
        # divide bcov by the scale parameter here
        results = GEEResults(self, mean_params, bcov / scale, scale,
                             cov_type=cov_type, use_t=False,
                             attr_kwds=res_kwds)

        # attributes not needed during results__init__
        results.fit_history = self._fit_history
        delattr(self, "_fit_history")
        results.score_norm = del_params
        results.converged = (del_params < ctol)
        results.cov_struct = self.cov_struct
        results.params_niter = params_niter
        results.first_dep_update = first_dep_update
        results.ctol = ctol
        results.maxiter = maxiter

        # These will be copied over to subclasses when upgrading.
        results._props = ["cov_type", "use_t",
                          "cov_params_default", "cov_robust",
                          "cov_naive", "cov_robust_bc",
                          "fit_history",
                          "score_norm", "converged", "cov_struct",
                          "params_niter", "first_dep_update", "ctol",
                          "maxiter"]

        return GEEResultsWrapper(results)

    fit.__doc__ = _gee_fit_doc

    def _handle_constraint(self, mean_params, bcov):
        """
        Expand the parameter estimate `mean_params` and covariance matrix
        `bcov` to the coordinate system of the unconstrained model.

        Parameters
        ----------
        mean_params : array-like
            A parameter vector estimate for the reduced model.
        bcov : array-like
            The covariance matrix of mean_params.

        Returns
        -------
        mean_params : array-like
            The input parameter vector mean_params, expanded to the
            coordinate system of the full model
        bcov : array-like
            The input covariance matrix bcov, expanded to the
            coordinate system of the full model
        """

        # The number of variables in the full model
        red_p = len(mean_params)
        full_p = self.constraint.lhs.shape[1]
        mean_params0 = np.r_[mean_params, np.zeros(full_p - red_p)]

        # Get the score vector under the full model.
        save_exog_li = self.exog_li
        self.exog_li = self.constraint.exog_fulltrans_li
        import copy
        save_cached_means = copy.deepcopy(self.cached_means)
        self.update_cached_means(mean_params0)
        _, score = self._update_mean_params()

        if score is None:
            warnings.warn("Singular matrix encountered in GEE score test",
                          ConvergenceWarning)
            return None, None

        _, ncov1, cmat = self._covmat()
        scale = self.estimate_scale()
        cmat = cmat / scale ** 2
        score2 = score[red_p:] / scale

        amat = np.linalg.inv(ncov1)

        bmat_11 = cmat[0:red_p, 0:red_p]
        bmat_22 = cmat[red_p:, red_p:]
        bmat_12 = cmat[0:red_p, red_p:]
        amat_11 = amat[0:red_p, 0:red_p]
        amat_12 = amat[0:red_p, red_p:]

        score_cov = bmat_22 - np.dot(amat_12.T,
                                     np.linalg.solve(amat_11, bmat_12))
        score_cov -= np.dot(bmat_12.T,
                            np.linalg.solve(amat_11, amat_12))
        score_cov += np.dot(amat_12.T,
                            np.dot(np.linalg.solve(amat_11, bmat_11),
                                   np.linalg.solve(amat_11, amat_12)))

        from scipy.stats.distributions import chi2
        score_statistic = np.dot(score2,
                                 np.linalg.solve(score_cov, score2))
        score_df = len(score2)
        score_pvalue = 1 - chi2.cdf(score_statistic, score_df)
        self.score_test_results = {"statistic": score_statistic,
                                   "df": score_df,
                                   "p-value": score_pvalue}

        mean_params = self.constraint.unpack_param(mean_params)
        bcov = self.constraint.unpack_cov(bcov)

        self.exog_li = save_exog_li
        self.cached_means = save_cached_means
        self.exog = self.constraint.restore_exog()

        return mean_params, bcov

    def _update_assoc(self, params):
        """
        Update the association parameters
        """

        self.cov_struct.update(params)

    def _derivative_exog(self, params, exog=None, transform='dydx',
                         dummy_idx=None, count_idx=None):
        """
        For computing marginal effects, returns dF(XB) / dX where F(.)
        is the fitted mean.

        transform can be 'dydx', 'dyex', 'eydx', or 'eyex'.

        Not all of these make sense in the presence of discrete regressors,
        but checks are done in the results in get_margeff.
        """
        # This form should be appropriate for group 1 probit, logit,
        # logistic, cloglog, heckprob, xtprobit.
        offset_exposure = None
        if exog is None:
            exog = self.exog
            offset_exposure = self._offset_exposure

        margeff = self.mean_deriv_exog(exog, params, offset_exposure)

        if 'ex' in transform:
            margeff *= exog
        if 'ey' in transform:
            margeff /= self.predict(params, exog)[:, None]
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

    __doc__ = (
        "This class summarizes the fit of a marginal regression model "
        "using GEE.\n" + _gee_results_doc)

    def __init__(self, model, params, cov_params, scale,
                 cov_type='robust', use_t=False, **kwds):

        super(GEEResults, self).__init__(
            model, params, normalized_cov_params=cov_params,
            scale=scale)

        # not added by super
        self.df_resid = model.df_resid
        self.df_model = model.df_model
        self.family = model.family

        attr_kwds = kwds.pop('attr_kwds', {})
        self.__dict__.update(attr_kwds)

        # we don't do this if the cov_type has already been set
        # subclasses can set it through attr_kwds
        if not (hasattr(self, 'cov_type') and
                hasattr(self, 'cov_params_default')):
            self.cov_type = cov_type  # keep alias
            covariance_type = self.cov_type.lower()
            allowed_covariances = ["robust", "naive", "bias_reduced"]
            if covariance_type not in allowed_covariances:
                msg = ("GEE: `cov_type` must be one of " +
                       ", ".join(allowed_covariances))
                raise ValueError(msg)

            if cov_type == "robust":
                cov = self.cov_robust
            elif cov_type == "naive":
                cov = self.cov_naive
            elif cov_type == "bias_reduced":
                cov = self.cov_robust_bc

            self.cov_params_default = cov
        else:
            if self.cov_type != cov_type:
                raise ValueError('cov_type in argument is different from '
                                 'already attached cov_type')

    def standard_errors(self, cov_type="robust"):
        """
        This is a convenience function that returns the standard
        errors for any covariance type.  The value of `bse` is the
        standard errors for whichever covariance type is specified as
        an argument to `fit` (defaults to "robust").

        Parameters
        ----------
        cov_type : string
            One of "robust", "naive", or "bias_reduced".  Determines
            the covariance used to compute standard errors.  Defaults
            to "robust".
        """

        # Check covariance_type
        covariance_type = cov_type.lower()
        allowed_covariances = ["robust", "naive", "bias_reduced"]
        if covariance_type not in allowed_covariances:
            msg = ("GEE: `covariance_type` must be one of " +
                   ", ".join(allowed_covariances))
            raise ValueError(msg)

        if covariance_type == "robust":
            return np.sqrt(np.diag(self.cov_robust))
        elif covariance_type == "naive":
            return np.sqrt(np.diag(self.cov_naive))
        elif covariance_type == "bias_reduced":
            if self.cov_robust_bc is None:
                raise ValueError(
                    "GEE: `bias_reduced` covariance not available")
            return np.sqrt(np.diag(self.cov_robust_bc))

    # Need to override to allow for different covariance types.
    @cache_readonly
    def bse(self):
        return self.standard_errors(self.cov_type)

    @cache_readonly
    def resid(self):
        """
        Returns the residuals, the endogeneous data minus the fitted
        values from the model.
        """
        return self.model.endog - self.fittedvalues

    @cache_readonly
    def resid_split(self):
        """
        Returns the residuals, the endogeneous data minus the fitted
        values from the model.  The residuals are returned as a list
        of arrays containing the residuals for each cluster.
        """
        sresid = []
        for v in self.model.group_labels:
            ii = self.model.group_indices[v]
            sresid.append(self.resid[ii])
        return sresid

    @cache_readonly
    def resid_centered(self):
        """
        Returns the residuals centered within each group.
        """
        cresid = self.resid.copy()
        for v in self.model.group_labels:
            ii = self.model.group_indices[v]
            cresid[ii] -= cresid[ii].mean()
        return cresid

    @cache_readonly
    def resid_centered_split(self):
        """
        Returns the residuals centered within each group.  The
        residuals are returned as a list of arrays containing the
        centered residuals for each cluster.
        """
        sresid = []
        for v in self.model.group_labels:
            ii = self.model.group_indices[v]
            sresid.append(self.centered_resid[ii])
        return sresid

    # FIXME: alias to be removed, temporary backwards compatibility
    split_resid = resid_split
    centered_resid = resid_centered
    split_centered_resid = resid_centered_split

    @cache_readonly
    def resid_response(self):
        return self.model.endog - self.fittedvalues

    @cache_readonly
    def resid_pearson(self):
        val = self.model.endog - self.fittedvalues
        val = val / np.sqrt(self.family.variance(self.fittedvalues))
        return val

    @cache_readonly
    def resid_working(self):
        val = self.resid_response
        val = val / self.family.link.deriv(self.fittedvalues)
        return val

    @cache_readonly
    def resid_anscombe(self):
        return self.family.resid_anscombe(self.model.endog, self.fittedvalues)

    @cache_readonly
    def resid_deviance(self):
        return self.family.resid_dev(self.model.endog, self.fittedvalues)

    @cache_readonly
    def fittedvalues(self):
        """
        Returns the fitted values from the model.
        """
        return self.model.family.link.inverse(np.dot(self.model.exog,
                                                     self.params))

    def plot_added_variable(self, focus_exog, resid_type=None,
                            use_glm_weights=True, fit_kwargs=None,
                            ax=None):
        # Docstring attached below

        from statsmodels.graphics.regressionplots import plot_added_variable

        fig = plot_added_variable(self, focus_exog,
                                  resid_type=resid_type,
                                  use_glm_weights=use_glm_weights,
                                  fit_kwargs=fit_kwargs, ax=ax)

        return fig

    plot_added_variable.__doc__ = _plot_added_variable_doc % {
        'extra_params_doc': ''}

    def plot_partial_residuals(self, focus_exog, ax=None):
        # Docstring attached below

        from statsmodels.graphics.regressionplots import plot_partial_residuals

        return plot_partial_residuals(self, focus_exog, ax=ax)

    plot_partial_residuals.__doc__ = _plot_partial_residuals_doc % {
        'extra_params_doc': ''}

    def plot_ceres_residuals(self, focus_exog, frac=0.66, cond_means=None,
                             ax=None):
        # Docstring attached below

        from statsmodels.graphics.regressionplots import plot_ceres_residuals

        return plot_ceres_residuals(self, focus_exog, frac,
                                    cond_means=cond_means, ax=ax)

    plot_ceres_residuals.__doc__ = _plot_ceres_residuals_doc % {
        'extra_params_doc': ''}

    def conf_int(self, alpha=.05, cols=None, cov_type=None):
        """
        Returns confidence intervals for the fitted parameters.

        Parameters
        ----------
        alpha : float, optional
             The `alpha` level for the confidence interval.  i.e., The
             default `alpha` = .05 returns a 95% confidence interval.
        cols : array-like, optional
             `cols` specifies which confidence intervals to return
        cov_type : string
             The covariance type used for computing standard errors;
             must be one of 'robust', 'naive', and 'bias reduced'.
             See `GEE` for details.

        Notes
        -----
        The confidence interval is based on the Gaussian distribution.
        """
        # super doesn't allow to specify cov_type and method is not
        # implemented,
        # FIXME: remove this method here
        if cov_type is None:
            bse = self.bse
        else:
            bse = self.standard_errors(cov_type=cov_type)
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
        return np.asarray(lzip(lower, upper))

    def summary(self, yname=None, xname=None, title=None, alpha=.05):
        """
        Summarize the GEE regression results

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
        cov_type : string
            The covariance type used to compute the standard errors;
            one of 'robust' (the usual robust sandwich-type covariance
            estimate), 'naive' (ignores dependence), and 'bias
            reduced' (the Mancl/DeRouen estimate).

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
                    ('Method:', ['Generalized']),
                    ('', ['Estimating Equations']),
                    ('Family:', [self.model.family.__class__.__name__]),
                    ('Dependence structure:',
                     [self.model.cov_struct.__class__.__name__]),
                    ('Date:', None),
                    ('Covariance type: ', [self.cov_type, ])
                    ]

        NY = [len(y) for y in self.model.endog_li]

        top_right = [('No. Observations:', [sum(NY)]),
                     ('No. clusters:', [len(self.model.endog_li)]),
                     ('Min. cluster size:', [min(NY)]),
                     ('Max. cluster size:', [max(NY)]),
                     ('Mean cluster size:', ["%.1f" % np.mean(NY)]),
                     ('Num. iterations:', ['%d' %
                                           len(self.fit_history['params'])]),
                     ('Scale:', ["%.3f" % self.scale]),
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

        # Override the dataframe names if xname is provided as an
        # argument.
        if xname is not None:
            xna = xname
        else:
            xna = self.model.exog_names

        # Create summary table instance
        from statsmodels.iolib.summary import Summary
        smry = Summary()
        smry.add_table_2cols(self, gleft=top_left, gright=top_right,
                             yname=self.model.endog_names, xname=xna,
                             title=title)
        smry.add_table_params(self, yname=yname, xname=xna,
                              alpha=alpha, use_t=False)
        smry.add_table_2cols(self, gleft=diagn_left,
                             gright=diagn_right, yname=yname,
                             xname=xna, title="")

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
            - 'all', The marginal effects at each observation. If `at` is 'all'
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

        if self.model.constraint is not None:
            warnings.warn("marginal effects ignore constraints",
                          ValueWarning)

        return GEEMargins(self, (at, method, atexog, dummy, count))

    def plot_isotropic_dependence(self, ax=None, xpoints=10,
                                  min_n=50):
        """
        Create a plot of the pairwise products of within-group
        residuals against the corresponding time differences.  This
        plot can be used to assess the possible form of an isotropic
        covariance structure.

        Parameters
        ----------
        ax : Matplotlib axes instance
            An axes on which to draw the graph.  If None, new
            figure and axes objects are created
        xpoints : scalar or array-like
            If scalar, the number of points equally spaced points on
            the time difference axis used to define bins for
            calculating local means.  If an array, the specific points
            that define the bins.
        min_n : integer
            The minimum sample size in a bin for the mean residual
            product to be included on the plot.
        """

        from statsmodels.graphics import utils as gutils

        resid = self.model.cluster_list(self.resid)
        time = self.model.cluster_list(self.model.time)

        # All within-group pairwise time distances (xdt) and the
        # corresponding products of scaled residuals (xre).
        xre, xdt = [], []
        for re, ti in zip(resid, time):
            ix = np.tril_indices(re.shape[0], 0)
            re = re[ix[0]] * re[ix[1]] / self.scale ** 2
            xre.append(re)
            dists = np.sqrt(((ti[ix[0], :] - ti[ix[1], :]) ** 2).sum(1))
            xdt.append(dists)

        xre = np.concatenate(xre)
        xdt = np.concatenate(xdt)

        if ax is None:
            fig, ax = gutils.create_mpl_ax(ax)
        else:
            fig = ax.get_figure()

        # Convert to a correlation
        ii = np.flatnonzero(xdt == 0)
        v0 = np.mean(xre[ii])
        xre /= v0

        # Use the simple average to smooth, since fancier smoothers
        # that trim and downweight outliers give biased results (we
        # need the actual mean of a skewed distribution).
        if np.isscalar(xpoints):
            xpoints = np.linspace(0, max(xdt), xpoints)
        dg = np.digitize(xdt, xpoints)
        dgu = np.unique(dg)
        hist = np.asarray([np.sum(dg == k) for k in dgu])
        ii = np.flatnonzero(hist >= min_n)
        dgu = dgu[ii]
        dgy = np.asarray([np.mean(xre[dg == k]) for k in dgu])
        dgx = np.asarray([np.mean(xdt[dg == k]) for k in dgu])

        ax.plot(dgx, dgy, '-', color='orange', lw=5)
        ax.set_xlabel("Time difference")
        ax.set_ylabel("Product of scaled residuals")

        return fig

    def sensitivity_params(self, dep_params_first,
                           dep_params_last, num_steps):
        """
        Refits the GEE model using a sequence of values for the
        dependence parameters.

        Parameters
        ----------
        dep_params_first : array-like
            The first dep_params in the sequence
        dep_params_last : array-like
            The last dep_params in the sequence
        num_steps : int
            The number of dep_params in the sequence

        Returns
        -------
        results : array-like
            The GEEResults objects resulting from the fits.
        """

        model = self.model

        import copy
        cov_struct = copy.deepcopy(self.model.cov_struct)

        # We are fixing the dependence structure in each run.
        update_dep = model.update_dep
        model.update_dep = False

        dep_params = []
        results = []
        for x in np.linspace(0, 1, num_steps):

            dp = x * dep_params_last + (1 - x) * dep_params_first
            dep_params.append(dp)

            model.cov_struct = copy.deepcopy(cov_struct)
            model.cov_struct.dep_params = dp
            rslt = model.fit(start_params=self.params,
                             ctol=self.ctol,
                             params_niter=self.params_niter,
                             first_dep_update=self.first_dep_update,
                             cov_type=self.cov_type)
            results.append(rslt)

        model.update_dep = update_dep

        return results

    # FIXME: alias to be removed, temporary backwards compatibility
    params_sensitivity = sensitivity_params


class GEEResultsWrapper(lm.RegressionResultsWrapper):
    _attrs = {
        'centered_resid': 'rows',
    }
    _wrap_attrs = wrap.union_dicts(lm.RegressionResultsWrapper._wrap_attrs,
                                   _attrs)
wrap.populate_wrapper(GEEResultsWrapper, GEEResults)


class OrdinalGEE(GEE):

    __doc__ = (
        "    Estimation of ordinal response marginal regression models\n"
        "    using Generalized Estimating Equations (GEE).\n" +
        _gee_init_doc % {'extra_params': base._missing_param_doc,
                         'family_doc': _gee_ordinal_family_doc,
                         'example': _gee_ordinal_example})

    def __init__(self, endog, exog, groups, time=None, family=None,
                 cov_struct=None, missing='none', offset=None,
                 dep_data=None, constraint=None, **kwargs):

        if family is None:
            family = families.Binomial()
        else:
            if not isinstance(family, families.Binomial):
                raise ValueError("ordinal GEE must use a Binomial family")

        if cov_struct is None:
            cov_struct = cov_structs.OrdinalIndependence()

        endog, exog, groups, time, offset = self.setup_ordinal(
            endog, exog, groups, time, offset)

        super(OrdinalGEE, self).__init__(endog, exog, groups, time,
                                         family, cov_struct, missing,
                                         offset, dep_data, constraint)

    def setup_ordinal(self, endog, exog, groups, time, offset):
        """
        Restructure ordinal data as binary indicators so that they can
        be analysed using Generalized Estimating Equations.
        """

        self.endog_orig = endog.copy()
        self.exog_orig = exog.copy()
        self.groups_orig = groups.copy()
        if offset is not None:
            self.offset_orig = offset.copy()
        else:
            self.offset_orig = None
            offset = np.zeros(len(endog))
        if time is not None:
            self.time_orig = time.copy()
        else:
            self.time_orig = None
            time = np.zeros((len(endog), 1))

        exog = np.asarray(exog)
        endog = np.asarray(endog)
        groups = np.asarray(groups)
        time = np.asarray(time)
        offset = np.asarray(offset)

        # The unique outcomes, except the greatest one.
        self.endog_values = np.unique(endog)
        endog_cuts = self.endog_values[0:-1]
        ncut = len(endog_cuts)

        nrows = ncut * len(endog)
        exog_out = np.zeros((nrows, exog.shape[1]),
                            dtype=np.float64)
        endog_out = np.zeros(nrows, dtype=np.float64)
        intercepts = np.zeros((nrows, ncut), dtype=np.float64)
        groups_out = np.zeros(nrows, dtype=groups.dtype)
        time_out = np.zeros((nrows, time.shape[1]),
                            dtype=np.float64)
        offset_out = np.zeros(nrows, dtype=np.float64)

        jrow = 0
        zipper = zip(exog, endog, groups, time, offset)
        for (exog_row, endog_value, group_value, time_value,
             offset_value) in zipper:

            # Loop over thresholds for the indicators
            for thresh_ix, thresh in enumerate(endog_cuts):

                exog_out[jrow, :] = exog_row
                endog_out[jrow] = (int(endog_value > thresh))
                intercepts[jrow, thresh_ix] = 1
                groups_out[jrow] = group_value
                time_out[jrow] = time_value
                offset_out[jrow] = offset_value
                jrow += 1

        exog_out = np.concatenate((intercepts, exog_out), axis=1)

        # exog column names, including intercepts
        xnames = ["I(y>%.1f)" % v for v in endog_cuts]
        if type(self.exog_orig) == pd.DataFrame:
            xnames.extend(self.exog_orig.columns)
        else:
            xnames.extend(["x%d" % k for k in range(1, exog.shape[1] + 1)])
        exog_out = pd.DataFrame(exog_out, columns=xnames)

        # Preserve the endog name if there is one
        if type(self.endog_orig) == pd.Series:
            endog_out = pd.Series(endog_out, name=self.endog_orig.name)

        return endog_out, exog_out, groups_out, time_out, offset_out

    def _starting_params(self):
        model = GEE(self.endog, self.exog, self.groups,
                    time=self.time, family=families.Binomial(),
                    offset=self.offset, exposure=self.exposure)
        result = model.fit()
        return result.params

    def fit(self, maxiter=60, ctol=1e-6, start_params=None,
            params_niter=1, first_dep_update=0,
            cov_type='robust'):

        rslt = super(OrdinalGEE, self).fit(maxiter, ctol, start_params,
                                           params_niter, first_dep_update,
                                           cov_type=cov_type)

        rslt = rslt._results   # use unwrapped instance
        res_kwds = dict(((k, getattr(rslt, k)) for k in rslt._props))
        # Convert the GEEResults to an OrdinalGEEResults
        ord_rslt = OrdinalGEEResults(self, rslt.params,
                                     rslt.cov_params() / rslt.scale,
                                     rslt.scale,
                                     cov_type=cov_type,
                                     attr_kwds=res_kwds)
        # for k in rslt._props:
        #    setattr(ord_rslt, k, getattr(rslt, k))

        return OrdinalGEEResultsWrapper(ord_rslt)

    fit.__doc__ = _gee_fit_doc


class OrdinalGEEResults(GEEResults):

    __doc__ = (
        "This class summarizes the fit of a marginal regression model"
        "for an ordinal response using GEE.\n"
        + _gee_results_doc)

    def plot_distribution(self, ax=None, exog_values=None):
        """
        Plot the fitted probabilities of endog in an ordinal model,
        for specifed values of the predictors.

        Parameters
        ----------
        ax : Matplotlib axes instance
            An axes on which to draw the graph.  If None, new
            figure and axes objects are created
        exog_values : array-like
            A list of dictionaries, with each dictionary mapping
            variable names to values at which the variable is held
            fixed.  The values P(endog=y | exog) are plotted for all
            possible values of y, at the given exog value.  Variables
            not included in a dictionary are held fixed at the mean
            value.

        Example:
        --------
        We have a model with covariates 'age' and 'sex', and wish to
        plot the probabilities P(endog=y | exog) for males (sex=0) and
        for females (sex=1), as separate paths on the plot.  Since
        'age' is not included below in the map, it is held fixed at
        its mean value.

        >>> ev = [{"sex": 1}, {"sex": 0}]
        >>> rslt.distribution_plot(exog_values=ev)
        """

        from statsmodels.graphics import utils as gutils

        if ax is None:
            fig, ax = gutils.create_mpl_ax(ax)
        else:
            fig = ax.get_figure()

        # If no covariate patterns are specified, create one with all
        # variables set to their mean values.
        if exog_values is None:
            exog_values = [{}, ]

        exog_means = self.model.exog.mean(0)
        ix_icept = [i for i, x in enumerate(self.model.exog_names) if
                    x.startswith("I(")]

        for ev in exog_values:

            for k in ev.keys():
                if k not in self.model.exog_names:
                    raise ValueError("%s is not a variable in the model"
                                     % k)

            # Get the fitted probability for each level, at the given
            # covariate values.
            pr = []
            for j in ix_icept:

                xp = np.zeros_like(self.params)
                xp[j] = 1.
                for i, vn in enumerate(self.model.exog_names):
                    if i in ix_icept:
                        continue
                    # User-specified value
                    if vn in ev:
                        xp[i] = ev[vn]
                    # Mean value
                    else:
                        xp[i] = exog_means[i]

                p = 1 / (1 + np.exp(-np.dot(xp, self.params)))
                pr.append(p)

            pr.insert(0, 1)
            pr.append(0)
            pr = np.asarray(pr)
            prd = -np.diff(pr)

            ax.plot(self.model.endog_values, prd, 'o-')

        ax.set_xlabel("Response value")
        ax.set_ylabel("Probability")
        ax.set_ylim(0, 1)

        return fig


class OrdinalGEEResultsWrapper(GEEResultsWrapper):
    pass
wrap.populate_wrapper(OrdinalGEEResultsWrapper, OrdinalGEEResults)


class NominalGEE(GEE):

    __doc__ = (
        "    Estimation of nominal response marginal regression models\n"
        "    using Generalized Estimating Equations (GEE).\n" +
        _gee_init_doc % {'extra_params': base._missing_param_doc,
                         'family_doc': _gee_nominal_family_doc,
                         'example': _gee_nominal_example})

    def __init__(self, endog, exog, groups, time=None, family=None,
                 cov_struct=None, missing='none', offset=None,
                 dep_data=None, constraint=None, **kwargs):

        endog, exog, groups, time, offset = self.setup_nominal(
            endog, exog, groups, time, offset)

        if family is None:
            family = _Multinomial(self.ncut + 1)

        if cov_struct is None:
            cov_struct = cov_structs.NominalIndependence()

        super(NominalGEE, self).__init__(
            endog, exog, groups, time, family, cov_struct, missing,
            offset, dep_data, constraint)

    def _starting_params(self):
        model = GEE(self.endog, self.exog, self.groups,
                    time=self.time, family=families.Binomial(),
                    offset=self.offset, exposure=self.exposure)
        result = model.fit()
        return result.params

    def setup_nominal(self, endog, exog, groups, time, offset):
        """
        Restructure nominal data as binary indicators so that they can
        be analysed using Generalized Estimating Equations.
        """

        self.endog_orig = endog.copy()
        self.exog_orig = exog.copy()
        self.groups_orig = groups.copy()
        if offset is not None:
            self.offset_orig = offset.copy()
        else:
            self.offset_orig = None
            offset = np.zeros(len(endog))
        if time is not None:
            self.time_orig = time.copy()
        else:
            self.time_orig = None
            time = np.zeros((len(endog), 1))

        exog = np.asarray(exog)
        endog = np.asarray(endog)
        groups = np.asarray(groups)
        time = np.asarray(time)
        offset = np.asarray(offset)

        # The unique outcomes, except the greatest one.
        self.endog_values = np.unique(endog)
        endog_cuts = self.endog_values[0:-1]
        ncut = len(endog_cuts)
        self.ncut = ncut

        nrows = len(endog_cuts) * exog.shape[0]
        ncols = len(endog_cuts) * exog.shape[1]
        exog_out = np.zeros((nrows, ncols), dtype=np.float64)
        endog_out = np.zeros(nrows, dtype=np.float64)
        groups_out = np.zeros(nrows, dtype=np.float64)
        time_out = np.zeros((nrows, time.shape[1]),
                            dtype=np.float64)
        offset_out = np.zeros(nrows, dtype=np.float64)

        jrow = 0
        zipper = zip(exog, endog, groups, time, offset)
        for (exog_row, endog_value, group_value, time_value,
             offset_value) in zipper:

            # Loop over thresholds for the indicators
            for thresh_ix, thresh in enumerate(endog_cuts):

                u = np.zeros(len(endog_cuts), dtype=np.float64)
                u[thresh_ix] = 1
                exog_out[jrow, :] = np.kron(u, exog_row)
                endog_out[jrow] = (int(endog_value == thresh))
                groups_out[jrow] = group_value
                time_out[jrow] = time_value
                offset_out[jrow] = offset_value
                jrow += 1

        # exog names
        if type(self.exog_orig) == pd.DataFrame:
            xnames_in = self.exog_orig.columns
        else:
            xnames_in = ["x%d" % k for k in range(1, exog.shape[1] + 1)]
        xnames = []
        for tr in endog_cuts:
            xnames.extend(["%s[%.1f]" % (v, tr) for v in xnames_in])
        exog_out = pd.DataFrame(exog_out, columns=xnames)
        exog_out = pd.DataFrame(exog_out, columns=xnames)

        # Preserve endog name if there is one
        if type(self.endog_orig) == pd.Series:
            endog_out = pd.Series(endog_out, name=self.endog_orig.name)

        return endog_out, exog_out, groups_out, time_out, offset_out

    def mean_deriv(self, exog, lin_pred):
        """
        Derivative of the expected endog with respect to the parameters.

        Parameters
        ----------
        exog : array-like
           The exogeneous data at which the derivative is computed,
           number of rows must be a multiple of `ncut`.
        lin_pred : array-like
           The values of the linear predictor, length must be multiple
           of `ncut`.

        Returns
        -------
        The derivative of the expected endog with respect to the
        parameters.
        """

        expval = np.exp(lin_pred)

        # Reshape so that each row contains all the indicators
        # corresponding to one multinomial observation.
        expval_m = np.reshape(expval, (len(expval) // self.ncut,
                                       self.ncut))

        # The normalizing constant for the multinomial probabilities.
        denom = 1 + expval_m.sum(1)
        denom = np.kron(denom, np.ones(self.ncut, dtype=np.float64))

        # The multinomial probabilities
        mprob = expval / denom

        # First term of the derivative: denom * expval' / denom^2 =
        # expval' / denom.
        dmat = mprob[:, None] * exog

        # Second term of the derivative: -expval * denom' / denom^2
        ddenom = expval[:, None] * exog
        dmat -= mprob[:, None] * ddenom / denom[:, None]

        return dmat

    def mean_deriv_exog(self, exog, params, offset_exposure=None):
        """
        Derivative of the expected endog with respect to exog for the
        multinomial model, used in analyzing marginal effects.

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

        Notes
        -----
        offset_exposure must be set at None for the multinoial family.
        """

        if offset_exposure is not None:
            warnings.warn("Offset/exposure ignored for the multinomial family",
                          ValueWarning)

        lpr = np.dot(exog, params)
        expval = np.exp(lpr)

        expval_m = np.reshape(expval, (len(expval) // self.ncut,
                                       self.ncut))

        denom = 1 + expval_m.sum(1)
        denom = np.kron(denom, np.ones(self.ncut, dtype=np.float64))

        bmat0 = np.outer(np.ones(exog.shape[0]), params)

        # Masking matrix
        qmat = []
        for j in range(self.ncut):
            ee = np.zeros(self.ncut, dtype=np.float64)
            ee[j] = 1
            qmat.append(np.kron(ee, np.ones(len(params) // self.ncut)))
        qmat = np.array(qmat)
        qmat = np.kron(np.ones((exog.shape[0] // self.ncut, 1)), qmat)
        bmat = bmat0 * qmat

        dmat = expval[:, None] * bmat / denom[:, None]

        expval_mb = np.kron(expval_m, np.ones((self.ncut, 1)))
        expval_mb = np.kron(expval_mb, np.ones((1, self.ncut)))

        dmat -= expval[:, None] * (bmat * expval_mb) / denom[:, None] ** 2

        return dmat

    def fit(self, maxiter=60, ctol=1e-6, start_params=None,
            params_niter=1, first_dep_update=0,
            cov_type='robust'):

        rslt = super(NominalGEE, self).fit(maxiter, ctol, start_params,
                                           params_niter, first_dep_update,
                                           cov_type=cov_type)
        if rslt is None:
            warnings.warn("GEE updates did not converge",
                          ConvergenceWarning)
            return None

        rslt = rslt._results   # use unwrapped instance
        res_kwds = dict(((k, getattr(rslt, k)) for k in rslt._props))
        # Convert the GEEResults to a NominalGEEResults
        nom_rslt = NominalGEEResults(self, rslt.params,
                                     rslt.cov_params() / rslt.scale,
                                     rslt.scale,
                                     cov_type=cov_type,
                                     attr_kwds=res_kwds)
        # for k in rslt._props:
        #    setattr(nom_rslt, k, getattr(rslt, k))

        return NominalGEEResultsWrapper(nom_rslt)

    fit.__doc__ = _gee_fit_doc


class NominalGEEResults(GEEResults):

    __doc__ = (
        "This class summarizes the fit of a marginal regression model"
        "for a nominal response using GEE.\n"
        + _gee_results_doc)

    def plot_distribution(self, ax=None, exog_values=None):
        """
        Plot the fitted probabilities of endog in an nominal model,
        for specifed values of the predictors.

        Parameters
        ----------
        ax : Matplotlib axes instance
            An axes on which to draw the graph.  If None, new
            figure and axes objects are created
        exog_values : array-like
            A list of dictionaries, with each dictionary mapping
            variable names to values at which the variable is held
            fixed.  The values P(endog=y | exog) are plotted for all
            possible values of y, at the given exog value.  Variables
            not included in a dictionary are held fixed at the mean
            value.

        Example:
        --------
        We have a model with covariates 'age' and 'sex', and wish to
        plot the probabilities P(endog=y | exog) for males (sex=0) and
        for females (sex=1), as separate paths on the plot.  Since
        'age' is not included below in the map, it is held fixed at
        its mean value.

        >>> ex = [{"sex": 1}, {"sex": 0}]
        >>> rslt.distribution_plot(exog_values=ex)
        """

        from statsmodels.graphics import utils as gutils

        if ax is None:
            fig, ax = gutils.create_mpl_ax(ax)
        else:
            fig = ax.get_figure()

        # If no covariate patterns are specified, create one with all
        # variables set to their mean values.
        if exog_values is None:
            exog_values = [{}, ]

        link = self.model.family.link.inverse
        ncut = self.model.family.ncut

        k = int(self.model.exog.shape[1] / ncut)
        exog_means = self.model.exog.mean(0)[0:k]
        exog_names = self.model.exog_names[0:k]
        exog_names = [x.split("[")[0] for x in exog_names]

        params = np.reshape(self.params,
                            (ncut, len(self.params) // ncut))

        for ev in exog_values:

            exog = exog_means.copy()

            for k in ev.keys():
                if k not in exog_names:
                    raise ValueError("%s is not a variable in the model"
                                     % k)

                ii = exog_names.index(k)
                exog[ii] = ev[k]

            lpr = np.dot(params, exog)
            pr = link(lpr)
            pr = np.r_[pr, 1 - pr.sum()]

            ax.plot(self.model.endog_values, pr, 'o-')

        ax.set_xlabel("Response value")
        ax.set_ylabel("Probability")
        ax.set_xticks(self.model.endog_values)
        ax.set_xticklabels(self.model.endog_values)
        ax.set_ylim(0, 1)

        return fig


class NominalGEEResultsWrapper(GEEResultsWrapper):
    pass
wrap.populate_wrapper(NominalGEEResultsWrapper, NominalGEEResults)


class _MultinomialLogit(Link):
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

    call and derivative use a private method _clean to trim p by 1e-10
    so that p is in (0, 1)
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

        denom = 1 + np.reshape(expval, (len(expval) // self.ncut,
                                        self.ncut)).sum(1)
        denom = np.kron(denom, np.ones(self.ncut, dtype=np.float64))

        prob = expval / denom

        return prob


class _Multinomial(families.Family):
    """
    Pseudo-link function for fitting nominal multinomial models with
    GEE.  Not for use outside the GEE class.
    """

    links = [_MultinomialLogit, ]
    variance = varfuncs.binary
    safe_links = [_MultinomialLogit, ]

    def __init__(self, nlevels):
        """
        Parameters
        ----------
        nlevels : integer
            The number of distinct categories for the multinomial
            distribution.
        """
        self.initialize(nlevels)

    def initialize(self, nlevels):
        self.ncut = nlevels - 1
        self.link = _MultinomialLogit(self.ncut)


from statsmodels.discrete.discrete_margins import (
    _get_margeff_exog, _check_margeff_args, _effects_at, margeff_cov_with_se,
    _check_at_is_all, _transform_names, _check_discrete_args,
    _get_dummy_index, _get_count_index)


class GEEMargins(object):
    """
    Estimated marginal effects for a regression model fit with GEE.

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
        ind = self.results.model.exog.var(0) != 0  # True if not a constant
        exog_names = self.results.model.exog_names
        var_names = [name for i, name in enumerate(exog_names) if ind[i]]
        table = np.column_stack((self.margeff, self.margeff_se, self.tvalues,
                                 self.pvalues, self.conf_int(alpha)))
        return DataFrame(table, columns=names, index=var_names)

    @cache_readonly
    def pvalues(self):
        _check_at_is_all(self.margeff_options)
        return stats.norm.sf(np.abs(self.tvalues)) * 2

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
        q = stats.norm.ppf(1 - alpha / 2)
        lower = self.margeff - q * me_se
        upper = self.margeff + q * me_se
        return np.asarray(lzip(lower, upper))

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
                    ('At:', [self.margeff_options['at']]), ]

        from statsmodels.iolib.summary import (Summary, summary_params,
                                               table_extend)
        exog_names = model.exog_names[:]  # copy
        smry = Summary()

        const_idx = model.data.const_idx
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

        # NOTE: add_table_params is not general enough yet for margeff
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
                restup = (results, margeff[:, eq], margeff_se[:, eq],
                          tvalues[:, eq], pvalues[:, eq], conf_int[:, :, eq])
                tble = summary_params(restup, yname=yname_list[eq],
                                      xname=exog_names, alpha=alpha,
                                      use_t=False,
                                      skip_header=True)
                tble.title = yname_list[eq]
                # overwrite coef with method name
                header = ['', _transform_names[method], 'std err', 'z',
                          'P>|z|',
                          '[%3.1f%% Conf. Int.]' % (100 - alpha * 100)]
                tble.insert_header_row(0, header)
                # from IPython.core.debugger import Pdb; Pdb().set_trace()
                table.append(tble)

            table = table_extend(table, keep_headers=True)
        else:
            restup = (results, margeff, margeff_se, tvalues, pvalues, conf_int)
            table = summary_params(restup, yname=yname, xname=exog_names,
                                   alpha=alpha, use_t=False, skip_header=True)
            header = ['', _transform_names[method], 'std err', 'z',
                      'P>|z|', '[%3.1f%% Conf. Int.]' % (100 - alpha * 100)]
            table.insert_header_row(0, header)

        smry.tables.append(table)
        return smry

    def get_margeff(self, at='overall', method='dydx', atexog=None,
                    dummy=False, count=False):

        self._reset()  # always reset the cache when this is called
        # TODO: if at is not all or overall, we can also put atexog values
        # in summary table head
        method = method.lower()
        at = at.lower()
        _check_margeff_args(at, method)
        self.margeff_options = dict(method=method, at=at)
        results = self.results
        model = results.model
        params = results.params
        exog = model.exog.copy()  # copy because values are changed
        effects_idx = exog.var(0) != 0
        const_idx = model.data.const_idx

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
        effects = _effects_at(effects, at)

        if at == 'all':
            self.margeff = effects[:, effects_idx]
        else:
            # Set standard error of the marginal effects by Delta method.
            margeff_cov, margeff_se = margeff_cov_with_se(
                model, params, exog, results.cov_params(), at,
                model._derivative_exog, dummy_idx, count_idx,
                method, 1)

            # don't care about at constant
            self.margeff_cov = margeff_cov[effects_idx][:, effects_idx]
            self.margeff_se = margeff_se[effects_idx]
            self.margeff = effects[effects_idx]
