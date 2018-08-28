# -*- coding: utf-8 -*-

"""
This module implements likelihood-based estimation (MLE) of Gaussian
models for finite-dimensional observations made on infinite-dimensional
processes.

The implementation is tailored to regression-style analyses in which the
mean and covariance structures are parameterized in terms of covariates.
The fitting is based on a grouped dataset.  The repeated observations within
a group are related through the Gaussian covariance model.
"""

import numpy as np
import pandas as pd
import patsy
import statsmodels.base.model as base
import statsmodels.api as sm
import collections
from statsmodels.compat.python import string_types
from scipy.optimize import minimize
from statsmodels.iolib import summary2
from statsmodels.tools.numdiff import approx_fprime
import warnings


class ProcessCovariance(object):
    """
    A covariance model for linear regression with dependent data.

    This class represents (possibly) non-stationary covariance models
    constructed by modifying correlation kernels, as described in the
    work of Paciorek et al.

    This is a base class that does not implement any functionality.
    See GaussianCovariance for a concrete implementation.

    An implementation of this class is based on a positive definite
    correlation function h that maps real numbers to the interval [0,
    1], such as the Gaussian (squared exponential) correlation function
    :math:`\exp(-x^2)`.  It also depends on a positive scaling function
    `s` and a positive smoothness function `u`.  Following Paciorek et
    al, the covariance between observations with index `i` and `j` is
    given by:

    .. math::

      s[i] \cdot s[j] \cdot h(|time[i] - time[j]| / \sqrt{(u[i] + u[j]) /
      2}) \cdot \frac{u[i]^{1/})u[j]^{1/4}}{\sqrt{(u[i] + u[j])/2}}

    The ProcessMLE class allows linear models with this
    covariance structure to be fit using maximum likelihood (ML), which
    is equivalent to generalized least squares (GLS) in this setting.

    The mean and covariance parameters of the model are fit jointly.
    The mean, scaling, and smoothing parameters can be linked to
    covariates.  The mean parameters are linked linearly, and  the
    scaling and smoothing parameters use an exponential link to preserve
    positivity.

    The reference of Paciorek et al. below provides more details.  Note
    that here we only implement the 1-dimensional version of their
    approach.

    References
    ----------
    Paciorek, C. J. and Schervish, M. J. (2006). Spatial modeling using
    a new class of nonstationary covariance functions. Environmetrics,
    17:483–506.
    https://papers.nips.cc/paper/2350-nonstationary-covariance-functions-for-gaussian-process-regression.pdf
    """

    def get_cov(self, time, sc, sm):
        """
        Returns the covariance matrix for given time values.

        Parameters
        ----------
        time : array-like
            The time points for the observations.  If len(time) = p,
            a pxp covariance matrix is returned.
        sc : array-like
            The scaling parameters for the observations.
        sm : array-like
            The smoothness parameters for the observation.  See class
            docstring for details.
        """
        raise NotImplementedError

    def jac(self, time, sc, sm):
        """
        The Jacobian of the covariance respect to the parameters.

        See get_cov for parameters.

        Returns
        -------
        jsc : list-like
            jsc[i] is the derivative of the covariance matrix
            with respect to the i^th scaling parameter.
        jsm : list-like
            jsm[i] is the derivative of the covariance matrix
            with respect to the i^th smoothness parameter.
        """
        raise NotImplementedError


class GaussianCovariance(ProcessCovariance):
    """
    An implementation of ProcessCovariance using the Gaussian kernel.

    This is sometimes called the "squared exponential"
    covariance model.
    """

    def get_cov(self, time, sc, sm):

        da = np.subtract.outer(time, time)
        ds = np.add.outer(sm, sm) / 2

        qmat = da * da / ds
        cm = np.exp(-qmat / 2) / np.sqrt(ds)
        cm *= np.outer(sm, sm)**0.25
        cm *= np.outer(sc, sc)

        return cm

    def jac(self, time, sc, sm):

        da = np.subtract.outer(time, time)
        ds = np.add.outer(sm, sm) / 2
        sds = np.sqrt(ds)
        daa = da * da
        qmat = daa / ds
        p = len(time)
        eqm = np.exp(-qmat / 2)
        sm4 = np.outer(sm, sm)**0.25
        cmx = eqm * sm4 / sds
        dq0 = -daa / ds**2
        di = np.zeros((p, p))
        fi = np.zeros((p, p))
        scc = np.outer(sc, sc)

        # Derivatives with respect to the smoothing parameters.
        jsm = []
        for i, _ in enumerate(sm):
            di *= 0
            di[i, :] += 0.5
            di[:, i] += 0.5
            dbottom = 0.5 * di / sds
            dtop = -0.5 * eqm * dq0 * di
            b = dtop / sds - eqm * dbottom / ds
            c = eqm / sds
            v = 0.25 * sm**0.25 / sm[i]**0.75
            fi *= 0
            fi[i, :] = v
            fi[:, i] = v
            fi[i, i] = 0.5 / sm[i]**0.5
            b = c * fi + b * sm4
            b *= scc
            jsm.append(b)

        # Derivatives with respect to the scaling parameters.
        jsc = []
        for i in range(0, len(sc)):
            b = np.zeros((p, p))
            b[i, :] = cmx[i, :] * sc
            b[:, i] += cmx[:, i] * sc
            jsc.append(b)

        return jsc, jsm


def _check_args(endog, exog, exog_scale, exog_smooth, time, groups):

    v = [
        len(endog), exog.shape[0], exog_scale.shape[0],
        exog_smooth.shape[0],
        len(time),
        len(groups)
    ]
    if min(v) != max(v):
        msg = ("The leading dimensions of all array arguments " +
               "must be equal.")
        raise ValueError(msg)


class ProcessMLE(base.LikelihoodModel):
    """
    Fit a Gaussian mean/variance regression model.

    This class fits a Gaussian model with parameterized mean and
    covariance structures to data.  The mean structure is a linear
    relationship as in least squares regression.  The covariance model
    belongs to a parameterized family of covariances, such that the
    covariance between two observations in the same group is a function
    of the distance between the two observations.  The distance is based
    on a given quantity, called "time" below, but it could be any
    appropriate unidimensional quantity.

    The data should be provided in "long form", with a group label to
    indicate which observations belong to the same group.  Observations
    in different groups are always independent.

    Parameters
    ----------
    endog : array-like
        The dependent variable.
    exog : array-like
        The design matrix for the mean structure
    exog_scale : array-like
        The design matrix for the scaling structure
    exog_smooth : array-like
        The design matrix for the smoothness structure
    time : array-like (1-dimensional)
        The 'time' values, used to calculate distances
        between observations in the same group, and hence
        their correlations.
    groups : array-like (1-dimensional)
        The group values.
    cov : a ProcessCovariance instance
        Defaults to GaussianCovariance.
    """

    def __init__(self,
                 endog,
                 exog,
                 exog_scale,
                 exog_smooth,
                 time,
                 groups,
                 cov=None,
                 **kwargs):

        super(ProcessMLE, self).__init__(
            endog,
            exog,
            exog_scale=exog_scale,
            exog_smooth=exog_smooth,
            time=time,
            groups=groups,
            **kwargs)

        # Create parameter names
        xnames = []
        if hasattr(exog, "columns"):
            xnames = list(exog.columns)
        else:
            xnames = ["Mean%d" % j for j in range(exog.shape[1])]
        if hasattr(exog_scale, "columns"):
            xnames += list(exog_scale.columns)
        else:
            xnames += ["Scale%d" % j for j in range(exog_scale.shape[1])]
        if hasattr(exog_smooth, "columns"):
            xnames += list(exog_smooth.columns)
        else:
            xnames += ["Smooth%d" % j for j in range(exog_smooth.shape[1])]
        self.data.param_names = xnames

        if cov is None:
            cov = GaussianCovariance()
        self.cov = cov

        _check_args(endog, exog, exog_scale, exog_smooth, time, groups)

        groups_ix = collections.defaultdict(lambda: [])
        for i, g in enumerate(groups):
            groups_ix[g].append(i)
        self._groups_ix = groups_ix

        self.verbose = False

    @classmethod
    def from_formula(cls,
                     formula,
                     data,
                     subset=None,
                     drop_cols=None,
                     *args,
                     **kwargs):

        if "scale_formula" in kwargs:
            scale_formula = kwargs["scale_formula"]
        else:
            raise ValueError("scale_formula is a required argument")

        if "smooth_formula" in kwargs:
            smooth_formula = kwargs["smooth_formula"]
        else:
            raise ValueError("smooth_formula is a required argument")

        if "time" in kwargs:
            time = kwargs["time"]
        else:
            raise ValueError("time is a required argument")

        if "groups" in kwargs:
            groups = kwargs["groups"]
        else:
            raise ValueError("groups is a required argument")

        if subset is not None:
            warnings.warn("'subset' is ignored")

        if drop_cols is not None:
            warnings.warn("'drop_cols' is ignored")

        if isinstance(time, string_types):
            time = np.asarray(data[time])

        if isinstance(groups, string_types):
            groups = np.asarray(data[groups])

        exog_scale = patsy.dmatrix(scale_formula, data)
        scale_design_info = exog_scale.design_info
        scale_names = scale_design_info.column_names
        exog_scale = np.asarray(exog_scale)

        exog_smooth = patsy.dmatrix(smooth_formula, data)
        smooth_design_info = exog_smooth.design_info
        smooth_names = smooth_design_info.column_names
        exog_smooth = np.asarray(exog_smooth)

        mod = super(ProcessMLE, cls).from_formula(
            formula,
            data=data,
            subset=None,
            exog_scale=exog_scale,
            exog_smooth=exog_smooth,
            time=time,
            groups=groups)

        mod.data.scale_design_info = scale_design_info
        mod.data.smooth_design_info = smooth_design_info
        mod.data.param_names = mod.exog_names + scale_names + smooth_names

        return mod

    def unpack(self, z):
        """
        Split the packed parameter vector into blocks.
        """

        # Mean parameters
        pm = self.exog.shape[1]
        mnpar = z[0:pm]

        # Standard deviation parameters
        pv = self.exog_scale.shape[1]
        sdpar = z[pm:pm + pv]

        # Smoothness parameters
        smpar = z[pm + pv:]

        return mnpar, sdpar, smpar

    def _get_start(self):

        model = sm.OLS(self.endog, self.exog)
        result = model.fit()

        par = np.concatenate(
            (result.params,
             np.zeros(self.exog_scale.shape[1] + self.exog_smooth.shape[1])))

        return par

    def loglike(self, params):
        """
        Calculate the log-likelihood function for the model.

        Parameters
        ----------
        params : array-like
            The packed parameters for the model.

        Returns
        -------
        The log-likelihood value at the given parameter point.

        Notes
        -----
        The mean, scaling, and smoothing parameters are packed into
        a vector.  Use `unpack` to access the component vectors.
        """

        mnpar, scpar, smpar = self.unpack(params)

        # Residuals
        resid = self.endog - np.dot(self.exog, mnpar)

        # Scaling parameters
        sc = np.exp(np.dot(self.exog_scale, scpar))

        # Smoothness parameters
        sm = np.exp(np.dot(self.exog_smooth, smpar))

        # Get the log-likelihood
        ll = 0.
        for _, ix in self._groups_ix.items():

            # Get the covariance matrix for this person.
            cm = self.cov.get_cov(self.time[ix], sc[ix], sm[ix])

            re = resid[ix]
            ll -= 0.5 * np.linalg.slogdet(cm)[1]
            ll -= 0.5 * np.dot(re, np.linalg.solve(cm, re))

        if self.verbose:
            print("L=", ll)

        return ll

    def score(self, params):
        """
        Calculate the score function for the model.

        Parameters
        ----------
        params : array-like
            The packed parameters for the model.

        Returns
        -------
        The score vector at the given parameter point.

        Notes
        -----
        The mean, scaling, and smoothing parameters are packed into
        a vector.  Use `unpack` to access the component vectors.
        """

        mnpar, sdpar, smpar = self.unpack(params)
        pm, pv = len(mnpar), len(sdpar)

        # Residuals
        resid = self.endog - np.dot(self.exog, mnpar)

        # Standard deviations
        sd = np.exp(np.dot(self.exog_scale, sdpar))

        # Smoothness
        sm = np.exp(np.dot(self.exog_smooth, smpar))

        # Get the log-likelihood
        score = np.zeros(len(mnpar) + len(sdpar) + len(smpar))
        for _, ix in self._groups_ix.items():

            sd_i = sd[ix]
            sm_i = sm[ix]
            resid_i = resid[ix]
            time_i = self.time[ix]
            exog_i = self.exog[ix, :]
            exog_scale_i = self.exog_scale[ix, :]
            exog_smooth_i = self.exog_smooth[ix, :]

            # Get the covariance matrix for this person.
            cm = self.cov.get_cov(time_i, sd_i, sm_i)
            cmi = np.linalg.inv(cm)

            jacv, jacs = self.cov.jac(time_i, sd_i, sm_i)

            # The derivatives for the mean parameters.
            dcr = np.linalg.solve(cm, resid_i)
            score[0:pm] += np.dot(exog_i.T, dcr)

            # The derivatives for the standard deviation parameters.
            rx = np.outer(resid_i, resid_i)
            qm = np.linalg.solve(cm, rx)
            qm = 0.5 * np.linalg.solve(cm, qm.T)
            sdx = sd_i[:, None] * exog_scale_i
            for i, _ in enumerate(ix):
                jq = np.sum(jacv[i] * qm)
                score[pm:pm + pv] += jq * sdx[i, :]
                score[pm:pm + pv] -= 0.5 * np.sum(jacv[i] * cmi) * sdx[i, :]

            # The derivatives for the smoothness parameters.
            smx = sm_i[:, None] * exog_smooth_i
            for i, _ in enumerate(ix):
                jq = np.sum(jacs[i] * qm)
                score[pm + pv:] += jq * smx[i, :]
                score[pm + pv:] -= 0.5 * np.sum(jacs[i] * cmi) * smx[i, :]

        if self.verbose:
            print("|G|=", np.sqrt(np.sum(score * score)))

        return score

    def fit(self, start_params=None, method=None, maxiter=None,
            full_output=True, disp=True, fargs=(), callback=None,
            retall=False, skip_hessian=False, **kwargs):

        if "verbose" in kwargs:
            self.verbose = kwargs["verbose"]

        minim_opts = {}
        if "minim_opts" in kwargs:
            minim_opts = kwargs["minim_opts"]

        if start_params is None:
            start_params = self._get_start()

        if isinstance(method, str):
            method = [method]
        elif method is None:
            method = ["powell", "bfgs"]

        for j, meth in enumerate(method):

            jac = None
            if meth not in ("powell",):
                jac = lambda x: -self.score(x)

            if maxiter is not None:
                if np.isscalar(maxiter):
                    minim_opts["maxiter"] = maxiter
                else:
                    minim_opts["maxiter"] = maxiter[j % len(maxiter)]

            f = minimize(
                lambda x: -self.loglike(x),
                method=meth,
                x0=start_params,
                jac=jac,
                options=minim_opts)

            if not f.success:
                msg = "Fitting did not converge"
                if jac is not None:
                    msg += ", |gradient|=%.6f" % np.sqrt(np.sum(f.jac**2))
                if j < len(method) - 1:
                    msg += ", trying %s next..." % method[j+1]
                warnings.warn(msg)

            if np.isfinite(f.x).all():
                start_params = f.x

        hess = approx_fprime(f.x, self.score)
        try:
            cov_params = -np.linalg.inv(hess)
        except Exception:
            cov_params = None

        class rslt:
            pass

        r = rslt()
        r.params = f.x
        r.normalized_cov_params = cov_params
        r.optim_retvals = f
        r.scale = 1

        rslt = ProcessMLEResults(self, r)

        return rslt

    def covariance(self, time, scale_params, smooth_params, scale_data,
                   smooth_data):
        """
        Returns a fitted covariance matrix.

        Parameters
        ----------
        time : array-like
            The time points at which the fitted covariance
            matrix is calculated.
        scale_params : array-like
            The regression parameters for the scaling part
            of the covariance structure.
        smooth_params : array-like
            The regression parameters for the smoothing part
            of the covariance structure.
        scale_data : Dataframe
            The data used to determine the scale parameter,
            must have len(time) rows.
        smooth_data: Dataframe
            The data used to determine the smoothness parameter,
            must have len(time) rows.

        Returns
        -------
        A covariance matrix.

        Notes
        -----
        If the model was fit using formulas, `scale` and `smooth` should
        be Dataframes, containing all variables that were present in the
        respective scaling and smoothing formulas used to fit the model.
        Otherwise, `scale` and `smooth` should contain data arrays whose
        columns align with the fitted scaling and smoothing parameters.
        """

        if not hasattr(self.data, "scale_design_info"):
            sca = np.dot(scale_data, scale_params)
            smo = np.dot(smooth_data, smooth_params)
        else:
            sc = patsy.dmatrix(self.data.scale_design_info, scale_data)
            sm = patsy.dmatrix(self.data.smooth_design_info, smooth_data)
            sca = np.exp(np.dot(sc, scale_params))
            smo = np.exp(np.dot(sm, smooth_params))

        return self.cov.get_cov(time, sca, smo)

    def predict(self, params, exog=None, *args, **kwargs):
        """
        Obtain predictions of the mean structure.

        Parameters
        ----------
        params : array-like
            The model parameters, may be truncated to include only mean
            parameters.
        exog : array-like
            The design matrix for the mean structure.  If not provided,
            the model's design matrix is used.
        """

        if exog is None:
            if hasattr(self.data, "frame"):
                exog = self.data.frame
            else:
                exog = self.exog

        if hasattr(self.data, "design_info"):
            exog = patsy.dmatrix(self.data.design_info, exog)

        if len(params) > exog.shape[1]:
            params = params[0:exog.shape[1]]

        return np.dot(exog, params)


class ProcessMLEResults(base.GenericLikelihoodModelResults):
    """
    Results class for Gaussian process regression models.
    """

    def __init__(self, model, mlefit):

        super(ProcessMLEResults, self).__init__(
            model, mlefit)

        pa = model.unpack(mlefit.params)

        self.mean_params = pa[0]
        self.scale_params = pa[1]
        self.smooth_params = pa[2]

        self.df_resid = model.endog.shape[0] - len(mlefit.params)

    def predict(self, exog=None, transform=True, *args, **kwargs):

        if not transform:
            warnings.warn("'transform=False' is ignored in predict")

        if len(args) > 0 or len(kwargs) > 0:
            warnings.warn("extra arguments ignored in 'predict'")

        return self.model.predict(self.params, exog)

    def covariance(self, time, scale, smooth):
        """
        Returns a fitted covariance matrix.

        Parameters
        ----------
        time : array-like
            The time points at which the fitted covariance
            matrix is calculated.
        scale : array-like
            The data used to determine the scale parameter,
            must have len(time) rows.
        smooth: array-like
            The data used to determine the smoothness parameter,
            must have len(time) rows.

        Returns
        -------
        A covariance matrix.

        Notes
        -----
        If the model was fit using formulas, `scale` and `smooth` should
        be Dataframes, containing all variables that were present in the
        respective scaling and smoothing formulas used to fit the model.
        Otherwise, `scale` and `smooth` should be data arrays whose
        columns align with the fitted scaling and smoothing parameters.
        """

        return self.model.covariance(time, self.scale_params,
                                     self.smooth_params, scale, smooth)

    def summary(self, yname=None, xname=None, title=None, alpha=0.05):

        df = pd.DataFrame()
        pm = self.model.exog.shape[1]
        pv = self.model.exog_scale.shape[1]
        ps = self.model.exog_smooth.shape[1]

        df["Type"] = ["Mean"] * pm + ["Scale"] * pv + ["Smooth"] * ps
        df["coef"] = self.params

        try:
            df["std err"] = np.sqrt(np.diag(self.cov_params()))
        except Exception:
            df["std err"] = np.nan

        from scipy.stats.distributions import norm
        df["t"] = df.coef / df["std err"]
        df["P>|t|"] = norm.cdf(2 * (1 - np.abs(df.t)))

        f = norm.ppf(1 - alpha / 2)
        df["[%.3f" % (alpha / 2)] = df.coef - f * df["std err"]
        df["%.3f]" % (1 - alpha / 2)] = df.coef + f * df["std err"]

        df.index = self.model.data.param_names

        summ = summary2.Summary()
        if title is None:
            title = "Gaussian process regression results"
        summ.add_title(title)
        summ.add_df(df)

        return summ
