from __future__ import division
import numpy as np
from scipy.optimize import minimize
from scipy import sparse
from statsmodels.iolib import summary2
import pandas as pd
import statsmodels
import warnings

# Gauss-Legendre weights
glw = [[0.2955242247147529, -0.1488743389816312],
       [0.2955242247147529, 0.1488743389816312],
       [0.2692667193099963, -0.4333953941292472],
       [0.2692667193099963, 0.4333953941292472],
       [0.2190863625159820, -0.6794095682990244],
       [0.2190863625159820, 0.6794095682990244],
       [0.1494513491505806, -0.8650633666889845],
       [0.1494513491505806, 0.8650633666889845],
       [0.0666713443086881, -0.9739065285171717],
       [0.0666713443086881, 0.9739065285171717]]


_init_doc = r"""
    Fit a generalized linear mixed model using Bayesian methods.
{fit_method}
    Parameters
    ----------
    endog : array-like
        Vector of response values.
    exog_fe : array-like
        Array of covariates for the fixed effects part of the mean
        structure.
    exog_vc : array-like
        Array of covariates for the random part of the model.  A
        scipy.sparse array may be provided, or else the passed
        array will be converted to sparse internally.
    ident : array-like
        Array of labels showing which random terms have a common
        variance.
    vc_p : float
        Prior standard deviation for variance component parameters
        (the prior standard deviation of log(s) is vc_p, where s is
        the standard deviation of a random effect).
    fe_p : float
        Prior standard deviation for fixed effects parameters.
    family : statsmodels.genmod.families instance
        The GLM family.
    fep_names : list of strings
        The names of the fixed effects parameters (corresponding
        to columns of exog_fe).
    vcp_names : list of strings
        The names of the variance component parameters (corresponding
        to distinct labels in ident).

    Returns
    -------
    MixedGLMResults object

    Notes
    -----
    There are three types of values in the posterior distribution:
    fixed effects parameters (fep), corresponding to the columns of
    `exog_fe`, random effects realizations (vc), corresponding to the
    columns of `exog_vc`, and the variances of the random effects
    realizations (vcp), corresponding to the unique labels in `ident`.

    All random effects are modeled as being independent Gaussian
    values (given the variance parameters).  Every column of `exog_vc`
    has a distinct realized random effect that is used to form the
    linear predictors.  The elements of `ident` determine the distinct
    random effect variance parameters.  Two random effect realizations
    that have the same value in `ident` are constrained to have the
    same variance.

    The random effect standard deviation parameters (vcp) have
    log-normal prior distributions with mean 0 and standard deviation
    `vcp_p`.

    The prior for the fixed effects parameters is Gaussian with mean 0
    and standard deviation `fe_p`.
    """

_laplace_fit_method = """
    The class implements the Laplace approximation to the posterior
    distribution.  See subclasses, e.g. BinomialBayesMixedGLM for
    other estimation approaches.
"""

_vb_fit_method = """
    The class implements a variational Bayes approximation to the
    posterior.  See the docstring to `fit_vb` for more information.
"""


class BayesMixedGLM(object):

    __doc__ = _init_doc.format(fit_method=_laplace_fit_method)

    def __init__(self, endog, exog_fe, exog_vc, ident, vcp_p=0.5,
                 fe_p=0.5, family=None, fep_names=None,
                 vcp_names=None):

        if family is None:
            family = statsmodels.genmod.families.Gaussian()
            warnings.Warn("Defaulting to Gaussian family")

        # Get the fixed effects parameter names
        if fep_names is None:
            if hasattr(exog_fe, "columns"):
                fep_names = exog_fe.columns.tolist()
            else:
                fep_names = ["FE_%d" % (k + 1)
                             for k in range(exog_fe.shape[1])]
        self.fep_names = fep_names

        # Get the variance parameter names
        if vcp_names is None:
            if hasattr(exog_vc, "columns"):
                vcp_names = exog_vc.columns.tolist()
            else:
                vcp_names = ["VC_%d" % (k + 1)
                             for k in range(int(max(ident)) + 1)]
        self.vcp_names = vcp_names

        self.endog = np.asarray(endog)
        self.exog_fe = np.asarray(exog_fe)

        if sparse.issparse(exog_vc):
            self.exog_vc = exog_vc
        else:
            self.exog_vc = sparse.csr_matrix(exog_vc)

        self.ident = ident.astype(np.int)
        self.family = family
        self.vcp_p = float(vcp_p)
        self.fe_p = float(fe_p)

        # Number of fixed effects parameters
        if self.exog_fe is None:
            self.k_fep = 0
        else:
            self.k_fep = exog_fe.shape[1]

        # Number of variance component structure parameters and
        # variance component realizations.
        if self.exog_vc is None:
            self.k_vc = 0
            self.k_vcp = 0
        else:
            self.k_vc = exog_vc.shape[1]
            self.k_vcp = max(self.ident) + 1

    def _unpack(self, vec):

        ii = 0

        # Fixed effects parameters
        fep = vec[:ii+self.k_fep]
        ii += self.k_fep

        # Variance component structure parameters (standard
        # deviations).  These are on the log scale.  The standard
        # deviation for random effect j is exp(vcp[ident[j]]).
        vcp = vec[ii:ii+self.k_vcp]
        ii += self.k_vcp

        # Random effect realizations
        vc = vec[ii:]

        return fep, vcp, vc

    def logposterior(self, params):
        """
        The overall log-density: log p(y, fe, vc, vcp).

        This differs by an additive constant from the log posterior
        log p(fe, vc, vcp | y).
        """

        fep, vcp, vc = self._unpack(params)

        # Contributions from p(y | vc)
        lp = 0
        if self.k_fep > 0:
            lp += np.dot(self.exog_fe, fep)
        if self.k_vc > 0:
            lp += self.exog_vc.dot(vc)

        mu = self.family.link.inverse(lp)
        ll = self.family.loglike(self.endog, mu)

        if self.k_vc > 0:

            # Contribution from p(vc | vcp)
            vcp0 = vcp[self.ident]
            s = np.exp(vcp0)
            ll -= 0.5 * np.sum(vc**2 / s**2) + np.sum(vcp0)

            # Prior for vc parameters
            ll -= 0.5 * np.sum(vcp**2 / self.vcp_p**2)

        # Contributions from p(fep)
        if self.k_fep > 0:
            ll -= 0.5 * np.sum(fep**2 / self.fe_p**2)

        return ll

    def logposterior_grad(self, params):
        """
        The gradient of the log posterior.
        """

        fep, vcp, vc = self._unpack(params)

        lp = 0
        if self.k_fep > 0:
            lp += np.dot(self.exog_fe, fep)
        if self.k_vc > 0:
            lp += self.exog_vc.dot(vc)

        mu = self.family.link.inverse(lp)

        score_factor = (self.endog - mu) / self.family.link.deriv(mu)
        score_factor /= self.family.variance(mu)

        te = [None, None, None]

        # Contributions from p(y | x, z, vc)
        if self.k_fep > 0:
            te[0] = np.dot(score_factor, self.exog_fe)
        if self.k_vc > 0:
            te[2] = self.exog_vc.transpose().dot(score_factor)

        if self.k_vc > 0:
            # Contributions from p(vc | vcp)
            # vcp0 = vcp[self.ident]
            # s = np.exp(vcp0)
            # ll -= 0.5 * np.sum(vc**2 / s**2) + np.sum(vcp0)
            vcp0 = vcp[self.ident]
            s = np.exp(vcp0)
            u = vc**2 / s**2 - 1
            te[1] = np.bincount(self.ident, weights=u)
            te[2] -= vc / s**2

            # Contributions from p(vcp)
            # ll -= 0.5 * np.sum(vcp**2 / self.vcp_p**2)
            te[1] -= vcp / self.vcp_p**2

        # Contributions from p(fep)
        if self.k_fep > 0:
            te[0] -= fep / self.fe_p**2

        te = [x for x in te if x is not None]
        return np.concatenate(te)

    def _get_start(self):
        start_fep = np.zeros(self.k_fep)
        start_vcp = np.ones(self.k_vcp)
        start_vc = np.random.normal(size=self.k_vc)
        start = np.concatenate((start_fep, start_vcp, start_vc))
        return start

    def fit_map(self, method="BFGS", minim_opts=None):
        """
        Construct the Laplace approximation to the posterior
        distribution.
        """

        def fun(params):
            return -self.logposterior(params)

        def grad(params):
            return -self.logposterior_grad(params)

        start = self._get_start()

        r = minimize(fun, start, method=method, jac=grad, options=minim_opts)
        if not r.success:
            msg = ("Laplace fitting did not converge, |gradient|=%.6f" %
                   np.sqrt(np.sum(r.jac**2)))
            warnings.warn(msg)

        return BayesMixedGLMResults(self, r.x, r.hess_inv, optim_retvals=r)


class _VariationalBayesMixedGLM(BayesMixedGLM):

    # Returns the mean and variance of the linear predictor under the
    # given distribution parameters.
    def _lp_stats(self, fep_mean, fep_sd, vc_mean, vc_sd):

        tm = np.dot(self.exog_fe, fep_mean)
        tv = np.dot(self.exog_fe**2, fep_sd**2)
        tm += self.exog_vc.dot(vc_mean)
        tv += self.exog_vc.power(2).dot(vc_sd**2)

        return tm, tv

    def fit_vb(self, mean=None, sd=None, minim_opts=None, verbose=False):
        """
        Fit a model using the variational Bayes mean field approximation.

        Parameters:
        -----------
        mean : array-like
            Starting value for VB mean vector
        sd : array-like
            Starting value for VB standard deviation vector
        minim_opts : dict-like
            Options passed to scipy.minimize
        verbose : bool
            If True, print the gradient norm to the screen each time
            it is calculated.

        Notes
        -----
        The goal is to find a factored Gaussian approximation
        q1*q2*...  to the posterior distribution, approximately
        minimizing the KL divergence from the factored approximation
        to the actual posterior.  The KL divergence, or ELBO function
        has the form

            E* log p(y, fe, vcp, vc) - E* log q

        where E* is expectation with respect to the product of qj.

        References
        ----------
        https://arxiv.org/pdf/1601.00670.pdf
        """

        self.verbose = verbose

        n = self.k_fep + self.k_vcp + self.k_vc
        if mean is None:
            m = np.zeros(n)
        else:
            m = mean
        if sd is None:
            s = -0.5 + 0.1 * np.random.normal(size=n)
        else:
            # s is parameterized on the log-scale internally when
            # optimizing the ELBO function (this is transparent to the
            # caller)
            s = np.log(sd)

        # Don't allow the variance parameter starting mean values to
        # be too small.
        i1, i2 = self.k_fep, self.k_fep + self.k_vcp
        m[i1:i2] = np.where(m[i1:i2] < -1, -1, m[i1:i2])

        # Don't allow the posterior standard deviation starting values
        # to be too small.
        s = np.where(s < -1, -1, s)

        def elbo(x):
            n = len(x) // 2
            return -self.vb_elbo(x[:n], np.exp(x[n:]))

        def elbo_grad(x):
            n = len(x) // 2
            gm, gs = self.vb_elbo_grad(x[:n], np.exp(x[n:]))
            gs *= np.exp(x[n:])
            return -np.concatenate((gm, gs))

        start = np.concatenate((m, s))
        mm = minimize(elbo, start, jac=elbo_grad, method="bfgs",
                      options=minim_opts)
        if not mm.success:
            warnings.warn("VB fitting did not converge")

        n = len(mm.x) // 2
        return BayesMixedGLMResults(self, mm.x[0:n], np.exp(2*mm.x[n:]), mm)

    # Handle terms in the ELBO that are common to all models.
    def _elbo_common(self, fep_mean, fep_sd, vcp_mean, vcp_sd, vc_mean, vc_sd):

        iv = 0

        # p(vc | vcp) contributions
        m = vcp_mean[self.ident]
        s = vcp_sd[self.ident]
        iv -= np.sum((vc_mean**2 + vc_sd**2) * np.exp(-2*m + 2*s**2)) / 2
        iv -= np.sum(m)

        # p(vcp) contributions
        iv -= 0.5 * (vcp_mean**2 + vcp_sd**2).sum() / self.vcp_p**2

        # p(b) contributions
        iv -= 0.5 * (fep_mean**2 + fep_sd**2).sum() / self.fe_p**2

        return iv

    def _elbo_grad_common(self, fep_mean, fep_sd, vcp_mean, vcp_sd,
                          vc_mean, vc_sd):

        # p(vc | vcp) contributions
        m = vcp_mean[self.ident]
        s = vcp_sd[self.ident]
        u = vc_mean**2 + vc_sd**2
        ve = np.exp(2*(s**2 - m))
        dm = u * ve - 1
        ds = -2 * u * ve * s
        vcp_mean_grad = np.bincount(self.ident, weights=dm)
        vcp_sd_grad = np.bincount(self.ident, weights=ds)

        vc_mean_grad = -vc_mean.copy() * ve
        vc_sd_grad = -vc_sd.copy() * ve

        # p(vcp) contributions
        vcp_mean_grad -= vcp_mean / self.vcp_p**2
        vcp_sd_grad -= vcp_sd / self.vcp_p**2

        # p(b) contributions
        fep_mean_grad = -fep_mean.copy() / self.fe_p**2
        fep_sd_grad = -fep_sd.copy() / self.fe_p**2

        return (fep_mean_grad, fep_sd_grad, vcp_mean_grad, vcp_sd_grad,
                vc_mean_grad, vc_sd_grad)


class BayesMixedGLMResults(object):

    def __init__(self, model, params, cov_params,
                 optim_retvals=None):

        self.model = model
        self.params = params
        self.cov_params = cov_params
        self.optim_retvals = optim_retvals

        self.fe_params, self.vcp_params, self.vc_params = (
            model._unpack(params))

        if cov_params.ndim == 2:
            cp = np.diag(cov_params)
        else:
            cp = cov_params
        self.fe_sd, self.vcp_sd, self.vc_sd = model._unpack(cp)
        self.fe_sd = np.sqrt(self.fe_sd)
        self.vcp_sd = np.sqrt(self.vcp_sd)
        self.vc_sd = np.sqrt(self.vc_sd)

    def summary(self):

        df = pd.DataFrame()
        m = self.model.k_fep + self.model.k_vcp
        df["Type"] = (["F" for k in range(self.model.k_fep)] +
                      ["R" for k in range(self.model.k_vcp)])

        df["Post. Mean"] = self.params[0:m]

        if self.cov_params.ndim == 2:
            v = np.diag(self.cov_params)[0:m]
            df["Post. SD"] = np.sqrt(v)
        else:
            df["Post. SD"] = np.sqrt(self.cov_params[0:m])

        # Convert variance parameters to natural scale
        df["VC"] = np.exp(df["Post. Mean"])
        df["VC (LB)"] = np.exp(df["Post. Mean"] - 2*df["Post. SD"])
        df["VC (UB)"] = np.exp(df["Post. Mean"] + 2*df["Post. SD"])
        df["VC"] = ["%.3f" % x for x in df.VC]
        df["VC (LB)"] = ["%.3f" % x for x in df["VC (LB)"]]
        df["VC (UB)"] = ["%.3f" % x for x in df["VC (UB)"]]
        df.loc[df.index < self.model.k_fep, "VC"] = ""
        df.loc[df.index < self.model.k_fep, "VC (LB)"] = ""
        df.loc[df.index < self.model.k_fep, "VC (UB)"] = ""

        df.index = self.model.fep_names + self.model.vcp_names

        summ = summary2.Summary()
        summ.add_title(self.model.family.__class__.__name__ +
                       " Mixed GLM Results")
        summ.add_df(df)

        return summ


class BinomialBayesMixedGLM(_VariationalBayesMixedGLM):

    __doc__ = _init_doc.format(fit_method=_vb_fit_method)

    # Integration range (from -rng to +rng).  The integrals are with
    # respect to a standard Gaussian distribution so (-5, 5) will be
    # sufficient in many cases.
    rng = 5

    verbose = False

    def __init__(self, endog, exog_fe, exog_vc, ident, vcp_p=0.5,
                 fe_p=0.5, fep_names=None, vcp_names=None):

        super(BinomialBayesMixedGLM, self).__init__(
            endog=endog, exog_fe=exog_fe, exog_vc=exog_vc,
            ident=ident, vcp_p=vcp_p, fe_p=fe_p,
            family=statsmodels.genmod.families.Binomial(),
            fep_names=fep_names, vcp_names=vcp_names)

    def vb_elbo(self, vb_mean, vb_sd):
        """
        Returns the evidence lower bound (ELBO) for the model.
        """

        fep_mean, vcp_mean, vc_mean = self._unpack(vb_mean)
        fep_sd, vcp_sd, vc_sd = self._unpack(vb_sd)

        tm, tv = self._lp_stats(fep_mean, fep_sd, vc_mean, vc_sd)

        def h(z):
            x = np.log(1 + np.exp(tm + np.sqrt(tv)*z))
            x *= np.exp(-z**2 / 2)
            x /= np.sqrt(2*np.pi)
            return x

        # p(y | vc) contributions
        iv = 0
        for w in glw:
            iv += h(self.rng * w[1]) * w[0]
        iv *= -self.rng
        iv += self.endog * tm
        iv = iv.sum()

        iv += self._elbo_common(fep_mean, fep_sd, vcp_mean, vcp_sd,
                                vc_mean, vc_sd)

        return iv + np.sum(np.log(vb_sd))

    def vb_elbo_grad(self, vb_mean, vb_sd):
        """
        Returns the gradient of the model's evidence lower bound (ELBO).
        """

        fep_mean, vcp_mean, vc_mean = self._unpack(vb_mean)
        fep_sd, vcp_sd, vc_sd = self._unpack(vb_sd)

        tm, tv = self._lp_stats(fep_mean, fep_sd, vc_mean, vc_sd)

        def h(z):
            x = np.log(1 + np.exp(tm + np.sqrt(tv)*z))
            x *= np.exp(-z**2 / 2)
            x /= np.sqrt(2*np.pi)
            return x

        def h1(z):
            u = tm + np.sqrt(tv)*z
            x = np.exp(u) / (1 + np.exp(u))
            x *= np.exp(-z**2 / 2)
            x /= np.sqrt(2*np.pi)
            return x

        fep_mean_grad = 0.
        fep_sd_grad = 0.
        vcp_mean_grad = 0.
        vcp_sd_grad = 0.
        vc_mean_grad = 0.
        vc_sd_grad = 0.

        # p(y | vc) contributions
        for w in glw:
            x = self.rng * w[1]
            u = h1(x)
            r = u / np.sqrt(tv)
            fep_mean_grad += w[0] * np.dot(u, self.exog_fe)
            vc_mean_grad += w[0] * self.exog_vc.transpose().dot(u)
            fep_sd_grad += w[0] * x * np.dot(r, self.exog_fe**2 * fep_sd)
            v = self.exog_vc.power(2).multiply(vc_sd).transpose().dot(r)
            v = np.squeeze(np.asarray(v))
            vc_sd_grad += w[0] * x * v
        fep_mean_grad *= -self.rng
        vc_mean_grad *= -self.rng
        fep_sd_grad *= -self.rng
        vc_sd_grad *= -self.rng
        fep_mean_grad += np.dot(self.endog, self.exog_fe)
        vc_mean_grad += self.exog_vc.transpose().dot(self.endog)

        (fep_mean_grad_i, fep_sd_grad_i, vcp_mean_grad_i, vcp_sd_grad_i,
         vc_mean_grad_i, vc_sd_grad_i) = self._elbo_grad_common(
            fep_mean, fep_sd, vcp_mean, vcp_sd, vc_mean, vc_sd)

        fep_mean_grad += fep_mean_grad_i
        fep_sd_grad += fep_sd_grad_i
        vcp_mean_grad += vcp_mean_grad_i
        vcp_sd_grad += vcp_sd_grad_i
        vc_mean_grad += vc_mean_grad_i
        vc_sd_grad += vc_sd_grad_i

        mean_grad = np.concatenate((fep_mean_grad, vcp_mean_grad,
                                    vc_mean_grad))
        sd_grad = np.concatenate((fep_sd_grad, vcp_sd_grad, vc_sd_grad))

        sd_grad += 1 / vb_sd

        if self.verbose:
            print("|G|=%f" % np.sqrt(np.sum(mean_grad**2) +
                                     np.sum(sd_grad**2)))

        return mean_grad, sd_grad
