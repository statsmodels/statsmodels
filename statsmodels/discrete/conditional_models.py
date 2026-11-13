"""
Conditional logistic, Poisson, and multinomial logit regression
"""

import collections
import itertools
import warnings

import numpy as np
from scipy.special import expit, logsumexp

import statsmodels.base.model as base
import statsmodels.base.wrapper as wrap
from statsmodels.discrete.discrete_model import (
    MultinomialResults,
    MultinomialResultsWrapper,
)
from statsmodels.formula.formulatools import advance_eval_env
import statsmodels.regression.linear_model as lm
from statsmodels.tools.rng_qrng import check_random_state
from statsmodels.tools.sm_exceptions import ModelWarning
from statsmodels.tools.validation import string_like


def _check_cov_type(kwargs):
    """
    Raise if a robust covariance type is requested.

    The conditional likelihood is computed by group and there is no
    observation level score, so only the nonrobust covariance is available.
    """
    cov_type = kwargs.get("cov_type", "nonrobust")
    if cov_type != "nonrobust":
        raise NotImplementedError(
            f"cov_type={cov_type!r} is not available for conditional models, "
            "only cov_type='nonrobust' is supported."
        )


class _ConditionalModel(base.LikelihoodModel):

    def __init__(self, endog, exog, missing="none", **kwargs):

        if "groups" not in kwargs:
            raise ValueError("'groups' is a required argument")
        groups = kwargs["groups"]

        if groups.size != endog.size:
            msg = "'endog' and 'groups' should have the same dimensions"
            raise ValueError(msg)

        if exog.shape[0] != endog.size:
            msg = "The leading dimension of 'exog' should equal the length of 'endog'"
            raise ValueError(msg)

        super().__init__(endog, exog, missing=missing, **kwargs)

        if self.data.const_idx is not None:
            msg = "Conditional models should not have an intercept in the design matrix"
            raise ValueError(msg)

        exog = self.exog
        self.k_params = exog.shape[1]

        # Get the row indices for each group
        row_ix = {}
        for i, g in enumerate(groups):
            if g not in row_ix:
                row_ix[g] = []
            row_ix[g].append(i)

        # Split the data into groups and remove groups with no variation
        endog, exog = np.asarray(endog), np.asarray(exog)
        offset = kwargs.get("offset")
        self._endog_grp = []
        self._exog_grp = []
        self._groupsize = []
        if offset is not None:
            offset = np.asarray(offset)
            self._offset_grp = []
        self._offset = []
        self._sumy = []
        self.nobs = 0
        drops = [0, 0]
        for _, ix in row_ix.items():
            y = endog[ix].flat
            if np.std(y) == 0:
                drops[0] += 1
                drops[1] += len(y)
                continue
            self.nobs += len(y)
            self._endog_grp.append(y)
            if offset is not None:
                self._offset_grp.append(offset[ix])
            self._groupsize.append(len(y))
            self._exog_grp.append(exog[ix, :])
            self._sumy.append(np.sum(y))

        if drops[0] > 0:
            msg = (
                f"Dropped {drops[0]} groups and {drops[1]} observations for "
                f"having no within-group variance"
            )
            warnings.warn(msg, ModelWarning, stacklevel=2)

        # This can be pre-computed
        if offset is not None:
            self._endofs = []
            for k, ofs in enumerate(self._offset_grp):
                self._endofs.append(np.dot(self._endog_grp[k], ofs))

        # Number of groups
        self._n_groups = len(self._endog_grp)

        # These are the sufficient statistics
        self._xy = []
        self._n1 = []
        for g in range(self._n_groups):
            self._xy.append(np.dot(self._endog_grp[g], self._exog_grp[g]))
            self._n1.append(np.sum(self._endog_grp[g]))

    def hessian(self, params):
        """
        Returns numerical approximation to the Hessian.

        Parameters
        ----------
        params : array_like
            The model parameters.

        Returns
        -------
        ndarray
            The Hessian matrix, approximated numerically from the
            score function.
        """

        from statsmodels.tools.numdiff import approx_fprime

        hess = approx_fprime(params, self.score)
        hess = np.atleast_2d(hess)
        return hess

    def fit(
        self,
        start_params=None,
        method="BFGS",
        maxiter=100,
        full_output=True,
        disp=False,
        fargs=(),
        callback=None,
        retall=False,
        skip_hessian=False,
        **kwargs,
    ):
        """
        Fit the conditional model.

        Parameters
        ----------
        start_params : array_like, optional
            Initial guess of the solution for the loglikelihood
            maximization. The default is an array of zeros.
        method : str, optional
            The `method` determines which solver from `scipy.optimize`
            is used, see `LikelihoodModel.fit` for more information.
        maxiter : int, optional
            The maximum number of iterations to perform.
        full_output : bool, optional
            Set to True to have all available output in the Results
            object's mle_retvals attribute.
        disp : bool, optional
            Set to True to print convergence messages.
        fargs : tuple, optional
            Extra arguments passed to the likelihood function, its score and
            its Hessian. The conditional models do not take any, so this is
            only useful in a subclass that overrides these methods.
        callback : callable, optional
            Called after each iteration, as callback(xk), where xk is
            the current parameter vector.
        retall : bool, optional
            Set to True to return list of solutions at each iteration.
        skip_hessian : bool, optional
            If False, the covariance matrix is calculated using the
            numerical Hessian after the optimization.  If True, the
            Hessian is not calculated and the returned results have
            no covariance matrix.
        **kwargs
            Additional keyword arguments used by the solver, see
            `LikelihoodModel.fit`. This includes ``warn_convergence``.
            Robust covariance estimators are not available, so ``cov_type``
            must be "nonrobust" if it is passed.

            .. versionchanged:: 0.15.1

               ``fargs``, ``callback``, ``retall`` and the additional
               keyword arguments are forwarded to the solver. They were
               previously ignored.

        Returns
        -------
        ConditionalResultsWrapper
            The fitted model results.
        """

        _check_cov_type(kwargs)
        rslt = super().fit(
            start_params=start_params,
            method=method,
            maxiter=maxiter,
            full_output=full_output,
            disp=disp,
            fargs=fargs,
            callback=callback,
            retall=retall,
            skip_hessian=skip_hessian,
            **kwargs,
        )

        if skip_hessian:
            cov_params = None
        else:
            cov_params = rslt.cov_params()

        crslt = ConditionalResults(self, rslt.params, cov_params, 1)
        crslt.mle_retvals = rslt.mle_retvals
        crslt.mle_settings = rslt.mle_settings
        crslt.method = method
        crslt.nobs = self.nobs
        crslt.n_groups = self._n_groups
        crslt._group_stats = [
            f"{min(self._groupsize):d}",
            f"{max(self._groupsize):d}",
            f"{np.mean(self._groupsize):.1f}",
        ]
        rslt = ConditionalResultsWrapper(crslt)
        return rslt

    def fit_regularized(
        self, method="elastic_net", alpha=0.0, start_params=None, refit=False, **kwargs
    ):
        """
        Return a regularized fit to a linear regression model.

        Parameters
        ----------
        method : {'elastic_net'}, optional
            Only the `elastic_net` approach is currently implemented.
        alpha : scalar or array_like, optional
            The penalty weight.  If a scalar, the same penalty weight
            applies to all variables in the model.  If a vector, it
            must have the same length as `params`, and contains a
            penalty weight for each coefficient.
        start_params : array_like, optional
            Starting values for `params`.
        refit : bool, optional
            If True, the model is refit using only the variables that
            have non-zero coefficients in the regularized fit.  The
            refitted model is not regularized.
        **kwargs
            Additional keyword argument that are used when fitting the model.

        Returns
        -------
        Results
            A results instance.
        """

        from statsmodels.base.elastic_net import fit_elasticnet

        method = string_like(method, "method", options=("elastic_net",), lower=False)

        defaults = {"maxiter": 50, "L1_wt": 1, "cnvrg_tol": 1e-10, "zero_tol": 1e-10}
        defaults.update(kwargs)

        return fit_elasticnet(
            self,
            method=method,
            alpha=alpha,
            start_params=start_params,
            refit=refit,
            **defaults,
        )

    # Override to allow groups to be passed as a variable name.
    @classmethod
    def from_formula(cls, formula, data, subset=None, drop_cols=None, *args, **kwargs):

        try:
            groups = kwargs["groups"]
            del kwargs["groups"]
        except KeyError as err:
            raise ValueError("'groups' is a required argument") from err

        if isinstance(groups, str):
            groups = data[groups]

        if "0+" not in formula.replace(" ", ""):
            warnings.warn(
                "Conditional models should not include an intercept",
                ModelWarning,
                stacklevel=2,
            )
        advance_eval_env(kwargs)
        model = super().from_formula(formula, data, *args, groups=groups, **kwargs)

        return model


class ConditionalLogit(_ConditionalModel):
    """
    Fit a conditional logistic regression model to grouped data.

    Every group is implicitly given an intercept, but the model is fit using
    a conditional likelihood in which the intercepts are not present.  Thus,
    intercept estimates are not given, but the other parameter estimates can
    be interpreted as being adjusted for any group-level confounders.

    Parameters
    ----------
    endog : array_like
        The response variable, must contain only 0 and 1.
    exog : array_like
        The array of covariates.  Do not include an intercept
        in this array.
    groups : array_like
        Codes defining the groups. This is a required keyword parameter.
    missing : {'none', 'drop', 'raise'}, optional
        If 'none', no nan checking is done. If 'drop', any observations
        with nans are dropped. If 'raise', an error is raised.

    Notes
    -----
    .. versionchanged:: 0.15.1

       The conditional likelihood uses iterative, scaled log-space recurrences
       for large groups and batched calculations for matched small groups.
    """

    def __init__(self, endog, exog, missing="none", **kwargs):

        super().__init__(endog, exog, missing=missing, **kwargs)

        if np.any(np.unique(self.endog) != np.r_[0, 1]):
            msg = "endog must be coded as 0, 1"
            raise ValueError(msg)

        self.K = self.exog.shape[1]
        # i.e., self.k_params, for compatibility with MNLogit

        # A matched pair is an ordinary logistic likelihood for the difference
        # between its case and control. Batch these very common small groups.
        pair_exog, pair_offset = [], []
        single_groups = collections.defaultdict(list)
        self._nonpair_groups = []
        for g, size in enumerate(self._groupsize):
            if size != 2:
                if self._n1[g] == 1 or self._n1[g] == size - 1:
                    single_groups[size].append(g)
                else:
                    self._nonpair_groups.append(g)
                continue
            direction = 1 if self._endog_grp[g][0] == 1 else -1
            exog = self._exog_grp[g]
            exog = np.asarray(exog, dtype=np.result_type(exog, float))
            pair_exog.append(direction * (exog[0] - exog[1]))
            if hasattr(self, "offset"):
                offset = self._offset_grp[g]
                offset = np.asarray(offset, dtype=np.result_type(offset, float))
                pair_offset.append(direction * (offset[0] - offset[1]))
            else:
                pair_offset.append(0.0)
        self._pair_exog = np.asarray(pair_exog).reshape(-1, self.K)
        self._pair_offset = np.asarray(pair_offset)

        self._single_batches = []
        for size, groups in single_groups.items():
            # Bound each design/offset batch to about 512 KiB for float64.
            batch_size = 65536 // (size * (self.K + 1))
            if len(groups) == 1 or batch_size < 2:
                self._nonpair_groups.extend(groups)
                continue
            for start in range(0, len(groups), batch_size):
                indices = groups[start : start + batch_size]
                design, offsets = [], []
                for g in indices:
                    direction = 1 if self._n1[g] == 1 else -1
                    endog = np.asarray(self._endog_grp[g])
                    reference = np.argmax(endog if direction == 1 else 1 - endog)
                    exog = self._exog_grp[g]
                    exog = np.asarray(exog, dtype=np.result_type(exog, float))
                    design.append(direction * (exog - exog[reference]))
                    if hasattr(self, "offset"):
                        offset = self._offset_grp[g]
                        offset = np.asarray(offset, dtype=np.result_type(offset, float))
                        offsets.append(direction * (offset - offset[reference]))
                    else:
                        offsets.append(np.zeros(size))
                self._single_batches.append((np.asarray(design), np.asarray(offsets)))
        self._nonpair_groups.sort()

    def loglike(self, params):
        """
        Log-likelihood of the conditional logistic model.

        Parameters
        ----------
        params : array_like
            The parameters of the model.

        Returns
        -------
        float
            The log-likelihood value at `params`, summed over all
            groups.
        """

        if (
            getattr(self.loglike_grp, "__func__", None)
            is not ConditionalLogit.loglike_grp
        ):
            return sum(self.loglike_grp(g, params) for g in range(self._n_groups))

        linear = self._pair_exog @ params + self._pair_offset
        if np.iscomplexobj(linear):
            ll = -logsumexp(np.stack((np.zeros_like(linear), -linear)), axis=0).sum()
        else:
            ll = -np.logaddexp(0, -linear).sum()
        for design, offset in self._single_batches:
            linear = design @ params + offset
            ll -= logsumexp(linear, axis=1).sum()
        for g in self._nonpair_groups:
            ll += self.loglike_grp(g, params)
        return ll

    def score(self, params):
        """
        Score vector of the conditional logistic model.

        Parameters
        ----------
        params : array_like
            The parameters of the model.

        Returns
        -------
        ndarray
            The score vector at `params`, summed over all groups.
        """

        score = np.zeros(self.K, dtype=np.result_type(np.asarray(params), float))
        if getattr(self.score_grp, "__func__", None) is not ConditionalLogit.score_grp:
            for g in range(self._n_groups):
                score += self.score_grp(g, params)
            return score

        linear = self._pair_exog @ params + self._pair_offset
        if np.iscomplexobj(linear):
            weights = np.exp(
                -logsumexp(np.stack((np.zeros_like(linear), linear)), axis=0)
            )
        else:
            weights = expit(-linear)
        score = self._pair_exog.T @ weights
        for design, offset in self._single_batches:
            linear = design @ params + offset
            shifted = linear - np.max(linear.real, axis=1, keepdims=True)
            weights = np.exp(shifted)
            weights /= weights.sum(axis=1, keepdims=True)
            score -= np.einsum("ij,ijk->k", weights, design)
        for g in self._nonpair_groups:
            score += self.score_grp(g, params)
        return score

    def _denom(self, grp, params, ofs=None):

        if ofs is None:
            ofs = 0

        exb = np.exp(np.dot(self._exog_grp[grp], params) + ofs)

        nobs = len(exb)
        n1 = int(self._n1[grp])
        denom = np.zeros(n1 + 1, dtype=exb.dtype)
        denom[0] = 1
        for t, value in enumerate(exb, 1):
            # Update only states that can reach the final number of successes.
            lower = max(1, n1 - nobs + t)
            upper = min(t, n1)
            # D(t, k) = D(t-1, k) + value * D(t-1, k-1).
            # The right-hand side is evaluated before updating the slice.
            denom[lower : upper + 1] += value * denom[lower - 1 : upper]

        return denom[n1]

    def _denom_grad(self, grp, params, ofs=None):

        if ofs is None:
            ofs = 0

        ex = self._exog_grp[grp]
        exb = np.exp(np.dot(ex, params) + ofs)

        nobs = len(exb)
        n1 = int(self._n1[grp])
        denom = np.zeros(n1 + 1, dtype=exb.dtype)
        grad = np.zeros((n1 + 1, self.k_params), dtype=exb.dtype)
        denom[0] = 1
        for t, value in enumerate(exb, 1):
            lower = max(1, n1 - nobs + t)
            upper = min(t, n1)
            # Update the gradient first, using the preceding denominator state.
            grad[lower : upper + 1] += value * (
                grad[lower - 1 : upper] + denom[lower - 1 : upper, None] * ex[t - 1]
            )
            denom[lower : upper + 1] += value * denom[lower - 1 : upper]

        return denom[n1], grad[n1]

    def _loglike_score_grp(self, grp, params, ofs=None, calc_score=False):
        """Evaluate a group's likelihood without forming its normalizer.

        The denominator is an elementary symmetric polynomial in exp(X beta).
        Each log-polynomial state is normalized by its largest subset term,
        retaining small likelihood differences even at large predictor contrasts.
        """
        exog = self._exog_grp[grp]
        exog = np.asarray(exog, dtype=np.result_type(exog, float))
        endog = np.asarray(self._endog_grp[grp])
        nobs = len(endog)
        n1 = int(self._n1[grp])

        # Conditioning on failures gives the same likelihood with -X beta.
        # Use the smaller number of selected observations to reduce the state.
        direction = 1
        if n1 > nobs // 2:
            direction = -1
            endog = 1 - endog
            n1 = nobs - n1

        # Remove a common group predictor before taking exponentials or sums.
        # This also removes constant offsets without cancellation at the end.
        reference = np.argmax(endog) if n1 == 1 else 0
        design = direction * (exog - exog[reference])
        linear = design @ params
        if ofs is not None and np.ndim(ofs):
            ofs = np.asarray(ofs, dtype=np.result_type(ofs, float))
            linear = linear + direction * (ofs - ofs[reference])

        if n1 == 1:
            log_denom = logsumexp(linear)
            if calc_score:
                # Reconstructing a large log-normalizer can round away log(2)
                # for tied leading terms. Normalize shifted weights directly.
                weights = np.exp(linear - np.max(linear.real))
                weights /= weights.sum()
                return -log_denom, -(weights @ design)
            return -log_denom

        dtype = np.result_type(linear, float)
        log_denom = np.full(n1 + 1, -np.inf, dtype=dtype)
        log_denom[0] = 0
        if calc_score:
            mean = np.zeros((n1 + 1, self.K), dtype=dtype)

        # For state k, subtract the sum of the k largest preceding predictors.
        # The difference between the two candidate reference sums is just
        # value - kth, so neither large cumulative sum needs to be formed.
        top = np.full(n1, -np.inf, dtype=dtype)
        top_indices = np.full(n1, -1, dtype=int)
        complex_params = np.iscomplexobj(linear)
        for t, value in enumerate(linear, 1):
            lower = max(1, n1 - nobs + t)
            upper = min(t, n1)
            current = slice(lower, upper + 1)
            previous = slice(lower - 1, upper)
            kth = top[lower - 1 : upper]
            difference = value - kth
            choose_include = value.real > kth.real

            # Use one complementary branch for both shifts. At equal real
            # parts this retains the imaginary difference for complex-step
            # derivatives, consistently with the stable top-k insertion below.
            exclude = log_denom[current] + np.where(choose_include, -difference, 0)
            include = log_denom[previous] + np.where(choose_include, 0, difference)
            if complex_params:
                shift = np.maximum(exclude.real, include.real)
                weight_exclude = np.exp(exclude - shift)
                weight_include = np.exp(include - shift)
                total = weight_exclude + weight_include
                combined = shift + np.log(total)
                if calc_score:
                    weight_exclude /= total
                    weight_include /= total
            else:
                combined = np.logaddexp(exclude, include)
                if calc_score:
                    difference = include - exclude
                    weight_exclude = expit(-difference)
                    weight_include = expit(difference)

            if calc_score:
                mean[current] = weight_exclude[:, None] * mean[
                    current
                ] + weight_include[:, None] * (mean[previous] + design[t - 1])
            log_denom[current] = combined

            insertion = np.searchsorted(-top.real, -value.real, side="right")
            if insertion < n1:
                top[insertion + 1 :] = top[insertion:-1]
                top_indices[insertion + 1 :] = top_indices[insertion:-1]
                top[insertion] = value
                top_indices[insertion] = t - 1

        # Cancel observations common to the observed and reference subsets
        # before summing. Every remaining real contrast is nonpositive, so
        # small likelihood differences cannot disappear in a subtraction of
        # two large accumulated predictor sums.
        top_mask = np.zeros(nobs, dtype=bool)
        top_mask[top_indices] = True
        observed_only = linear[(endog != 0) & ~top_mask]
        top_only = linear[(endog == 0) & top_mask]
        ll = np.sum(observed_only - top_only) - log_denom[n1]
        if calc_score:
            return ll, np.dot(endog, design) - mean[n1]
        return ll

    def loglike_grp(self, grp, params):

        ofs = None
        if hasattr(self, "offset"):
            ofs = self._offset_grp[grp]

        return self._loglike_score_grp(grp, params, ofs)

    def score_grp(self, grp, params):

        ofs = None
        if hasattr(self, "offset"):
            ofs = self._offset_grp[grp]

        return self._loglike_score_grp(grp, params, ofs, calc_score=True)[1]


class ConditionalPoisson(_ConditionalModel):
    """
    Fit a conditional Poisson regression model to grouped data.

    Every group is implicitly given an intercept, but the model is fit using
    a conditional likelihood in which the intercepts are not present.  Thus,
    intercept estimates are not given, but the other parameter estimates can
    be interpreted as being adjusted for any group-level confounders.

    Parameters
    ----------
    endog : array_like
        The response variable
    exog : array_like
        The covariates
    groups : array_like
        Codes defining the groups. This is a required keyword parameter.
    missing : {'none', 'drop', 'raise'}, optional
        If 'none', no nan checking is done. If 'drop', any observations
        with nans are dropped. If 'raise', an error is raised.
    """

    def loglike(self, params):
        """
        Log-likelihood of the conditional Poisson model.

        Parameters
        ----------
        params : array_like
            The parameters of the model.

        Returns
        -------
        float
            The log-likelihood value at `params`, summed over all
            groups.
        """

        ofs = None
        if hasattr(self, "offset"):
            ofs = self._offset_grp

        ll = 0.0

        for i in range(len(self._endog_grp)):

            xb = np.dot(self._exog_grp[i], params)
            if ofs is not None:
                xb += ofs[i]
            exb = np.exp(xb)
            y = self._endog_grp[i]
            ll += np.dot(y, xb)
            s = exb.sum()
            ll -= self._sumy[i] * np.log(s)

        return ll

    def score(self, params):
        """
        Score vector of the conditional Poisson model.

        Parameters
        ----------
        params : array_like
            The parameters of the model.

        Returns
        -------
        ndarray
            The score vector at `params`, summed over all groups.
        """

        ofs = None
        if hasattr(self, "offset"):
            ofs = self._offset_grp

        score = 0.0

        for i in range(len(self._endog_grp)):

            x = self._exog_grp[i]
            xb = np.dot(x, params)
            if ofs is not None:
                xb += ofs[i]
            exb = np.exp(xb)
            s = exb.sum()
            y = self._endog_grp[i]
            score += np.dot(y, x)
            score -= self._sumy[i] * np.dot(exb, x) / s

        return score


class ConditionalResults(base.LikelihoodModelResults):
    def __init__(self, model, params, normalized_cov_params, scale):

        super().__init__(
            model, params, normalized_cov_params=normalized_cov_params, scale=scale
        )

    def summary(self, yname=None, xname=None, title=None, alpha=0.05):
        """
        Summarize the fitted model.

        Parameters
        ----------
        yname : str, optional
            Default is `y`
        xname : list of str, optional
            Names for the exogenous variables, default is "var_xx".
            Must match the number of parameters in the model
        title : str, optional
            Title for the top table. If not None, then this replaces the
            default title
        alpha : float, optional
            Significance level for the confidence intervals

        Returns
        -------
        smry : Summary instance
            This holds the summary tables and text, which can be printed or
            converted to various output formats.

        See Also
        --------
        statsmodels.iolib.summary.Summary : class to hold summary
            results
        """

        top_left = [
            ("Dep. Variable:", None),
            ("Model:", None),
            ("Log-Likelihood:", None),
            ("Method:", [self.method]),
            ("Date:", None),
            ("Time:", None),
        ]

        top_right = [
            ("No. Observations:", None),
            ("No. groups:", [self.n_groups]),
            ("Min group size:", [self._group_stats[0]]),
            ("Max group size:", [self._group_stats[1]]),
            ("Mean group size:", [self._group_stats[2]]),
        ]

        if title is None:
            title = "Conditional Logit Model Regression Results"

        # create summary tables
        from statsmodels.iolib.summary import Summary

        smry = Summary()
        smry.add_table_2cols(
            self,
            gleft=top_left,
            gright=top_right,  # [],
            yname=yname,
            xname=xname,
            title=title,
        )
        smry.add_table_params(
            self, yname=yname, xname=xname, alpha=alpha, use_t=self.use_t
        )

        return smry


class ConditionalMNLogit(_ConditionalModel):
    """
    Fit a conditional multinomial logit model to grouped data.

    Parameters
    ----------
    endog : array_like
        The dependent variable, must be integer-valued, coded
        0, 1, ..., c-1, where c is the number of response
        categories.
    exog : array_like
        The independent variables.
    groups : array_like
        Codes defining the groups. This is a required keyword parameter.
    missing : {'none', 'drop', 'raise'}, optional
        If 'none', no nan checking is done. If 'drop', any observations
        with nans are dropped. If 'raise', an error is raised.

    Notes
    -----
    Equivalent to femlogit in Stata.

    References
    ----------
    Gary Chamberlain (1980).  Analysis of covariance with qualitative
    data. The Review of Economic Studies.  Vol. 47, No. 1, pp. 225-238.
    """

    def __init__(self, endog, exog, missing="none", **kwargs):

        super().__init__(endog, exog, missing=missing, **kwargs)

        # endog must be integers
        self.endog = self.endog.astype(int)

        self.k_cat = self.endog.max() + 1
        self.df_model = (self.k_cat - 1) * self.exog.shape[1]
        self.df_resid = self.nobs - self.df_model
        self._ynames_map = {j: str(j) for j in range(self.k_cat)}
        self.J = self.k_cat  # Unfortunate name, needed for results
        self.K = self.exog.shape[1]  # for compatibility with MNLogit

        if self.endog.min() < 0:
            msg = "endog may not contain negative values"
            raise ValueError(msg)

        grx = collections.defaultdict(list)
        for k, v in enumerate(self.groups):
            grx[v].append(k)
        self._group_labels = list(grx.keys())
        self._group_labels.sort()
        self._grp_ix = [grx[k] for k in self._group_labels]

    def fit(
        self,
        start_params=None,
        method="BFGS",
        maxiter=100,
        full_output=True,
        disp=False,
        fargs=(),
        callback=None,
        retall=False,
        skip_hessian=False,
        rng=None,
        **kwargs,
    ):
        """
        Fit the conditional multinomial logit model.

        Parameters
        ----------
        start_params : array_like, optional
            Initial guess of the solution for the loglikelihood
            maximization.  If None, random values are drawn using
            `generator`.
        method : str, optional
            The `method` determines which solver from `scipy.optimize`
            is used, see `LikelihoodModel.fit` for more information.
        maxiter : int, optional
            The maximum number of iterations to perform.
        full_output : bool, optional
            Set to True to have all available output in the Results
            object's mle_retvals attribute.
        disp : bool, optional
            Set to True to print convergence messages.
        fargs : tuple, optional
            Extra arguments passed to the likelihood function, its score and
            its Hessian. The conditional models do not take any, so this is
            only useful in a subclass that overrides these methods.
        callback : callable, optional
            Called after each iteration, as callback(xk), where xk is
            the current parameter vector.
        retall : bool, optional
            Set to True to return list of solutions at each iteration.
        skip_hessian : bool, optional
            If False, the covariance matrix is calculated using the
            numerical Hessian after the optimization.  If True, the
            Hessian is not calculated and the returned results have
            no covariance matrix.
        rng : int, array_like of int, numpy.random.Generator, or numpy.random.RandomState, optional
            Used to draw random starting values for `start_params` when
            `start_params` is None. If `rng` is None, the legacy global
            (singleton) ``RandomState`` provided by ``numpy.random`` is
            used; this behavior is deprecated and will change to
            creating a new ``Generator`` using fresh entropy from the
            operating system in a future release. If `rng` is an int or
            array of ints, a new ``Generator`` is created, seeded with
            `rng`. If `rng` is already a ``Generator`` or ``RandomState``
            instance, that instance is used.

            .. deprecated:: 0.15

               After statsmodels 0.15 is released, the default
               (``rng=None``) method for producing random start_params
               will change to a new instance of a
               ``numpy.random.Generator``. To obtain deterministic
               starting values today, either pass a ``Generator`` or
               ``RandomState`` explicitly using `rng`, or call
               ``numpy.random.seed()`` beforehand.

        **kwargs
            Additional keyword arguments used by the solver, see
            `LikelihoodModel.fit`. This includes ``warn_convergence``.
            Robust covariance estimators are not available, so ``cov_type``
            must be "nonrobust" if it is passed.

            .. versionchanged:: 0.15.1

               ``fargs``, ``callback``, ``retall`` and the additional
               keyword arguments are forwarded to the solver. They were
               previously ignored.

        Returns
        -------
        MultinomialResultsWrapper
            The fitted model results.
        """

        _check_cov_type(kwargs)

        if start_params is None:
            q = self.exog.shape[1]
            c = self.k_cat - 1
            if rng is not None:
                generator = check_random_state(rng)
                start_params = generator.normal(size=q * c)
            else:
                warnings.warn(
                    "When start_params is not specified, random values are "
                    "used. After statsmodels 0.15 is released, the default method "
                    "for producing random values will be a new instance of a "
                    "numpy.random.Generator. To control the generation of "
                    "start_param using random values, pass a Generator or "
                    "RandomState using the ``generator`` keyword argument. ",
                    FutureWarning,
                    stacklevel=2,
                )
                start_params = np.random.normal(size=q * c)

        # Do not call super(...).fit because it cannot handle the 2d-params.
        rslt = base.LikelihoodModel.fit(
            self,
            start_params=start_params,
            method=method,
            maxiter=maxiter,
            full_output=full_output,
            disp=disp,
            fargs=fargs,
            callback=callback,
            retall=retall,
            skip_hessian=skip_hessian,
            **kwargs,
        )

        rslt.params = rslt.params.reshape((self.exog.shape[1], -1))
        rslt = MultinomialResults(self, rslt)

        # Not clear what the null likelihood should be, there is no intercept
        # so the null model is not clearly defined.  This is needed for summary
        # to work.
        rslt.set_null_options(llnull=np.nan)

        return MultinomialResultsWrapper(rslt)

    def loglike(self, params):
        """
        Log-likelihood of the conditional multinomial logit model.

        Parameters
        ----------
        params : ndarray
            The flattened parameter array, of length
            ``exog.shape[1] * (k_cat - 1)``.

        Returns
        -------
        float
            The log-likelihood value at `params`, summed over all
            groups.
        """

        q = self.exog.shape[1]
        c = self.k_cat - 1

        pmat = params.reshape((q, c))
        pmat = np.concatenate((np.zeros((q, 1)), pmat), axis=1)
        lpr = np.dot(self.exog, pmat)

        ll = 0.0

        # denom - immediately calculate the sums for the selected elements

        for ii in self._grp_ix:
            x = lpr[ii, :]
            jj = np.arange(x.shape[0], dtype=int)
            y = self.endog[ii]
            denom = np.sum(x[jj, list(itertools.permutations(y))], axis=1)
            ll += x[(jj, y)].sum() - logsumexp(denom)

        return ll

    def score(self, params):
        """
        Score vector of the conditional multinomial logit model.

        Parameters
        ----------
        params : ndarray
            The flattened parameter array, of length
            ``exog.shape[1] * (k_cat - 1)``.

        Returns
        -------
        ndarray
            The flattened score vector at `params`, summed over all
            groups.
        """

        q = self.exog.shape[1]
        c = self.k_cat - 1

        pmat = params.reshape((q, c))
        pmat = np.concatenate((np.zeros((q, 1)), pmat), axis=1)
        lpr = np.dot(self.exog, pmat)

        grad = np.zeros((q, c))
        for ii in self._grp_ix:
            x = lpr[ii, :]
            jj = np.arange(x.shape[0], dtype=int)
            y = self.endog[ii]
            denomg = np.zeros((q, c)).T

            # Extract itertools.permutations(y) to the list
            iter_ = np.array(list(itertools.permutations(y)))

            # Instead of iterative exponential value of sums of
            # selected elements and their product by selected
            # elements from self.exog, we calculate them at
            # once (exp_sum, exog_exp_multy).
            exp_sum = np.exp(np.sum(x[jj, iter_], axis=1))
            denom = np.sum(exp_sum)

            ind_exog = np.arange(iter_.shape[1], dtype=np.int32)
            hist = len(iter_)
            mask = iter_ != 0
            iexog = np.take(ind_exog, mask.nonzero()[1])
            iexog = iexog.reshape((hist, int(len(iexog) / hist)))
            ii_ = np.take(ii, iexog)
            exog_exp_multy = self.exog[ii_, :] * exp_sum[:, np.newaxis, np.newaxis]
            ind_iter = iter_[mask].reshape((hist, int(len(iter_[mask]) / hist))) - 1
            np.add.at(denomg, ind_iter, exog_exp_multy)
            denomg = denomg.T

            for i, r in enumerate(y):
                if r != 0:
                    grad[:, r - 1] += self.exog[ii[i], :]

            grad -= denomg / denom

        return grad.flatten()


class ConditionalResultsWrapper(lm.RegressionResultsWrapper):
    pass


wrap.populate_wrapper(ConditionalResultsWrapper, ConditionalResults)
