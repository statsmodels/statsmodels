"""
Divide-and-conquer generalized estimating equations.

References
----------
Lin, N. and Xi, R. (2011). Aggregated estimating equation estimation.
Statistics and Its Interface, 4(1), 73-83.
"""

from collections.abc import Mapping
import copy
import warnings

import numpy as np

from statsmodels.base.model import LikelihoodModelResults
from statsmodels.genmod import cov_struct as cov_structs, families
from statsmodels.genmod.generalized_estimating_equations import GEE
from statsmodels.tools.sm_exceptions import ConvergenceWarning
from statsmodels.tools.validation import bool_like, float_like, int_like

_UNSUPPORTED = ("weights", "constraint")
_RESERVED = ("family", "cov_struct", "missing", "update_dep")


class _NamesData:
    """The variable names that results methods read from ``model.data``."""

    def __init__(self, xnames, ynames):
        self.xnames = self.param_names = self.cov_names = xnames
        self.ynames = ynames


def _partition_stats(model, params):
    """
    Information, score and robust meat of one partition at ``params``.

    Returns ``(A, U, C)`` with ``A = sum_i D_i' V_i^{-1} D_i``,
    ``U = sum_i D_i' V_i^{-1} r_i`` and ``C = sum_i s_i s_i'`` where
    ``s_i = D_i' V_i^{-1} r_i``, summed over the clusters of the partition.
    """
    model.update_cached_means(params)
    varfunc = model.family.variance
    k = len(params)
    amat, score, cmat = np.zeros((k, k)), np.zeros(k), np.zeros((k, k))
    for i in range(model.num_group):
        expval, lpr = model.cached_means[i]
        resid = model.endog_li[i] - expval
        dmat = model.mean_deriv(model.exog_li[i], lpr)
        sdev = np.sqrt(varfunc(expval))
        rslt = model.cov_struct.covariance_matrix_solve(expval, i, sdev, (dmat, resid))
        if rslt is None:
            raise ValueError(
                f"The working covariance of group {model.group_labels[i]} is singular."
            )
        vinv_d, vinv_resid = rslt
        amat += dmat.T @ vinv_d
        s = dmat.T @ vinv_resid
        score += s
        cmat += np.outer(s, s)
    return amat, score, cmat


class DistributedGEE:
    """
    Divide-and-conquer generalized estimating equations

    Fits a GEE to data that are split into partitions of whole clusters,
    holding one partition in memory at a time. A first pass fits a ``GEE``
    to each partition and combines the estimates with the aggregated
    estimating equations of Lin and Xi (2011). Further passes take Newton
    steps on the estimating equations of all the data, so that with an
    independence working correlation the estimate converges to the
    full-data ``GEE`` estimate.

    .. versionadded:: 0.15.1

    Parameters
    ----------
    partitions : sequence or callable
        The partitions of the data. Either a sequence that can be iterated
        over more than once, or a callable without arguments that returns a
        new iterable over the partitions at each call, for example a
        function that reads the partitions from disk. Each partition is a
        tuple ``(endog, exog, groups)`` or a dict of keyword arguments of
        ``GEE``, which must include ``endog``, ``exog`` and ``groups`` and
        can include ``offset``, ``exposure``, ``time`` and ``dep_data``.
    family : Family, optional
        The family of the model. The default is Gaussian.
    cov_struct : CovStruct, optional
        The working dependence structure. A copy is used in each partition.
        The default is independence.
    missing : str, optional
        How each partition handles missing values, see ``GEE``. The default
        is "none".
    update_dep : bool, optional
        If True (the default), each partition estimates the dependence
        parameters. If False, each partition keeps the dependence parameters
        that ``cov_struct`` has after it is initialized for the partition,
        as ``GEE`` with ``update_dep=False`` does. Some structures, such as
        ``Unstructured``, reset them at initialization.

    See Also
    --------
    statsmodels.genmod.generalized_estimating_equations.GEE
        The GEE model fit to each partition.
    statsmodels.base.distributed_estimation.DistributedModel
        Averages estimates of models fit to the partitions.

    Notes
    -----
    A cluster must lie in one partition, since GEE treats clusters as
    independent and the robust covariance sums over clusters. Group labels
    must therefore be distinct across partitions.

    In the first pass each partition ``k`` returns its estimate ``b_k``, its
    information ``A_k = sum_i D_i' V_i^{-1} D_i`` and its robust meat
    ``C_k = sum_i s_i s_i'``, where ``s_i = D_i' V_i^{-1} r_i`` is the score
    of cluster ``i``. The combined estimate is
    ``b = A^{-1} sum_k A_k b_k`` with ``A = sum_k A_k``. Each further pass
    evaluates ``A_k``, the score ``U_k`` and ``C_k`` of every partition at
    ``b`` and updates ``b <- b + A^{-1} sum_k U_k``. The covariance is the
    robust sandwich ``A^{-1} (sum_k C_k) A^{-1}`` from the last pass. If
    the refinement does not converge, one more pass evaluates it, and the
    dependence parameters, at the returned estimate. The one-pass estimate
    (``maxiter=0``) uses the sandwich of the first pass, with ``C_k`` at
    ``b_k``.

    With a working dependence structure other than independence, each
    partition estimates its own dependence parameters, at ``b`` in each
    pass. The estimate then solves the estimating equations of all the
    data with these partition-specific working correlations. It is
    consistent and its robust covariance is valid, but it differs from the
    full-data ``GEE`` estimate, which uses one set of dependence
    parameters. With ``update_dep=False`` and a structure that keeps its
    dependence parameters at initialization, such as ``Exchangeable``, all
    partitions share them, and the estimate converges to that of ``GEE``
    with ``update_dep=False``. The partition-specific parameters change
    with ``b``, so the refinement converges more slowly than with
    independence.

    The first pass fits a ``GEE`` to each partition, so the design matrix
    of each partition must have full rank. A column that is constant zero
    in a partition, such as the indicator of a category absent from it,
    makes that fit fail or warn. The number of clusters is known only after
    the first pass, so ``use_t`` with too few clusters raises after it.

    Weights and constraints are not supported, and only the robust
    covariance is available.

    References
    ----------
    .. [*] Lin, N. and Xi, R. (2011). Aggregated estimating equation
       estimation. Statistics and Its Interface, 4(1), 73-83.

    Examples
    --------
    >>> import numpy as np
    >>> import statsmodels.api as sm
    >>> from statsmodels.genmod.api import DistributedGEE
    >>> rng = np.random.default_rng(0)
    >>> parts = []
    >>> for k in range(4):
    ...     groups = np.repeat(np.arange(50) + 50 * k, 3)
    ...     exog = sm.add_constant(rng.normal(size=(150, 2)))
    ...     endog = rng.poisson(np.exp(exog @ [0.2, 0.3, -0.2]))
    ...     parts.append((endog, exog, groups))
    >>> model = DistributedGEE(parts, family=sm.families.Poisson())
    >>> results = model.fit()
    """

    # the model keeps no data, used by remove_data of the results
    _data_attr = []

    def __init__(
        self,
        partitions,
        family=None,
        cov_struct=None,
        missing="none",
        update_dep=True,
    ):
        if not callable(partitions):
            if iter(partitions) is partitions:
                raise TypeError(
                    "partitions is an iterator, which can be consumed only "
                    "once. Pass a sequence, or a callable that returns a new "
                    "iterable over the partitions."
                )
        self.partitions = partitions
        self.family = families.Gaussian() if family is None else family
        if cov_struct is None:
            cov_struct = cov_structs.Independence()
        self.cov_struct = cov_struct
        # copied for each partition, without the model and data of an
        # earlier fit that cov_struct may hold
        memo = {id(getattr(cov_struct, "model", None)): None}
        self._cov_template = copy.deepcopy(cov_struct, memo)
        self.missing = missing
        self.update_dep = bool_like(update_dep, "update_dep")

    def _iter_models(self):
        """Yield a ``GEE`` model for each partition."""
        parts = self.partitions() if callable(self.partitions) else self.partitions
        for part in parts:
            if isinstance(part, Mapping):
                kwds = dict(part)
            else:
                endog, exog, groups = part
                kwds = dict(endog=endog, exog=exog, groups=groups)
            for key in _UNSUPPORTED:
                if kwds.get(key) is not None:
                    raise ValueError(f"DistributedGEE does not support {key}.")
            for key in _RESERVED:
                if key in kwds:
                    raise ValueError(
                        f"Set {key} in DistributedGEE, not in a partition."
                    )
            yield GEE(
                family=self.family,
                cov_struct=copy.deepcopy(self._cov_template),
                missing=self.missing,
                update_dep=self.update_dep,
                **kwds,
            )

    def _solve(self, amat, vec):
        if np.linalg.matrix_rank(amat) < amat.shape[0]:
            raise ValueError(
                "The combined information matrix is singular, check the "
                "design matrix for collinear columns."
            )
        return np.linalg.solve(amat, vec)

    def _first_pass(self):
        """Fit each partition and combine the estimates."""
        amat = cmat = amat_b = 0
        nobs = num_group = 0
        labels = set()
        dep_params = []
        self.exog_names = None
        for model in self._iter_models():
            if self.exog_names is None:
                self.exog_names = list(model.exog_names)
                self.endog_names = model.endog_names
            elif list(model.exog_names) != self.exog_names:
                raise ValueError(
                    "All partitions must have the same exog columns, found "
                    f"{list(model.exog_names)} and {self.exog_names}."
                )
            new = set(model.group_labels.tolist())
            if not labels.isdisjoint(new):
                raise ValueError(
                    "A group appears in more than one partition. Each cluster "
                    "must lie in one partition, with group labels distinct "
                    "across partitions."
                )
            labels |= new
            nobs += model.nobs
            num_group += model.num_group

            rslt = model.fit()
            if rslt is None:
                raise ValueError(f"GEE failed to fit partition {len(dep_params)}.")
            params_k = np.asarray(rslt.params)
            amat_k, _, cmat_k = _partition_stats(model, params_k)
            amat = amat + amat_k
            amat_b = amat_b + amat_k @ params_k
            cmat = cmat + cmat_k
            dep_params.append(copy.deepcopy(model.cov_struct.dep_params))

        if self.exog_names is None:
            raise ValueError("partitions is empty.")
        self.nobs = nobs
        self.num_group = num_group
        self.n_partitions = len(dep_params)
        self.k_params = len(self.exog_names)
        self.data = _NamesData(self.exog_names, self.endog_names)
        params = self._solve(amat, amat_b)
        return params, amat, cmat, dep_params

    def _refine_pass(self, params):
        """Evaluate the combined information, score and meat at ``params``."""
        amat = score = cmat = 0
        dep_params = []
        for model in self._iter_models():
            # set as GEE.fit does, used by estimate_scale in cov_struct.update
            model.scaletype = None
            model.ddof_scale = self.k_params
            model.scaling_factor = 1.0
            model.update_cached_means(params)
            if model.update_dep:
                model.cov_struct.update(params)
            amat_k, score_k, cmat_k = _partition_stats(model, params)
            amat = amat + amat_k
            score = score + score_k
            cmat = cmat + cmat_k
            dep_params.append(copy.deepcopy(model.cov_struct.dep_params))
        if len(dep_params) != self.n_partitions:
            raise ValueError(
                f"partitions returned {len(dep_params)} partitions, but "
                f"{self.n_partitions} in the first pass."
            )
        return amat, score, cmat, dep_params

    def fit(self, maxiter=50, ctol=1e-8, use_t=False):
        """
        Fit the model

        Parameters
        ----------
        maxiter : int, optional
            The largest number of refinement passes over the data after the
            first pass. With 0, the one-pass estimate is returned. The
            default is 50.
        ctol : float, optional
            The refinement stops when no parameter changes by more than
            ``ctol`` times its standard error in a pass. The default is
            1e-8.
        use_t : bool, optional
            If True, p-values and confidence intervals use the t
            distribution with the number of clusters minus the number of
            parameters as degrees of freedom. The default is False.

        Returns
        -------
        DistributedGEEResults
            The results of the fit.
        """
        maxiter = int_like(maxiter, "maxiter")
        if maxiter < 0:
            raise ValueError("maxiter must be non-negative.")
        ctol = float_like(ctol, "ctol")
        use_t = bool_like(use_t, "use_t")

        params, amat, cmat, dep_params = self._first_pass()
        if use_t and self.num_group <= self.k_params:
            raise ValueError(
                f"use_t requires more groups ({self.num_group}) than "
                f"parameters ({self.k_params})."
            )

        converged = False
        n_iter = 0
        while n_iter < maxiter:
            n_iter += 1
            amat, score, cmat, dep_params = self._refine_pass(params)
            step = self._solve(amat, score)
            bse = np.sqrt(np.diag(self._sandwich(amat, cmat)))
            params = params + step
            if np.all(np.abs(step) <= ctol * bse):
                converged = True
                break
        if maxiter > 0 and not converged:
            warnings.warn(
                f"The refinement did not converge in {maxiter} passes.",
                ConvergenceWarning,
                stacklevel=2,
            )
            # inference at the returned params, not before the last step
            amat, _, cmat, dep_params = self._refine_pass(params)

        cov = self._sandwich(amat, cmat)
        return DistributedGEEResults(
            self,
            params,
            cov,
            use_t=use_t,
            converged=converged,
            n_iter=n_iter,
            dep_params=dep_params,
        )

    def _sandwich(self, amat, cmat):
        amati = np.linalg.inv(amat)
        return amati @ cmat @ amati

    def predict(self, params, exog, offset=None):
        """
        Predict the mean of the response

        Parameters
        ----------
        params : array_like
            The parameters of the model.
        exog : array_like
            The design matrix at which to predict.
        offset : array_like, optional
            Offset of the linear predictor. Include the log of an exposure
            here.

        Returns
        -------
        ndarray
            The predicted mean.
        """
        lin_pred = np.asarray(exog, dtype=float) @ np.asarray(params)
        if offset is not None:
            lin_pred = lin_pred + np.asarray(offset, dtype=float)
        return self.family.link.inverse(lin_pred)


class DistributedGEEResults(LikelihoodModelResults):
    """
    Results of a divide-and-conquer GEE fit

    .. versionadded:: 0.15.1

    Parameters
    ----------
    model : DistributedGEE
        The fitted model.
    params : ndarray
        The estimated parameters.
    cov_params : ndarray
        The robust covariance of the parameters.
    use_t : bool
        Whether inference uses the t distribution.
    converged : bool
        Whether the refinement converged. False if there was none.
    n_iter : int
        The number of refinement passes.
    dep_params : list
        The dependence parameters of each partition in the last pass.

    Attributes
    ----------
    cov_type : str
        The covariance type, always "robust".
    nobs : int
        The number of observations over all partitions.
    num_group : int
        The number of clusters over all partitions.
    n_partitions : int
        The number of partitions.
    df_resid_inference : int
        The degrees of freedom used if ``use_t`` is True, the number of
        clusters minus the number of parameters.
    converged : bool
        Whether the refinement converged. False if there was none.
    n_iter : int
        The number of refinement passes.
    dep_params : list
        The dependence parameters of each partition in the last pass.
    """

    def __init__(self, model, params, cov_params, use_t, converged, n_iter, dep_params):
        super().__init__(
            model, params, normalized_cov_params=cov_params, scale=1.0, use_t=use_t
        )
        self.cov_type = "robust"
        self.nobs = model.nobs
        self.num_group = model.num_group
        self.n_partitions = model.n_partitions
        self.df_resid = model.nobs - model.k_params
        self.df_resid_inference = model.num_group - model.k_params
        self.converged = converged
        self.n_iter = n_iter
        self.dep_params = dep_params

    def predict(self, exog, offset=None):
        """
        Predict the mean of the response

        Parameters
        ----------
        exog : array_like
            The design matrix at which to predict. It is required since the
            model does not keep the data.
        offset : array_like, optional
            Offset of the linear predictor. Include the log of an exposure
            here.

        Returns
        -------
        ndarray
            The predicted mean.
        """
        return self.model.predict(self.params, exog, offset=offset)

    def summary(self, yname=None, xname=None, title=None, alpha=0.05):
        """
        Summarize the results

        Parameters
        ----------
        yname : str, optional
            The name of the response. The default is the name in the first
            partition.
        xname : list of str, optional
            The names of the parameters. The default is the exog names of
            the first partition.
        title : str, optional
            The title of the top table.
        alpha : float, optional
            The intervals have coverage ``1 - alpha``. The default is 0.05.

        Returns
        -------
        Summary
            The summary tables.
        """
        from statsmodels.iolib.summary import Summary

        model = self.model
        if yname is None:
            yname = model.endog_names
        if xname is None:
            xname = model.exog_names
        if title is None:
            title = "DistributedGEE Regression Results"

        top_left = [
            ("Dep. Variable:", None),
            ("Model:", ["DistributedGEE"]),
            ("Family:", [model.family.__class__.__name__]),
            ("Dependence structure:", [model.cov_struct.__class__.__name__]),
            ("Covariance type:", [self.cov_type]),
            ("Date:", None),
        ]
        top_right = [
            ("No. Observations:", [self.nobs]),
            ("No. clusters:", [self.num_group]),
            ("No. partitions:", [self.n_partitions]),
            ("Refinement passes:", [self.n_iter]),
            ("Converged:", [str(self.converged)]),
            ("Time:", None),
        ]
        smry = Summary()
        smry.add_table_2cols(
            self,
            gleft=top_left,
            gright=top_right,
            yname=yname,
            xname=xname,
            title=title,
        )
        smry.add_table_params(
            self, yname=yname, xname=xname, alpha=alpha, use_t=self.use_t
        )
        return smry
