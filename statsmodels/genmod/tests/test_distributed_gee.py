import numpy as np
from numpy.testing import assert_allclose, assert_equal
import pandas as pd
import pytest

from statsmodels.genmod import cov_struct as cs, families
from statsmodels.genmod.api import DistributedGEE
from statsmodels.genmod.generalized_estimating_equations import GEE
from statsmodels.tools.sm_exceptions import ConvergenceWarning, SingularMatrixWarning
from statsmodels.tools.tools import add_constant

N_GROUPS, GROUP_SIZE, N_PARTS = 300, 4, 5
FAMILIES = {
    "gaussian": families.Gaussian,
    "poisson": families.Poisson,
    "binomial": families.Binomial,
}
# GEE stops on the norm of the score, so converge the reference tightly
FULL_FIT = dict(ctol=1e-12, maxiter=200)


def make_data(family, seed=0):
    rng = np.random.default_rng(seed)
    groups = np.repeat(np.arange(N_GROUPS), GROUP_SIZE)
    exog = add_constant(rng.normal(size=(len(groups), 2)))
    effect = np.repeat(rng.normal(scale=0.5, size=N_GROUPS), GROUP_SIZE)
    lin_pred = exog @ [0.3, 0.4, -0.3] + effect
    if family == "gaussian":
        endog = lin_pred + rng.normal(size=len(groups))
    elif family == "poisson":
        endog = rng.poisson(np.exp(lin_pred)).astype(float)
    else:
        endog = rng.binomial(1, 1 / (1 + np.exp(-lin_pred))).astype(float)
    return endog, exog, groups


def split(endog, exog, groups, n_parts=N_PARTS, **extra):
    parts = []
    for labels in np.array_split(np.unique(groups), n_parts):
        idx = np.isin(groups, labels)
        part = dict(endog=endog[idx], exog=exog[idx], groups=groups[idx])
        part.update({key: val[idx] for key, val in extra.items()})
        parts.append(part)
    return parts


def as_tuples(parts):
    return [(p["endog"], p["exog"], p["groups"]) for p in parts]


@pytest.mark.parametrize("family", list(FAMILIES))
def test_matches_gee_independence(family):
    endog, exog, groups = make_data(family)
    fam = FAMILIES[family]()
    full = GEE(endog, exog, groups, family=fam).fit(**FULL_FIT)
    res = DistributedGEE(as_tuples(split(endog, exog, groups)), family=fam).fit()
    assert res.converged
    assert_allclose(res.params, full.params, rtol=1e-10)
    # the covariance is from the last pass, before its step of up to
    # ctol=1e-8 standard errors
    assert_allclose(res.cov_params(), full.cov_robust, rtol=1e-7)
    assert_allclose(res.bse, full.bse, rtol=1e-7)
    assert_allclose(res.pvalues, full.pvalues, rtol=1e-7)
    assert_allclose(res.conf_int(), full.conf_int(), rtol=1e-10)
    assert_equal((res.nobs, res.num_group, res.n_partitions), (1200, 300, 5))


def test_newton_passes_converge():
    # the error of each pass is roughly the square of the previous one
    endog, exog, groups = make_data("poisson")
    fam = families.Poisson()
    full = GEE(endog, exog, groups, family=fam).fit(**FULL_FIT)
    parts = as_tuples(split(endog, exog, groups))
    errs = []
    for maxiter in range(4):
        with pytest.warns(ConvergenceWarning) if maxiter else _no_warning():
            res = DistributedGEE(parts, family=fam).fit(maxiter=maxiter, ctol=0)
        errs.append(np.max(np.abs(res.params - full.params) / full.bse))
    assert 1e-3 < errs[0] < 1
    assert errs[1] < 1e-2
    assert errs[2] < 1e-6
    assert errs[3] < 1e-12


class _no_warning:
    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False


def test_one_pass_gaussian_is_exact():
    # for linear estimating equations the aggregation is exact
    endog, exog, groups = make_data("gaussian")
    full = GEE(endog, exog, groups).fit(**FULL_FIT)
    res = DistributedGEE(as_tuples(split(endog, exog, groups))).fit(maxiter=0)
    assert_allclose(res.params, full.params, rtol=1e-10)
    assert not res.converged
    assert res.n_iter == 0


def test_one_pass_bse_close():
    endog, exog, groups = make_data("poisson")
    fam = families.Poisson()
    full = GEE(endog, exog, groups, family=fam).fit(**FULL_FIT)
    res = DistributedGEE(as_tuples(split(endog, exog, groups)), family=fam).fit(
        maxiter=0
    )
    assert_allclose(res.bse, full.bse, rtol=0.1)


def test_fixed_dependence_matches_gee():
    endog, exog, groups = make_data("poisson")
    fam = families.Poisson()
    cov = cs.Exchangeable()
    cov.dep_params = 0.3
    full = GEE(endog, exog, groups, family=fam, cov_struct=cov, update_dep=False)
    full = full.fit(**FULL_FIT)
    mod = DistributedGEE(
        as_tuples(split(endog, exog, groups)),
        family=fam,
        cov_struct=cov,
        update_dep=False,
    )
    res = mod.fit()
    assert_allclose(res.params, full.params, rtol=1e-10)
    assert_allclose(res.bse, full.bse, rtol=1e-7)
    assert_allclose(res.dep_params, [0.3] * N_PARTS)


def test_single_partition_exchangeable_matches_gee():
    endog, exog, groups = make_data("poisson")
    fam = families.Poisson()
    full = GEE(endog, exog, groups, family=fam, cov_struct=cs.Exchangeable())
    full = full.fit(**FULL_FIT)
    mod = DistributedGEE(
        [(endog, exog, groups)], family=fam, cov_struct=cs.Exchangeable()
    )
    res = mod.fit(ctol=1e-12, maxiter=50)
    assert_allclose(res.params, full.params, rtol=1e-8)
    assert_allclose(res.bse, full.bse, rtol=1e-6)
    assert_allclose(res.dep_params[0], full.cov_struct.dep_params, rtol=1e-6)


def test_partition_dependence_solves_equations():
    # with partition-specific dependence parameters the estimate solves the
    # combined estimating equations at those parameters
    endog, exog, groups = make_data("binomial")
    mod = DistributedGEE(
        as_tuples(split(endog, exog, groups)),
        family=families.Binomial(),
        cov_struct=cs.Exchangeable(),
    )
    res = mod.fit(ctol=1e-10, maxiter=50)
    assert res.converged
    amat, score, _, dep_params = mod._refine_pass(res.params)
    assert np.all(np.abs(np.linalg.solve(amat, score)) < 1e-8 * res.bse)
    assert len(set(dep_params)) == N_PARTS
    full = GEE(
        endog, exog, groups, family=families.Binomial(), cov_struct=cs.Exchangeable()
    ).fit(**FULL_FIT)
    # an approximation of the full-data GEE
    assert np.all(np.abs(res.params - full.params) < full.bse)


def test_exposure_in_dict_partitions():
    endog, exog, groups = make_data("poisson")
    exposure = np.random.default_rng(1).uniform(0.5, 2, size=len(endog))
    fam = families.Poisson()
    full = GEE(endog, exog, groups, family=fam, exposure=exposure).fit(**FULL_FIT)
    parts = split(endog, exog, groups, exposure=exposure)
    res = DistributedGEE(parts, family=fam).fit()
    assert_allclose(res.params, full.params, rtol=1e-10)
    assert_allclose(res.bse, full.bse, rtol=1e-7)
    pred = res.predict(exog, offset=np.log(exposure))
    assert_allclose(pred, full.predict(exog, exposure=exposure), rtol=1e-10)


def test_callable_partitions_and_names():
    endog, exog, groups = make_data("poisson")
    exog = pd.DataFrame(exog, columns=["const", "age", "dose"])
    endog = pd.Series(endog, name="count")
    parts = []
    for labels in np.array_split(np.unique(groups), N_PARTS):
        idx = np.isin(groups, labels)
        parts.append((endog[idx], exog[idx], groups[idx]))
    calls = []

    def reader():
        calls.append(1)
        return iter(parts)

    fam = families.Poisson()
    res = DistributedGEE(reader, family=fam).fit()
    res_list = DistributedGEE(parts, family=fam).fit()
    assert_allclose(res.params, res_list.params, rtol=1e-14)
    assert len(calls) == 1 + res.n_iter
    assert res.model.exog_names == ["const", "age", "dose"]
    text = res.summary().as_text()
    assert "dose" in text
    assert "count" in text
    assert "No. partitions:" in text
    assert_allclose(res.t_test("dose = 0").pvalue, res.pvalues[2], rtol=1e-12)


def test_use_t():
    endog, exog, groups = make_data("binomial")
    fam = families.Binomial()
    full = GEE(endog, exog, groups, family=fam).fit(use_t=True, **FULL_FIT)
    res = DistributedGEE(as_tuples(split(endog, exog, groups)), family=fam).fit(
        use_t=True
    )
    assert res.df_resid_inference == N_GROUPS - 3
    assert_allclose(res.pvalues, full.pvalues, rtol=1e-7)
    assert_allclose(res.conf_int(), full.conf_int(), rtol=1e-7)
    tt = res.t_test(np.eye(3))
    assert_allclose(tt.pvalue, full.pvalues, rtol=1e-7)
    assert_allclose(res.t_test("x1 = 0").pvalue, full.pvalues[1], rtol=1e-7)


def test_use_t_too_few_groups():
    endog, exog, groups = make_data("gaussian")
    idx = groups < 3
    with pytest.raises(ValueError, match="use_t requires more groups"):
        DistributedGEE([(endog[idx], exog[idx], groups[idx])]).fit(use_t=True)


def test_convergence_warning():
    endog, exog, groups = make_data("poisson")
    mod = DistributedGEE(
        as_tuples(split(endog, exog, groups)), family=families.Poisson()
    )
    with pytest.warns(ConvergenceWarning, match="did not converge in 1 passes"):
        res = mod.fit(maxiter=1)
    assert not res.converged


def test_iterator_raises():
    endog, exog, groups = make_data("gaussian")
    parts = iter(as_tuples(split(endog, exog, groups)))
    with pytest.raises(TypeError, match="consumed only once"):
        DistributedGEE(parts)


def test_overlapping_groups_raise():
    endog, exog, groups = make_data("gaussian")
    idx = np.arange(len(endog))
    # the cluster of the middle row is split across the two partitions
    first, second = idx < 601, idx >= 601
    parts = [
        (endog[first], exog[first], groups[first]),
        (endog[second], exog[second], groups[second]),
    ]
    with pytest.raises(ValueError, match="more than one partition"):
        DistributedGEE(parts).fit()


@pytest.mark.parametrize("key", ["weights", "constraint"])
def test_unsupported_raise(key):
    endog, exog, groups = make_data("gaussian")
    part = dict(endog=endog, exog=exog, groups=groups)
    part[key] = np.ones(len(endog)) if key == "weights" else (np.eye(3), 0)
    with pytest.raises(ValueError, match=f"does not support {key}"):
        DistributedGEE([part]).fit()


def test_mismatched_columns_raise():
    endog, exog, groups = make_data("gaussian")
    parts = as_tuples(split(endog, exog, groups, n_parts=2))
    cols = ["const", "a", "b"]
    parts = [
        (parts[0][0], pd.DataFrame(parts[0][1], columns=cols), parts[0][2]),
        (parts[1][0], pd.DataFrame(parts[1][1], columns=cols[::-1]), parts[1][2]),
    ]
    with pytest.raises(ValueError, match="same exog columns"):
        DistributedGEE(parts).fit()


def test_changed_partition_count_raises():
    endog, exog, groups = make_data("poisson")
    parts = as_tuples(split(endog, exog, groups))
    calls = []

    def reader():
        calls.append(1)
        return parts if len(calls) == 1 else parts[:-1]

    with pytest.raises(ValueError, match="4 partitions, but 5"):
        DistributedGEE(reader, family=families.Poisson()).fit()


def test_empty_raises():
    with pytest.raises(ValueError, match="partitions is empty"):
        DistributedGEE([]).fit()


def test_collinear_raises():
    endog, exog, groups = make_data("gaussian")
    exog = np.column_stack([exog, exog[:, 1]])
    # the GLM start values of each partition warn first
    with (
        pytest.warns(SingularMatrixWarning),
        pytest.raises(ValueError, match="information matrix is singular"),
    ):
        DistributedGEE(as_tuples(split(endog, exog, groups))).fit()


def test_negative_maxiter_raises():
    endog, exog, groups = make_data("gaussian")
    with pytest.raises(ValueError, match="maxiter must be non-negative"):
        DistributedGEE([(endog, exog, groups)]).fit(maxiter=-1)


def test_remove_data():
    endog, exog, groups = make_data("gaussian")
    res = DistributedGEE(as_tuples(split(endog, exog, groups))).fit()
    params = res.params.copy()
    res.remove_data()
    assert_allclose(res.params, params, rtol=1e-15)


def test_not_converged_inference_at_params():
    endog, exog, groups = make_data("poisson")
    mod = DistributedGEE(
        as_tuples(split(endog, exog, groups)), family=families.Poisson()
    )
    with pytest.warns(ConvergenceWarning):
        res = mod.fit(maxiter=1)
    amat, _, cmat, dep_params = mod._refine_pass(res.params)
    assert_allclose(res.cov_params(), mod._sandwich(amat, cmat), rtol=1e-12)


def test_default_maxiter_partition_dependence():
    # partition-specific dependence parameters slow the refinement down,
    # here a covariate left out of the model shifts across partitions
    rng = np.random.default_rng(0)
    n_groups, size, n_parts = 200, 5, 4
    groups = np.repeat(np.arange(n_groups), size)
    shift = np.repeat(np.linspace(-2, 2, n_parts), n_groups * size // n_parts)
    x = rng.normal(size=len(groups))
    effect = np.repeat(rng.normal(size=n_groups), size)
    endog = 1 + 0.5 * x + shift + rng.normal(size=len(groups)) + effect
    endog = endog + rng.normal(size=len(groups))
    parts = as_tuples(split(endog, add_constant(x), groups, n_parts=n_parts))
    res = DistributedGEE(parts, cov_struct=cs.Exchangeable()).fit()
    assert res.converged
    assert res.n_iter > 10


def test_update_dep_false_follows_gee_initialization():
    # Unstructured resets the dependence parameters at initialization, in
    # GEE as in each partition
    endog, exog, groups = make_data("gaussian")
    time = np.tile(np.arange(GROUP_SIZE), N_GROUPS)
    cov = cs.Unstructured()
    cov.dep_params = 0.3 + 0.7 * np.eye(GROUP_SIZE)
    full = GEE(endog, exog, groups, time=time, cov_struct=cov, update_dep=False)
    full = full.fit(**FULL_FIT)
    parts = split(endog, exog, groups, time=time)
    res = DistributedGEE(parts, cov_struct=cov, update_dep=False).fit()
    for dep in res.dep_params:
        assert_allclose(dep, full.cov_struct.dep_params, rtol=1e-15)
    assert_allclose(res.params, full.params, rtol=1e-10)


def test_fitted_cov_struct_not_copied_with_model():
    endog, exog, groups = make_data("poisson")
    fam = families.Poisson()
    fitted = GEE(endog, exog, groups, family=fam, cov_struct=cs.Exchangeable())
    cov = fitted.fit(**FULL_FIT).model.cov_struct
    mod = DistributedGEE(
        as_tuples(split(endog, exog, groups)),
        family=fam,
        cov_struct=cov,
        update_dep=False,
    )
    assert mod._cov_template.model is None
    assert cov.model is fitted
    res = mod.fit()
    fresh = cs.Exchangeable()
    fresh.dep_params = cov.dep_params
    ref = DistributedGEE(
        as_tuples(split(endog, exog, groups)),
        family=fam,
        cov_struct=fresh,
        update_dep=False,
    ).fit()
    assert_allclose(res.params, ref.params, rtol=1e-14)


@pytest.mark.parametrize("key", ["family", "cov_struct", "missing", "update_dep"])
def test_reserved_keys_raise(key):
    endog, exog, groups = make_data("gaussian")
    part = dict(endog=endog, exog=exog, groups=groups)
    part[key] = None
    with pytest.raises(ValueError, match=f"Set {key} in DistributedGEE"):
        DistributedGEE([part]).fit()
