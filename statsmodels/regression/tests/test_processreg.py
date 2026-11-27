"""Tests for process regression models."""

from statsmodels.compat.platform import PLATFORM_OSX

import collections
import inspect

import numpy as np
from numpy.testing import assert_allclose, assert_equal
import pandas as pd
import pytest

from statsmodels.iolib.summary2 import Summary
from statsmodels.regression.process_regression import GaussianCovariance, ProcessMLE
from statsmodels.tools.docstring import Docstring
import statsmodels.tools.numdiff as nd
from statsmodels.tools.sm_exceptions import MissingDataError


# Parameters for a test model, with or without additive
# noise.
def model1(noise):

    mn_par = np.r_[1, 0, -1, 0]
    sc_par = np.r_[1, 1]
    sm_par = np.r_[0.5, 0.1]

    if noise:
        no_par = np.r_[0.25, 0.25]
    else:
        no_par = np.array([])

    return mn_par, sc_par, sm_par, no_par


def setup1(n, get_model, noise, rng):
    mn_par, sc_par, sm_par, no_par = get_model(noise)

    groups = np.kron(np.arange(n // 5), np.ones(5))
    time = np.kron(np.ones(n // 5), np.arange(5))
    time_z = (time - time.mean()) / time.std()

    x_mean = rng.normal(size=(n, len(mn_par)))
    x_sc = rng.normal(size=(n, len(sc_par)))
    x_sc[:, 0] = 1
    x_sc[:, 1] = time_z
    x_sm = rng.normal(size=(n, len(sm_par)))
    x_sm[:, 0] = 1
    x_sm[:, 1] = time_z

    mn = np.dot(x_mean, mn_par)
    sc = np.exp(np.dot(x_sc, sc_par))
    sm = np.exp(np.dot(x_sm, sm_par))

    if noise:
        x_no = rng.normal(size=(n, len(no_par)))
        x_no[:, 0] = 1
        x_no[:, 1] = time_z
        no = np.exp(np.dot(x_no, no_par))
    else:
        x_no = None

    y = mn.copy()

    gc = GaussianCovariance()

    ix = collections.defaultdict(list)
    for i, g in enumerate(groups):
        ix[g].append(i)

    for ii in ix.values():
        c = gc.get_cov(time[ii], sc[ii], sm[ii])
        r = np.linalg.cholesky(c)
        y[ii] += np.dot(r, rng.normal(size=len(ii)))

    # Additive white noise
    if noise:
        y += no * rng.normal(size=y.shape)

    return y, x_mean, x_sc, x_sm, x_no, time, groups


def run_arrays(n, get_model, noise, rng):

    y, x_mean, x_sc, x_sm, x_no, time, groups = setup1(n, get_model, noise, rng)

    preg = ProcessMLE(y, x_mean, x_sc, x_sm, x_no, time, groups)

    return preg.fit()


@pytest.mark.slow
@pytest.mark.high_memory
@pytest.mark.parametrize("noise", [False, True])
def test_arrays(noise):

    rs = np.random.RandomState(8234)

    f = run_arrays(1000, model1, noise, rng=rs)
    mod = f.model

    f.summary()  # Smoke test

    # Compare the parameter estimates to population values.
    epar = np.concatenate(model1(noise))
    assert_allclose(f.params, epar, atol=0.3, rtol=0.3)

    # Test the fitted covariance matrix
    cv = f.covariance(mod.time[0:5], mod.exog_scale[0:5, :], mod.exog_smooth[0:5, :])
    assert_allclose(cv, cv.T)  # Check symmetry
    a, _ = np.linalg.eig(cv)
    assert_equal(a > 0, True)  # Check PSD

    # Test predict
    yhat = f.predict()
    assert_equal(np.corrcoef(yhat, mod.endog)[0, 1] > 0.2, True)
    yhatm = f.predict(exog=mod.exog)
    assert_allclose(yhat, yhatm, rtol=1e-10)
    yhat0 = mod.predict(params=f.params, exog=mod.exog)
    assert_allclose(yhat, yhat0, rtol=1e-10)

    # Smoke test t-test
    f.t_test(np.eye(len(f.params)))


def run_formula(n, get_model, noise, rng):

    y, x_mean, x_sc, x_sm, x_no, time, groups = setup1(n, get_model, noise, rng)

    df = pd.DataFrame(
        {
            "y": y,
            "x1": x_mean[:, 0],
            "x2": x_mean[:, 1],
            "x3": x_mean[:, 2],
            "x4": x_mean[:, 3],
            "xsc1": x_sc[:, 0],
            "xsc2": x_sc[:, 1],
            "xsm1": x_sm[:, 0],
            "xsm2": x_sm[:, 1],
            "time": time,
            "groups": groups,
        }
    )

    if noise:
        df["xno1"] = x_no[:, 0]
        df["xno2"] = x_no[:, 1]

    mean_formula = "y ~ 0 + x1 + x2 + x3 + x4"
    scale_formula = "0 + xsc1 + xsc2"
    smooth_formula = "0 + xsm1 + xsm2"

    if noise:
        noise_formula = "0 + xno1 + xno2"
    else:
        noise_formula = None

    preg = ProcessMLE.from_formula(
        mean_formula,
        data=df,
        scale_formula=scale_formula,
        smooth_formula=smooth_formula,
        noise_formula=noise_formula,
        time="time",
        groups="groups",
    )
    f = preg.fit()

    return f, df


@pytest.mark.slow
@pytest.mark.high_memory
@pytest.mark.parametrize("noise", [False, True])
def test_formulas(noise):

    rs = np.random.RandomState(8789)

    f, df = run_formula(1000, model1, noise, rng=rs)
    mod = f.model

    f.summary()  # Smoke test

    # Compare the parameter estimates to population values.
    epar = np.concatenate(model1(noise))
    assert_allclose(f.params, epar, atol=0.1, rtol=1)

    # Test the fitted covariance matrix
    exog_scale = pd.DataFrame(mod.exog_scale[0:5, :], columns=["xsc1", "xsc2"])
    exog_smooth = pd.DataFrame(mod.exog_smooth[0:5, :], columns=["xsm1", "xsm2"])
    cv = f.covariance(mod.time[0:5], exog_scale, exog_smooth)
    assert_allclose(cv, cv.T)
    a, _ = np.linalg.eig(cv)
    assert_equal(a > 0, True)

    # Test predict
    yhat = f.predict()
    assert_equal(np.corrcoef(yhat, mod.endog)[0, 1] > 0.2, True)
    yhatm = f.predict(exog=df)
    assert_allclose(yhat, yhatm, rtol=1e-10)
    yhat0 = mod.predict(params=f.params, exog=df)
    assert_allclose(yhat, yhat0, rtol=1e-10)

    # Smoke test t-test
    f.t_test(np.eye(len(f.params)))


# Test the score functions using numerical derivatives.
@pytest.mark.parametrize("noise", [False, True])
def test_score_numdiff(noise):
    rs = np.random.RandomState(3422121)
    y, x_mean, x_sc, x_sm, x_no, time, groups = setup1(1000, model1, noise, rng=rs)

    preg = ProcessMLE(y, x_mean, x_sc, x_sm, x_no, time, groups)

    def loglike(x):
        return preg.loglike(x)

    q = x_mean.shape[1] + x_sc.shape[1] + x_sm.shape[1]
    if noise:
        q += x_no.shape[1]

    rs = np.random.RandomState(342)

    atol = 2e-3 if PLATFORM_OSX else 1e-2
    for _ in range(5):
        par0 = preg._get_start()
        par = par0 + 0.1 * rs.normal(size=q)
        score = preg.score(par)
        score_nd = nd.approx_fprime(par, loglike, epsilon=1e-7)
        assert_allclose(score, score_nd, atol=atol, rtol=1e-4)


def test_summary_after_remove_data():
    # summary() must still work after remove_data() has been called
    rs = np.random.RandomState(8234)
    res = run_arrays(50, model1, False, rng=rs)

    assert isinstance(res.summary(), Summary)
    res.remove_data()
    assert isinstance(res.summary(), Summary)


def test_split_param_names_partitions_xnames():
    rs = np.random.RandomState(8234)
    y, x_mean, x_sc, x_sm, x_no, time, groups = setup1(50, model1, True, rs)
    mod = ProcessMLE(y, x_mean, x_sc, x_sm, x_no, time, groups)

    mean_names, scale_names, smooth_names, noise_names = mod._split_param_names()
    assert list(mean_names) + list(scale_names) + list(smooth_names) + \
        list(noise_names) == list(mod.data.param_names)
    assert len(mean_names) == mod.k_exog
    assert len(scale_names) == mod.k_scale
    assert len(smooth_names) == mod.k_smooth
    assert len(noise_names) == mod.k_noise


def test_covariance_group_matches_direct_call():
    rs = np.random.RandomState(8234)
    res = run_arrays(50, model1, False, rng=rs)
    mod = res.model

    group = next(iter(mod._groups_ix))
    ix = mod._groups_ix[group]
    cov = res.covariance_group(group)

    assert cov.shape == (len(ix), len(ix))
    assert_allclose(cov, cov.T)
    assert np.all(np.linalg.eigvalsh(cov) > -1e-8 * np.abs(cov).max())

    # rebuild it "by hand" from the model's own covariance() kernel to
    # check covariance_group's indexing/column-selection, independent of
    # its own internal computation of scale_data/smooth_data
    _, scale_names, smooth_names, _ = mod._split_param_names()
    scale_data = pd.DataFrame(mod.exog_scale[ix, :], columns=scale_names)
    smooth_data = pd.DataFrame(mod.exog_smooth[ix, :], columns=smooth_names)
    expected = mod.covariance(
        mod.time[ix], res.scale_params, res.smooth_params, scale_data, smooth_data)
    assert_allclose(cov, expected)

    with pytest.raises(ValueError, match="does not exist"):
        res.covariance_group("not-a-real-group")


def test_covariance_matches_gaussian_kernel_formula():
    # ProcessMLE.covariance and ProcessMLEResults.covariance had no direct
    # test coverage of their own: they were previously only exercised
    # (with real assertions) inside the @pytest.mark.slow test_arrays/
    # test_formulas, and only incidentally, as the "expected" ground truth,
    # by test_covariance_group_matches_direct_call above.
    #
    # Reproduce GaussianCovariance's documented squared-exponential kernel
    # formula by hand -- independent of GaussianCovariance.get_cov's own
    # code -- and check both covariance() methods against it.
    rng = np.random.default_rng(20230)
    res = run_arrays(50, model1, False, rng=rng)
    mod = res.model

    idx = np.arange(5)
    t0 = mod.time[idx]
    scale_data0 = mod.exog_scale[idx, :]
    smooth_data0 = mod.exog_smooth[idx, :]

    # scale/smooth use a log link to preserve positivity (GaussianCovariance
    # docstring; also matches ProcessMLE.loglike/score's sc/sm computation).
    sca = np.exp(scale_data0 @ np.asarray(res.scale_params))
    smo = np.exp(smooth_data0 @ np.asarray(res.smooth_params))
    da = np.subtract.outer(t0, t0)
    ds = np.add.outer(smo, smo) / 2
    qmat = da * da / ds
    expected = np.exp(-qmat / 2) / np.sqrt(ds)
    expected *= np.outer(smo, smo) ** 0.25
    expected *= np.outer(sca, sca)

    # ProcessMLE.covariance is the model-level method.
    cv = mod.covariance(
        t0, res.scale_params, res.smooth_params, scale_data0, smooth_data0
    )
    assert_allclose(cv, cv.T)
    assert_allclose(cv, expected, rtol=1e-10)

    # ProcessMLEResults.covariance forwards to model.covariance using the
    # fitted scale/smooth params.
    cv_res = res.covariance(t0, scale_data0, smooth_data0)
    assert_allclose(cv_res, cv, rtol=1e-12)


def test_predict_matches_mean_structure_formula():
    # ProcessMLE.predict and ProcessMLEResults.predict had no direct test
    # coverage of their own: they were previously only exercised inside the
    # @pytest.mark.slow test_arrays/test_formulas. Both are documented to
    # return the linear mean structure exog @ mean_params (no conditioning
    # on the fitted Gaussian process); verify against that formula by hand.
    rng = np.random.default_rng(20231)
    res = run_arrays(50, model1, False, rng=rng)
    mod = res.model

    exog0 = mod.exog[:6, :]
    expected = exog0 @ np.asarray(res.mean_params)

    # ProcessMLE.predict is the model-level method.
    yhat_model = mod.predict(res.params, exog=exog0)
    assert_allclose(yhat_model, expected, rtol=1e-10)

    # ProcessMLEResults.predict forwards to model.predict using the fitted
    # params.
    yhat_res = res.predict(exog=exog0)
    assert_allclose(yhat_res, yhat_model, rtol=1e-12)

    # default exog (None) falls back to the model's own exog.
    assert_allclose(res.predict(), mod.exog @ np.asarray(res.mean_params), rtol=1e-10)


class _DoubledCovariance(GaussianCovariance):
    # Twice the Gaussian covariance, so that the log-likelihood depends on
    # which covariance the model uses
    def get_cov(self, time, sc, sm):
        return 2 * super().get_cov(time, sc, sm)


def _formula_data(nobs=40):
    rng = np.random.default_rng(20261126)
    return pd.DataFrame(
        {
            "y": rng.standard_normal(nobs),
            "x": rng.standard_normal(nobs),
            "g": np.repeat(np.arange(nobs // 5), 5),
            "time": np.tile(np.arange(5.0), nobs // 5),
        }
    )


def test_from_formula_documented_parameters_in_signature():
    # Every documented parameter is a parameter of the signature, and every
    # parameter of the signature is documented
    doc = Docstring(ProcessMLE.from_formula.__doc__)
    documented = [param.name for param in doc._ds["Parameters"]]
    signature = []
    for param in inspect.signature(ProcessMLE.from_formula).parameters.values():
        prefix = "**" if param.kind is param.VAR_KEYWORD else ""
        signature.append(prefix + param.name)
    assert documented == signature


def test_from_formula_required_and_positional_arguments():
    data = _formula_data()
    with pytest.raises(TypeError, match="scale_formula"):
        ProcessMLE.from_formula(
            "y ~ x", data, smooth_formula="0 + x", time="time", groups="g"
        )
    # Arguments after drop_cols are keyword-only (positional ones were ignored)
    with pytest.raises(TypeError, match="positional"):
        ProcessMLE.from_formula(
            "y ~ x",
            data,
            None,
            None,
            "0 + x",
            scale_formula="0 + x",
            smooth_formula="0 + x",
            time="time",
            groups="g",
        )


def test_from_formula_passes_other_keywords():
    # Keywords of the base model, such as hasconst, are used
    data = _formula_data()
    kwargs = dict(
        scale_formula="0 + x", smooth_formula="0 + x", time="time", groups="g"
    )
    # The intercept is found as the constant by default
    assert ProcessMLE.from_formula("y ~ x", data, **kwargs).k_constant == 1
    mod = ProcessMLE.from_formula("y ~ x", data, hasconst=False, **kwargs)
    assert mod.k_constant == 0


def test_from_formula_passes_model_keywords():
    # cov, and keywords that from_formula does not use itself, are passed to
    # the model
    data = _formula_data()
    nobs = data.shape[0]
    cov = _DoubledCovariance()
    mod = ProcessMLE.from_formula(
        "y ~ x",
        data,
        scale_formula="0 + x",
        smooth_formula="0 + x",
        time="time",
        groups="g",
        cov=cov,
    )
    assert mod.cov is cov

    exog = np.column_stack([np.ones(nobs), data["x"]])
    x = data[["x"]].to_numpy()
    time = data["time"].to_numpy()
    groups = data["g"].to_numpy()
    args = (data["y"].to_numpy(), exog, x, x, None, time, groups)
    direct = ProcessMLE(*args, cov=_DoubledCovariance())
    default = ProcessMLE(*args)
    params = np.array([0.1, -0.2, 0.3, 0.2])
    expected = direct.loglike(params)
    assert_allclose(mod.loglike(params), expected, rtol=1e-12)
    # The covariance changes the log-likelihood
    assert abs(expected - default.loglike(params)) > 1.0


def _missing_data():
    data = _formula_data()
    data["z"] = np.random.default_rng(11231).standard_normal(data.shape[0])
    return data


@pytest.mark.parametrize(
    "nan_cells, scale, smooth, noise",
    [
        # A missing value in the mean formula, with one- and two-column
        # scale and smoothness formulas
        ([(3, "y")], "0 + x", "0 + x", None),
        ([(3, "y")], "x", "x", None),
        # Missing values only in the scale, smoothness or noise formula
        ([(7, "z")], "z", "x", None),
        ([(7, "z")], "x", "z", None),
        ([(7, "z")], "x", "x", "z"),
        # Missing values in several formulas, in the same and other rows
        ([(3, "y"), (7, "z"), (7, "x"), (12, "z")], "z", "x", "x"),
    ],
)
def test_from_formula_drops_missing(nan_cells, scale, smooth, noise):
    # The rows with a missing value in any of the formulas are dropped from
    # all of the arrays, as if they were dropped from the data
    data = _missing_data()
    for row, col in nan_cells:
        data.loc[row, col] = np.nan
    dropped = sorted({row for row, _ in nan_cells})
    kwargs = dict(
        scale_formula=scale,
        smooth_formula=smooth,
        noise_formula=noise,
        time="time",
        groups="g",
    )
    mod = ProcessMLE.from_formula("y ~ x", data, **kwargs)
    expected = ProcessMLE.from_formula("y ~ x", data.drop(index=dropped), **kwargs)

    nobs = data.shape[0] - len(dropped)
    assert mod.endog.shape[0] == nobs
    for name in ("exog", "exog_scale", "exog_smooth", "time", "groups"):
        assert_allclose(getattr(mod, name), getattr(expected, name))
    if noise is not None:
        assert_allclose(mod.exog_noise, expected.exog_noise)
    assert_equal(dict(mod._groups_ix), dict(expected._groups_ix))
    params = np.linspace(-0.3, 0.3, len(mod.data.param_names))
    assert_allclose(mod.loglike(params), expected.loglike(params), rtol=1e-12)


def test_from_formula_missing_raise():
    data = _missing_data()
    data.loc[7, "z"] = np.nan
    with pytest.raises(MissingDataError):
        ProcessMLE.from_formula(
            "y ~ x",
            data,
            scale_formula="z",
            smooth_formula="x",
            time="time",
            groups="g",
            missing="raise",
        )


@pytest.mark.parametrize("k", [1, 2])
@pytest.mark.parametrize("nan_in", ["endog", "exog_scale", "time"])
def test_arrays_drop_missing(k, nan_in):
    data = _missing_data()
    nobs = data.shape[0]
    arrays = {
        "endog": data["y"].to_numpy(),
        "exog": np.column_stack([np.ones(nobs), data["x"]]),
        "exog_scale": np.column_stack([np.ones(nobs), data["z"]])[:, :k],
        "exog_smooth": np.column_stack([np.ones(nobs), data["x"]])[:, :k],
        "exog_noise": None,
        "time": data["time"].to_numpy(),
        "groups": data["g"].to_numpy(),
    }
    keep = np.ones(nobs, dtype=bool)
    keep[5] = False
    expected = ProcessMLE(
        **{key: None if v is None else v[keep] for key, v in arrays.items()}
    )
    arrays[nan_in] = arrays[nan_in].copy()
    arrays[nan_in][5] = np.nan
    mod = ProcessMLE(**arrays, missing="drop")

    assert mod.endog.shape[0] == nobs - 1
    assert_equal(dict(mod._groups_ix), dict(expected._groups_ix))
    params = np.linspace(-0.3, 0.3, 2 + 2 * k)
    assert_allclose(mod.loglike(params), expected.loglike(params), rtol=1e-12)
    with pytest.raises(MissingDataError):
        ProcessMLE(**arrays, missing="raise")
