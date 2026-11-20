"""
Array layouts at the Cython entry points

Cython functions that take typed buffers or memoryviews reject arrays that are
not in native byte order, and memoryviews declared contiguous reject strided
views and arrays in the other memory order. The public functions and models
below end up in Cython code, so they have to convert what they are given.

Every case is evaluated once with a native, C-contiguous array and once with
the same values stored in a different layout. The results must agree.
"""

import numpy as np
from numpy.testing import assert_allclose
import pytest

from statsmodels.nonparametric.kde import KDEUnivariate, kdensityfft
from statsmodels.nonparametric.smoothers_lowess import lowess
from statsmodels.robust.scale import qn_scale
from statsmodels.tsa.exponential_smoothing.ets import ETSModel
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.innovations import arma_innovations
from statsmodels.tsa.regime_switching.markov_autoregression import (
    MarkovAutoregression,
)
from statsmodels.tsa.regime_switching.markov_regression import MarkovRegression
from statsmodels.tsa.seasonal import MSTL, STL
from statsmodels.tsa.statespace import tools as ss_tools
from statsmodels.tsa.statespace.dynamic_factor import DynamicFactor
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.statespace.structural import UnobservedComponents
from statsmodels.tsa.statespace.varmax import VARMAX
from statsmodels.tsa.stattools import acovf, innovations_algo, innovations_filter


def swap_byte_order(a):
    # Same values, stored in the byte order that is not native on this machine
    a = np.asarray(a)
    return a.byteswap().view(a.dtype.newbyteorder())


def row_strided(a):
    # A view that skips every other row (every other element in 1-d)
    a = np.asarray(a)
    buf = np.zeros((2 * a.shape[0],) + a.shape[1:], dtype=a.dtype)
    buf[::2] = a
    return buf[::2]


def column_strided(a):
    a = np.asarray(a)
    buf = np.zeros((a.shape[0], 2 * a.shape[1]) + a.shape[2:], dtype=a.dtype)
    buf[:, ::2] = a
    return buf[:, ::2]


LAYOUTS_1D = {
    "swapped": swap_byte_order,
    "strided": row_strided,
    "swapped-strided": lambda a: row_strided(swap_byte_order(a)),
}
LAYOUTS_2D = {
    "swapped": swap_byte_order,
    "fortran": np.asfortranarray,
    "swapped-fortran": lambda a: np.asfortranarray(swap_byte_order(a)),
    "row-strided": row_strided,
    "column-strided": column_strided,
}

RNG = np.random.default_rng(20261118)
NOBS = 120
X = RNG.standard_normal(NOBS)
Y = np.sin(X) + 0.1 * RNG.standard_normal(NOBS)
XVALS = np.linspace(-2, 2, 7)
WEIGHTS = RNG.uniform(0.5, 1.5, NOBS)
EXOG = RNG.standard_normal((NOBS, 2))

_eps = RNG.standard_normal(NOBS)
ARMA = np.zeros(NOBS)
for t in range(1, NOBS):
    ARMA[t] = 0.5 * ARMA[t - 1] + _eps[t] + 0.3 * _eps[t - 1]
ARMA2 = np.column_stack([ARMA, 0.5 * ARMA + RNG.standard_normal(NOBS)])
ARMA2_MISSING = ARMA2.copy()
ARMA2_MISSING[5, 0] = np.nan
ARMA2_MISSING[10, 1] = np.nan
ACOV = acovf(ARMA, nlag=10)
THETA = innovations_algo(acovf(ARMA[:40], nlag=39), nobs=40)[0]
AR_MA_SIGMA2 = np.array([0.5, 0.3, 1.0])

SEASON = np.tile(np.sin(np.arange(12) / 12 * 2 * np.pi), 8)
SEASONAL = 10 + 0.05 * np.arange(96) + SEASON + 0.2 * RNG.standard_normal(96)

REGIMES = np.r_[RNG.normal(0, 1, 60), RNG.normal(3, 1, 60)]
TVTP = np.column_stack([np.ones(NOBS), EXOG[:, 0]])

SARIMAX_PARAMS = np.array([0.2, -0.1, 0.5, 0.3, 1.0])
SHOCKS = RNG.standard_normal((20, 1))
UNCONSTRAINED = RNG.standard_normal((2, 4))
CONSTRAINED = ss_tools.constrain_stationary_multivariate(UNCONSTRAINED, np.eye(2))[0]
MATRICES = RNG.standard_normal((2, 2, 10))
MATRICES_B = RNG.standard_normal((2, 2, 10))
VECTORS = RNG.standard_normal((2, 10))
VECTORS_B = RNG.standard_normal((2, 10))
MISSING = (RNG.uniform(size=(2, 10)) < 0.4).astype(np.intc)
INDEX = (RNG.uniform(size=(2, 10)) < 0.6).astype(np.intc)
# lltrend has two states
CFA_VARIATES = np.random.default_rng(10795).standard_normal((2, 96))


def _ets_model(endog, error="add", trend="add", seasonal="add"):
    return ETSModel(
        endog, error=error, trend=trend, seasonal=seasonal, seasonal_periods=12
    )


ETS_PARAMS = _ets_model(SEASONAL)._start_params


def _ets(endog, error, trend, seasonal):
    mod = _ets_model(endog, error, trend, seasonal)
    params = mod._start_params
    return mod.loglike(params), mod.smooth(params).fittedvalues


def _holt_winters(endog, trend, seasonal):
    # All smoothing parameters are fixed and the initial states are heuristic,
    # so there is no optimizer whose path could depend on the last bit of a
    # reduction over endog.
    mod = ExponentialSmoothing(
        endog,
        trend=trend,
        damped_trend=trend is not None,
        seasonal=seasonal,
        seasonal_periods=12,
        initialization_method="heuristic",
    )
    fixed = {"smoothing_level": 0.4}
    if trend is not None:
        fixed.update(smoothing_trend=0.1, damping_trend=0.9)
    if seasonal is not None:
        fixed["smoothing_seasonal"] = 0.2
    res = mod.fit(optimized=False, **fixed)
    return res.params_formatted["param"].to_numpy(float), res.fittedvalues


def _markov(endog=REGIMES, exog=None, exog_tvtp=None):
    mod = MarkovRegression(endog, k_regimes=2, exog=exog, exog_tvtp=exog_tvtp)
    start = MarkovRegression(
        REGIMES,
        k_regimes=2,
        exog=EXOG if exog is not None else None,
        exog_tvtp=TVTP if exog_tvtp is not None else None,
    ).start_params
    return mod.smooth(start).smoothed_marginal_probabilities


def _varmax(endog, univariate=False):
    mod = VARMAX(endog, order=(1, 0))
    if univariate:
        mod.ssm.set_filter_method(filter_univariate=True)
    params = VARMAX(ARMA2, order=(1, 0)).start_params
    return mod.smooth(params).smoothed_state


def _simulation_smoother(endog):
    mod = UnobservedComponents(endog, "lltrend")
    mod.update(UnobservedComponents(SEASONAL, "lltrend").start_params)
    sim = mod.simulation_smoother()
    sim.simulate(
        measurement_disturbance_variates=np.zeros(mod.nobs),
        state_disturbance_variates=np.zeros(mod.nobs * mod.ssm.k_posdef),
        initial_state_variates=np.zeros(mod.k_states),
    )
    return sim.simulated_state


def _cfa_simulation_smoother(endog, variates=CFA_VARIATES):
    mod = UnobservedComponents(endog, "lltrend")
    mod.update(UnobservedComponents(SEASONAL, "lltrend").start_params)
    sim = mod.simulation_smoother(method="cfa")
    sim.simulate(variates=variates)
    return sim.simulated_state, sim.posterior_mean


SARIMAX_MOD = SARIMAX(ARMA, exog=EXOG, order=(1, 0, 1))
SARIMAX_RES = SARIMAX_MOD.smooth(SARIMAX_PARAMS)

# (id, callable, arguments, position of the argument that changes layout)
CASES = [
    # nonparametric
    ("kdensityfft", lambda x: kdensityfft(x)[:2], (X,), 0),
    ("KDEUnivariate fft", lambda x: KDEUnivariate(x).fit(fft=True).density, (X,), 0),
    (
        "KDEUnivariate weights",
        lambda w: KDEUnivariate(X).fit(fft=False, weights=w).density,
        (WEIGHTS,),
        0,
    ),
    ("lowess endog", lowess, (Y, X), 0),
    ("lowess exog", lowess, (Y, X), 1),
    ("lowess xvals", lambda xv: lowess(Y, X, xvals=xv), (XVALS,), 0),
    # robust
    ("qn_scale", qn_scale, (X,), 0),
    ("qn_scale 2-d", qn_scale, (EXOG,), 0),
    # innovations
    ("innovations_algo", lambda a: innovations_algo(a)[0], (ACOV,), 0),
    ("innovations_filter endog", innovations_filter, (ARMA[:40], THETA), 0),
    ("innovations_filter theta", innovations_filter, (ARMA[:40], THETA), 1),
    (
        "arma_innovations",
        lambda y: arma_innovations.arma_innovations(y, [0.5], [0.3])[0],
        (ARMA,),
        0,
    ),
    (
        "arma_loglikeobs endog",
        lambda y: arma_innovations.arma_loglikeobs(y, [0.5], [0.3]),
        (ARMA,),
        0,
    ),
    (
        "arma_loglike params",
        lambda p: arma_innovations.arma_loglike(ARMA, p[:1], p[1:2], sigma2=p[2]),
        (AR_MA_SIGMA2,),
        0,
    ),
    (
        "arma_score params",
        lambda p: arma_innovations.arma_score(ARMA, p[:1], p[1:2], sigma2=p[2]),
        (AR_MA_SIGMA2,),
        0,
    ),
    # seasonal decomposition
    (
        "STL",
        lambda y: [
            getattr(STL(y, period=12, robust=True).fit(), a)
            for a in ("trend", "seasonal", "weights")
        ],
        (SEASONAL,),
        0,
    ),
    ("MSTL", lambda y: MSTL(y, periods=(6, 12)).fit().seasonal, (SEASONAL,), 0),
    # exponential smoothing
    *[
        (
            f"ETSModel {e}/{t}/{s}",
            lambda y, e=e, t=t, s=s: _ets(y, e, t, s),
            (SEASONAL,),
            0,
        )
        for e, t, s in [
            ("add", "add", "add"),
            ("add", "add", "mul"),
            ("mul", "add", "add"),
            ("mul", "mul", "mul"),
        ]
    ],
    (
        "ETSModel.loglike params",
        lambda p: _ets_model(SEASONAL).loglike(p),
        (ETS_PARAMS,),
        0,
    ),
    (
        "ETSModel.smooth params",
        lambda p: _ets_model(SEASONAL).smooth(p).fittedvalues,
        (ETS_PARAMS,),
        0,
    ),
    (
        "ETSModel.fit start_params",
        lambda p: (
            _ets_model(SEASONAL).fit(start_params=p, maxiter=1, disp=False).params
        ),
        (ETS_PARAMS,),
        0,
    ),
    *[
        (
            f"ExponentialSmoothing {t}/{s}",
            lambda y, t=t, s=s: _holt_winters(y, t, s),
            (SEASONAL,),
            0,
        )
        for t in (None, "add", "mul")
        for s in (None, "add", "mul")
    ],
    # regime switching
    ("MarkovRegression endog", lambda y: _markov(endog=y), (REGIMES,), 0),
    ("MarkovRegression exog", lambda x: _markov(exog=x), (EXOG,), 0),
    ("MarkovRegression exog_tvtp", lambda x: _markov(exog_tvtp=x), (TVTP,), 0),
    (
        "MarkovAutoregression",
        lambda y: (
            MarkovAutoregression(y, k_regimes=2, order=2)
            .smooth(MarkovAutoregression(REGIMES, k_regimes=2, order=2).start_params)
            .smoothed_marginal_probabilities
        ),
        (REGIMES,),
        0,
    ),
    # state space
    (
        "SARIMAX endog",
        lambda y: (
            SARIMAX(y, exog=EXOG, order=(1, 0, 1)).smooth(SARIMAX_PARAMS).smoothed_state
        ),
        (ARMA,),
        0,
    ),
    (
        "SARIMAX exog",
        lambda x: (
            SARIMAX(ARMA, exog=x, order=(1, 0, 1)).smooth(SARIMAX_PARAMS).smoothed_state
        ),
        (EXOG,),
        0,
    ),
    ("SARIMAX params", SARIMAX_MOD.loglike, (SARIMAX_PARAMS,), 0),
    (
        "SARIMAX simulate",
        lambda s: SARIMAX_MOD.simulate(
            SARIMAX_PARAMS,
            20,
            measurement_shocks=np.zeros(20),
            state_shocks=s,
            initial_state=np.zeros(SARIMAX_MOD.k_states),
        ),
        (SHOCKS,),
        0,
    ),
    (
        "SARIMAX forecast exog",
        lambda x: SARIMAX_RES.forecast(5, exog=x),
        (EXOG[:5],),
        0,
    ),
    (
        "SARIMAX diffuse",
        lambda y: (
            SARIMAX(y, order=(1, 1, 1), initialization="diffuse")
            .smooth([0.5, 0.3, 1.0])
            .smoothed_state
        ),
        (ARMA,),
        0,
    ),
    ("VARMAX", _varmax, (ARMA2,), 0),
    ("VARMAX missing", _varmax, (ARMA2_MISSING,), 0),
    (
        "VARMAX missing univariate",
        lambda y: _varmax(y, univariate=True),
        (ARMA2_MISSING,),
        0,
    ),
    (
        "DynamicFactor",
        lambda y: (
            DynamicFactor(y, k_factors=1, factor_order=1)
            .smooth(DynamicFactor(ARMA2, k_factors=1, factor_order=1).start_params)
            .smoothed_state
        ),
        (ARMA2,),
        0,
    ),
    (
        "UnobservedComponents",
        lambda y: (
            UnobservedComponents(y, "lltrend", freq_seasonal=[{"period": 12}])
            .smooth(
                UnobservedComponents(
                    SEASONAL, "lltrend", freq_seasonal=[{"period": 12}]
                ).start_params
            )
            .smoothed_state
        ),
        (SEASONAL,),
        0,
    ),
    ("simulation smoother", _simulation_smoother, (SEASONAL,), 0),
    ("CFA simulation smoother endog", _cfa_simulation_smoother, (SEASONAL,), 0),
    (
        "CFA simulation smoother variates",
        lambda v: _cfa_simulation_smoother(SEASONAL, v),
        (CFA_VARIATES,),
        0,
    ),
    (
        "compute_smoothed_state_weights",
        lambda y: ss_tools.compute_smoothed_state_weights(
            SARIMAX(y, order=(1, 0, 0)).smooth([0.5, 1.0])
        )[0],
        (ARMA[:40],),
        0,
    ),
    (
        "constrain_stationary_multivariate",
        lambda u: ss_tools.constrain_stationary_multivariate(u, np.eye(2))[0],
        (UNCONSTRAINED,),
        0,
    ),
    (
        "unconstrain_stationary_multivariate",
        lambda c: ss_tools.unconstrain_stationary_multivariate(c, np.eye(2))[0],
        (CONSTRAINED,),
        0,
    ),
    (
        "constrain_stationary_univariate",
        ss_tools.constrain_stationary_univariate,
        (np.array([0.2, -0.4, 0.1]),),
        0,
    ),
    (
        "VARMAX start_params",
        lambda y: VARMAX(y, order=(1, 0)).start_params,
        (ARMA2,),
        0,
    ),
    *[
        (
            f"reorder_missing_matrix {i}",
            lambda a, m: ss_tools.reorder_missing_matrix(a, m, reorder_rows=True),
            (MATRICES, MISSING),
            i,
        )
        for i in range(2)
    ],
    *[
        (
            f"reorder_missing_vector {i}",
            ss_tools.reorder_missing_vector,
            (VECTORS, MISSING),
            i,
        )
        for i in range(2)
    ],
    *[
        (
            f"copy_missing_matrix {i}",
            lambda a, b, m: ss_tools.copy_missing_matrix(a, b, m, missing_rows=True),
            (MATRICES, MATRICES_B, MISSING),
            i,
        )
        for i in range(3)
    ],
    *[
        (
            f"copy_missing_vector {i}",
            ss_tools.copy_missing_vector,
            (VECTORS, VECTORS_B, MISSING),
            i,
        )
        for i in range(3)
    ],
    *[
        (
            f"copy_index_matrix {i}",
            lambda a, b, m: ss_tools.copy_index_matrix(a, b, m, index_rows=True),
            (MATRICES, MATRICES_B, INDEX),
            i,
        )
        for i in range(3)
    ],
    *[
        (
            f"copy_index_vector {i}",
            ss_tools.copy_index_vector,
            (VECTORS, VECTORS_B, INDEX),
            i,
        )
        for i in range(3)
    ],
]


def _params():
    for name, func, args, pos in CASES:
        layouts = LAYOUTS_1D if np.ndim(args[pos]) == 1 else LAYOUTS_2D
        for layout in layouts:
            yield pytest.param(func, args, pos, layout, id=f"{name}-{layout}")


def _as_arrays(res):
    if isinstance(res, (tuple, list)):
        return [np.asarray(r, dtype=float) for r in res]
    return [np.asarray(res, dtype=float)]


# ETSModel.fit is cut short to keep the tests fast. Both fits see the same
# values, so they stop at the same point.
@pytest.mark.filterwarnings(
    "ignore::statsmodels.tools.sm_exceptions.ConvergenceWarning"
)
@pytest.mark.parametrize("func, args, pos, layout", list(_params()))
def test_array_layout(func, args, pos, layout):
    layouts = LAYOUTS_1D if np.ndim(args[pos]) == 1 else LAYOUTS_2D
    expected = _as_arrays(func(*args))
    changed = list(args)
    changed[pos] = layouts[layout](args[pos])
    assert_allclose(changed[pos], args[pos], rtol=0, atol=0)
    actual = _as_arrays(func(*changed))
    for a, e in zip(actual, expected, strict=True):
        assert_allclose(a, e, rtol=1e-10, atol=0)


# (function, whether the array that is changed is the second one)
INPLACE = {
    "reorder_missing_matrix": (
        lambda a, b, **kw: ss_tools.reorder_missing_matrix(
            a, MISSING, reorder_rows=True, **kw
        ),
        False,
    ),
    "reorder_missing_vector": (
        lambda a, b, **kw: ss_tools.reorder_missing_vector(a, MISSING, **kw),
        False,
    ),
    "copy_missing_matrix": (
        lambda a, b, **kw: ss_tools.copy_missing_matrix(
            a, b, MISSING, missing_rows=True, **kw
        ),
        True,
    ),
    "copy_missing_vector": (
        lambda a, b, **kw: ss_tools.copy_missing_vector(a, b, MISSING, **kw),
        True,
    ),
    "copy_index_matrix": (
        lambda a, b, **kw: ss_tools.copy_index_matrix(
            a, b, INDEX, index_rows=True, **kw
        ),
        True,
    ),
    "copy_index_vector": (
        lambda a, b, **kw: ss_tools.copy_index_vector(a, b, INDEX, **kw),
        True,
    ),
}


def _inplace_args(name):
    func, changes_b = INPLACE[name]
    a, b = (MATRICES, MATRICES_B) if name.endswith("matrix") else (VECTORS, VECTORS_B)
    return func, changes_b, a, b


@pytest.mark.parametrize("name", list(INPLACE))
def test_inplace_fortran(name):
    # inplace=True changes the caller's array, which has to be a native
    # float64 array in Fortran order, and returns that array
    func, changes_b, a0, b0 = _inplace_args(name)
    expected = func(a0, b0)
    a = np.asfortranarray(a0)
    b = np.asfortranarray(b0)
    target, other = (b, a) if changes_b else (a, b)
    untouched = other.copy()
    assert not np.shares_memory(func(a, b), target)
    assert_allclose(target, b0 if changes_b else a0, rtol=0, atol=0)
    result = func(a, b, inplace=True)
    assert result is target
    assert_allclose(target, expected, rtol=0, atol=0)
    assert_allclose(other, untouched, rtol=0, atol=0)


@pytest.mark.parametrize("name", list(INPLACE))
def test_inplace_c_order_raises(name):
    # a C-ordered array cannot be changed in place: this raises, as it did
    # before the conversions were added, and leaves the array as it was
    func, changes_b, a0, b0 = _inplace_args(name)
    a = np.ascontiguousarray(a0)
    b = np.ascontiguousarray(b0)
    target = b if changes_b else a
    before = target.copy()
    with pytest.raises(ValueError, match="contiguous"):
        func(a, b, inplace=True)
    assert_allclose(target, before, rtol=0, atol=0)
