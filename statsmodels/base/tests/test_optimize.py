import warnings

import numpy as np
from numpy.testing import assert_, assert_allclose, assert_almost_equal
import pytest

from statsmodels.base.optimizer import (
    _fit_basinhopping,
    _fit_bfgs,
    _fit_cg,
    _fit_lbfgs,
    _fit_minimize,
    _fit_ncg,
    _fit_newton,
    _fit_nm,
    _fit_powell,
)

fit_funcs = {
    "newton": _fit_newton,
    "nm": _fit_nm,  # Nelder-Mead
    "bfgs": _fit_bfgs,
    "cg": _fit_cg,
    "ncg": _fit_ncg,
    "powell": _fit_powell,
    "lbfgs": _fit_lbfgs,
    "basinhopping": _fit_basinhopping,
    "minimize": _fit_minimize,
}


def dummy_func(x):
    return x**2


def dummy_score(x):
    return 2.0 * x


def dummy_hess(x):
    return [[2.0]]


def dummy_bounds_constraint_func(x):
    return (x[0] - 1) ** 2 + (x[1] - 2.5) ** 2


def dummy_bounds():
    return ((0, None), (0, None))


def dummy_bounds_tight():
    return ((2, None), (3.5, None))


def dummy_constraints():
    cons = (
        {"type": "ineq", "fun": lambda x: x[0] - 2 * x[1] + 2},
        {"type": "ineq", "fun": lambda x: -x[0] - 2 * x[1] + 6},
        {"type": "ineq", "fun": lambda x: -x[0] + 2 * x[1] + 2},
    )
    return cons


@pytest.mark.smoke
@pytest.mark.parametrize("method", fit_funcs.keys())
def test_full_output_false(method):
    # newton needs f, score, start, fargs, kwargs
    # bfgs needs f, score start, fargs, kwargs
    # nm needs ""
    # cg ""
    # ncg ""
    # powell ""
    func = fit_funcs[method]
    kwargs = {"seed": 32839013} if method == "basinhopping" else {}
    if method == "newton":
        xopt, retvals = func(
            dummy_func,
            dummy_score,
            [1.0],
            (),
            kwargs=kwargs,
            hess=dummy_hess,
            full_output=False,
            disp=0,
        )

    else:
        xopt, retvals = func(
            dummy_func,
            dummy_score,
            [1.0],
            (),
            kwargs=kwargs,
            full_output=False,
            disp=0,
        )
    assert_(retvals is None)
    assert_(len(xopt) == 1)


@pytest.mark.parametrize("method", fit_funcs.keys())
def test_full_output(method):
    func = fit_funcs[method]
    kwargs = {"seed": 32839013} if method == "basinhopping" else {}
    if method == "newton":
        xopt, retvals = func(
            dummy_func,
            dummy_score,
            [1.0],
            (),
            kwargs,
            hess=dummy_hess,
            full_output=True,
            disp=0,
        )

    else:
        xopt, retvals = func(
            dummy_func,
            dummy_score,
            [1.0],
            (),
            kwargs,
            full_output=True,
            disp=0,
        )

    assert_(retvals is not None)
    assert_("converged" in retvals)
    assert_(len(xopt) == 1)


def test_minimize_scipy_slsqp():
    func = fit_funcs["minimize"]
    xopt, _ = func(
        dummy_bounds_constraint_func,
        None,
        (2.0, 0.0),
        (),
        {
            "min_method": "SLSQP",
            "bounds": dummy_bounds(),
            "constraints": dummy_constraints(),
        },
        hess=None,
        full_output=False,
        disp=0,
    )
    assert_almost_equal(xopt, [1.4, 1.7], 4)


def test_minimize_scipy_powell():
    func = fit_funcs["minimize"]
    xopt, _ = func(
        dummy_bounds_constraint_func,
        None,
        (3, 4.5),
        (),
        {
            "min_method": "Powell",
            "bounds": dummy_bounds_tight(),
        },
        hess=None,
        full_output=False,
        disp=0,
    )
    assert_almost_equal(xopt, [2, 3.5], 4)


def test_minimize_scipy_nm():
    func = fit_funcs["minimize"]
    xopt, _ = func(
        dummy_bounds_constraint_func,
        None,
        (3, 4.5),
        (),
        {
            "min_method": "Nelder-Mead",
            "bounds": dummy_bounds_tight(),
        },
        hess=None,
        full_output=False,
        disp=0,
    )
    assert_almost_equal(xopt, [2, 3.5], 4)


@pytest.mark.parametrize("min_method", ["L-BFGS-B", "TNC"])
def test_minimize_no_hess_method(min_method):
    # GH#9140: L-BFGS-B and TNC do not accept a Hessian, so `hess` must not
    # be forwarded to scipy.optimize.minimize (warning or error depending
    # on the scipy version).
    from statsmodels.discrete.discrete_model import Poisson

    exog = np.column_stack((np.ones(10), np.arange(10) / 10.0))
    endog = np.array([1, 2, 1, 3, 2, 4, 3, 5, 4, 6])
    res_default = Poisson(endog, exog).fit(disp=0)

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        res = Poisson(endog, exog).fit(
            method="minimize", min_method=min_method, disp=0
        )
    hess_warnings = [
        wrn for wrn in w if "hess" in str(wrn.message).lower()
    ]
    assert hess_warnings == []
    assert res.mle_retvals["converged"]
    assert_allclose(res.params, res_default.params, rtol=1e-4)


def test_fit_invalid_method_raises():
    from statsmodels.discrete.discrete_model import Poisson

    exog = np.column_stack((np.ones(10), np.arange(10) / 10.0))
    endog = np.array([1, 2, 1, 3, 2, 4, 3, 5, 4, 6])
    mod = Poisson(endog, exog)

    with pytest.raises(ValueError, match="method"):
        mod.fit(method="not-a-method", disp=0)

    # case-insensitive, matching the previous manual `method.lower()`
    res_upper = mod.fit(method="BFGS", disp=0)
    res_lower = mod.fit(method="bfgs", disp=0)
    assert_allclose(res_upper.params, res_lower.params)


def test_lbfgs_disp_false_no_output(capsys):
    xopt, _ = _fit_lbfgs(
        dummy_func,
        dummy_score,
        [1.0],
        (),
        {},
        full_output=False,
        disp=False,
    )
    captured = capsys.readouterr()
    assert captured.out == ""


def _newton_funcs(region, curvature):
    """Quadratic objective that is nan outside of |x| <= region"""

    def f(x, *args):
        x = np.asarray(x)
        return 0.5 * float(x @ x) if np.all(np.abs(x) <= region) else np.nan

    def score(x):
        return np.asarray(x, dtype=float)

    def hess(x):
        return curvature * np.eye(len(x))

    return f, score, hess


def test_newton_step_halving_nonfinite():
    # The Hessian is too small by a factor of 100, so that the first Newton
    # step leaves the region where f is finite. The step is halved.
    f, score, hess = _newton_funcs(1.0, 0.01)
    params, retvals = _fit_newton(f, score, np.array([0.5]), (), {}, hess=hess)
    assert np.all(np.isfinite(params))
    assert retvals["converged"]
    assert retvals["warnflag"] == 0
    assert_allclose(params, 0.0, atol=1e-7)


def test_newton_step_halving_increase():
    # f = sum(sqrt(1 + x**2)) is convex and finite everywhere, but Newton's
    # method diverges from |x| > 1 since the Newton update is -x**3
    def f(x, *args):
        return float(np.sum(np.sqrt(1 + x**2)))

    def score(x):
        return x / np.sqrt(1 + x**2)

    def hess(x):
        return np.diag((1 + x**2) ** -1.5)

    start = np.array([2.0, -1.5])
    params, retvals = _fit_newton(
        f, score, start, (), {}, hess=hess, maxiter=100, retall=True
    )
    assert retvals["converged"]
    assert_allclose(params, 0.0, atol=1e-7)
    # f does not increase by more than the tolerance in any iteration
    values = [f(v) for v in retvals["allvecs"][1:]]
    assert np.all(np.diff(values) <= 1e-12 * np.abs(values[:-1]))

    # the full Newton steps diverge
    _, retvals = _fit_newton(
        f,
        score,
        start,
        (),
        {"step_halving": False},
        hess=hess,
        maxiter=5,
        retall=True,
    )
    assert f(retvals["allvecs"][-1]) > f(start)
    assert not retvals["converged"]


def test_newton_step_halving_disabled():
    # without step halving the first step leaves the region where f is finite
    f, score, hess = _newton_funcs(1.0, 0.01)
    params, retvals = _fit_newton(
        f,
        score,
        np.array([0.5]),
        (),
        {"step_halving": False},
        hess=hess,
        maxiter=5,
    )
    assert np.isnan(retvals["fopt"])
    assert retvals["warnflag"] == 2
    assert not retvals["converged"]


def test_newton_no_finite_step_is_not_converged():
    # f is nan at every point but the start. The iteration stops at the start
    # and reports that it did not converge, instead of returning nan.
    start = np.array([0.5, 0.25])

    def f(x, *args):
        return 1.0 if np.all(np.asarray(x) == start) else np.nan

    f_ok, score, hess = _newton_funcs(1.0, 1.0)
    params, retvals = _fit_newton(f, score, start, (), {}, hess=hess)
    assert_allclose(params, start)
    assert not retvals["converged"]
    assert retvals["warnflag"] == 2
    assert retvals["iterations"] == 0


def test_newton_nonfinite_start_is_not_converged():
    # f, the score and the Hessian are not finite at the start
    start = np.array([0.5, 0.25])

    def f(x, *args):
        return np.nan

    def score(x):
        return np.full(2, np.nan)

    def hess(x):
        return np.full((2, 2), np.nan) + np.eye(2)

    params, retvals = _fit_newton(f, score, start, (), {}, hess=hess)
    assert np.all(np.isfinite(params))
    assert not retvals["converged"]
    assert retvals["warnflag"] == 2
