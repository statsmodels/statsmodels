import warnings

import numpy as np
from numpy.testing import assert_, assert_allclose, assert_almost_equal
import pytest

from statsmodels.base.model import GenericLikelihoodModel
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
    _newton_step_halving,
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


@pytest.fixture
def shifted_poisson():
    from statsmodels.base.model import GenericLikelihoodModel
    from statsmodels.discrete.discrete_model import Poisson

    rng = np.random.default_rng(2836)
    exog = np.column_stack((np.ones(50), rng.standard_normal(50)))
    endog = rng.poisson(np.exp(exog @ [0.5, 0.3]))
    poisson = Poisson(endog, exog)

    # The loglikelihood of Poisson at params - shift, so that its maximizer is
    # that of Poisson plus shift. The argument is required, so that a call of
    # the loglikelihood, the score or the Hessian without fargs fails.
    def loglike(params, shift):
        return poisson.loglike(params - shift)

    def score(params, shift):
        return poisson.score(params - shift)

    def hessian(params, shift):
        return poisson.hessian(params - shift)

    model = GenericLikelihoodModel(
        endog, exog, loglike=loglike, score=score, hessian=hessian
    )
    return model, poisson


def test_newton_fargs_optimum(shifted_poisson):
    # Newton passed fargs to the objective but not to the score and the Hessian
    model, poisson = shifted_poisson
    shift = np.array([0.25, -0.5])
    desired = poisson.fit(method="bfgs", gtol=1e-10, disp=0)
    res = model.fit(
        start_params=shift, method="newton", fargs=(shift,), disp=0
    )
    assert res.mle_retvals["converged"]
    assert_allclose(res.params, desired.params + shift, rtol=1e-8)
    assert res.mle_settings["fargs"] == (shift,)


def test_newton_fargs_derivatives(shifted_poisson):
    # The score and the Hessian that Newton reports are those of Poisson at
    # params - shift, as derivatives of the objective -loglike / nobs
    model, poisson = shifted_poisson
    shift = np.array([0.25, -0.5])
    start = np.array([0.4, 0.1])
    res = model.fit(
        start_params=start,
        method="newton",
        fargs=(shift,),
        maxiter=0,
        disp=0,
        warn_convergence=False,
    )
    nobs = poisson.endog.shape[0]
    assert_allclose(
        res.mle_retvals["score"], -poisson.score(start - shift) / nobs, rtol=1e-12
    )
    assert_allclose(
        res.mle_retvals["Hessian"], -poisson.hessian(start - shift) / nobs, rtol=1e-12
    )


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


class _ShiftedLeastSquares(GenericLikelihoodModel):
    # The Gaussian loglikelihood of endog - shift, up to a constant. The shift
    # is a required argument of the loglikelihood, the score and the Hessian,
    # so that a solver that calls one of them without fargs raises. The fits
    # skip the Hessian after the optimization, which is computed without fargs.
    def loglike(self, params, shift):
        resid = self.endog - shift - self.exog @ params
        return -0.5 * (resid @ resid)

    def score(self, params, shift):
        return self.exog.T @ (self.endog - shift - self.exog @ params)

    def hessian(self, params, shift):
        return -self.exog.T @ self.exog


@pytest.mark.parametrize(
    "method", ["nm", "bfgs", "lbfgs", "powell", "cg", "ncg", "basinhopping", "minimize"]
)
def test_fit_fargs(method):
    # cg did not pass fargs to the objective and the score
    rng = np.random.default_rng(12345)
    exog = np.column_stack((np.ones(50), rng.standard_normal(50)))
    endog = exog @ [1.0, 0.5] + rng.standard_normal(50)
    shift = 3.0
    # The maximizer is least squares of endog - shift on exog. Without the
    # shift, the intercept would be larger by 3.
    desired = np.linalg.lstsq(exog, endog - shift, rcond=None)[0]
    # The default tolerances of Nelder-Mead stop at a change of 1e-4 in the
    # parameters. Those of the other solvers reach the maximizer to about 2e-8.
    options = {
        "nm": {"xtol": 1e-8, "ftol": 1e-10},
        "basinhopping": {"niter": 5, "seed": 32839013},
    }
    res = _ShiftedLeastSquares(endog, exog).fit(
        method=method,
        fargs=(shift,),
        disp=0,
        skip_hessian=True,
        **options.get(method, {}),
    )
    assert res.mle_retvals["converged"]
    assert_allclose(res.params, desired, rtol=1e-6)


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
    # f is nan at every point but the start. The fit reports that it did not
    # converge and returns a point with a finite value, instead of nan.
    start = np.array([0.5, 0.25])

    def f(x, *args):
        return 1.0 if np.all(np.asarray(x) == start) else np.nan

    f_ok, score, hess = _newton_funcs(1.0, 1.0)
    params, retvals = _fit_newton(f, score, start, (), {}, hess=hess)
    assert not retvals["converged"]
    assert retvals["warnflag"] == 2
    assert np.isfinite(f(params))
    assert np.isfinite(retvals["fopt"])


def test_newton_result_does_not_share_start_params():
    # the parameters of a fit without a step are not the array of the user
    start = np.array([0.5, 0.25])

    def f(x, *args):
        return 1.0 if np.all(np.asarray(x) == start) else np.nan

    f_ok, score, hess = _newton_funcs(1.0, 1.0)
    params, _ = _fit_newton(f, score, start, (), {}, hess=hess)
    assert not np.shares_memory(params, start)
    params[0] = 99.0
    assert_allclose(start, [0.5, 0.25])

    # integer starting values give float parameters
    params, _ = _fit_newton(
        f_ok, score, np.array([1, 0]), (), {}, hess=hess, maxiter=1
    )
    assert params.dtype == np.float64


@pytest.mark.parametrize("step_halving", [True, False, "finite"])
def test_newton_function_evaluations(step_halving):
    # f is evaluated at the start if the steps have to decrease it, once for a
    # step, and not again for the results
    calls = []
    f_ok, score, hess = _newton_funcs(10.0, 1.0)

    def f(x, *args):
        calls.append(1)
        return f_ok(x, *args)

    params, retvals = _fit_newton(
        f, score, np.array([0.5]), (), {"step_halving": step_halving}, hess=hess
    )
    assert retvals["converged"]
    iterations = retvals["iterations"]
    if step_halving is True:
        expected = iterations + 1
    elif step_halving == "finite":
        expected = iterations
    else:
        expected = 1
    assert len(calls) == expected
    assert retvals["fopt"] == f_ok(params)


@pytest.mark.parametrize("value", ["False", "no", "Finite ", 1, 0, None, 1.0, [True]])
def test_newton_step_halving_invalid(value):
    f, score, hess = _newton_funcs(1.0, 1.0)
    exception = ValueError if isinstance(value, str) else TypeError
    with pytest.raises(exception, match="step_halving"):
        _fit_newton(
            f, score, np.array([0.5]), (), {"step_halving": value}, hess=hess
        )


@pytest.mark.parametrize("value", [True, False, "finite", "FINITE"])
def test_newton_step_halving_valid(value):
    f, score, hess = _newton_funcs(1.0, 1.0)
    _, retvals = _fit_newton(
        f, score, np.array([0.5]), (), {"step_halving": value}, hess=hess
    )
    assert retvals["converged"]


def _double_well():
    """f = x**4 / 4 - x**2 / 2, with a negative curvature at x = 0.3"""

    def f(x, *args):
        x = np.asarray(x)
        return float(np.sum(x**4 / 4 - x**2 / 2))

    def score(x):
        return np.asarray(x) ** 3 - np.asarray(x)

    def hess(x):
        return np.diag(3 * np.asarray(x) ** 2 - 1.0)

    return f, score, hess


def test_newton_not_descent_direction_does_not_increase_f():
    # The Hessian is negative at the start, so that the Newton step goes to
    # the maximum of f at 0 and f increases for all of its shortened steps.
    f, score, hess = _double_well()
    start = np.array([0.3])
    params, retvals = _fit_newton(f, score, start, (), {}, hess=hess)
    assert not retvals["converged"]
    assert retvals["warnflag"] == 2
    assert f(params) <= f(start)

    # the step is taken if only a finite value is required
    params, retvals = _fit_newton(
        f, score, start, (), {"step_halving": "finite"}, hess=hess
    )
    assert f(params) > f(start)


def test_newton_shortened_step_is_not_convergence():
    # f is finite for x <= 1 + 1e-9 only and its minimum at 2 is outside of
    # that region. The shortened steps end at the border, where they are
    # smaller than tol, but the gradient is -2 and the full Newton step is 1.
    limit = 1 + 1e-9

    def f(x, *args):
        x = np.asarray(x)
        return float(np.sum((x - 2) ** 2)) if np.all(x <= limit) else np.nan

    def score(x):
        return 2 * (np.asarray(x, dtype=float) - 2)

    def hess(x):
        return 2 * np.eye(len(x))

    params, retvals = _fit_newton(
        f, score, np.array([0.0]), (), {}, hess=hess, maxiter=100
    )
    assert not retvals["converged"]
    assert retvals["warnflag"] == 2
    assert_allclose(params, 1.0, atol=2e-9)
    assert_allclose(score(params), -2.0, atol=1e-8)


def test_newton_shortened_steps_converge_if_the_newton_step_is_small():
    # The Hessian is wrong by a factor of 100, so that every step is shortened.
    # The iteration converges anyway, when the full Newton step is below tol.
    f, score, hess = _newton_funcs(1.0, 0.01)
    _, retvals = _fit_newton(
        f, score, np.array([0.5]), (), {"tol": 1e-4}, hess=hess, retall=True
    )
    assert retvals["converged"]
    last, previous = retvals["allvecs"][-1], retvals["allvecs"][-2]
    # the last step is shortened, its length is far below that of the full step
    assert abs(last - previous) < 1e-4 / 10
    assert abs(100 * previous) <= 1e-4


def _line_f(values):
    """f along the ray x = 1, 1/2, 1/4, ... given by `values`, else nan"""
    table = {1.0 / 2**k: v for k, v in enumerate(values)}

    def f(x, *args):
        return table.get(float(x[0]), np.nan)

    return f


def _halve_newton(values, fold, **kwargs):
    # the update is oldparams - step = 1, then 1 / 2 and so on
    return _newton_step_halving(
        _line_f(values), (), np.array([0.0]), np.array([-1.0]), fold, **kwargs
    )


def test_newton_step_halving_tolerance():
    # an increase of f of 1e-14 is rounding noise and is accepted, the larger
    # increase of 1e-6 is not
    params, fnew, success, shortened = _halve_newton([1 + 1e-14, 0.5], 1.0)
    assert success
    assert not shortened
    assert_allclose(params, 1.0)
    assert fnew == 1 + 1e-14

    params, fnew, success, shortened = _halve_newton([1 + 1e-6, 0.5], 1.0)
    assert success
    assert shortened
    assert_allclose(params, 0.5)
    assert fnew == 0.5


def test_newton_step_halving_no_decrease_found():
    # f does not decrease for any step, so that the first step with a finite
    # value is used
    params, fnew, success, shortened = _halve_newton([2.0, 3.0], 1.0)
    assert success
    assert not shortened
    assert_allclose(params, 1.0)
    assert fnew == 2.0

    params, fnew, success, shortened = _halve_newton([np.nan, 3.0, 2.0], 1.0)
    assert success
    assert shortened
    assert_allclose(params, 0.5)
    assert fnew == 3.0


def test_newton_step_halving_no_decrease_found_without_first_finite():
    # for a direction that is not one of descent there is no step if f does
    # not decrease for any step, and a step that decreases f is used
    params, fnew, success, shortened = _halve_newton(
        [2.0, 3.0], 1.0, use_first_finite=False
    )
    assert not success
    assert not shortened
    assert_allclose(params, 0.0)
    assert fnew == 1.0

    params, fnew, success, shortened = _halve_newton(
        [2.0, 0.5], 1.0, use_first_finite=False
    )
    assert success
    assert shortened
    assert_allclose(params, 0.5)
    assert fnew == 0.5


def test_newton_step_halving_value_not_needed():
    # fold is nan if the steps do not have to decrease f. Then any step with a
    # finite value is accepted, and a step that is not finite is not.
    params, fnew, success, shortened = _halve_newton([1e9, 0.5], np.nan)
    assert success
    assert not shortened
    assert fnew == 1e9

    params, fnew, success, shortened = _halve_newton([np.nan, 1e9], np.nan)
    assert success
    assert shortened
    assert_allclose(params, 0.5)

    params, fnew, success, shortened = _halve_newton([np.nan, np.nan], 1.0)
    assert not success
    assert not shortened
    assert_allclose(params, 0.0)
    assert fnew == 1.0


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
