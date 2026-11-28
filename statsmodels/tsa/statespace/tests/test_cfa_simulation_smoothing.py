"""
Tests for CFA simulation smoothing

Author: Chad Fulton
License: BSD-3
"""
from pathlib import Path

import numpy as np
from numpy.testing import assert_allclose
import pandas as pd
import pytest
from scipy.linalg import cho_solve_banded

from statsmodels import datasets
from statsmodels.tsa.statespace import (
    dynamic_factor,
    mlemodel,
    sarimax,
    structural,
    varmax,
)
from statsmodels.tsa.statespace.initialization import Initialization

current_path = Path(__file__).resolve().parent
dta = datasets.macrodata.load_pandas().data
dta.index = pd.period_range("1959Q1", "2009Q3", freq="Q")
dta = np.log(dta[["realcons", "realgdp", "cpi"]]).diff().iloc[1:] * 400


class CheckPosteriorMoments:
    @classmethod
    def setup_class(cls, model_class, missing=None, mean_atol=0, cov_atol=0,
                    use_complex=False, *args, **kwargs):
        cls.mean_atol = mean_atol
        cls.cov_atol = cov_atol

        endog = dta.copy()

        if missing == "all":
            endog.iloc[0:50, :] = np.nan
        elif missing == "partial":
            endog.iloc[0:50, 0] = np.nan
        elif missing == "mixed":
            endog.iloc[0:50, 0] = np.nan
            endog.iloc[19:70, 1] = np.nan
            endog.iloc[39:90, 2] = np.nan
            endog.iloc[119:130, 0] = np.nan
            endog.iloc[119:130, 2] = np.nan
            endog.iloc[-10:, :] = np.nan

        if model_class in [sarimax.SARIMAX, structural.UnobservedComponents]:
            endog = endog.iloc[:, 2]

        cls.mod = model_class(endog, *args, **kwargs)
        params = cls.mod.start_params
        if use_complex:
            params = params + 0j
        cls.res = cls.mod.smooth(params)

        cls.sim_cfa = cls.mod.simulation_smoother(method="cfa")
        cls.sim_cfa.simulate()
        prefix = "z" if use_complex else "d"
        cls._sim_cfa = cls.sim_cfa._simulation_smoothers[prefix]

    def test_posterior_mean(self):
        # Test the values from the Cython results
        actual = np.array(self._sim_cfa.posterior_mean, copy=True)
        assert_allclose(actual, self.res.smoothed_state, atol=self.mean_atol)

        # Test the values from the CFASimulationSmoother wrapper results
        assert_allclose(self.sim_cfa.posterior_mean, self.res.smoothed_state,
                        atol=self.mean_atol)

    def test_posterior_cov(self):
        # Test the values from the Cython results
        inv_chol = np.array(self._sim_cfa.posterior_cov_inv_chol, copy=True)
        actual = cho_solve_banded((inv_chol, True), np.eye(inv_chol.shape[1]))

        for t in range(self.mod.nobs):
            tm = t * self.mod.k_states
            t1m = tm + self.mod.k_states
            assert_allclose(actual[tm:t1m, tm:t1m],
                            self.res.smoothed_state_cov[..., t],
                            atol=self.cov_atol)

        # Test the values from the CFASimulationSmoother wrapper results
        actual = self.sim_cfa.posterior_cov

        for t in range(self.mod.nobs):
            tm = t * self.mod.k_states
            t1m = tm + self.mod.k_states
            assert_allclose(actual[tm:t1m, tm:t1m],
                            self.res.smoothed_state_cov[..., t],
                            atol=self.cov_atol)


class TestDFM(CheckPosteriorMoments):
    @classmethod
    def setup_class(cls, missing=None, *args, **kwargs):
        kwargs["k_factors"] = 1
        kwargs["factor_order"] = 1
        super().setup_class(
            dynamic_factor.DynamicFactor, missing, *args, **kwargs
        )


class TestDFMComplex(CheckPosteriorMoments):
    @classmethod
    def setup_class(cls, missing=None, *args, **kwargs):
        kwargs["k_factors"] = 1
        kwargs["factor_order"] = 1
        super().setup_class(
            dynamic_factor.DynamicFactor, missing, *args, use_complex=True, **kwargs
        )


class TestDFMAllMissing(TestDFM):
    def setup_class(cls, missing="all", *args, **kwargs):
        super().setup_class(*args, missing=missing, **kwargs)


class TestDFMPartialMissing(TestDFM):
    def setup_class(cls, missing="partial", *args, **kwargs):
        super().setup_class(*args, missing=missing, **kwargs)


class TestDFMMixedMissing(TestDFM):
    def setup_class(cls, missing="mixed", *args, **kwargs):
        super().setup_class(*args, missing=missing, **kwargs)


class TestVARME(CheckPosteriorMoments):
    # Test VAR model with Measurement Error
    # Note: this includes a trend
    # Note: have to use measurement error, due to the restriction that all
    # shocks must be non-degenerate for the CFA algorithm
    @classmethod
    def setup_class(cls, missing=None, *args, **kwargs):
        kwargs["order"] = (1, 0)
        kwargs["measurement_error"] = True
        super().setup_class(varmax.VARMAX, *args, missing=missing, **kwargs)


class TestVARMEAllMissing(TestVARME):
    def setup_class(cls, missing="all", *args, **kwargs):
        super().setup_class(*args, missing=missing, **kwargs)


class TestVARMEPartialMissing(TestVARME):
    def setup_class(cls, missing="partial", *args, **kwargs):
        super().setup_class(*args, missing=missing, **kwargs)


class TestVARMEMixedMissing(TestVARME):
    def setup_class(cls, missing="mixed", *args, **kwargs):
        super().setup_class(*args, missing=missing, **kwargs)


class TestSARIMAXME(CheckPosteriorMoments):
    # Test SARIMAX model with Measurement Error
    # Note: have to use measurement error, due to the restriction that all
    # shocks must be non-degenerate for the CFA algorithm
    @classmethod
    def setup_class(cls, missing=None, *args, **kwargs):
        kwargs["order"] = (1, 0, 0)
        kwargs["measurement_error"] = True
        super().setup_class(sarimax.SARIMAX, *args, missing=missing, **kwargs)


class TestSARIMAXMEMissing(TestSARIMAXME):
    def setup_class(cls, missing="mixed", *args, **kwargs):
        super().setup_class(*args, missing=missing, **kwargs)


class TestUnobservedComponents(CheckPosteriorMoments):
    # Test UC model, with exog
    @classmethod
    def setup_class(cls, missing=None, *args, **kwargs):
        kwargs["level"] = "llevel"
        kwargs["exog"] = np.arange(dta.shape[0])
        kwargs["autoregressive"] = 1
        super().setup_class(
            structural.UnobservedComponents, *args, missing=missing, **kwargs
        )


class TestUnobservedComponentsMissing(TestUnobservedComponents):
    def setup_class(cls, missing="mixed", *args, **kwargs):
        super().setup_class(*args, missing=missing, **kwargs)


def test_dfm():
    mod = dynamic_factor.DynamicFactor(dta, k_factors=2, factor_order=1)
    mod.update(mod.start_params)
    sim_cfa = mod.simulation_smoother(method="cfa")
    res = mod.ssm.smooth()

    # Test zero variates
    sim_cfa.simulate(np.zeros((mod.k_states, mod.nobs)))
    assert_allclose(sim_cfa.simulated_state, res.smoothed_state)


def _ar1_measurement_error():
    endog = np.random.default_rng(7123).standard_normal(20)
    mod = sarimax.SARIMAX(endog, order=(1, 0, 0), measurement_error=True)
    mod.update([0.5, 0.5, 1.0])
    return mod


@pytest.mark.parametrize("kind", ["int", "generator", "random_state"])
def test_cfa_rng(kind):
    # The CFA simulation smoother ignored `rng`: every simulate() drew its
    # variates from a new generator seeded with fresh entropy. The draws use
    # `rng`, so they equal draws with explicit variates from the same
    # generator, and successive draws continue it.
    mod = _ar1_measurement_error()

    def make_rng():
        if kind == "int":
            return 2943
        elif kind == "generator":
            return np.random.default_rng(2943)
        return np.random.RandomState(2943)

    sim = mod.simulation_smoother(method="cfa", rng=make_rng())
    same_seed = mod.simulation_smoother(method="cfa", rng=make_rng())
    explicit = mod.simulation_smoother(method="cfa")
    rng = np.random.default_rng(2943) if kind == "int" else make_rng()
    for _ in range(2):
        sim.simulate()
        same_seed.simulate()
        variates = rng.normal(size=mod.k_states * mod.nobs)
        explicit.simulate(variates.reshape(mod.k_states, mod.nobs, order="F"))
        assert_allclose(sim.simulated_state, same_seed.simulated_state, rtol=1e-13)
        assert_allclose(sim.simulated_state, explicit.simulated_state, rtol=1e-13)


# Blocks of a block initialization of three states as (index, kind), in the
# order in which they are set
MIXED_INITIALIZATIONS = {
    "diffuse-known": [((0, 1), "diffuse"), ((1, 3), "known")],
    "known-diffuse": [((1, 3), "known"), ((0, 1), "diffuse")],
    "known-diffuse-known": [((0, 1), "known"), ((1, 2), "diffuse"), ((2, 3), "known")],
    "diffuse-known-known": [((1, 2), "diffuse"), ((2, 3), "known"), ((0, 1), "known")],
}
BLOCK_INITIALIZATIONS = {
    **MIXED_INITIALIZATIONS,
    "known": [((0, 3), "known")],
    "diffuse": [((0, 3), "diffuse")],
}


def _mixed_initialization_model(blocks, known_scale=1.0):
    endog = np.random.default_rng(4410).standard_normal((30, 2))
    mod = mlemodel.MLEModel(endog, k_states=3, k_posdef=3)
    mod["design"] = [[1.0, 0.5, 0.0], [0.3, 0.0, 1.0]]
    mod["obs_cov"] = np.diag([0.8, 1.2])
    mod["transition"] = [[0.9, 0.1, 0.0], [0.0, 0.5, 0.2], [0.1, 0.0, -0.4]]
    mod["selection"] = np.eye(3)
    mod["state_cov"] = np.diag([0.5, 1.0, 0.7])
    # A known initialization of all states, of which the blocks are taken
    constant = np.array([3.0, -1.0, 2.0])
    cov = known_scale * np.array([[1.0, 0.3, 0.2], [0.3, 0.5, 0.1], [0.2, 0.1, 0.8]])
    init = Initialization(3)
    for (start, stop), kind in blocks:
        if kind == "known":
            init.set(
                (start, stop),
                "known",
                constant=constant[start:stop],
                stationary_cov=cov[start:stop, start:stop],
            )
        else:
            init.set((start, stop), "diffuse")
    mod.ssm.initialization = init
    return mod


@pytest.mark.parametrize("blocks", BLOCK_INITIALIZATIONS.values(),
                         ids=BLOCK_INITIALIZATIONS.keys())
def test_cfa_mixed_diffuse_initialization(blocks):
    # With a block initialization in which some blocks are diffuse and others
    # known, only the last block set decided whether the CFA treated the whole
    # initial state as diffuse (prior precision zero) or as known (inverting a
    # singular covariance matrix raised LinAlgError). The diffuse states have
    # prior precision zero and the others the inverse of their covariance
    # block, so the posterior moments equal those of the exact diffuse Kalman
    # smoother (also when all or none of the states are diffuse).
    mod = _mixed_initialization_model(blocks)
    res = mod.ssm.smooth()
    sim = mod.simulation_smoother(method="cfa")
    # The smoother first runs with a known initialization of all states
    initialization = mod.ssm.initialization
    mod.ssm.initialize_known(np.zeros(3), np.eye(3))
    sim.simulate(np.zeros((mod.k_states, mod.nobs)))
    mod.ssm.initialization = initialization
    sim.simulate(np.zeros((mod.k_states, mod.nobs)))

    # The CFA and the Kalman smoother agree to about 1e-8 relative for this
    # model, also with a known initialization of all states
    assert_allclose(sim.posterior_mean, res.smoothed_state, rtol=1e-6)
    assert_allclose(sim.simulated_state, res.smoothed_state, rtol=1e-6)
    posterior_cov = np.array(
        [
            sim.posterior_cov[3 * t : 3 * (t + 1), 3 * t : 3 * (t + 1)]
            for t in range(mod.nobs)
        ]
    ).transpose(1, 2, 0)
    assert_allclose(posterior_cov, res.smoothed_state_cov, rtol=1e-6)


@pytest.mark.parametrize("blocks", MIXED_INITIALIZATIONS.values(),
                         ids=MIXED_INITIALIZATIONS.keys())
def test_cfa_mixed_diffuse_initialization_singular_known(blocks):
    # A known block with a singular covariance matrix raises, as a known
    # initialization of the whole state does
    mod = _mixed_initialization_model(blocks, known_scale=0.0)
    sim = mod.simulation_smoother(method="cfa")
    with pytest.raises(np.linalg.LinAlgError, match="Non-positive-definite initial"):
        sim.simulate()


def bivariate_model(endog=None, diffuse=False, **matrices):
    if endog is None:
        rng = np.random.default_rng(20261122)
        endog = rng.standard_normal((40, 2)).cumsum(0) * 0.1 + rng.standard_normal(
            (40, 2)
        )
    mod = mlemodel.MLEModel(endog, k_states=2, k_posdef=2)
    mod["design"] = np.array([[1.0, 0.0], [0.5, 1.0]])
    mod["obs_intercept"] = np.array([0.1, -0.2])
    mod["obs_cov"] = matrices.get("obs_cov", np.diag([1.0, 0.7]))
    mod["transition"] = np.array([[0.9, 0.1], [0.0, 0.8]])
    mod["state_intercept"] = np.array([0.05, 0.02])
    mod["selection"] = np.eye(2)
    mod["state_cov"] = matrices.get("state_cov", np.array([[0.5, 0.2], [0.2, 0.4]]))
    if diffuse:
        mod.ssm.initialize_diffuse()
    else:
        initial_state_cov = matrices.get("initial_state_cov", np.eye(2))
        mod.ssm.initialize_known(np.array([0.3, -0.1]), initial_state_cov)
    return mod


def banded_to_dense(banded):
    # Lower triangular matrix from its banded storage (diagonals in rows)
    n = banded.shape[1]
    dense = np.zeros((n, n))
    for k in range(banded.shape[0]):
        ix = np.arange(n - k)
        dense[ix + k, ix] = banded[k, : n - k]
    return dense


def test_exact_diffuse():
    # With exact diffuse initialization the prior precision of the initial
    # state is zero, and the posterior moments equal the smoothed moments
    mod = bivariate_model(diffuse=True)
    res = mod.ssm.smooth()
    assert res.nobs_diffuse > 0
    sim = mod.simulation_smoother(method="cfa")
    sim.simulate(np.zeros((mod.k_states, mod.nobs)))

    # The algorithms differ in rounding, about 1e-10 here for entries of
    # order one; atol for entries close to zero
    assert_allclose(sim.posterior_mean, res.smoothed_state, rtol=1e-8, atol=1e-9)
    assert_allclose(sim.simulated_state, res.smoothed_state, rtol=1e-8, atol=1e-9)
    cov = sim.posterior_cov
    for t in range(mod.nobs):
        block = cov[2 * t : 2 * t + 2, 2 * t : 2 * t + 2]
        assert_allclose(block, res.smoothed_state_cov[..., t], rtol=1e-8, atol=1e-9)


def test_single_observation():
    mod = bivariate_model(endog=np.zeros((1, 2)))
    sim = mod.simulation_smoother(method="cfa")
    with pytest.raises(NotImplementedError, match="single observation"):
        sim.simulate()


@pytest.mark.parametrize(
    "name, match",
    [
        ("initial_state_cov", "Non-positive-definite initial state covariance"),
        ("state_cov", "Non-positive-definite selected state covariance"),
        ("obs_cov", "Non-positive-definite observation covariance"),
    ],
)
def test_singular_covariance(name, match):
    # A positive semi-definite covariance matrix with a zero variance has no
    # Cholesky factor (the CFA needs its inverse); the KFS smoother works
    mod = bivariate_model(**{name: np.diag([1.0, 0.0])})
    assert np.all(np.isfinite(mod.ssm.smooth().smoothed_state))
    sim = mod.simulation_smoother(method="cfa")
    with pytest.raises(np.linalg.LinAlgError, match=match):
        sim.simulate()


def test_posterior_cov_and_update_posterior():
    mod = bivariate_model()
    res = mod.ssm.smooth()
    sim = mod.simulation_smoother(method="cfa")
    sim.simulate(np.zeros((mod.k_states, mod.nobs)))
    mean = sim.posterior_mean

    # L L' is the inverse of the posterior covariance, and its diagonal blocks
    # are the smoothed state covariances; both properties are cached
    for _ in range(2):
        chol = banded_to_dense(sim.posterior_cov_inv_chol_sparse)
        cov = sim.posterior_cov
        # atol: the off-diagonal entries of the identity are zero
        assert_allclose(cov @ (chol @ chol.T), np.eye(2 * mod.nobs), atol=1e-12)
        for t in [0, 10, mod.nobs - 1]:
            block = cov[2 * t : 2 * t + 2, 2 * t : 2 * t + 2]
            # The algorithms differ in rounding; atol for small covariances
            assert_allclose(
                block, res.smoothed_state_cov[..., t], rtol=1e-8, atol=1e-9
            )

    # Without updating the posterior moments, a new draw is
    # x = mean + (L')^{-1} u, with the posterior moments of the last update
    variates = np.random.default_rng(1122).standard_normal((mod.k_states, mod.nobs))
    sim.simulate(variates, update_posterior=False)
    desired = mean.ravel(order="F") + np.linalg.solve(
        chol.T, variates.ravel(order="F")
    )
    assert_allclose(sim.simulated_state, desired.reshape(2, -1, order="F"), rtol=1e-10)
    assert sim.posterior_mean is mean


def test_posterior_moments_before_simulate():
    # The posterior moments are available before simulate() is called (they
    # raised AttributeError): they equal the Kalman smoother output for the
    # current parameters, and simulate(update_posterior=False) then uses them
    endog = dta[["realgdp", "cpi"]].iloc[:40]

    def get_model():
        mod = varmax.VARMAX(endog, order=(1, 0), measurement_error=True)
        mod.update(mod.start_params)
        return mod

    res = get_model().ssm.smooth()
    nobs, k_states = res.nobs, res.k_states

    def get_simulation_smoother():
        # The representation of the model was last filtered at other
        # parameters, so the moments need the current matrices and
        # initialization
        mod = get_model()
        mod.update(mod.start_params * 0.5)
        mod.ssm.smooth()
        mod.update(mod.start_params)
        return mod.simulation_smoother(method="cfa")

    sim_cfa = get_simulation_smoother()
    assert_allclose(sim_cfa.posterior_mean, res.smoothed_state, rtol=1e-8)
    assert sim_cfa.simulated_state is None

    sim_cfa = get_simulation_smoother()
    posterior_cov = sim_cfa.posterior_cov
    blocks = [
        posterior_cov[k_states * t : k_states * (t + 1), k_states * t : k_states * (t + 1)]
        for t in range(nobs)
    ]
    assert_allclose(
        np.array(blocks), res.smoothed_state_cov.transpose(2, 0, 1), rtol=1e-8
    )

    sim_cfa = get_simulation_smoother()
    inv_chol = sim_cfa.posterior_cov_inv_chol_sparse
    actual = cho_solve_banded((inv_chol, True), np.eye(nobs * k_states))
    # atol: covariances of states far apart are close to zero
    assert_allclose(actual, posterior_cov, rtol=1e-8, atol=1e-14)

    # With zero variates the draw is the posterior mean
    sim_cfa.simulate(variates=np.zeros((k_states, nobs)), update_posterior=False)
    assert_allclose(sim_cfa.simulated_state, res.smoothed_state, rtol=1e-8)
