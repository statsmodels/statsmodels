"""
Tests for simulation smoothing

Author: Chad Fulton
License: Simplified-BSD
"""
from pathlib import Path

import numpy as np
from numpy.testing import assert_allclose, assert_equal
import pandas as pd
import pytest

from statsmodels import datasets
from statsmodels.tsa.statespace import mlemodel, sarimax, structural
from statsmodels.tsa.statespace.simulation_smoother import (
    SIMULATION_ALL,
    SIMULATION_DISTURBANCE,
    SIMULATION_STATE,
    SimulationSmoother,


    SimulationSmoother,
    SimulationSmoothResults,
)

current_path = Path(__file__).resolve().parent


class MultivariateVARKnown:
    """
    Tests for simulation smoothing values in a couple of special cases of
    variates. Both computed values and KFAS values are used for comparison
    against the simulation smoother output.
    """

    @classmethod
    def setup_class(cls, missing=None, test_against_KFAS=True, *args, **kwargs):
        cls.test_against_KFAS = test_against_KFAS
        # Data
        dta = datasets.macrodata.load_pandas().data
        dta.index = pd.date_range(start="1959-01-01", end="2009-7-01", freq="QS")
        obs = np.log(dta[["realgdp", "realcons", "realinv"]]).diff().iloc[1:]

        if missing == "all":
            obs.iloc[0:50, :] = np.nan
        elif missing == "partial":
            obs.iloc[0:50, 0] = np.nan
        elif missing == "mixed":
            obs.iloc[0:50, 0] = np.nan
            obs.iloc[19:70, 1] = np.nan
            obs.iloc[39:90, 2] = np.nan
            obs.iloc[119:130, 0] = np.nan
            obs.iloc[119:130, 2] = np.nan
            obs.iloc[-10:, :] = np.nan

        if test_against_KFAS:
            obs = obs.iloc[:9]

        # Create the model
        mod = mlemodel.MLEModel(obs, k_states=3, k_posdef=3, **kwargs)
        mod["design"] = np.eye(3)
        mod["obs_cov"] = np.array(
            [
                [0.0000640649, 0.0, 0.0],
                [0.0, 0.0000572802, 0.0],
                [0.0, 0.0, 0.0017088585],
            ]
        )
        mod["transition"] = np.array(
            [
                [-0.1119908792, 0.8441841604, 0.0238725303],
                [0.2629347724, 0.4996718412, -0.0173023305],
                [-3.2192369082, 4.1536028244, 0.4514379215],
            ]
        )
        mod["selection"] = np.eye(3)
        mod["state_cov"] = np.array(
            [
                [0.0000640649, 0.0000388496, 0.0002148769],
                [0.0000388496, 0.0000572802, 0.000001555],
                [0.0002148769, 0.000001555, 0.0017088585],
            ]
        )
        mod.initialize_approximate_diffuse(1e6)
        mod.ssm.filter_univariate = True
        cls.model = mod
        cls.results = mod.smooth([], return_ssm=True)

        cls.sim = cls.model.simulation_smoother()

    def test_loglike(self):
        assert_allclose(np.sum(self.results.llf_obs), self.true_llf)

    def test_simulate_0(self):
        n = 10

        # Test with all inputs as zeros
        measurement_shocks = np.zeros((n, self.model.k_endog))
        state_shocks = np.zeros((n, self.model.ssm.k_posdef))
        initial_state = np.zeros(self.model.k_states)
        obs, states = self.model.ssm.simulate(
            nsimulations=n,
            measurement_shocks=measurement_shocks,
            state_shocks=state_shocks,
            initial_state=initial_state,
        )

        assert_allclose(obs, np.zeros((n, self.model.k_endog)))
        assert_allclose(states, np.zeros((n, self.model.k_states)))

    def test_simulate_1(self):
        n = 10

        # Test with np.arange / 10 measurement shocks only
        measurement_shocks = np.reshape(
            np.arange(n * self.model.k_endog) / 10.0, (n, self.model.k_endog)
        )
        state_shocks = np.zeros((n, self.model.ssm.k_posdef))
        initial_state = np.zeros(self.model.k_states)
        obs, states = self.model.ssm.simulate(
            nsimulations=n,
            measurement_shocks=measurement_shocks,
            state_shocks=state_shocks,
            initial_state=initial_state,
        )

        assert_allclose(
            obs,
            np.reshape(
                np.arange(n * self.model.k_endog) / 10.0, (n, self.model.k_endog)
            ),
        )
        assert_allclose(states, np.zeros((n, self.model.k_states)))

    def test_simulate_2(self):
        n = 10
        Z = self.model["design"]
        T = self.model["transition"]

        # Test with non-zero state shocks and initial state
        measurement_shocks = np.zeros((n, self.model.k_endog))
        state_shocks = np.ones((n, self.model.ssm.k_posdef))
        initial_state = np.ones(self.model.k_states) * 2.5
        obs, states = self.model.ssm.simulate(
            nsimulations=n,
            measurement_shocks=measurement_shocks,
            state_shocks=state_shocks,
            initial_state=initial_state,
        )

        desired_obs = np.zeros((n, self.model.k_endog))
        desired_state = np.zeros((n, self.model.k_states))
        desired_state[0] = initial_state
        desired_obs[0] = np.dot(Z, initial_state)
        for i in range(1, n):
            desired_state[i] = np.dot(T, desired_state[i - 1]) + state_shocks[i]
            desired_obs[i] = np.dot(Z, desired_state[i])

        assert_allclose(obs, desired_obs)
        assert_allclose(states, desired_state)

    @pytest.mark.thread_unsafe(reason="statespace cython is not thread safe")
    def test_simulation_smoothing_0(self):
        # Simulation smoothing when setting all variates to zeros
        # In this case:
        # - unconditional disturbances are zero, because they are simply
        #   transformed to have the appropriate variance matrix, but keep the
        #   same mean - of zero
        # - generated states are zeros, because initial state is
        #   zeros and all state disturbances are zeros
        # - generated observations are zeros, because states are zeros and all
        #    measurement disturbances are zeros
        # - The simulated state is equal to the smoothed state from the
        #   original model, because
        #   simulated state = (generated state - smoothed generated state +
        #                      smoothed state)
        #   and here generated state = smoothed generated state = 0
        # - The simulated measurement disturbance is equal to the smoothed
        #   measurement disturbance for very similar reasons, because
        #   simulated measurement disturbance = (
        #       generated measurement disturbance -
        #       smoothed generated measurement disturbance +
        #       smoothed measurement disturbance)
        #   and here generated measurement disturbance and
        #   smoothed generated measurement disturbance are zero.
        # - The simulated state disturbance is equal to the smoothed
        #   state disturbance for exactly the same reason as above.
        sim = self.sim
        Z = self.model["design"]

        nobs = self.model.nobs
        k_endog = self.model.k_endog
        k_posdef = self.model.ssm.k_posdef
        k_states = self.model.k_states

        # Test against known quantities (see above for description)
        sim.simulate(
            measurement_disturbance_variates=np.zeros(nobs * k_endog),
            state_disturbance_variates=np.zeros(nobs * k_posdef),
            initial_state_variates=np.zeros(k_states),
        )
        assert_allclose(sim.generated_measurement_disturbance, 0)
        assert_allclose(sim.generated_state_disturbance, 0)
        assert_allclose(sim.generated_state, 0)
        assert_allclose(sim.generated_obs, 0)
        assert_allclose(sim.simulated_state, self.results.smoothed_state)
        if not self.model.ssm.filter_collapsed:
            assert_allclose(
                sim.simulated_measurement_disturbance,
                self.results.smoothed_measurement_disturbance,
            )
        assert_allclose(
            sim.simulated_state_disturbance, self.results.smoothed_state_disturbance
        )

        # Test against R package KFAS values
        if self.test_against_KFAS:
            path = Path(current_path).joinpath("results", "results_simulation_smoothing0.csv")
            true = pd.read_csv(path)

            assert_allclose(
                sim.simulated_state, true[["state1", "state2", "state3"]].T, atol=1e-7
            )
            assert_allclose(
                sim.simulated_measurement_disturbance,
                true[["eps1", "eps2", "eps3"]].T,
                atol=1e-7,
            )
            assert_allclose(
                sim.simulated_state_disturbance,
                true[["eta1", "eta2", "eta3"]].T,
                atol=1e-7,
            )
            signals = np.zeros((3, self.model.nobs))
            for t in range(self.model.nobs):
                signals[:, t] = np.dot(Z, sim.simulated_state[:, t])
            assert_allclose(
                signals, true[["signal1", "signal2", "signal3"]].T, atol=1e-7
            )

    @pytest.mark.thread_unsafe(reason="statespace cython is not thread safe")
    def test_simulation_smoothing_1(self):
        # Test with measurement disturbance as np.arange / 10., all other
        # disturbances are zeros
        sim = self.sim

        Z = self.model["design"]

        nobs = self.model.nobs
        k_endog = self.model.k_endog
        k_posdef = self.model.ssm.k_posdef
        k_states = self.model.k_states

        # Construct the variates
        measurement_disturbance_variates = np.reshape(
            np.arange(nobs * k_endog) / 10.0, (nobs, k_endog)
        )
        state_disturbance_variates = np.zeros(nobs * k_posdef)

        # Compute some additional known quantities
        generated_measurement_disturbance = np.zeros(
            measurement_disturbance_variates.shape
        )
        chol = np.linalg.cholesky(self.model["obs_cov"])
        for t in range(self.model.nobs):
            generated_measurement_disturbance[t] = np.dot(
                chol, measurement_disturbance_variates[t]
            )
        y = generated_measurement_disturbance.copy()
        y[np.isnan(self.model.endog)] = np.nan

        generated_model = mlemodel.MLEModel(y, k_states=k_states, k_posdef=k_posdef)
        for name in ["design", "obs_cov", "transition", "selection", "state_cov"]:
            generated_model[name] = self.model[name]

        generated_model.initialize_approximate_diffuse(1e6)
        generated_model.ssm.filter_univariate = True
        generated_res = generated_model.ssm.smooth()
        simulated_state = 0 - generated_res.smoothed_state + self.results.smoothed_state
        if not self.model.ssm.filter_collapsed:
            simulated_measurement_disturbance = (
                generated_measurement_disturbance.T
                - generated_res.smoothed_measurement_disturbance
                + self.results.smoothed_measurement_disturbance
            )
        simulated_state_disturbance = (
            0
            - generated_res.smoothed_state_disturbance
            + self.results.smoothed_state_disturbance
        )

        # Test against known values
        sim.simulate(
            measurement_disturbance_variates=(measurement_disturbance_variates),
            state_disturbance_variates=state_disturbance_variates,
            initial_state_variates=np.zeros(k_states),
        )
        assert_allclose(
            sim.generated_measurement_disturbance, generated_measurement_disturbance
        )
        assert_allclose(sim.generated_state_disturbance, 0)
        assert_allclose(sim.generated_state, 0)
        assert_allclose(sim.generated_obs, generated_measurement_disturbance.T)
        assert_allclose(sim.simulated_state, simulated_state)
        if not self.model.ssm.filter_collapsed:
            assert_allclose(
                sim.simulated_measurement_disturbance, simulated_measurement_disturbance
            )
        assert_allclose(sim.simulated_state_disturbance, simulated_state_disturbance)

        # Test against R package KFAS values
        if self.test_against_KFAS:
            path = Path(current_path).joinpath("results", "results_simulation_smoothing1.csv")
            true = pd.read_csv(path)
            assert_allclose(
                sim.simulated_state, true[["state1", "state2", "state3"]].T, atol=1e-7
            )
            assert_allclose(
                sim.simulated_measurement_disturbance,
                true[["eps1", "eps2", "eps3"]].T,
                atol=1e-7,
            )
            assert_allclose(
                sim.simulated_state_disturbance,
                true[["eta1", "eta2", "eta3"]].T,
                atol=1e-7,
            )
            signals = np.zeros((3, self.model.nobs))
            for t in range(self.model.nobs):
                signals[:, t] = np.dot(Z, sim.simulated_state[:, t])
            assert_allclose(
                signals, true[["signal1", "signal2", "signal3"]].T, atol=1e-7
            )

    @pytest.mark.thread_unsafe(reason="statespace cython is not thread safe")
    def test_simulation_smoothing_2(self):
        # Test with measurement and state disturbances as np.arange / 10.,
        # initial state variates are zeros.
        sim = self.sim

        Z = self.model["design"]
        T = self.model["transition"]

        nobs = self.model.nobs
        k_endog = self.model.k_endog
        k_posdef = self.model.ssm.k_posdef
        k_states = self.model.k_states

        # Construct the variates
        measurement_disturbance_variates = np.reshape(
            np.arange(nobs * k_endog) / 10.0, (nobs, k_endog)
        )
        state_disturbance_variates = np.reshape(
            np.arange(nobs * k_posdef) / 10.0, (nobs, k_posdef)
        )
        initial_state_variates = np.zeros(k_states)

        # Compute some additional known quantities
        generated_measurement_disturbance = np.zeros(
            measurement_disturbance_variates.shape
        )
        chol = np.linalg.cholesky(self.model["obs_cov"])
        for t in range(self.model.nobs):
            generated_measurement_disturbance[t] = np.dot(
                chol, measurement_disturbance_variates[t]
            )

        generated_state_disturbance = np.zeros(state_disturbance_variates.shape)
        chol = np.linalg.cholesky(self.model["state_cov"])
        for t in range(self.model.nobs):
            generated_state_disturbance[t] = np.dot(chol, state_disturbance_variates[t])

        generated_obs = np.zeros((self.model.k_endog, self.model.nobs))
        generated_state = np.zeros((self.model.k_states, self.model.nobs + 1))
        chol = np.linalg.cholesky(self.results.initial_state_cov)
        generated_state[:, 0] = self.results.initial_state + np.dot(
            chol, initial_state_variates
        )
        for t in range(self.model.nobs):
            generated_state[:, t + 1] = (
                np.dot(T, generated_state[:, t]) + generated_state_disturbance.T[:, t]
            )
            generated_obs[:, t] = (
                np.dot(Z, generated_state[:, t])
                + generated_measurement_disturbance.T[:, t]
            )
        y = generated_obs.copy().T
        y[np.isnan(self.model.endog)] = np.nan

        generated_model = mlemodel.MLEModel(y, k_states=k_states, k_posdef=k_posdef)
        for name in ["design", "obs_cov", "transition", "selection", "state_cov"]:
            generated_model[name] = self.model[name]

        generated_model.initialize_approximate_diffuse(1e6)
        generated_model.ssm.filter_univariate = True
        generated_res = generated_model.ssm.smooth()
        simulated_state = (
            generated_state[:, :-1]
            - generated_res.smoothed_state
            + self.results.smoothed_state
        )
        if not self.model.ssm.filter_collapsed:
            simulated_measurement_disturbance = (
                generated_measurement_disturbance.T
                - generated_res.smoothed_measurement_disturbance
                + self.results.smoothed_measurement_disturbance
            )
        simulated_state_disturbance = (
            generated_state_disturbance.T
            - generated_res.smoothed_state_disturbance
            + self.results.smoothed_state_disturbance
        )

        # Test against known values
        sim.simulate(
            measurement_disturbance_variates=(measurement_disturbance_variates),
            state_disturbance_variates=state_disturbance_variates,
            initial_state_variates=np.zeros(k_states),
        )

        assert_allclose(
            sim.generated_measurement_disturbance, generated_measurement_disturbance
        )
        assert_allclose(sim.generated_state_disturbance, generated_state_disturbance)
        assert_allclose(sim.generated_state, generated_state)
        assert_allclose(sim.generated_obs, generated_obs)
        assert_allclose(sim.simulated_state, simulated_state, atol=1e-7)
        if not self.model.ssm.filter_collapsed:
            assert_allclose(
                sim.simulated_measurement_disturbance.T,
                simulated_measurement_disturbance.T,
            )
        assert_allclose(sim.simulated_state_disturbance, simulated_state_disturbance)

        # Test against R package KFAS values
        if self.test_against_KFAS:
            path = Path(current_path).joinpath("results", "results_simulation_smoothing2.csv")
            true = pd.read_csv(path)
            assert_allclose(
                sim.simulated_state.T, true[["state1", "state2", "state3"]], atol=1e-7
            )
            assert_allclose(
                sim.simulated_measurement_disturbance,
                true[["eps1", "eps2", "eps3"]].T,
                atol=1e-7,
            )
            assert_allclose(
                sim.simulated_state_disturbance,
                true[["eta1", "eta2", "eta3"]].T,
                atol=1e-7,
            )
            signals = np.zeros((3, self.model.nobs))
            for t in range(self.model.nobs):
                signals[:, t] = np.dot(Z, sim.simulated_state[:, t])
            assert_allclose(
                signals, true[["signal1", "signal2", "signal3"]].T, atol=1e-7
            )


class TestMultivariateVARKnown(MultivariateVARKnown):
    @classmethod
    def setup_class(cls, *args, **kwargs):
        super().setup_class()
        cls.true_llf = 39.01246166


class TestMultivariateVARKnownMissingAll(MultivariateVARKnown):
    """
    Notes
    -----
    Cannot test against KFAS because they have a different behavior for
    missing entries. When an entry is missing, KFAS does not draw a simulation
    smoothed value for that entry, whereas we draw from the unconditional
    distribution. It appears there is nothing to definitively recommend one
    approach over the other, but it makes it difficult to line up the variates
    correctly in order to replicate results.
    """

    @classmethod
    def setup_class(cls, *args, **kwargs):
        super().setup_class(missing="all", test_against_KFAS=False)
        cls.true_llf = 1305.739288


class TestMultivariateVARKnownMissingPartial(MultivariateVARKnown):
    @classmethod
    def setup_class(cls, *args, **kwargs):
        super().setup_class(missing="partial", test_against_KFAS=False)
        cls.true_llf = 1518.449598


class TestMultivariateVARKnownMissingMixed(MultivariateVARKnown):
    @classmethod
    def setup_class(cls, *args, **kwargs):
        super().setup_class(missing="mixed", test_against_KFAS=False)
        cls.true_llf = 1117.265303


class TestDFM(TestMultivariateVARKnown):
    test_against_KFAS = False

    @classmethod
    def setup_class(cls, which="none", *args, **kwargs):
        # Data
        dta = datasets.macrodata.load_pandas().data
        dta.index = pd.date_range(start="1959-01-01", end="2009-7-01", freq="QS")
        levels = dta[["realgdp", "realcons", "realinv"]]
        obs = np.log(levels).diff().iloc[1:] * 400

        if which == "all":
            obs.iloc[:50, :] = np.nan
            obs.iloc[119:130, :] = np.nan
        elif which == "partial":
            obs.iloc[0:50, 0] = np.nan
            obs.iloc[119:130, 0] = np.nan
        elif which == "mixed":
            obs.iloc[0:50, 0] = np.nan
            obs.iloc[19:70, 1] = np.nan
            obs.iloc[39:90, 2] = np.nan
            obs.iloc[119:130, 0] = np.nan
            obs.iloc[119:130, 2] = np.nan

        # Create the model with typical state space
        mod = mlemodel.MLEModel(obs, k_states=2, k_posdef=2, **kwargs)
        mod["design"] = np.array(
            [
                [-32.47143586, 17.33779024],
                [-7.40264169, 1.69279859],
                [-209.04702853, 125.2879374],
            ]
        )
        mod["obs_cov"] = np.diag(np.array([0.0622668, 1.95666886, 58.37473642]))
        mod["transition"] = np.array(
            [[0.29935707, 0.33289005], [-0.7639868, 1.2844237]]
        )
        mod["selection"] = np.eye(2)
        mod["state_cov"] = np.array([[1.2, -0.25], [-0.25, 1.1]])
        mod.initialize_approximate_diffuse(1e6)
        mod.ssm.filter_univariate = True
        mod.ssm.filter_collapsed = True
        cls.model = mod
        cls.results = mod.smooth([], return_ssm=True)

        cls.sim = cls.model.simulation_smoother()

    def test_loglike(self):
        pass


class MultivariateVAR:
    """
    More generic tests for simulation smoothing; use actual N(0,1) variates
    """

    @classmethod
    def setup_class(cls, missing="none", *args, **kwargs):
        # Data
        dta = datasets.macrodata.load_pandas().data
        dta.index = pd.date_range(start="1959-01-01", end="2009-7-01", freq="QS")
        obs = np.log(dta[["realgdp", "realcons", "realinv"]]).diff().iloc[1:]

        if missing == "all":
            obs.iloc[0:50, :] = np.nan
        elif missing == "partial":
            obs.iloc[0:50, 0] = np.nan
        elif missing == "mixed":
            obs.iloc[0:50, 0] = np.nan
            obs.iloc[19:70, 1] = np.nan
            obs.iloc[39:90, 2] = np.nan
            obs.iloc[119:130, 0] = np.nan
            obs.iloc[119:130, 2] = np.nan
            obs.iloc[-10:, :] = np.nan

        # Create the model
        mod = mlemodel.MLEModel(obs, k_states=3, k_posdef=3, **kwargs)
        mod["design"] = np.eye(3)
        mod["obs_cov"] = np.array(
            [
                [0.0000640649, 0.0, 0.0],
                [0.0, 0.0000572802, 0.0],
                [0.0, 0.0, 0.0017088585],
            ]
        )
        mod["transition"] = np.array(
            [
                [-0.1119908792, 0.8441841604, 0.0238725303],
                [0.2629347724, 0.4996718412, -0.0173023305],
                [-3.2192369082, 4.1536028244, 0.4514379215],
            ]
        )
        mod["selection"] = np.eye(3)
        mod["state_cov"] = np.array(
            [
                [0.0000640649, 0.0000388496, 0.0002148769],
                [0.0000388496, 0.0000572802, 0.000001555],
                [0.0002148769, 0.000001555, 0.0017088585],
            ]
        )
        mod.initialize_approximate_diffuse(1e6)
        mod.ssm.filter_univariate = True
        cls.model = mod
        cls.results = mod.smooth([], return_ssm=True)

        cls.sim = cls.model.simulation_smoother()

    def test_loglike(self):
        assert_allclose(np.sum(self.results.llf_obs), self.true_llf)

    @pytest.mark.thread_unsafe(reason="statespace cython is not thread safe")
    def test_simulation_smoothing(self):
        sim = self.sim

        Z = self.model["design"]

        nobs = self.model.nobs
        k_endog = self.model.k_endog

        # Simulate with known variates
        # TODO
        sim.simulate(
            measurement_disturbance_variates=(self.variates[: nobs * k_endog]),
            state_disturbance_variates=(self.variates[nobs * k_endog : -3]),
            initial_state_variates=self.variates[-3:],
        )

        # Test against R package KFAS values
        assert_allclose(
            sim.simulated_state.T, self.true[["state1", "state2", "state3"]], atol=1e-7
        )
        assert_allclose(
            sim.simulated_measurement_disturbance,
            self.true[["eps1", "eps2", "eps3"]].T,
            atol=1e-7,
        )
        assert_allclose(
            sim.simulated_state_disturbance,
            self.true[["eta1", "eta2", "eta3"]].T,
            atol=1e-7,
        )
        signals = np.zeros((3, self.model.nobs))
        for t in range(self.model.nobs):
            signals[:, t] = np.dot(Z, sim.simulated_state[:, t])
        assert_allclose(
            signals, self.true[["signal1", "signal2", "signal3"]].T, atol=1e-7
        )


class TestMultivariateVAR(MultivariateVAR):
    @classmethod
    def setup_class(cls):
        super().setup_class()
        path = Path(current_path).joinpath("results", "results_simulation_smoothing3_variates.csv")
        cls.variates = pd.read_csv(path).values.squeeze()
        path = Path(current_path).joinpath("results", "results_simulation_smoothing3.csv")
        cls.true = pd.read_csv(path)
        cls.true_llf = 1695.34872


def test_misc():

    # Create the model and simulation smoother
    dta = datasets.macrodata.load_pandas().data
    dta.index = pd.date_range(start="1959-01-01", end="2009-7-01", freq="QS")
    obs = np.log(dta[["realgdp", "realcons", "realinv"]]).diff().iloc[1:]

    mod = sarimax.SARIMAX(obs["realgdp"], order=(1, 0, 0))
    mod["design", 0, 0] = 0.0
    mod["obs_cov", 0, 0] = 1.0
    mod.update(np.r_[0.5, 1.0])
    sim = mod.simulation_smoother()

    # Test that the simulation smoother is drawing variates correctly
    rs = np.random.RandomState(1234)
    n_disturbance_variates = mod.nobs * (mod.k_endog + mod.k_states)
    variates = rs.normal(size=n_disturbance_variates)
    rs = np.random.RandomState(1234)
    sim.simulate(rng=rs)
    assert_allclose(sim.generated_measurement_disturbance[:, 0], variates[: mod.nobs])
    assert_allclose(sim.generated_state_disturbance[:, 0], variates[mod.nobs :])

    # Test that we can change the options of the simulations smoother
    assert_equal(sim.simulation_output, mod.ssm.smoother_output)
    sim.simulation_output = 0
    assert_equal(sim.simulation_output, 0)

    sim.simulate_state = True
    assert_equal(sim.simulation_output, SIMULATION_STATE)
    sim.simulate_state = False
    assert_equal(sim.simulation_output, 0)

    sim.simulate_disturbance = True
    assert_equal(sim.simulation_output, SIMULATION_DISTURBANCE)
    sim.simulate_disturbance = False
    assert_equal(sim.simulation_output, 0)

    sim.simulate_all = True
    assert_equal(sim.simulation_output, SIMULATION_ALL)
    sim.simulate_all = False
    assert_equal(sim.simulation_output, 0)


def test_simulation_smooth_results_output_flag_properties():
    # `simulate_state`, `simulate_disturbance`, and `simulate_all` are
    # boolean flag properties backed by the `simulation_output` bitmask
    # (the same OptionWrapper-style pattern used elsewhere in
    # tsa.statespace, e.g. `smoother_state`/`smoother_all` in
    # kalman_smoother.py). `test_misc` above exercises the *setters*
    # indirectly (by checking `simulation_output` afterwards) but never
    # calls the getters, so check the getters directly here against
    # independently-computed bitmask values.
    endog = np.arange(10) * 1.0
    mod = sarimax.SARIMAX(endog, order=(1, 0, 0))
    mod.update([0.5, 1.0])
    sim = mod.simulation_smoother()

    sim.simulation_output = 0
    assert sim.simulate_state is False
    assert sim.simulate_disturbance is False
    assert sim.simulate_all is False

    sim.simulation_output = SIMULATION_STATE
    assert sim.simulate_state is True
    assert sim.simulate_disturbance is False
    # SIMULATION_ALL = SIMULATION_STATE | SIMULATION_DISTURBANCE, so
    # `simulate_all` reads True whenever *either* component bit is set
    # (matching the analogous `smoother_all`/OptionWrapper "any-bit"
    # semantics used throughout this subpackage), not only when both are.
    assert sim.simulate_all is True

    sim.simulation_output = SIMULATION_DISTURBANCE
    assert sim.simulate_state is False
    assert sim.simulate_disturbance is True
    assert sim.simulate_all is True

    sim.simulation_output = SIMULATION_ALL
    assert sim.simulate_state is True
    assert sim.simulate_disturbance is True
    assert sim.simulate_all is True

    # Setters round-trip through the same bitmask; verify by re-reading the
    # flag itself (not just `simulation_output`).
    sim.simulate_state = False
    assert sim.simulation_output == SIMULATION_DISTURBANCE
    assert sim.simulate_state is False
    assert sim.simulate_disturbance is True

    sim.simulate_disturbance = False
    assert sim.simulation_output == 0
    assert sim.simulate_all is False


def test_simulation_smoother_invalid_method_raises():
    dta = datasets.macrodata.load_pandas().data
    dta.index = pd.date_range(start="1959-01-01", end="2009-7-01", freq="QS")
    obs = np.log(dta[["realgdp"]]).diff().iloc[1:]

    mod = sarimax.SARIMAX(obs["realgdp"], order=(1, 0, 0))
    with pytest.raises(ValueError, match="method"):
        mod.simulation_smoother(method="invalid")


def test_simulation_smoothing_obs_intercept():
    nobs = 10
    intercept = 100
    endog = np.ones(nobs) * intercept
    mod = structural.UnobservedComponents(endog, "rwalk", exog=np.ones(nobs))
    mod.update([1, intercept])
    sim = mod.simulation_smoother()
    sim.simulate(
        measurement_disturbance_variates=np.zeros(mod.nobs),
        state_disturbance_variates=np.zeros(mod.nobs),
        initial_state_variates=np.zeros(1),
    )
    assert_equal(sim.simulated_state[0], 0)


def test_simulation_smoothing_state_intercept():
    nobs = 10
    intercept = 100
    endog = np.ones(nobs) * intercept

    mod = sarimax.SARIMAX(endog, order=(0, 0, 0), trend="c", measurement_error=True)
    mod.initialize_known([100], [[0]])
    mod.update([intercept, 1.0, 1.0])

    sim = mod.simulation_smoother()
    sim.simulate(
        measurement_disturbance_variates=np.zeros(mod.nobs),
        state_disturbance_variates=np.zeros(mod.nobs),
        initial_state_variates=np.zeros(1),
    )
    assert_equal(sim.simulated_state[0], intercept)


def test_simulation_smoothing_state_intercept_diffuse():
    nobs = 10
    intercept = 100
    endog = np.ones(nobs) * intercept

    # Test without missing values
    mod = sarimax.SARIMAX(
        endog,
        order=(0, 0, 0),
        trend="c",
        measurement_error=True,
        initialization="diffuse",
    )
    mod.update([intercept, 1.0, 1.0])

    sim = mod.simulation_smoother()
    sim.simulate(
        measurement_disturbance_variates=np.zeros(mod.nobs),
        state_disturbance_variates=np.zeros(mod.nobs),
        initial_state_variates=np.zeros(1),
    )
    assert_equal(sim.simulated_state[0], intercept)

    # Test with missing values
    endog[5] = np.nan
    mod = sarimax.SARIMAX(
        endog,
        order=(0, 0, 0),
        trend="c",
        measurement_error=True,
        initialization="diffuse",
    )
    mod.update([intercept, 1.0, 1.0])

    sim = mod.simulation_smoother()
    sim.simulate(
        measurement_disturbance_variates=np.zeros(mod.nobs),
        state_disturbance_variates=np.zeros(mod.nobs),
        initial_state_variates=np.zeros(1),
    )
    assert_equal(sim.simulated_state[0], intercept)


def test_nan():
    """
    This is a very slow test to check that the distribution of simulated states
    (from the posterior) is correct in the presense of NaN values. Here, it
    checks the marginal distribution of the drawn states against the values
    computed from the smoother and prints the result.

    With the fixed simulation smoother, it prints:

    True values:
    [1.         0.66666667 0.66666667 1.        ]
    [0.         0.95238095 0.95238095 0.        ]

    Simulated values:
    [1.         0.66699187 0.66456719 1.        ]
    [0.       0.953608 0.953198 0.      ]

    Previously, it would have printed:

    True values:
    [1.         0.66666667 0.66666667 1.        ]
    [0.         0.95238095 0.95238095 0.        ]

    Simulated values:
    [1.         0.66666667 0.66666667 1.        ]
    [0. 0. 0. 0.]
    """
    return
    mod = sarimax.SARIMAX([1, np.nan, np.nan, 1], order=(1, 0, 0), trend="c")
    res = mod.smooth([0, 0.5, 1.0])

    rs = np.random.RandomState(1234)
    sim = mod.simulation_smoother(rng=rs)

    n = 1000000
    out = np.zeros((n, mod.nobs))
    for i in range(n):
        sim.simulate()
        out[i] = sim.simulated_state

    print("True values:")
    print(res.smoothed_state[0])
    print(res.smoothed_state_cov[0, 0])
    print()
    print("Simulated values:")
    print(np.mean(out, axis=0))
    print(np.var(out, axis=0).round(6))


@pytest.mark.parametrize(
    "dtype, prefix",
    [(np.float32, "s"), (np.float64, "d"), (np.complex64, "c"), (complex, "z")],
)
def test_simulate_default_variates_dtype(dtype, prefix):
    # Without explicit variates, simulate draws them from rng. The draws are
    # float64, and the float32, complex64 and complex128 simulation smoothers
    # raised "Buffer dtype mismatch" when assigning them.
    nobs, k_endog, k_states, k_posdef = 12, 2, 3, 2

    def build(dtype):
        endog = np.random.default_rng(1234).standard_normal((nobs, k_endog))
        mod = SimulationSmoother(k_endog, k_states, k_posdef=k_posdef, dtype=dtype)
        mod.bind(endog.astype(dtype))
        mod["design"] = np.array([[1.0, 0.0, 0.5], [0.3, 1.0, 0.0]], dtype=dtype)
        mod["obs_cov"] = np.array([[0.5, 0.1], [0.1, 0.4]], dtype=dtype)
        mod["transition"] = np.array(
            [[0.6, 0.2, 0.0], [0.0, 0.4, 0.1], [1.0, 0.0, 0.0]], dtype=dtype
        )
        mod["selection"] = np.array([[1.0, 0.0], [0.0, 1.0], [0.0, 0.0]], dtype=dtype)
        mod["state_cov"] = np.array([[1.0, 0.2], [0.2, 0.5]], dtype=dtype)
        mod.initialize_known(np.array([0.1, -0.2, 0.0]), np.diag([1.0, 0.5, 0.8]))
        return mod

    names = [
        "simulated_state",
        "simulated_measurement_disturbance",
        "simulated_state_disturbance",
    ]
    mod = build(dtype)
    sim = mod.simulation_smoother()
    sim.simulate(rng=np.random.default_rng(5678))
    assert mod.prefix == prefix

    # The same draws, in the order simulate draws them, passed explicitly
    rng = np.random.default_rng(5678)
    measurement = rng.normal(size=(nobs, k_endog))
    state = rng.normal(size=(nobs, k_posdef))
    initial = rng.normal(size=k_states)
    explicit = build(dtype).simulation_smoother()
    explicit.simulate(
        measurement_disturbance_variates=measurement.astype(dtype),
        state_disturbance_variates=state.astype(dtype),
        initial_state_variates=initial.astype(dtype),
    )
    # The float64 simulation smoother with the same draws
    desired = build(np.float64).simulation_smoother()
    desired.simulate(rng=np.random.default_rng(5678))

    # Single precision: the float64 results agree to about 1e-7 times the
    # largest value, so the atol for the smaller elements (down to 0.006)
    rtol = 1e-4 if prefix in ("s", "c") else 1e-12
    for name in names:
        actual = getattr(sim, name)
        assert actual.dtype == dtype
        # The same computation on the same values
        assert_allclose(actual, getattr(explicit, name), rtol=1e-6)
        expected = getattr(desired, name)
        atol = rtol * np.abs(expected).max()
        assert_allclose(actual.real, expected, rtol=rtol, atol=atol)
        # The imaginary part is zero in exact arithmetic; allow rounding
        # errors relative to the size of the values
        atol = 10 * np.finfo(dtype).eps * np.abs(actual.real).max()
        assert_allclose(actual.imag, 0, atol=atol)


def bivariate_model(nobs=60, time_varying_obs_cov=False):
    rng = np.random.default_rng(20261122)
    endog = rng.standard_normal((nobs, 2))
    mod = mlemodel.MLEModel(endog, k_states=2, k_posdef=2)
    mod["design"] = np.array([[1.0, 0.0], [0.5, 1.0]])
    mod["obs_intercept"] = np.array([0.1, -0.2])
    mod["obs_cov"] = np.diag([1.0, 0.7])
    mod["transition"] = np.array([[0.9, 0.1], [0.0, 0.8]])
    mod["state_intercept"] = np.array([0.05, 0.02])
    mod["selection"] = np.eye(2)
    mod["state_cov"] = np.array([[0.5, 0.2], [0.2, 0.4]])
    if time_varying_obs_cov:
        mod["obs_cov"] = np.repeat(np.diag([1.0, 0.7])[:, :, None], nobs, axis=2)
    mod.ssm.initialize_known(np.array([0.3, -0.1]), np.eye(2))
    return mod


def test_simulation_smooth_results_class():
    class CustomResults(SimulationSmoothResults):
        pass

    mod = bivariate_model()
    # Default class
    assert type(mod.simulation_smoother()) is SimulationSmoothResults
    # Class given to the model
    ssm = SimulationSmoother(
        2,
        2,
        simulation_smooth_results_class=CustomResults,
        **{name: mod[name] for name in mod.ssm.shapes if name != "obs"},
    )
    ssm.bind(np.ascontiguousarray(mod.endog))
    ssm.initialize_known(np.array([0.3, -0.1]), np.eye(2))
    sim = ssm.simulation_smoother()
    assert type(sim) is CustomResults
    # Class given to the method
    sim = mod.ssm.simulation_smoother(results_class=CustomResults)
    assert type(sim) is CustomResults
    # Both simulate the same draws
    variates = dict(
        measurement_disturbance_variates=np.zeros((mod.nobs, 2)),
        state_disturbance_variates=np.zeros((mod.nobs, 2)),
        initial_state_variates=np.zeros(2),
    )
    desired = mod.simulation_smoother()
    desired.simulate(**variates)
    sim.simulate(**variates)
    assert_allclose(sim.simulated_state, desired.simulated_state, rtol=1e-12)


def test_simulation_smoother_invalid_results_class():
    mod = bivariate_model()
    with pytest.raises(ValueError, match="Invalid results class"):
        mod.ssm.simulation_smoother(results_class=object)


@pytest.mark.parametrize(
    "kwargs, desired",
    [
        (dict(simulate_state=True), SIMULATION_STATE),
        (dict(simulate_disturbance=True), SIMULATION_DISTURBANCE),
        (dict(simulate_all=True), SIMULATION_ALL),
        (dict(simulate_state=True, simulate_disturbance=False), SIMULATION_STATE),
        # Without any option: the smoother output, which includes both
        ({}, SIMULATION_ALL),
    ],
)
def test_simulation_output_options(kwargs, desired):
    mod = bivariate_model()
    sim = mod.simulation_smoother(**kwargs)
    assert (sim.simulation_output & SIMULATION_ALL) == desired
    assert sim.simulate_state is bool(desired & SIMULATION_STATE)
    assert sim.simulate_disturbance is bool(desired & SIMULATION_DISTURBANCE)


@pytest.mark.parametrize(
    "kwargs",
    [
        dict(simulate_state=False),
        dict(simulate_disturbance=False),
        dict(simulate_all=False),
    ],
)
def test_simulation_output_options_no_output(kwargs):
    mod = bivariate_model()
    with pytest.raises(ValueError, match="Invalid simulation output options"):
        mod.simulation_smoother(**kwargs)


def test_cfa_simulation_output():
    mod = bivariate_model()
    with pytest.raises(ValueError, match="Can only retrieve simulations of the state"):
        mod.simulation_smoother(method="cfa", simulation_output=SIMULATION_DISTURBANCE)


def test_generated_variates_and_series():
    # With pretransformed variates the generated disturbances are the given
    # variates, and the generated series follow the state space recursions
    mod = bivariate_model(nobs=5)
    rng = np.random.default_rng(1122)
    eps = rng.standard_normal((5, 2))
    eta = rng.standard_normal((5, 2))
    initial_state = np.array([1.0, 2.0])
    sim = mod.simulation_smoother()
    sim.simulate(
        measurement_disturbance_variates=eps,
        state_disturbance_variates=eta,
        initial_state_variates=initial_state,
        pretransformed_measurement_disturbance_variates=True,
        pretransformed_state_disturbance_variates=True,
        pretransformed_initial_state_variates=True,
    )

    design, obs_intercept = mod["design"], mod["obs_intercept"]
    transition, state_intercept = mod["transition"], mod["state_intercept"]
    state = np.zeros((2, 6))
    state[:, 0] = initial_state
    obs = np.zeros((2, 5))
    for t in range(5):
        obs[:, t] = obs_intercept + design @ state[:, t] + eps[t]
        state[:, t + 1] = state_intercept + transition @ state[:, t] + eta[t]

    # Each property is computed on first access and then cached
    for _ in range(2):
        assert_allclose(sim.generated_measurement_disturbance, eps, rtol=1e-15)
        assert_allclose(sim.generated_state_disturbance, eta, rtol=1e-15)
        assert_allclose(sim.generated_obs, obs, rtol=1e-12)
        assert_allclose(sim.generated_state, state, rtol=1e-12)


def test_simulate_after_dimension_change():
    mod = bivariate_model()
    sim = mod.simulation_smoother()
    # A time-varying matrix replaces the compiled representation
    mod["obs_cov"] = np.repeat(np.diag([1.0, 0.7])[:, :, None], mod.nobs, axis=2)
    with pytest.raises(ValueError, match="cannot replace the underlying"):
        sim.simulate()


@pytest.mark.parametrize(
    "time_varying, kwargs, match",
    [
        (False, dict(nobs=0), "Invalid number of simulations"),
        (False, dict(nobs=10), "Setting the `nobs` variable is not compatible"),
        (True, dict(nobs=100, simulation_output=0), "cannot create more simulations"),
    ],
)
def test_simulation_smoother_invalid_nobs(time_varying, kwargs, match):
    mod = bivariate_model(time_varying_obs_cov=time_varying)
    with pytest.raises(ValueError, match=match):
        mod.ssm.simulation_smoother(**kwargs)
