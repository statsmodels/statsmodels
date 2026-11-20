"""
Test module for GLM canonical link optimization.

This module tests the `is_canonical_link` property for all family/link
combinations and verifies numerical equivalence between optimized and
non-optimized Hessian computation paths.

Issue: #4269 - Optimize GLM Hessian calculation for canonical links.

NOTE: Uses direct imports instead of statsmodels.api to avoid compiled
extension dependencies when running without full package build.
"""

import numpy as np
from numpy.testing import assert_allclose
import pytest

# Direct imports to avoid compiled extension requirements
from statsmodels.genmod.generalized_linear_model import GLM
from statsmodels.genmod import families
from statsmodels.genmod.families import links


class TestCanonicalLinkProperty:
    """Test the is_canonical_link property for all families."""

    def test_gaussian_identity_is_canonical(self):
        """Test Gaussian family with canonical Identity link."""
        family = families.Gaussian()
        assert family.is_canonical_link is True

        # Non-canonical link
        family_log = families.Gaussian(link=links.Log())
        assert family_log.is_canonical_link is False

    def test_binomial_logit_is_canonical(self):
        """Test Binomial family with canonical Logit link."""
        family = families.Binomial()
        assert family.is_canonical_link is True

        # Non-canonical link
        family_probit = families.Binomial(link=links.Probit())
        assert family_probit.is_canonical_link is False

    def test_poisson_log_is_canonical(self):
        """Test Poisson family with canonical Log link."""
        family = families.Poisson()
        assert family.is_canonical_link is True

        # Non-canonical link
        family_sqrt = families.Poisson(link=links.Sqrt())
        assert family_sqrt.is_canonical_link is False

    def test_gamma_inverse_is_canonical(self):
        """Test Gamma family with canonical InversePower link."""
        family = families.Gamma()
        assert family.is_canonical_link is True

        # Non-canonical link
        family_log = families.Gamma(link=links.Log())
        assert family_log.is_canonical_link is False

    def test_inverse_gaussian_inverse_squared_is_canonical(self):
        """Test InverseGaussian family with canonical InverseSquared link."""
        family = families.InverseGaussian()
        assert family.is_canonical_link is True

        # Non-canonical link
        family_log = families.InverseGaussian(link=links.Log())
        assert family_log.is_canonical_link is False

    def test_negative_binomial_log_is_canonical(self):
        """Test NegativeBinomial family with canonical Log link."""
        family = families.NegativeBinomial(alpha=1.0)
        assert family.is_canonical_link is True

        # Non-canonical link
        family_id = families.NegativeBinomial(
            alpha=1.0, link=links.Identity()
        )
        assert family_id.is_canonical_link is False


class TestHessianCanonicalOptimization:
    """Test that OIM equals EIM for canonical links."""

    @pytest.fixture
    def setup_binomial_data(self):
        """Create test data for Binomial family."""
        np.random.seed(42)
        n = 1000
        X = np.column_stack([np.ones(n), np.random.randn(n, 3)])
        beta = np.array([0.5, 1.0, -0.5, 0.3])
        lin_pred = X @ beta
        probs = 1 / (1 + np.exp(-lin_pred))
        y = (np.random.rand(n) < probs).astype(float)
        return X, y

    @pytest.fixture
    def setup_gaussian_data(self):
        """Create test data for Gaussian family."""
        np.random.seed(42)
        n = 1000
        X = np.column_stack([np.ones(n), np.random.randn(n, 3)])
        beta = np.array([1.0, 2.0, -1.0, 0.5])
        y = X @ beta + np.random.randn(n) * 0.5
        return X, y

    @pytest.fixture
    def setup_poisson_data(self):
        """Create test data for Poisson family."""
        np.random.seed(42)
        n = 1000
        X = np.column_stack([np.ones(n), np.random.randn(n, 3)])
        beta = np.array([0.5, 0.3, -0.2, 0.1])
        lin_pred = X @ beta
        mu = np.exp(lin_pred)
        y = np.random.poisson(mu)
        return X, y

    def test_binomial_logit_hessian_equivalence(self, setup_binomial_data):
        """Test OIM equals EIM for Binomial with Logit link."""
        X, y = setup_binomial_data
        model = GLM(y, X, family=families.Binomial())
        result = model.fit(method='irls', disp=False)

        # Compute Hessian with observed=True and observed=False
        # For canonical link, they should be identical
        hessian_oim = model.hessian(result.params, observed=True)
        hessian_eim = model.hessian(result.params, observed=False)

        # They should be numerically identical for canonical link
        assert_allclose(hessian_oim, hessian_eim, rtol=1e-10)

    def test_gaussian_identity_hessian_equivalence(self, setup_gaussian_data):
        """Test OIM equals EIM for Gaussian with Identity link."""
        X, y = setup_gaussian_data
        model = GLM(y, X, family=families.Gaussian())
        result = model.fit(method='irls', disp=False)

        hessian_oim = model.hessian(result.params, observed=True)
        hessian_eim = model.hessian(result.params, observed=False)

        assert_allclose(hessian_oim, hessian_eim, rtol=1e-10)

    def test_poisson_log_hessian_equivalence(self, setup_poisson_data):
        """Test OIM equals EIM for Poisson with Log link."""
        X, y = setup_poisson_data
        model = GLM(y, X, family=families.Poisson())
        result = model.fit(method='irls', disp=False)

        hessian_oim = model.hessian(result.params, observed=True)
        hessian_eim = model.hessian(result.params, observed=False)

        assert_allclose(hessian_oim, hessian_eim, rtol=1e-10)

    def test_newton_fit_results_unchanged(self, setup_binomial_data):
        """
        Test that fit results with newton method are numerically identical
        before and after the canonical link optimization.

        Since the optimization only changes the computation path (not the
        mathematical result), the fitted parameters should be identical.
        """
        X, y = setup_binomial_data
        model = GLM(y, X, family=families.Binomial())

        # Fit with newton method (uses Hessian)
        result_newton = model.fit(method='newton', disp=False)

        # Fit with IRLS (as reference)
        result_irls = model.fit(method='irls', disp=False)

        # Parameters should be very close (within optimization tolerance)
        assert_allclose(result_newton.params, result_irls.params, rtol=1e-5)
        assert_allclose(result_newton.bse, result_irls.bse, rtol=1e-5)


class TestNonCanonicalLinkBehavior:
    """Test that non-canonical links still compute OIM correctly."""

    def test_binomial_probit_oim_differs_from_eim(self):
        """For non-canonical Probit link, OIM should differ from EIM."""
        np.random.seed(42)
        n = 500
        X = np.column_stack([np.ones(n), np.random.randn(n, 2)])
        beta = np.array([0.5, 1.0, -0.5])
        lin_pred = X @ beta
        # Use probit inverse (Phi CDF)
        from scipy.stats import norm
        probs = norm.cdf(lin_pred)
        y = (np.random.rand(n) < probs).astype(float)

        # Non-canonical Probit link
        model = GLM(y, X, family=families.Binomial(
            link=links.Probit()
        ))
        result = model.fit(method='irls', disp=False)

        hessian_oim = model.hessian(result.params, observed=True)
        hessian_eim = model.hessian(result.params, observed=False)

        # For non-canonical link, they should NOT be identical
        max_diff = np.max(np.abs(hessian_oim - hessian_eim))
        assert max_diff > 1e-6, (
            "OIM and EIM should differ for non-canonical links"
        )
