"""
Tests corresponding to statsmodels.base._constraints

Ported from the historical `if __name__ == "__main__":` self-test block
in examples/try_fit_constrained.py.
"""
import numpy as np
from numpy.testing import assert_allclose
import pytest

from statsmodels.base._constraints import TransformRestriction
from statsmodels.regression.linear_model import OLS


class TestTransformRestriction:
    def setup_method(self):
        self.R = np.array([[1, 1, 0, 0, 0], [0, 0, 1, -1, 0]])
        self.q = [2, 0]
        self.tr = TransformRestriction(self.R, self.q)

    def test_expand_reduce_roundtrip(self):
        p_reduced = [1, 1, 1]
        assert_allclose(
            self.tr.reduce(self.tr.expand(p_reduced)), p_reduced, rtol=1e-14
        )

    def test_expand_satisfies_constraint(self):
        p_reduced = [1, 1, 1]
        p = self.tr.expand(p_reduced)
        assert_allclose(self.R.dot(p), self.q, rtol=1e-14)

    def test_homogeneous_restriction_with_forced_zero_solution(self):
        # inconsistent restrictions that nonetheless have a solution where
        # the relevant parameter is forced to zero (b1+b2=0 and b1+2*b2=0
        # implies b2=0)
        Ri = np.array([[1, 1, 0, 0, 0], [0, 0, 1, -1, 0], [0, 0, 1, -2, 0]])
        tri = TransformRestriction(Ri, [0, 1, 1])
        p = tri.expand([1, 1])
        assert_allclose(p[[2, 3]], [1.0, 0.0], atol=1e-10)

    def test_infeasible_restriction_raises(self):
        # L does not have full row rank here, so solving for the constant
        # fails with a singular matrix
        Ri2 = np.array([[0, 0, 0, 1, 0], [0, 0, 1, -1, 0], [0, 0, 1, -2, 0]])
        q = [1, 1]
        with pytest.raises(ValueError, match="possibly inconsistent constraints"):
            TransformRestriction(Ri2, q)


def test_transform_restriction_matches_direct_constrained_ols():
    # Fit OLS on a reduced/transformed exog implied by a linear equality
    # constraint on two of the parameters, then expand the estimated
    # reduced-space parameters back to the original space. The result
    # should match the parameters from an unconstrained OLS on the same
    # data (the constraint holds exactly for the data-generating process
    # by construction: exog sums to endog plus noise).
    rng = np.random.RandomState(123)
    nobs = 20
    x = 1 + rng.randn(nobs, 4)
    exog = np.column_stack((np.ones(nobs), x))
    endog = exog.sum(1) + rng.randn(nobs)

    res_direct = OLS(endog, exog).fit()

    transf = TransformRestriction([[0, 0, 0, 1, 1]], res_direct.params[-2:].sum())
    exog_reduced = transf.reduce(exog)
    offset = exog.dot(transf.constant.squeeze())
    res_reduced = OLS(endog - offset, exog_reduced).fit()
    params = transf.expand(res_reduced.params).squeeze()

    assert_allclose(params, res_direct.params, rtol=1e-13)
