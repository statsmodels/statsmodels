
import pytest
import numpy as np
from numpy.testing import assert_allclose

from statsmodels.robust import norms
from .results import results_norms as res_r

cases = [
    (norms.Hampel, (1.5, 3.5, 8.), res_r.res_hampel)
    ]

@pytest.mark.parametrize("case", cases)
def test_norm(case):
    ncls, args, res = case
    norm = ncls(*args)
    x = np.array([-9., -6, -2, -1, 0, 1, 2, 6, 9])

    assert_allclose(norm.weights(x), res.weights, rtol=1e-12, atol=1e-20)
    assert_allclose(norm.rho(x), res.rho, rtol=1e-12, atol=1e-20)
    assert_allclose(norm.psi(x), res.psi, rtol=1e-12, atol=1e-20)
    assert_allclose(norm.psi_deriv(x), res.psi_deriv, rtol=1e-12, atol=1e-20)
