
import numpy as np
from numpy.testing import assert_allclose, assert_equal

from statsmodels.stats.covariance import transform_corr_normal

def test_transform_corr_normal():

    # numbers from footnote of Table 1 Boudt, Cornelissen, Croux 2012
    vp, vgr, vs, vk, vmcd = np.array([[0.92, 0.92, 1.02, 1.01, 1.45],
                                      [0.13, 0.13, 0.16, 0.15, 0.20]]).T

    rho = np.array([0.2, 0.8])
    for method in ['pearson', 'gauss_rank']:
        r, v = transform_corr_normal(rho, method, return_var=True)
        assert_equal(r, rho)
        assert_allclose(v, vp, atol=0.05)

    method = 'kendal'
    r, v = transform_corr_normal(rho, method, return_var=True)
    # hardcoded transformation from BCC 2012 (round trip test)
    assert_allclose(2 / np.pi * np.arcsin(r), rho, atol=0.05)
    assert_allclose(v, vk, atol=0.05)

    method = 'quadrant'
    rho_ = 0.8  # Croux, Dehon 2010
    r, v = transform_corr_normal(rho_, method, return_var=True)
    # hardcoded transformation from BCC 2012 (round trip test)
    assert_allclose(2 / np.pi * np.arcsin(r), rho_, atol=0.05)
    assert_allclose(v, 0.58, atol=0.05)

    method = 'spearman'
    r, v = transform_corr_normal(rho, method, return_var=True)
    # hardcoded transformation from BCC 2012 (round trip test)
    assert_allclose(6 / np.pi * np.arcsin(r / 2), rho, atol=0.05)
    assert_allclose(v, vs, atol=0.05)


