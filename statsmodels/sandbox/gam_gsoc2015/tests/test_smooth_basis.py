
import numpy as np
from patsy.state import stateful_transform
from statsmodels.sandbox.gam_gsoc2015.smooth_basis import (make_poly_basis,
                                                           make_bsplines_basis, BS)


def test_make_basis():
    bs = stateful_transform(BS)
    df = 10
    degree = 4
    x = np.logspace(-1, 1, 100)
    result = bs(x, df=df, degree=degree, include_intercept=True)
    basis, der1, der2 = result   
    basis_old, der1_old, der2_old = make_bsplines_basis(x, df, degree)
    assert((basis == basis_old).all())
    assert((der1 == der1_old).all())
    assert((der2 == der2_old).all())
    return 

# test_make_basis()
