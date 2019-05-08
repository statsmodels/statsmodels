# -*- coding: utf-8 -*-
"""
Tests for Multinomial Models, specifically MNLogit
"""
import numpy as np
from numpy.testing import assert_allclose

from statsmodels.tools.numdiff import approx_fprime
from statsmodels.discrete.discrete_model import MNLogit


def test_pdf_equiv():
    # Check that two implementations of MNLogit.pdf agree
    nobs = 10**4
    J = 8
    k_exog = 10
    np.random.seed(8)
    exog = np.random.randn(nobs, k_exog)
    endog = np.random.randint(0, J, size=nobs)
    # wendog = pd.get_dummies(endog).values

    model = MNLogit(endog, exog)
    params = np.random.random((model.K * (model.J - 1)))
    params = params.reshape(model.K, -1, order='F')

    Xb = model.exog.dot(params)
    pdf1 = model.pdf(Xb)
    pdf2 = mnlogit_pdf(Xb)

    assert_allclose(pdf1, pdf2)

    # Check that differentiating cdf matches pdf
    pdf1 = np.zeros((J - 1, J - 1))
    for k in range(J - 1):
        func = lambda x: model.cdf(x.reshape(1, -1))[0][k]  # noqa:E731
        pdf1[k, :] = approx_fprime(Xb[0], func)

    pdf2 = model.pdf(Xb[:1])[0]
    # Differentiating model.cdf ignores the "base" column of wendog; i.e.
    # computes (J-1)x(J-1) partials.  model.pdf computes JxJ partials, but
    # the non-overlapping ones should be redundant because the partials
    # should sum to zero over both columns and rows.
    assert_allclose(pdf2.sum(0), 0, atol=1e-14)
    assert_allclose(pdf2.sum(1), 0, atol=1e-14)
    # We can therefore drop the inconveniently mismatched row and column
    pdf3 = pdf2[1:, :-1].T
    # TODO: clarify which axis means what
    assert_allclose(pdf3, pdf1, atol=5e-8)
    # Note: at tolerance of 5e-8, 10/1000 tests fail when running this
    # in a loop (and not setting np.random.seed)


# Used as an alternative implementation to double-check MNLogit.pdf
def mnlogit_pdf(Xb):
    r"""
    A broadcasting-based implementation of MNLogit.pdf that is surprisingly
    slower than the loop-based version.  This is retained for testing
    and exposition.

    nobs = 10**5
    exog = np.random.randn(nobs, 10)
    endog = np.random.randint(0, 8, size=nobs)
    model = MNLogit(endog, exog)
    #wendog = pd.get_dummies(endog).values

    We take a derivative of `cdf` using the quotient
    rule: (f'g - g'f) / g^2
    Here "g" is `denom` and "f" is `eXB`
    For each row, this derivative will be a nvars x nvars array, with
    the first dimension representing the coordinate of `cdf` being
    differentiated and the second dimension representing the variable
    doing the differentiation [AWK: how to phrase the last sentence?]

    i.e. row[:, i, j] == \frac{\partial cdf[:, i]}{\partial X[:, j]}
    """
    nobs = len(Xb)
    XB = np.column_stack((np.zeros(nobs), Xb))
    eXB = np.exp(XB)
    denom = eXB.sum(1)[:, None]
    # cdf = eXB / denom

    nvars = eXB.shape[1]  # equiv: self.J

    # Note: `broadcast_to` doesn't make copies.
    denom_deriv = np.broadcast_to(eXB[:, :, None], (nobs, nvars, nvars))

    # Notes from alternative (but more explicit) ways of doing this
    # calculation:
    #
    # numer_square = np.transpose(denom_deriv, (0, 2, 1))
    #              = np.broadcast_to(eXB[:, None, :], (nobs, nvars, nvars))
    # numer_deriv = numer_square * np.eye(nvars)
    #
    # Optional double-check:
    # We should have numer_deriv[n, m, m] == eXB[n, m]
    # and numer_deriv[n, j, k] == 0 for j != k.
    # ncheck = numer_deriv.diagonal(offset=0, axis1=1, axis2=2)
    # assert (ncheck == eXB).all()

    denom3d = denom[:, :, None]

    ddd3 = denom_deriv / denom3d
    # Non-trivial savings if we drop the first row/column and use
    # denom_deriv[:, 1:, 1:]
    nsd3 = np.transpose(ddd3, (0, 2, 1))

    deriv = nsd3 * (np.eye(nvars) - ddd3)

    # Equiv 1:
    #    deriv = (numer_deriv*denom3d - denom_deriv*numer_square) / denom3d**2
    # Equiv 2:
    #   deriv = numer_deriv/denom3d - denom_deriv * numer_square/denom3d**2
    # Equiv 3:
    #   deriv = nsd3 * (np.eye(nvars) - denom_deriv/denom3d)

    # Optional double-check: deriv.sum(1) should be zero, otherwise
    # we are not "conserving" probability.
    # dcheck = deriv.sum(1)
    # assert_allclose(dcheck, np.zeros(dcheck.shape),
    #                 rtol=1e-7, atol=1e-12)
    return deriv
