"""
Statistics for Complex Random Variables

This is currently mainly to collect some tools as reference.

Created on Nov. 28, 2023 11:04:50 a.m.

Author: Josef Perktold
License: BSD-3
"""


import numpy as np

from scipy import linalg as spla
from scipy import stats

from statsmodels.stats.base import HolderTuple


def transform_matrix_rvec2ext(k, inverse=False):
    """transformation matrix between bivariate real and extended complex

    The transformation matrix maps from real to complex space.

    T: R^{2k} -> C^{2k}

    If `z = x + j y`, then

    [z, z.conj] = T @ [x, y], where [z, z.conj] and [x, y] are column vectors.
    or `[x, y] @ T.T in case of row vectors (or 1-dim vectors)


    Notes
    -----
    The transformation is not a unitary matrix, but it is a double
    unitary matrix, i.e. $T^{-1} = 2 T.H$
    References differ in whether this transformation matrix or the inverse
    transformation is used in expressions and equations.
    This means that care has to be taken for whether terms have to be
    multiplied or divided by 2 if T is also used for it's inverse.

    """
    eyek = np.eye(k)
    eyekj = np.eye(k) * 1j
    t = np.block([[eyek, eyekj], [eyek, -eyekj]])
    if not inverse:
        return t
    else:
        return t.conj().T / 2


def cov_rvec(covz, pcovz):
    """covariance of real and imaginary part from cov and pcov
    """
    # covariance of real-real and imag-imag parts
    c11 = (covz + pcovz).real
    c22 = (covz - pcovz).real

    # covariance between real and imag parts
    c12 = (-covz + pcovz).imag
    c21 = (covz + pcovz).imag
    return np.block([[c11, c12], [c21, c22]]) / 2


def cov_from_rvec(cov_rvec):
    """cov and pcov from covariance of real and imaginary part
    """
    k = cov_rvec.shape[0] / 2
    if k != int(k):
        raise ValueError("shape of cov_rvec has to be multiple of 2")
    k = int(k)
    cov = cov_rvec  # shorthand
    crr, cri, cir, cii = cov[:k, :k], cov[:k, k:], cov[k:, :k], cov[k:, k:]
    if not np.allclose(cri.T, cir, rtol=1e-13, atol=1e-15):
        # TODO: maybe force symmetry instead of warning
        import warnings
        warnings.warn("cross-covariance not symmetric at tolerance")
    covz = crr + cii + 1j * (cri.T - cri)
    pcovz = crr - cii + 1j * (cri.T + cri)
    return covz, pcovz


def cov_ext(covz, pcovz):
    """covariance of extended complex variables from cov and pcov
    """
    return np.block([[covz, pcovz], [pcovz.conj(), covz.conj()]])


def circularity(covz, pcovz, nobs):
    """Second order circularity statistics and hypothesis test

    This function computes circularity coefficients and the likelihood ratio
    hypothesis test for second order circularity.

    Parameters
    ----------
    covz : ndarray, 2-dim, complex hermitian symmetric
    pcovz : ndarray, 2-dim complex symmetric
    nobs :
        number of pbservations use for estimatin cov and pcov

    Returns
    -------
    Holdertuple with the following main attributes:

    - circ_coef: circularity coefficients
    - llratio, statistic : test statistic for log-likelihood ratio test of
      circularity
    - pvalue : pvalue of LR test using chisquare distribution
    - df : degrees of freedom used in chisquare distribution of LRT
    - statistics_k : LR test for partial circularity
    - pvalues_k : pvalues for LRT for partial circularity
    - df_k : degrees of freedom for LRT for partial circularity

    Notes
    -----
    The likelihood ratio test is computed in two different ways.
    Most likely I will add an option to skip eigenvalue based computation.
    No verifying unit tests yet.

    References
    ----------


    """
    r = np.linalg.solve(covz, pcovz)
    # r_ = np.linalg.inv(covz) @ pcovz  # used for checking complex linalg
    # assert_allclose(r, r_)
    dim = r.shape[0]

    rrc = r @ r.conj()
    circ_coef = np.sort(np.real_if_close(np.sqrt(spla.eigvals(rrc))))
    # Note: we want partial sum of larger values,
    #       i.e. dropping small values in sequence
    statistic_k = - nobs * np.log(np.cumprod(1 - circ_coef[::-1]**2)[::-1])
    k = np.arange(dim)
    df_k = (dim - k) * (dim - k + 1)
    statistic2 = - nobs * np.log(np.prod(1 - circ_coef[::-1]**2))

    pvals_k = stats.chi2.sf(statistic_k, df_k)

    llratio = - nobs * np.linalg.slogdet(np.eye(dim) - rrc)[1]
    df = dim * (dim + 1)
    pval = stats.chi2.sf(llratio, df)
    pval2 = stats.chi2.sf(statistic2, df)


    res = HolderTuple(
        llratio=llratio,
        statistic=llratio,
        pvalue=pval,
        df=df,
        distr="chi-square",
        circ_coef=circ_coef,
        relmat=r,
        statistic_k=statistic_k,
        pvalues_k= pvals_k,
        df_k=df_k,
        statistic2=statistic2,
        pvalue2=pval2,
        )
    return res


class RandVarComplex():
    """Class to collect tools for complex random variables

    This currently only supports numpy arrays, data converted with `asarray`.
    This assumes bservations are in rows and variables in columns.

    Data has three representations:

     - complex random variable z = x + j y, (nobs, k) as input to class.
     - extended complex random variable [z, z*], (nobs, 2 k).
     - combined (bivariate) real [x, y], (nobs, 2 k).

    The underlying definitions for probability theory and statistics are based
    on the combined real representation.


    """

    def __init__(self, data, demean=True):
        data  = np.asarray(data)
        if demean:
            data = data - data.mean()
        self.data_complex = z = data
        self.nobs, self.k = z.shape
        self.data_ext = np.column_stack((z, z.conj()))
        self.data_real = np.column_stack((z.real, z.imag))

    def cov(self, data=None, mean=0, ddof=0):
        """Covariance or mean product matrix
        """
        if data is None:
            data = self.data_complex
            nobs = self.nobs
        else:
            data = np.asarray(data)
            nobs = data.shape[0]

        if mean != 0:
            data = data - mean

        return data.T @ data.conj() / (nobs - ddof)

    def pcov(self, data=None, mean=0, ddof=0):
        """Pseudo covariance or mean product matrix
        """
        if data is None:
            data = self.data_complex
            nobs = self.nobs
        else:
            data = np.asarray(data)
            nobs = data.shape[0]


        if mean != 0:
            data = data - mean

        return data.T @ data / (nobs - ddof)

    def cov_circular(self, data=None, mean=0, ddof=0):
        """Covariance or mean product matrix imposing circularity
        """
        if data is None:
            z_real = self.data_real
        else:
            z = np.asarray(data)
            z_real = np.column_stack((z.real, z.imag))

        if mean != 0:
            z_real = z_real - mean

        nobs, k = z_real.shape
        k = k // 2

        zd = np.vstack((z_real,
                        np.column_stack((- z_real[:, -k:], z_real[:, :k]))
                        ))

        return zd.T @ zd.conj() / 2 / (nobs - ddof)


    def cov_rvec(self, data=None, mean=0, ddof=0):
        """Covariance or mean product matrix of real representation
        """
        if data is None:
            z_real = self.data_real
        else:
            z = np.asarray(data)
            z_real = np.column_stack((z.real, z.imag))

        if mean != 0:
            z_real = z_real - mean

        nobs = z_real.shape[0]

        return z_real.T @ z_real / (nobs - ddof)

    def cov_ext(self, data=None, mean=0, ddof=0):
        """Covariance or mean product of extended complex variable
        """
        if data is None:
            z_ext = self.data_ext
        else:
            z = np.asarray(data)
            z_ext = np.column_stack((z, z.conj()))

        if mean != 0:
            z_ext = z_ext - mean

        nobs = z_ext.shape[0]

        return z_ext.T @ z_ext.conj() / (nobs - ddof)
