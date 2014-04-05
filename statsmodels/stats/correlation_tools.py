# -*- coding: utf-8 -*-
"""

Created on Fri Aug 17 13:10:52 2012

Author: Josef Perktold
License: BSD-3
"""

from statsmodels.tools.sm_exceptions import (IterationLimitWarning,
    iteration_limit_doc)
import numpy as np
import scipy.sparse as sparse
from scipy.sparse.linalg import svds
from scipy.optimize import fminbound

def clip_evals(x, value=0): #threshold=0, value=0):
    evals, evecs = np.linalg.eigh(x)
    clipped = np.any(evals < 0)
    x_new = np.dot(evecs * np.maximum(evals, value), evecs.T)
    return x_new, clipped


def corr_nearest(corr, threshold=1e-15, n_fact=100):
    '''
    Find the nearest correlation matrix that is positive semi-definite.

    The function iteratively adjust the correlation matrix by clipping the
    eigenvalues of a difference matrix. The diagonal elements are set to one.

    Parameters
    ----------
    corr : ndarray, (k, k)
        initial correlation matrix
    threshold : float
        clipping threshold for smallest eigenvalue, see Notes
    n_fact : int or float
        factor to determine the maximum number of iterations. The maximum
        number of iterations is the integer part of the number of columns in
        the correlation matrix times n_fact.

    Returns
    -------
    corr_new : ndarray, (optional)
        corrected correlation matrix

    Notes
    -----
    The smallest eigenvalue of the corrected correlation matrix is
    approximately equal to the ``threshold``.
    If the threshold=0, then the smallest eigenvalue of the correlation matrix
    might be negative, but zero within a numerical error, for example in the
    range of -1e-16.

    Assumes input correlation matrix is symmetric.

    Stops after the first step if correlation matrix is already positive
    semi-definite or positive definite, so that smallest eigenvalue is above
    threshold. In this case, the returned array is not the original, but
    is equal to it within numerical precision.

    See Also
    --------
    corr_clipped
    cov_nearest

    '''
    k_vars = corr.shape[0]
    if k_vars != corr.shape[1]:
        raise ValueError("matrix is not square")

    diff = np.zeros(corr.shape)
    x_new = corr.copy()
    diag_idx = np.arange(k_vars)

    for ii in range(int(len(corr) * n_fact)):
        x_adj = x_new - diff
        x_psd, clipped = clip_evals(x_adj, value=threshold)
        if not clipped:
            x_new = x_psd
            break
        diff = x_psd - x_adj
        x_new = x_psd.copy()
        x_new[diag_idx, diag_idx] = 1
    else:
        import warnings
        warnings.warn(iteration_limit_doc, IterationLimitWarning)

    return x_new

def corr_clipped(corr, threshold=1e-15):
    '''
    Find a near correlation matrix that is positive semi-definite

    This function clips the eigenvalues, replacing eigenvalues smaller than
    the threshold by the threshold. The new matrix is normalized, so that the
    diagonal elements are one.
    Compared to corr_nearest, the distance between the original correlation
    matrix and the positive definite correlation matrix is larger, however,
    it is much faster since it only computes eigenvalues once.

    Parameters
    ----------
    corr : ndarray, (k, k)
        initial correlation matrix
    threshold : float
        clipping threshold for smallest eigenvalue, see Notes

    Returns
    -------
    corr_new : ndarray, (optional)
        corrected correlation matrix


    Notes
    -----
    The smallest eigenvalue of the corrected correlation matrix is
    approximately equal to the ``threshold``. In examples, the
    smallest eigenvalue can be by a factor of 10 smaller than the threshold,
    e.g. threshold 1e-8 can result in smallest eigenvalue in the range
    between 1e-9 and 1e-8.
    If the threshold=0, then the smallest eigenvalue of the correlation matrix
    might be negative, but zero within a numerical error, for example in the
    range of -1e-16.

    Assumes input correlation matrix is symmetric. The diagonal elements of
    returned correlation matrix is set to ones.

    If the correlation matrix is already positive semi-definite given the
    threshold, then the original correlation matrix is returned.

    ``cov_clipped`` is 40 or more times faster than ``cov_nearest`` in simple
    example, but has a slightly larger approximation error.

    See Also
    --------
    corr_nearest
    cov_nearest

    '''
    x_new, clipped = clip_evals(corr, value=threshold)
    if not clipped:
        return corr

    #cov2corr
    x_std = np.sqrt(np.diag(x_new))
    x_new = x_new / x_std / x_std[:,None]
    return x_new


def cov_nearest(cov, method='clipped', threshold=1e-15, n_fact=100,
                return_all=False):

    '''
    Find the nearest covariance matrix that is postive (semi-) definite

    This leaves the diagonal, i.e. the variance, unchanged

    Parameters
    ----------
    cov : ndarray, (k,k)
        initial covariance matrix
    method : string
        if "clipped", then the faster but less accurate ``corr_clipped`` is used.
        if "nearest", then ``corr_nearest`` is used
    threshold : float
        clipping threshold for smallest eigen value, see Notes
    nfact : int or float
        factor to determine the maximum number of iterations in
        ``corr_nearest``. See its doc string
    return_all : bool
        if False (default), then only the covariance matrix is returned.
        If True, then correlation matrix and standard deviation are
        additionally returned.

    Returns
    -------
    cov_ : ndarray
        corrected covariance matrix
    corr_ : ndarray, (optional)
        corrected correlation matrix
    std_ : ndarray, (optional)
        standard deviation


    Notes
    -----
    This converts the covariance matrix to a correlation matrix. Then, finds
    the nearest correlation matrix that is positive semidefinite and converts
    it back to a covariance matrix using the initial standard deviation.

    The smallest eigenvalue of the intermediate correlation matrix is
    approximately equal to the ``threshold``.
    If the threshold=0, then the smallest eigenvalue of the correlation matrix
    might be negative, but zero within a numerical error, for example in the
    range of -1e-16.

    Assumes input covariance matrix is symmetric.

    See Also
    --------
    corr_nearest
    corr_clipped

    '''

    from statsmodels.stats.moment_helpers import cov2corr, corr2cov
    cov_, std_ = cov2corr(cov, return_std=True)
    if method == 'clipped':
        corr_ = corr_clipped(cov_, threshold=threshold)
    elif method == 'nearest':
        corr_ = corr_nearest(cov_, threshold=threshold, n_fact=n_fact)

    cov_ = corr2cov(corr_, std_)

    if return_all:
        return cov_, corr_, std_
    else:
        return cov_

def nmono_linesearch(obj, grad, x, d, obj_hist, M=10, sig1=0.1, sig2=0.9,
                     gam=1e-4, maxit=100):
    """
    Implements the non-monotone line search of Grippo et al. (1986),
    as described in Birgin, Martinez and Raydan (2013).

    Parameters
    ----------
    obj : real-valued function
        The objective function, to be minimized
    grad : vector-valued function
        The gradient of the objective function
    x : array_like
        The starting point for the line search
    d : array_like
        The search direction
    obj_hist : array_like
        Objective function history (must contain at least one value)
    M : positive integer
        Number of previous function points to consider (see references
        for details).
    sig1 : real
        Tuning parameter, see references for details.
    sig2 : real
        Tuning parameter, see references for details.
    gam : real
        Tuning parameter, see references for details.
    maxit : positive integer
        The maximum number of iterations; returns Nones if convergence
        does not occur by this point

    Returns
    -------
    alpha : real
        The step value
    x : Array_like
        The function argument at the final step
    obval : Real
        The function value at the final step
    g : Array_like
        The gradient at the final step

    Notes
    -----
    The basic idea is to take a big step in the direction of the
    gradient, even if the function value is not decreased (but there
    is a maximum allowed increase in terms of the recent history of
    the iterates).

    References
    ----------
    Grippo L, Lampariello F, Lucidi S (1986). A Nonmonotone Line
    Search Technique for Newton's Method. SIAM Journal on Numerical
    Analysis, 23, 707-716.

    E. Birgin, J.M. Martinez, and M. Raydan. Spectral projected
    gradient methods: Review and perspectives. Journal of Statistical
    Software (preprint).
    """

    alpha = 1.
    last_obval = obj(x)
    obj_max = max(obj_hist[-M:])

    for iter in range(maxit):

        obval = obj(x + alpha*d)
        g = grad(x)
        gtd = (g * d).sum()

        if obval <= obj_max + gam*alpha*gtd:
            return alpha, x + alpha*d, obval, g

        a1 = -0.5*alpha**2*gtd / (obval - last_obval - alpha*gtd)

        if (sig1 <= a1) and (a1 <= sig2*alpha):
            alpha = a1
        else:
            alpha /= 2.

        last_obval = obval

    return None, None, None, None


def spg_optim(func, grad, start, project, maxit=1e4, M=10, ctol=1e-3,
              maxit_nmls=200, lam_min=1e-30, lam_max=1e30, sig1=0.1,
              sig2=0.9, gam=1e-4):
    """
    Implements the spectral projected gradient method for minimizing a
    differentiable function on a convex domain.

    Parameters
    ----------
    func : real valued function
        The objective function to be minimized.
    grad : real array-valued function
        The gradient of the objective function
    start : array_like
        The starting point
    project : array_like
        In-place projection of the argument to the domain
        of func.
    ... See notes regarding additional arguments

    Returns
    -------
    rslt : dict
        rslt['X'] is the final iterate, other fields describe
        convergence status.

    Notes
    -----
    This can be an effective heuristic algorithm for problems where no
    gauranteed algorithm for computing a global minimizer is known.

    There are a number of tuning parameters, but these generally
    should not be changed except for maxit (positive integer) and
    ctol (small positive real).  See the Birgin et al reference for
    more information about the tuning parameters.

    Reference
    ---------
    E. Birgin, J.M. Martinez, and M. Raydan. Spectral projected
    gradient methods: Review and perspectives. Journal of Statistical
    Software (preprint).  Available at:
    http://www.ime.usp.br/~egbirgin/publications/bmr5.pdf
    """

    lam = min(10*lam_min, lam_max)

    X = start
    gval = grad(X)

    obj_hist = [func(X),]

    for itr in range(int(maxit)):

        # Check convergence
        df = X - gval
        project(df)
        df -= X
        if np.max(np.abs(df)) < ctol:
            return {"Converged": True, "X": X, "Message": "Converged successfully"}

        # The line search direction
        d = X - lam*gval
        project(d)
        d -= X

        # Carry out the nonmonotone line search
        alpha, X1, fval, gval1 = nmono_linesearch(func, grad, X, d,
                                                  obj_hist, M=M,
                                                  sig1=sig1,
                                                  sig2=sig2,
                                                  gam=gam,
                                                  maxit=maxit_nmls)
        if alpha is None:
            return {"Converged": False, "X": X, "Message": "Failed in nmono_linesearch"}

        obj_hist.append(fval)
        s = X1 - X
        y = gval1 - gval

        sy = (s*y).sum()
        if sy <= 0:
            lam = lam_max
        else:
            ss = (s*s).sum()
            lam = max(lam_min, min(ss/sy, lam_max))

        X = X1
        gval = gval1

    return {"Converged": False, "X": X, "Message": "spg_optim did not converge"}


def _project_correlation_factors(X):
    """
    Project a matrix into the domain of matrices whose row-wise sums
    of squares are less than or equal to 1.

    The input matrix is modified in-place.
    """
    nm = np.sqrt((X*X).sum(1))
    ii = np.flatnonzero(nm > 1)
    if len(ii) > 0:
        X[ii,:] /= nm[ii][:,None]


def corr_nearest_factor(mat, rank, ctol=1e-6, lam_min=1e-30,
                        lam_max=1e30, maxit=1000):
    """
    Attempts to find the nearest correlation matrix with factor
    structure to a given square matrix, see notes below for details.

    Parameters
    ----------
    mat : square array
        The target matrix (to which the nearest correlation matrix is
        sought).
    rank : positive integer
        The rank of the factor structure of the solution, i.e., the
        number of linearly independent columns of X.
    ctol : positive real
        Convergence criterion.

    Returns
    -------
    rslt : dict
        rslt["corr"] is the matrix X defining the estimated low rank
        structure.  To obtain the fitted correlation matrix use
        C = np.dot(X, X.T); np.fill_diagonal(C, 1).  rslt also has
        fields describing how the optimization terminated

    Example
    -------
    Hard thresholding a correlation matrix may result in a matrix that
    is not positive semidefinite.  We can approximate a hard
    thresholded correlation matrix with a PSD matrix as follows, where
    `cmat` is the input correlation matrix.

    >>> cmat = cmat * (np.abs(cmat) >= 0.3)
    >>> rslt = corr_nearest_factor(cmat, 3)

    Notes
    -----
    A correlation matrix has factor structure if it can be written in
    the form I + XX' - diag(XX'), where X is n x k with linearly
    independent columns, and with each row having sum of squares at
    most equal to 1.  The approximation is made in terms of the
    Frobenius norm.

    This routine is useful when one has an approximate correlation
    matrix that is not SPD, and there is need to estimate the inverse,
    square root, or inverse square root of the population correlation
    matrix.  The factor structure allows these tasks to be done
    without constructing any n x n matrices.

    This is a non-convex problem with no known guaranteed globally
    convergent algorithm for computing the solution.  Borsdof, Higham
    and Raydan (2010) compared several methods for this problem and
    found the spectral projected gradient (SPG) method (used here) to
    perform best.

    The input matrix `mat` can be a dense numpy array or any scipy
    sparse matrix.  The latter is useful if the input matrix is
    obtained by thresholding a very large sample correlation matrix.
    If `mat` is sparse, the calculations are optimized to save memory,
    so no working matrix with more than 10^6 elements is constructed.

    References
    ----------
    R Borsdof, N Higham, M Raydan (2010).  Computing a nearest
    correlation matrix with factor structure. SIAM J Matrix Anal
    Appl, 31:5, 2603-2622.
    http://eprints.ma.man.ac.uk/1523/01/covered/MIMS_ep2009_87.pdf
    """

    p,_ = mat.shape

    # Starting values (following the PCA method in BHR).
    u,s,vt = svds(mat, rank)
    X = u * np.sqrt(s)
    nm = np.sqrt((X**2).sum(1))
    ii = np.flatnonzero(nm > 1e-5)
    X[ii,:] /= nm[ii][:,None]

    # Zero the diagonal
    mat1 = mat.copy()
    if type(mat1) == np.ndarray:
        np.fill_diagonal(mat1, 0)
    elif sparse.issparse(mat1):
        mat1.setdiag(np.zeros(mat1.shape[0]))
        mat1.eliminate_zeros()
        mat1.sort_indices()
    else:
        raise ValueError("Matrix type not supported")

    # The gradient, from lemma 4.1 of BHR.
    def grad(X):
        gr = np.dot(X, np.dot(X.T, X))
        if type(mat1) == np.ndarray:
            gr -= np.dot(mat1, X)
        else:
            gr -= mat1.dot(X)
        gr -= (X*X).sum(1)[:,None] * X
        return 4*gr

    # The objective function (sum of squared deviations between fitted
    # and observed arrays).
    def func(X):
        if type(mat1) == np.ndarray:
            M = np.dot(X, X.T)
            np.fill_diagonal(M, 0)
            M -= mat1
            fval = (M*M).sum()
            return fval
        else:
            fval = 0.
            # Control the size of intermediates
            max_ws = 1e6
            bs = int(max_ws / X.shape[0])
            ir = 0
            while ir < X.shape[0]:
                ir2 = min(ir+bs, X.shape[0])
                u = np.dot(X[ir:ir2,:], X.T)
                ii = np.arange(u.shape[0])
                u[ii,ir+ii] = 0
                u -= np.asarray(mat1[ir:ir2,:].todense())
                fval += (u*u).sum()
                ir += bs
            return fval

    rslt = spg_optim(func, grad, X, _project_correlation_factors)
    rslt["corr"] = rslt["X"]
    del rslt["X"]
    return rslt


def cov_nearest_eye_factor(mat, rank):
    """
    Approximate a matrix with a factor-structured matrix of the form
    k*I + XX'.

    Parameters
    ----------
    mat : array-like
        The input array, must be square
    rank : positive integer
        The rank of the fitted factor structure

    Returns
    -------
    k : positive real
        The value of k in the fitted structure k*I + XX'
    X : array_like
        The value of X in the fitted structure k*I + XX'

    Notes
    -----
    This routine is useful if one has an estimated correlation matrix
    that is not SPD, and the ultimate goal is to estimate the inverse,
    square root, or inverse square root of the true correlation
    matrix. The factor structure allows these tasks to be performed
    without constructing any n x n matrices.

    The calculations use the fact that if k is known, then X can be
    determined from the eigen-decomposition of mat - k*I, which can in
    turn be easily obtained form the eigen-decomposition of mat.  Thus
    the problem can be reduced to a 1-dimensional search for k that
    does not require repeated eigen-decompositions.

    If the input matrix is sparse, then mat - k*I is also sparse, so
    the eigen-decomposition can be done effciciently using sparse
    routines.

    The one-dimensional search for the optimal value of k is not
    convex, so a local minimum could be obtained.

    Example
    -------
    Hard thresholding a covariance matrix may result in a matrix that
    is not positive semidefinite.  We can approximate a hard
    thresholded covariance matrix with a PSD matrix as follows:

    >>> cmat = cmat * (np.abs(cmat) >= 0.3)
    >>> rslt = cov_nearest_eye_factor(cmat, 3)
    """

    m,n = mat.shape

    Q,Lambda,_ = svds(mat, rank)

    if sparse.issparse(mat):
        QSQ = np.dot(Q.T, mat.dot(Q))
        ts = mat.diagonal().sum()
        tss = mat.dot(mat).diagonal().sum()
    else:
        QSQ = np.dot(Q.T, np.dot(mat, Q))
        ts = np.trace(mat)
        tss = np.trace(np.dot(mat, mat))

    def fun(k):
        Lambda_t = Lambda - k
        v = tss + m*(k**2) + np.sum(Lambda_t**2) - 2*k*ts
        v += 2*k*np.sum(Lambda_t) - 2*np.sum(np.diag(QSQ) * Lambda_t)
        return v

    # Get the optimal decomposition
    k_opt = fminbound(fun, 0, 1e5)
    Lambda_opt = Lambda - k_opt
    fac_opt = Q * np.sqrt(Lambda_opt)

    return k_opt, fac_opt

def corr_thresholded(mat, minabs, max_elt=1e7):
    """
    Construct a sparse matrix containing the thresholded row-wise
    correlation matrix of `mat`.

    Parameters
    ----------
    mat : array_like
        The data from which the row-wise thresholded correlation
        matrix is to be computed.
    minabs : non-negative real
        The threshold value; correlation coefficients smaller in
        magnitude than minabs are set to zero.

    Returns
    -------
    cormat : sparse.coo_matrix
        The thresholded correlation matrix, in COO format.

    Notes
    -----
    This is an alternative to C = np.corrcoef(mat); C *= (np.abs(C) >=
    absmin), suitable for very tall data matrices.

    No intermediate matrix with more than `max_elt` values will be
    constructed.  However memory use could still be high if a large
    number of correlation values exceed `minabs` in magnitude.

    The thresholded matrix is returned in COO format, which can easily
    be converted to other sparse formats.

    Example
    -------
    Here X is a tall data matrix (e.g. with 100,000 rows and 50
    columns).  The row-wise correlation matrix of X is calculated
    and stored in sparse form, with all entries smaller than 0.3
    treated as 0.

    >>> cmat = corr_thresholded(X, 0.3)
    """

    n,ncol = mat.shape

    # Row-standardize the data
    mat = mat.copy()
    mat -= mat.mean(1)[:,None]
    sd = mat.std(1, ddof=1)
    ii = np.flatnonzero(sd > 1e-5)
    mat[ii,:] /= sd[ii][:,None]
    ii = np.flatnonzero(sd <= 1e-5)
    mat[ii,:] = 0

    # Number of rows to process in one pass
    bs = int(np.floor(max_elt/n))

    ipos_all, jpos_all, data = [], [], []

    ir = 0
    while ir < n:
        ir2 = min(mat.shape[0], ir + bs)
        cm = np.dot(mat[ir:ir2,:], mat.T) / (ncol - 1)
        cma = np.abs(cm)
        ipos, jpos = np.nonzero(cma >= minabs)
        ipos_all.append(ipos + ir)
        jpos_all.append(jpos)
        data.append(cm[ipos, jpos])
        ir += bs

    ipos = np.concatenate(ipos_all)
    jpos = np.concatenate(jpos_all)
    data = np.concatenate(data)

    cmat = sparse.coo_matrix((data, (ipos, jpos)), (n,n))

    return cmat



if __name__ == '__main__':
    pass
