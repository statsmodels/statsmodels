"""
Created on Apr. 12, 2024 12:50:27 p.m.

Author: Josef Perktold
License: BSD-3
"""

import numpy as np
from scipy import linalg as splinalg


def corrdist(x1, x2):
    """Correlation coefficient without subtracting mean.

    Parameters
    ----------
    x1, x2 : ndarray
        Two one dimensional arrays.


    References
    ----------
    Herdin, M., N. Czink, H. Ozcelik, and E. Bonek. “Correlation Matrix Distance,
    a Meaningful Measure for Evaluation of Non-Stationary MIMO Channels.”
    In 2005 IEEE 61st Vehicular Technology Conference, 1:136-140 Vol. 1, 2005.
    https://doi.org/10.1109/VETECS.2005.1543265.

    """
    if x1.ndim !=1 or x2.ndim !=1:
        raise ValueError("data should be 1-dimensional")
    cross = x1.T @ x2
    s1 = (x1**2).sum(0)
    s2 = (x1**2).sum(0)
    cmd = 1 - cross / np.sqrt(s1 * s2)
    return cmd


def cov_distance(cov1, cov2, method="kl", compare_scatter=False):
    """Distance and divergence measures between two covariance matrices

    Notes
    -----
    Some measures require additional restrictions on the
    covariance matrix, e.g. symmetry, full rank, positive definite.
    Those restrictions are currently not checked and imposed.

    Some measures are not proper distance measures and violate
    properties such as symmetry d(c1, c2) == d(c2, c1).
    For some of those measures a symmetrized method is additionally
    available.

    Distance equal to zero means that the two matrices are equal or
    within the same equivalence class, for example they could be identical
    up to an arbitrary scaling factor as in scatter matrices,
    i.e. cov1 = k cov2 for some k>0.

    References
    ----------

    Cherian, Anoop, Suvrit Sra, Arindam Banerjee, and Nikolaos
        Papanikolopoulos. 2011. “Efficient Similarity Search for Covariance
        Matrices via the Jensen-Bregman LogDet Divergence.”
        In 2011 International Conference on Computer Vision, 2399–2406.
        https://doi.org/10.1109/ICCV.2011.6126523.

    ———. 2013. “Jensen-Bregman LogDet Divergence with Application to
        Efficient Similarity Search for Covariance Matrices.” IEEE Transactions
        on Pattern Analysis and Machine Intelligence 35 (9): 2161–74.
        https://doi.org/10.1109/TPAMI.2012.259.


    """

    cov1 = np.asarray(cov1)
    cov2 = np.asarray(cov2)
    if cov1.shape != cov2.shape:
        raise ValueError("Matrices cov1 and cov2 do not have the same shape.")

    k = cov1.shape[1]


    if compare_scatter:
        # normalize
        cov1 = cov1 / np.linalg.det(cov1)
        cov2 = cov2 / np.linalg.det(cov2)

    if method == "kl":
        dist = 0.5 * (np.trace(np.linalg.solve(cov1, cov2)) +
                      np.linalg.logdet(cov1) - np.linalg.logdet(cov2)) - k
    elif method == "kl-sym":
        dist = 0.5 * (np.trace(np.linalg.solve(cov1, cov2)) +
                      np.trace(np.linalg.solve(cov2, cov1))) - k
    elif method == "corrd":
        dist = corrdist(cov1.ravel(), cov2.ravel())
    elif method in ["Frobenius", "square"]:
        dist = splinalg.norm(cov1 - cov2)
    elif method == "relevals-trace":
        dist = np.trace(np.linalg.solve(cov1, cov2)) - k
    elif method == "relevals-logdet":
        dist = np.linalg.logdet(np.linalg.solve(cov1, cov2))
    elif method == "relevals-range":
        ev = np.linalg.evals(np.linalg.solve(cov1, cov2))
        dist = np.ptp(ev)
    elif method == "jb-logdet":
        # Jensen-Bregman LogDet Divergence, Cherian et al. 2013
        dist = (np.linalg.logdet((cov1 + cov2) / 2) -
                np.linalg.logdet(cov1 @ cov2)
                )
    else:
        raise ValueError("method not recognized")

    return dist
