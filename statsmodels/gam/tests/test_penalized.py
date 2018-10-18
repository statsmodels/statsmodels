
# coding: utf-8


import os

import numpy as np
from numpy.testing import assert_allclose
#import matplotlib.pyplot as plt
import pandas as pd

import patsy
import patsy.splines as bspl
import patsy.mgcv_cubic_splines as cspl

from statsmodels.regression.linear_model import OLS
from statsmodels.discrete.discrete_model import Poisson, Logit, Probit
from statsmodels.genmod.generalized_linear_model import GLM
from statsmodels.genmod.families import family
from statsmodels.sandbox.regression.penalized import TheilGLS
from statsmodels.base._penalized import PenalizedMixin
import statsmodels.base._penalties as smpen

from statsmodels.gam.smooth_basis import BSplines, CubicSplines, CyclicCubicSplines
from statsmodels.gam.gam import GLMGam, get_sqrt

from .results import results_pls

# temporary location
def matrix_sqrt(mat, inverse=False, full=False, nullspace=False, threshold=1e-15):
    """matrix square root for symmetric matrices

    Usage is for decomposing a covariance function S into a square root R
    such that

        R' R = S if inverse is False, or
        R' R = pinv(S) if inverse is True

    Parameters
    ----------
    mat : array_like, 2-d square
        symmetric square matrix for which square root or inverse square
        root is computed.
        There is no checking for whether the matrix is symmetric.
        A warning is issued if some singular values are negative, i.e.
        below the negative of the threshold.
    inverse : bool
        If False (default), then the matrix square root is returned.
        If inverse is True, then the matrix square root of the inverse
        matrix is returned.
    full : bool
        If full is False (default, then the square root has reduce number
        of rows if the matrix is singular, i.e. has singular values below
        the threshold.
    nullspace: bool
        If nullspace is true, then the matrix square root of the null space
        of the matrix is returned.
    threshold : float
        Singular values below the threshold are dropped.

    Returns
    -------
    msqrt : ndarray
        matrix square root or square root of inverse matrix.

    """
    # see also scipy.linalg null_space
    u, s, v = np.linalg.svd(mat)
    if np.any(s < -threshold):
        import warnings
        warnings.warn('some singular values are negative')

    if not nullspace:
        mask = s > threshold
        s[s < threshold] = 0
    else:
        mask = s < threshold
        s[s > threshold] = 0

    sqrt_s = np.sqrt(s[mask])
    if inverse:
        sqrt_s = 1 / np.sqrt(s[mask])

    if full:
        b = np.dot(u[:, mask], np.dot(np.diag(sqrt_s), v[mask]))
    else:
        b = np.dot(np.diag(sqrt_s), v[mask])
    return b



class PoissonPenalized(PenalizedMixin, Poisson):
    pass


class LogitPenalized(PenalizedMixin, Logit):
    pass


class ProbitPenalized(PenalizedMixin, Probit):
    pass


class GLMPenalized(PenalizedMixin, GLM):
    pass


cur_dir = os.path.dirname(os.path.abspath(__file__))

file_path = os.path.join(cur_dir, "results", "motorcycle.csv")
data_mcycle = pd.read_csv(file_path)


# In[12]:

from scipy import linalg

def transf_constraints(constraints):
    # constraints = spsc._constraints
    m = constraints.shape[0]
    q, r = linalg.qr(np.transpose(constraints))
    transf = q[:, m:]
    return transf



class CheckGAMMixin(object):

    @classmethod
    def _init(cls):
        # TODO: CyclicCubicSplines raises when using pandas
        cc_h = CyclicCubicSplines(np.asarray(data_mcycle['times']), df=[6])

        constraints = np.atleast_2d(cc_h.basis_.mean(0))
        transf = transf_constraints(constraints)

        exog = cc_h.basis_.dot(transf)
        penalty_matrix = transf.T.dot(cc_h.penalty_matrices_[0]).dot(transf)
        restriction = matrix_sqrt(penalty_matrix)
        return exog, penalty_matrix, restriction


    def test_params(self):
        res1 = self.res1
        res2 = self.res2
        assert_allclose(res1.params, res2.params, rtol=1e-5)
        assert_allclose(np.asarray(res1.cov_params()),
                        res2.Vp * self.covp_corrfact, rtol=1e-4)

    def test_fitted(self):
        res1 = self.res1
        res2 = self.res2
        assert_allclose(res1.fittedvalues, res2.fitted_values,
                        rtol=self.rtol_fitted)



class TestTheilPLS5(CheckGAMMixin):

    cov_type = 'data-prior'

    @classmethod
    def setup_class(cls):
        exog, penalty_matrix, restriction = cls._init()
        endog = data_mcycle['accel']
        modp = TheilGLS(endog, exog, r_matrix=restriction)
        # scaling of penweith in R mgcv
        s_scale_r = 0.02630734
        # Theil penweight uses preliminary sigma2_e to scale penweight
        sigma_e = 1405.7950179165323
        cls.pw = pw = 1 / sigma_e / s_scale_r
        cls.res1 = modp.fit(pen_weight=pw, cov_type=cls.cov_type)
        cls.res2 = results_pls.pls5

        cls.rtol_fitted = 1e-7
        cls.covp_corrfact = 0.99786932844203202


    def test_cov_robust(self):
        res1 = self.res1
        res2 = self.res2
        pw = res1.penalization_factor
        res1 = res1.model.fit(pen_weight=pw, cov_type='sandwich')
        assert_allclose(np.asarray(res1.cov_params()),
                        res2.Ve * self.covp_corrfact, rtol=1e-4)


class TestGLMPenalizedPLS5(CheckGAMMixin):

    cov_type = 'nonrobust'

    @classmethod
    def setup_class(cls):
        exog, penalty_matrix, restriction = cls._init()
        endog = data_mcycle['accel']
        pen = smpen.L2ContraintsPenalty(restriction=restriction)
        mod = GLMPenalized(endog, exog, family=family.Gaussian(),
                           penal=pen)
        # scaling of penweith in R mgcv
        s_scale_r = 0.02630734
        # set pen_weight to correspond to R mgcv example
        cls.pw = mod.pen_weight = 1 / s_scale_r / 2
        cls.res1 = mod.fit(cov_type=cls.cov_type, method='bfgs', maxiter=100,
                           disp=0, trim=False, scale='x2')
        cls.res2 = results_pls.pls5

        cls.rtol_fitted = 1e-5
        cls.covp_corrfact = 1.0025464444310588

    def _test_cov_robust(self):
        # TODO: HC0 differs from Theil sandwich, difference is large
        res1 = self.res1
        res2 = self.res2
        pw = res1.model.pen_weight
        res1 = res1.model.fit(pen_weight=pw, cov_type='HC0')
        assert_allclose(np.asarray(res1.cov_params()),
                        res2.Ve * self.covp_corrfact, rtol=1e-4)
