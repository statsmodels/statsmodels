import numpy as np
from numpy.testing import assert_array_almost_equal
from scipy import stats

from statsmodels.distributions.joint_distribution import (
    IndependentCopula, GaussianCopula, StudentCopula,
    ClaytonCopula, FrankCopula, GumbelCopula,
    JointDistribution
)


class CopulaTests:
    """Generic tests for copula."""
    copula = None
    dim = None

    def test_visualization(self):
        sample = self.copula.random(10000)
        assert sample.shape == (10000, 2)
        # h = sns.jointplot(sample[:, 0], sample[:, 1], kind='hex')
        # h.set_axis_labels('X1', 'X2', fontsize=16)

    def test_uniform_marginals(self):
        sample = self.copula.random(10000)
        assert_array_almost_equal(
            np.mean(sample, axis=0), np.repeat(0.5, self.dim), decimal=2
        )
        assert_array_almost_equal(
            np.percentile(sample, 25, axis=0), np.repeat(0.25, self.dim),
            decimal=2
        )
        assert_array_almost_equal(
            np.percentile(sample, 75, axis=0), np.repeat(0.75, self.dim),
            decimal=2
        )


class TestIndependentCopula(CopulaTests):
    copula = IndependentCopula()
    dim = 2


class TestGaussianCopula(CopulaTests):
    copula = GaussianCopula(cov=[[1., 0.5], [0.5, 1.]])
    dim = 2


class TestStudentCopula(CopulaTests):
    copula = StudentCopula(cov=[[1., 0.5], [0.5, 1.]])
    dim = 2


class TestClaytonCopula(CopulaTests):
    copula = ClaytonCopula(theta=2)
    dim = 2


class TestFrankCopula(CopulaTests):
    copula = FrankCopula(theta=2)
    dim = 2


class TestGumbelCopula(CopulaTests):
    copula = GumbelCopula(theta=2)
    dim = 2


def test_joint_distribution():
    dists = [stats.gamma(2), stats.norm]
    copula = GaussianCopula(cov=[[1., 0.5], [0.5, 1.]])
    joint_dist = JointDistribution(dists, copula)
    sample = joint_dist.random(512)
    assert sample.shape == (512, 2)
