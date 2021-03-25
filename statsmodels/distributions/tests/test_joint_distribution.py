from scipy import stats

from statsmodels.distributions.joint_distribution import (
    JointDistribution
)
from statsmodels.distributions.copula.elliptical import GaussianCopula


def test_joint_distribution():
    dists = [stats.gamma(2), stats.norm]
    copula = GaussianCopula(cov=[[1., 0.5], [0.5, 1.]])
    joint_dist = JointDistribution(dists, copula)
    sample = joint_dist.random(512)
    assert sample.shape == (512, 2)
