from numpy.testing import assert_almost_equal
import statsmodels.api as sm
from results.el_results import ANOVAResults

class TestANOVA():
    """
    Tests ANOVA difference in means
    """

    def __init__(self):
        self.data = sm.datasets.star98.load().exog[:30, 1:3]
        self.res1 = sm.emplike.ANOVA([self.data[:, 0], self.data[:, 1]])
        self.res2 = ANOVAResults()

    def test_anova(self):
        assert_almost_equal(self.res1.compute_ANOVA()[:2],
                            self.res2.compute_ANOVA[:2], 4)
        assert_almost_equal(self.res1.compute_ANOVA()[2],
                            self.res2.compute_ANOVA[2], 4)
        assert_almost_equal(self.res1.compute_ANOVA(return_weights=1)[3],
                            self.res2.compute_ANOVA[3], 4)
