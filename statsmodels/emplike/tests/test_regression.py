from numpy.testing import assert_almost_equal
import statsmodels.api as sm
from results.el_results import RegressionResults


class GenRes(object):
    """
    Loads data and creates class instance ot be tested

    """
    def __init__(self):
        data = sm.datasets.stackloss.load()
        data.exog = sm.add_constant(data.exog, prepend=1)
        self.res1 = sm.OLS(data.endog, data.exog).fit()
        self.res2 = RegressionResults()


class TestRegressionPowell(GenRes):
    """
    All confidence intervals are tested by conducting a hypothesis
    tests at the confidence interval values since hy_test_beta
    is already tested against Matlab

    See Also
    --------

    test_descriptive.py, test_ci_skew

    """
    def __init__(self):
        super(TestRegressionPowell, self).__init__()

    def test_hypothesis_beta0(self):
        beta0res = self.res1.el_test([-30], [0], return_weights=1, method='powell')
        assert_almost_equal(beta0res[:2], self.res2.test_beta0[:2], 4)
        assert_almost_equal(beta0res[2], self.res2.test_beta0[2], 4)

    def test_hypothesis_beta1(self):
        beta1res = self.res1.el_test([.5], [1], return_weights=1, method='powell')
        assert_almost_equal(beta1res[:2], self.res2.test_beta1[:2], 4)
        assert_almost_equal(beta1res[2], self.res2.test_beta1[2], 4)

    def test_hypothesis_beta2(self):
        beta2res = self.res1.el_test([1], [2], return_weights=1, method ='powell')
        assert_almost_equal(beta2res[:2], self.res2.test_beta2[:2], 4)
        assert_almost_equal(beta2res[2], self.res2.test_beta2[2], 4)

    def test_hypothesis_beta3(self):
        beta3res = self.res1.el_test([0], [3], return_weights=1, method='powell')
        assert_almost_equal(beta3res[:2], self.res2.test_beta3[:2], 4)
        assert_almost_equal(beta3res[2], self.res2.test_beta3[2], 4)

    def test_ci_beta0(self):
        beta0ci = self.res1.conf_int_el(0, method='powell')
        lower_lim = beta0ci[0]
        upper_lim = beta0ci[1]
        ul_pval = self.res1.el_test([upper_lim], [0])[1]
        ll_pval = self.res1.el_test([lower_lim], [0])[1]
        assert_almost_equal(ul_pval, .050000, 4)
        assert_almost_equal(ll_pval, .050000, 4)

    def test_ci_beta1(self):
        beta1ci = self.res1.conf_int_el(1, method='powell')
        lower_lim = beta1ci[0]
        upper_lim = beta1ci[1]
        ul_pval = self.res1.el_test([upper_lim], [1])[1]
        ll_pval = self.res1.el_test([lower_lim], [1])[1]
        assert_almost_equal(ul_pval, .050000, 4)
        assert_almost_equal(ll_pval, .050000, 4)

    def test_ci_beta2(self):
        beta2ci = self.res1.conf_int_el(2, method='powell')
        lower_lim = beta2ci[0]
        upper_lim = beta2ci[1]
        ul_pval = self.res1.el_test([upper_lim], [2])[1]
        ll_pval = self.res1.el_test([lower_lim], [2])[1]
        assert_almost_equal(ul_pval, .050000, 4)
        assert_almost_equal(ll_pval, .050000, 4)

    def test_ci_beta3(self):
        beta3ci = self.res1.conf_int_el(3, method='powell')
        lower_lim = beta3ci[0]
        upper_lim = beta3ci[1]
        ul_pval = self.res1.el_test([upper_lim], [3])[1]
        ll_pval = self.res1.el_test([lower_lim], [3])[1]
        assert_almost_equal(ul_pval, .050000, 4)
        assert_almost_equal(ll_pval, .050000, 4)


class TestRegressionNM(GenRes):
    """
    All confidence intervals are tested by conducting a hypothesis
    tests at the confidence interval values since el_test
    is already tested against Matlab

    See Also
    --------

    test_descriptive.py, test_ci_skew

    """
    def __init__(self):
        super(TestRegressionNM, self).__init__()

    def test_hypothesis_beta0(self):
        beta0res = self.res1.el_test([-30], [0], return_weights=1,
                                          method='nm')
        assert_almost_equal(beta0res[:2], self.res2.test_beta0[:2], 4)
        assert_almost_equal(beta0res[2], self.res2.test_beta0[2], 4)

    def test_hypothesis_beta1(self):
        beta1res = self.res1.el_test([.5], [1], return_weights=1,
                                          method='nm')
        assert_almost_equal(beta1res[:2], self.res2.test_beta1[:2], 4)
        assert_almost_equal(beta1res[2], self.res2.test_beta1[2], 4)

    def test_hypothesis_beta2(self):
        beta2res = self.res1.el_test([1], [2], return_weights=1,
                                          method='nm')
        assert_almost_equal(beta2res[:2], self.res2.test_beta2[:2], 4)
        assert_almost_equal(beta2res[2], self.res2.test_beta2[2], 4)

    def test_hypothesis_beta3(self):
        beta3res = self.res1.el_test([0], [3], return_weights=1,
                                          method='nm')
        assert_almost_equal(beta3res[:2], self.res2.test_beta3[:2], 4)
        assert_almost_equal(beta3res[2], self.res2.test_beta3[2], 4)

    def test_ci_beta0(self):
        """
        All confidence intervals are tested by conducting a hypothesis
        tests at the confidence interval values since el_test
        is already tested against Matlab

        See Also
        --------

        test_descriptive.py, test_ci_skew

        """

        beta0ci = self.res1.conf_int_el(0, method='nm', lower_bound=-60)
        # ^ doesn't converge at default bounds
        lower_lim = beta0ci[0]
        upper_lim = beta0ci[1]
        ul_pval = self.res1.el_test([upper_lim], [0], method='nm')[1]
        ll_pval = self.res1.el_test([lower_lim], [0], method='nm')[1]
        assert_almost_equal(ul_pval, .050000, 4)
        assert_almost_equal(ll_pval, .050000, 4)

    def test_ci_beta1(self):
        beta1ci = self.res1.conf_int_el(1, method='nm')
        lower_lim = beta1ci[0]
        upper_lim = beta1ci[1]
        ul_pval = self.res1.el_test([upper_lim], [1], method='nm')[1]
        ll_pval = self.res1.el_test([lower_lim], [1], method='nm')[1]
        assert_almost_equal(ul_pval, .050000, 4)
        assert_almost_equal(ll_pval, .050000, 4)

    def test_ci_beta2(self):
        beta2ci = self.res1.conf_int_el(2, method='nm')
        lower_lim = beta2ci[0]
        upper_lim = beta2ci[1]
        ul_pval = self.res1.el_test([upper_lim], [2], method='nm')[1]
        ll_pval = self.res1.el_test([lower_lim], [2], method='nm')[1]
        assert_almost_equal(ul_pval, .050000, 4)
        assert_almost_equal(ll_pval, .050000, 4)

    def test_ci_beta3(self):
        beta3ci = self.res1.conf_int_el(3)
        lower_lim = beta3ci[0]
        upper_lim = beta3ci[1]
        ul_pval = self.res1.el_test([upper_lim], [3], method='nm')[1]
        ll_pval = self.res1.el_test([lower_lim], [3], method='nm')[1]
        assert_almost_equal(ul_pval, .050000, 4)
        assert_almost_equal(ll_pval, .050000, 4)


# class TestANOVA(GenRes):
#     def __init__(self):
#         self.data = sm.datasets.star98.load().exog[:30, 1:3]
#         self.res1 = sm.emplike.ANOVA([self.data[:, 0], self.data[:, 1]])
#         self.res2 = RegressionResults()

#     def test_anova(self):
#         assert_almost_equal(self.res1.compute_ANOVA()[:2],
#                             self.res2.compute_ANOVA[:2], 4)
#         assert_almost_equal(self.res1.compute_ANOVA()[2],
#                             self.res2.compute_ANOVA[2], 4)
#         assert_almost_equal(self.res1.compute_ANOVA(return_weights=1)[3],
#                             self.res2.compute_ANOVA[3], 4)
