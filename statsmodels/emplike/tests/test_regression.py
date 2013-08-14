from numpy.testing import assert_almost_equal
from numpy.testing.decorators import slow
import statsmodels.api as sm
from results.el_results import RegressionResults


class GenRes(object):
    """
    Loads data and creates class instance ot be tested

    """
    def __init__(self):
        data = sm.datasets.stackloss.load()
        data.exog = sm.add_constant(data.exog)
        self.res1 = sm.OLS(data.endog, data.exog).fit()
        self.res2 = RegressionResults()


class TestRegressionPowell(GenRes):
    """
    All confidence intervals are tested by conducting a hypothesis
    tests at the confidence interval values.

    See Also
    --------

    test_descriptive.py, test_ci_skew

    """
    def __init__(self):
        super(TestRegressionPowell, self).__init__()

    @slow
    def test_hypothesis_beta0(self):
        beta0res = self.res1.el_test([-30], [0], return_weights=1,
                                     method='powell')
        assert_almost_equal(beta0res[:2], self.res2.test_beta0[:2], 4)
        assert_almost_equal(beta0res[2], self.res2.test_beta0[2], 4)

    @slow
    def test_hypothesis_beta1(self):
        beta1res = self.res1.el_test([.5], [1], return_weights=1,
                                     method='powell')
        assert_almost_equal(beta1res[:2], self.res2.test_beta1[:2], 4)
        assert_almost_equal(beta1res[2], self.res2.test_beta1[2], 4)

    def test_hypothesis_beta2(self):
        beta2res = self.res1.el_test([1], [2], return_weights=1,
                                     method='powell')
        assert_almost_equal(beta2res[:2], self.res2.test_beta2[:2], 4)
        assert_almost_equal(beta2res[2], self.res2.test_beta2[2], 4)

    def test_hypothesis_beta3(self):
        beta3res = self.res1.el_test([0], [3], return_weights=1,
                                     method='powell')
        assert_almost_equal(beta3res[:2], self.res2.test_beta3[:2], 4)
        assert_almost_equal(beta3res[2], self.res2.test_beta3[2], 4)

    # Confidence interval results obtained through hypothesis testing in Matlab
    def test_ci_beta0(self):
        beta0ci = self.res1.conf_int_el(0, lower_bound=-52.9,
                                        upper_bound=-24.1, method='powell')
        assert_almost_equal(beta0ci, self.res2.test_ci_beta0, 3)
        #  Slightly lower precision.  CI was obtained from nm method.

    def test_ci_beta1(self):
        beta1ci = self.res1.conf_int_el(1, lower_bound=.418, upper_bound=.986,
                                        method='powell')
        assert_almost_equal(beta1ci, self.res2.test_ci_beta1, 4)

    @slow
    def test_ci_beta2(self):
        beta2ci = self.res1.conf_int_el(2, lower_bound=.59,
                                    upper_bound=2.2, method='powell')
        assert_almost_equal(beta2ci, self.res2.test_ci_beta2, 5)

    @slow
    def test_ci_beta3(self):
        beta3ci = self.res1.conf_int_el(3, lower_bound=-.39, upper_bound=.01,
                                        method='powell')
        assert_almost_equal(beta3ci, self.res2.test_ci_beta3, 6)


class TestRegressionNM(GenRes):
    """
    All confidence intervals are tested by conducting a hypothesis
    tests at the confidence interval values.

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

    @slow
    def test_hypothesis_beta2(self):
        beta2res = self.res1.el_test([1], [2], return_weights=1,
                                          method='nm')
        assert_almost_equal(beta2res[:2], self.res2.test_beta2[:2], 4)
        assert_almost_equal(beta2res[2], self.res2.test_beta2[2], 4)

    @slow
    def test_hypothesis_beta3(self):
        beta3res = self.res1.el_test([0], [3], return_weights=1,
                                          method='nm')
        assert_almost_equal(beta3res[:2], self.res2.test_beta3[:2], 4)
        assert_almost_equal(beta3res[2], self.res2.test_beta3[2], 4)

    #  Confidence interval results obtained through hyp testing in Matlab

    @slow
    def test_ci_beta0(self):
        """
        All confidence intervals are tested by conducting a hypothesis
        tests at the confidence interval values since el_test
        is already tested against Matlab

        See Also
        --------

        test_descriptive.py, test_ci_skew

        """
        beta0ci = self.res1.conf_int_el(0, method='nm')
        assert_almost_equal(beta0ci, self.res2.test_ci_beta0, 6)

    @slow
    def test_ci_beta1(self):
        beta1ci = self.res1.conf_int_el(1, method='nm')
        assert_almost_equal(beta1ci, self.res2.test_ci_beta1, 6)

    def test_ci_beta2(self):
        beta2ci = self.res1.conf_int_el(2, lower_bound=.59, upper_bound=2.2,  method='nm')
        assert_almost_equal(beta2ci, self.res2.test_ci_beta2, 6)

    def test_ci_beta3(self):
        beta3ci = self.res1.conf_int_el(3, method='nm')
        assert_almost_equal(beta3ci, self.res2.test_ci_beta3, 6)
