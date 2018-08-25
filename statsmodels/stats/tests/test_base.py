from numpy.testing import assert_, assert_almost_equal

from statsmodels.stats.base import (Hypothesis, Statistics, CriticalValues,
                                    TestResult)


class TestBase:

    @classmethod
    def setup_class(cls):
        cls.hypothesis = Hypothesis(null="Null", alternative="Alternative")
        cls.statistics = Statistics(t=1.0, R=1.2, p=0.05)
        cls.critical_values = CriticalValues({"5%": 0.9, "10%": 1.5})

        cls.test_result = TestResult("Example test",
                                     hypothesis=cls.hypothesis,
                                     statistics=cls.statistics,
                                     critical_values=cls.critical_values)

    def test_hypothesis_properties(self):
        assert_(self.hypothesis.null == "Null")
        assert_(self.hypothesis.alternative == "Alternative")

    def test_hypothesis_str(self):
        assert_("H0" in str(self.hypothesis))
        assert_("Null" in str(self.hypothesis))

        assert_("H1" in str(self.hypothesis))
        assert_("Alternative" in str(self.hypothesis))

    def test_statistics_access(self):
        assert_almost_equal(self.statistics.t, 1.0)
        assert_almost_equal(self.statistics.R, 1.2)
        assert_almost_equal(self.statistics.p, 0.05)

    def test_statistics_str(self):
        assert_("R = 1.2" in str(self.statistics))
        assert_("t = 1.0" in str(self.statistics))
        assert_("p = 0.05" in str(self.statistics))

    def test_critical_values_properties(self):
        assert_(self.critical_values.crit_dict == {"5%": 0.9, "10%": 1.5})

    def test_critical_values_str(self):
        assert_("[5%] = 0.9" in str(self.critical_values))
        assert_("[10%] = 1.5" in str(self.critical_values))

    def test_critical_values_str_sorted(self):
        idx_5 = str(self.critical_values).find("5%")
        idx_10 = str(self.critical_values).find("10%")

        assert_(idx_5 < idx_10)

    def test_test_result_summary(self):
        summary = self.test_result.summary()

        assert_("Example test" in summary)  # test name

        assert_(str(self.hypothesis) in summary)
        assert_(str(self.statistics) in summary)
        assert_(str(self.critical_values) in summary)

    def test_test_result_str(self):
        assert_(str(self.test_result) == self.test_result.summary())
