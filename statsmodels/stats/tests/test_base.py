import warnings

from numpy.testing import (assert_, assert_almost_equal, assert_raises,
                           assert_warns, assert_equal)

from statsmodels.stats.base import (Hypothesis, Statistics, CriticalValues,
                                    TestResult)

warnings.simplefilter('always')  # should be as explicit as possible


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

    def test_statistics_print_filter(self):
        assert_("print_filter" not in str(self.statistics))

        self.statistics.print_filter = ["R"]

        assert_("R = 1.2" in str(self.statistics))
        assert_("t = 1.0" not in str(self.statistics))
        assert_("p = 0.05" not in str(self.statistics))

    def test_statistics_attributes(self):
        for item in ["R", "p", "t"]:
            assert_(item in self.statistics.attributes)

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

    def test_direct_access(self):
        assert_almost_equal(self.test_result.t, 1.0)
        assert_almost_equal(self.test_result.R, 1.2)
        assert_almost_equal(self.test_result.p, 0.05)

    def test_test_result_attributes(self):
        # These are the items we passed above, so they should be available
        on_test = ["statistics", "test_name", "hypothesis", "critical_values"]

        for item in on_test:
            assert_(item in self.test_result.attributes)

        assert_("some_value" not in self.test_result.attributes)

        # attributes explicitly compare against _options
        assert_(all(item in TestResult._options
                    for item in self.test_result.attributes))

    def test_raises_missing_value_access(self):
        with assert_raises(AttributeError):
            self.test_result.some_missing_statistic

    def test_warn_deprecation(self):
        with assert_warns(DeprecationWarning):
            assert_equal(self.test_result.R,
                         self.test_result.statistics.R)

    def test_warn_message_and_num_warnings(self):
        with warnings.catch_warnings(record=True) as warns:
            assert_equal(self.test_result.R,
                         self.test_result.statistics.R)

            # should only be the one DeprecationWarning, no more
            assert_equal(len(warns), 1)

            for warning in warns:
                assert_(self.test_result._warn
                        in str(warning.message))
