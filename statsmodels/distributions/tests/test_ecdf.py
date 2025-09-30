import numpy as np
import numpy.testing as npt
import pandas as pd
import pytest

from statsmodels.distributions import ECDFDiscrete, StepFunction, monotone_fn_inverter
from statsmodels.distributions.empirical_distribution import ECDF


class TestDistributions:

    def test_StepFunction(self):
        x = np.arange(20)
        y = np.arange(20)
        f = StepFunction(x, y)
        vals = f(np.array([[3.2, 4.5], [24, -3.1], [3.0, 4.0]]))
        npt.assert_almost_equal(vals, [[3, 4], [19, 0], [2, 3]])

    def test_StepFunctionBadShape(self):
        x = np.arange(20)
        y = np.arange(21)
        with pytest.raises(ValueError):
            StepFunction(x, y)
        x = np.zeros((2, 2))
        y = np.zeros((2, 2))
        with pytest.raises(ValueError):
            StepFunction(x, y)

    def test_StepFunctionValueSideRight(self):
        x = np.arange(20)
        y = np.arange(20)
        f = StepFunction(x, y, side="right")
        vals = f(np.array([[3.2, 4.5], [24, -3.1], [3.0, 4.0]]))
        npt.assert_almost_equal(vals, [[3, 4], [19, 0], [3, 4]])

    def test_StepFunctionRepeatedValues(self):
        x = [1, 1, 2, 2, 2, 3, 3, 3, 4, 5]
        y = [6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
        f = StepFunction(x, y)
        npt.assert_almost_equal(f([1, 2, 3, 4, 5]), [0, 7, 10, 13, 14])
        f2 = StepFunction(x, y, side="right")
        npt.assert_almost_equal(f2([1, 2, 3, 4, 5]), [7, 10, 13, 14, 15])

    def test_monotone_fn_inverter(self):
        x = [6, 7, 8, 9, 10, 11, 12, 13, 14, 15]

        def fn(x):
            return 1.0 / x

        y = fn(np.array(x))
        f = monotone_fn_inverter(fn, x)
        npt.assert_array_equal(f.y, x[::-1])
        npt.assert_array_equal(f.x, y[::-1])

    def test_ecdf_discrete(self):
        x = [3, 3, 1, 4]
        e = ECDFDiscrete(x)
        npt.assert_array_equal(e.x, [-np.inf, 1, 3, 4])
        npt.assert_array_equal(e.y, [0, 0.25, 0.75, 1])
        e1 = ECDFDiscrete([3.5, 3.5, 1.5, 1, 4])
        e2 = ECDFDiscrete([3.5, 1.5, 1, 4], freq_weights=[2, 1, 1, 1])
        npt.assert_array_equal(e1.x, e2.x)
        npt.assert_array_equal(e1.y, e2.y)

    def test_ecdf_data_modification(self):
        # GH9383
        now = pd.to_datetime("2024-01-01")
        weeks = 2
        testdata = pd.DataFrame(columns=["dates", "values", "othervalues"])
        testdata["dates"] = pd.date_range(start=now, periods=weeks * 7, freq="D")
        testdata["values"] = np.random.randint(0, 100, size=(weeks * 7))
        testdata["othervalues"] = np.random.randint(0, 100, size=(weeks * 7))
        orig_testadata = testdata.copy()

        ECDF(testdata["values"])
        pd.testing.assert_frame_equal(orig_testadata, testdata)
