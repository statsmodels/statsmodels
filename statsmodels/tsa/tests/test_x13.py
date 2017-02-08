from statsmodels.compat.testing import SkipTest
from numpy.testing import assert_
from statsmodels.tsa.x13 import _find_x12, x13_arima_select_order

x13path = _find_x12()

if x13path is False:
    raise SkipTest('X13/X12 not available')


class TestX13(object):
    @classmethod
    def setup_class(cls):
        import pandas as pd
        from statsmodels.datasets import macrodata, co2
        dta = macrodata.load_pandas().data
        index = pd.PeriodIndex(start='1959Q1', end='2009Q3', freq='Q')
        dta.index = index
        cls.quarterly_data = dta.dropna()

        dta = co2.load_pandas().data
        dta['co2'] = dta.co2.interpolate()
        cls.monthly_data = dta.resample('M')
        # change in pandas 0.18 resample is deferred object
        if not isinstance(cls.monthly_data, (pd.DataFrame, pd.Series)):
            cls.monthly_data = cls.monthly_data.mean()

        cls.monthly_start_data = dta.resample('MS')
        if not isinstance(cls.monthly_start_data, (pd.DataFrame, pd.Series)):
            cls.monthly_start_data = cls.monthly_start_data.mean()


    def test_x13_arima_select_order(self):
        res = x13_arima_select_order(self.monthly_data)
        assert_(isinstance(res.order, tuple))
        assert_(isinstance(res.sorder, tuple))
        res = x13_arima_select_order(self.monthly_start_data)
        assert_(isinstance(res.order, tuple))
        assert_(isinstance(res.sorder, tuple))
        res = x13_arima_select_order(self.monthly_data.co2)
        assert_(isinstance(res.order, tuple))
        assert_(isinstance(res.sorder, tuple))
        res = x13_arima_select_order(self.monthly_start_data.co2)
        assert_(isinstance(res.order, tuple))
        assert_(isinstance(res.sorder, tuple))

        res = x13_arima_select_order(self.quarterly_data[['realgdp']])
        assert_(isinstance(res.order, tuple))
        assert_(isinstance(res.sorder, tuple))
        res = x13_arima_select_order(self.quarterly_data.realgdp)
        assert_(isinstance(res.order, tuple))
        assert_(isinstance(res.sorder, tuple))
