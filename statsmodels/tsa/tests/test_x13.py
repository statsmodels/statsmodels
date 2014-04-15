from nose import SkipTest
from numpy.testing import assert_

from statsmodels.tsa.base.datetools import dates_from_range
from statsmodels.tsa.x12 import _find_x12, select_arima_order

x13path = _find_x12()

if x13path is False:
    _have_x13 = False
else:
    _have_x13 = True

class TestX13(object):
    @classmethod
    def setupClass(cls):
        if not _have_x13:
            raise SkipTest('X13/X12 not available')

        import pandas as pd
        from statsmodels.datasets import macrodata, co2
        dta = macrodata.load_pandas().data
        dates = dates_from_range('1959Q1', '2009Q3')
        index = pd.DatetimeIndex(dates)
        dta.index = index
        cls.quarterly_data = dta.dropna()

        dta = co2.load_pandas().data
        dta['co2'] = dta.co2.interpolate()
        cls.monthly_data = dta.resample('M')

        cls.monthly_start_data = dta.resample('MS')

    def test_select_arima_order(self):
        res = select_arima_order(self.monthly_data)
        assert_(isinstance(res.order, tuple))
        assert_(isinstance(res.sorder, tuple))
        res = select_arima_order(self.monthly_start_data)
        assert_(isinstance(res.order, tuple))
        assert_(isinstance(res.sorder, tuple))
        res = select_arima_order(self.monthly_data.co2)
        assert_(isinstance(res.order, tuple))
        assert_(isinstance(res.sorder, tuple))
        res = select_arima_order(self.monthly_start_data.co2)
        assert_(isinstance(res.order, tuple))
        assert_(isinstance(res.sorder, tuple))

        res = select_arima_order(self.quarterly_data[['realgdp']])
        assert_(isinstance(res.order, tuple))
        assert_(isinstance(res.sorder, tuple))
        res = select_arima_order(self.quarterly_data.realgdp)
        assert_(isinstance(res.order, tuple))
        assert_(isinstance(res.sorder, tuple))
