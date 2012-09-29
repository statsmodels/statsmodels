from pandas import Series
from statsmodels.tsa.base.tsa_model import TimeSeriesModel
import numpy.testing as npt

def test_pandas_nodates_index():
    from statsmodels.datasets import sunspots
    y = sunspots.load_pandas().data.SUNACTIVITY
    npt.assert_raises(ValueError, TimeSeriesModel, y)
