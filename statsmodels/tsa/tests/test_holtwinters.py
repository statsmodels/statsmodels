"""
Created on Wed Jul 12 09:44:01 2017

@author: tvzyl
"""

import numpy as np
import pandas as pd
from numpy.testing import assert_almost_equal, assert_equal
from statsmodels.tsa.holtwinters import (ExponentialSmoothing,
                                         SimpleExpSmoothing, Holt)
from pandas import DataFrame, DatetimeIndex

class TestHoltWinters(object):
    @classmethod
    def setup_class(cls):
        #Changed for backwards compatability with pandas

        #oildata_oil_json = '{"851990400000":446.6565229,"883526400000":454.4733065,"915062400000":455.662974,"946598400000":423.6322388,"978220800000":456.2713279,"1009756800000":440.5880501,"1041292800000":425.3325201,"1072828800000":485.1494479,"1104451200000":506.0481621,"1135987200000":526.7919833,"1167523200000":514.268889,"1199059200000":494.2110193}'
        #oildata_oil = pd.read_json(oildata_oil_json, typ='Series').sort_index()
        data = [446.65652290000003, 454.47330649999998, 455.66297400000002,
                423.63223879999998, 456.27132790000002, 440.58805009999998,
                425.33252010000001, 485.14944789999998, 506.04816210000001,
                526.79198329999997, 514.26888899999994, 494.21101929999998]
        index= ['1996-12-31 00:00:00', '1997-12-31 00:00:00', '1998-12-31 00:00:00',
                '1999-12-31 00:00:00', '2000-12-31 00:00:00', '2001-12-31 00:00:00',
                '2002-12-31 00:00:00', '2003-12-31 00:00:00', '2004-12-31 00:00:00',
                '2005-12-31 00:00:00', '2006-12-31 00:00:00', '2007-12-31 00:00:00']
        oildata_oil = pd.Series(data, index)
        oildata_oil.index = pd.DatetimeIndex(oildata_oil.index,
                                freq=pd.infer_freq(oildata_oil.index))
        cls.oildata_oil = oildata_oil

        #air_ausair_json = '{"662601600000":17.5534,"694137600000":21.8601,"725760000000":23.8866,"757296000000":26.9293,"788832000000":26.8885,"820368000000":28.8314,"851990400000":30.0751,"883526400000":30.9535,"915062400000":30.1857,"946598400000":31.5797,"978220800000":32.577569,"1009756800000":33.477398,"1041292800000":39.021581,"1072828800000":41.386432,"1104451200000":41.596552}'
        #air_ausair = pd.read_json(air_ausair_json, typ='Series').sort_index()
        data = [17.5534, 21.860099999999999, 23.886600000000001,
                26.929300000000001, 26.888500000000001, 28.831399999999999,
                30.075099999999999, 30.953499999999998, 30.185700000000001,
                31.579699999999999, 32.577568999999997, 33.477398000000001,
                39.021580999999998, 41.386431999999999, 41.596552000000003]
        index= ['1990-12-31 00:00:00', '1991-12-31 00:00:00', '1992-12-31 00:00:00',
                '1993-12-31 00:00:00', '1994-12-31 00:00:00', '1995-12-31 00:00:00',
                '1996-12-31 00:00:00', '1997-12-31 00:00:00', '1998-12-31 00:00:00',
                '1999-12-31 00:00:00', '2000-12-31 00:00:00', '2001-12-31 00:00:00',
                '2002-12-31 00:00:00', '2003-12-31 00:00:00', '2004-12-31 00:00:00']
        air_ausair = pd.Series(data, index)
        air_ausair.index = pd.DatetimeIndex(air_ausair.index,
                                            freq=pd.infer_freq(air_ausair.index))
        cls.air_ausair = air_ausair

        #livestock2_livestock_json = '{"31449600000":263.917747,"62985600000":268.307222,"94608000000":260.662556,"126144000000":266.639419,"157680000000":277.515778,"189216000000":283.834045,"220838400000":290.309028,"252374400000":292.474198,"283910400000":300.830694,"315446400000":309.286657,"347068800000":318.331081,"378604800000":329.37239,"410140800000":338.883998,"441676800000":339.244126,"473299200000":328.600632,"504835200000":314.255385,"536371200000":314.459695,"567907200000":321.413779,"599529600000":329.789292,"631065600000":346.385165,"662601600000":352.297882,"694137600000":348.370515,"725760000000":417.562922,"757296000000":417.12357,"788832000000":417.749459,"820368000000":412.233904,"851990400000":411.946817,"883526400000":394.697075,"915062400000":401.49927,"946598400000":408.270468,"978220800000":414.2428}'
        #livestock2_livestock = pd.read_json(livestock2_livestock_json, typ='Series').sort_index()
        data = [263.91774700000002, 268.30722200000002, 260.662556,
                266.63941899999998, 277.51577800000001, 283.834045,
                290.30902800000001, 292.474198, 300.83069399999999,
                309.28665699999999, 318.33108099999998, 329.37239,
                338.88399800000002, 339.24412599999999, 328.60063200000002,
                314.25538499999999, 314.45969500000001, 321.41377899999998,
                329.78929199999999, 346.38516499999997, 352.29788200000002,
                348.37051500000001, 417.56292200000001, 417.12356999999997,
                417.749459, 412.233904, 411.94681700000001, 394.69707499999998,
                401.49927000000002, 408.27046799999999, 414.24279999999999]
        index= ['1970-12-31 00:00:00', '1971-12-31 00:00:00', '1972-12-31 00:00:00',
                '1973-12-31 00:00:00', '1974-12-31 00:00:00', '1975-12-31 00:00:00',
                '1976-12-31 00:00:00', '1977-12-31 00:00:00', '1978-12-31 00:00:00',
                '1979-12-31 00:00:00', '1980-12-31 00:00:00', '1981-12-31 00:00:00',
                '1982-12-31 00:00:00', '1983-12-31 00:00:00', '1984-12-31 00:00:00',
                '1985-12-31 00:00:00', '1986-12-31 00:00:00', '1987-12-31 00:00:00',
                '1988-12-31 00:00:00', '1989-12-31 00:00:00', '1990-12-31 00:00:00',
                '1991-12-31 00:00:00', '1992-12-31 00:00:00', '1993-12-31 00:00:00',
                '1994-12-31 00:00:00', '1995-12-31 00:00:00', '1996-12-31 00:00:00',
                '1997-12-31 00:00:00', '1998-12-31 00:00:00', '1999-12-31 00:00:00',
                '2000-12-31 00:00:00']
        livestock2_livestock = pd.Series(data, index)
        livestock2_livestock.index = pd.DatetimeIndex(
                livestock2_livestock.index,
                freq=pd.infer_freq(livestock2_livestock.index))
        cls.livestock2_livestock = livestock2_livestock

        #aust_json = '{"1104537600000":41.727458,"1112313600000":24.04185,"1120176000000":32.328103,"1128124800000":37.328708,"1136073600000":46.213153,"1143849600000":29.346326,"1151712000000":36.48291,"1159660800000":42.977719,"1167609600000":48.901525,"1175385600000":31.180221,"1183248000000":37.717881,"1191196800000":40.420211,"1199145600000":51.206863,"1207008000000":31.887228,"1214870400000":40.978263,"1222819200000":43.772491,"1230768000000":55.558567,"1238544000000":33.850915,"1246406400000":42.076383,"1254355200000":45.642292,"1262304000000":59.76678,"1270080000000":35.191877,"1277942400000":44.319737,"1285891200000":47.913736}'
        #aust = pd.read_json(aust_json, typ='Series').sort_index()
        data = [41.727457999999999, 24.04185, 32.328102999999999,
                37.328707999999999, 46.213152999999998, 29.346326000000001,
                36.482909999999997, 42.977719, 48.901524999999999,
                31.180221, 37.717880999999998, 40.420211000000002,
                51.206862999999998, 31.887228, 40.978262999999998,
                43.772491000000002, 55.558566999999996, 33.850915000000001,
                42.076383, 45.642291999999998, 59.766779999999997,
                35.191876999999998, 44.319737000000003, 47.913736]
        index= ['2005-03-01 00:00:00', '2005-06-01 00:00:00', '2005-09-01 00:00:00',
                '2005-12-01 00:00:00', '2006-03-01 00:00:00', '2006-06-01 00:00:00',
                '2006-09-01 00:00:00', '2006-12-01 00:00:00', '2007-03-01 00:00:00',
                '2007-06-01 00:00:00', '2007-09-01 00:00:00', '2007-12-01 00:00:00',
                '2008-03-01 00:00:00', '2008-06-01 00:00:00', '2008-09-01 00:00:00',
                '2008-12-01 00:00:00', '2009-03-01 00:00:00', '2009-06-01 00:00:00',
                '2009-09-01 00:00:00', '2009-12-01 00:00:00', '2010-03-01 00:00:00',
                '2010-06-01 00:00:00', '2010-09-01 00:00:00', '2010-12-01 00:00:00']
        aust = pd.Series(data, index)
        aust.index = pd.DatetimeIndex(aust.index,
                                      freq=pd.infer_freq(aust.index))
        cls.aust = aust

    def test_predict(self):
        fit1 = ExponentialSmoothing(self.aust, seasonal_periods=4, trend='add',
                                    seasonal='mul').fit()
        fit2 = ExponentialSmoothing(self.aust, seasonal_periods=4, trend='add',
                                    seasonal='mul').fit()
#        fit3 = ExponentialSmoothing(self.aust, seasonal_periods=4, trend='add', seasonal='mul').fit(remove_bias=True, use_basinhopping=True)
        assert_almost_equal(fit1.predict('2011-03-01 00:00:00',
                                         '2011-12-01 00:00:00'),
                            [61.3083,37.3730,46.9652,51.5578], 3)
        assert_almost_equal(fit2.predict(end='2011-12-01 00:00:00'),
                            [61.3083,37.3730,46.9652,51.5578], 3)
#        assert_almost_equal(fit3.predict('2010-10-01 00:00:00', '2010-10-01 00:00:00'), [49.087], 3)

    def test_ndarray(self):
        fit1 = ExponentialSmoothing(self.aust.values, seasonal_periods=4,
                                    trend='add', seasonal='mul').fit()
        assert_almost_equal(fit1.forecast(4), [61.3083,37.3730,46.9652,51.5578], 3)

    def test_forecast(self):
        fit1 = ExponentialSmoothing(self.aust, seasonal_periods=4, trend='add',
                                    seasonal='add').fit()
        assert_almost_equal(fit1.forecast(steps=4),
                            [60.9542,36.8505,46.1628,50.1272], 3)

    def test_simple_exp_smoothing(self):
        fit1 = SimpleExpSmoothing(self.oildata_oil).fit(0.2,optimized=False)
        fit2 = SimpleExpSmoothing(self.oildata_oil).fit(0.6,optimized=False)
        fit3 = SimpleExpSmoothing(self.oildata_oil).fit()
        assert_almost_equal(fit1.forecast(1), [484.802468], 4)
        assert_almost_equal(fit1.level,
                            [446.65652290,448.21987962,449.7084985,
                             444.49324656,446.84886283,445.59670028,
                             441.54386424,450.26498098,461.4216172,
                             474.49569042,482.45033014,484.80246797], 4)
        assert_almost_equal(fit2.forecast(1), [501.837461], 4)
        assert_almost_equal(fit3.forecast(1), [496.493543], 4)
        assert_almost_equal(fit3.params['smoothing_level'], 0.891998, 4)
        #has to be 3 for old python2.7 scipy versions
        assert_almost_equal(fit3.params['initial_level'], 447.478440, 3)

    def test_holt(self):
        fit1 = Holt(self.air_ausair).fit(smoothing_level=0.8,
                                         smoothing_slope=0.2, optimized=False)
        fit2 = Holt(self.air_ausair, exponential=True).fit(
                    smoothing_level=0.8, smoothing_slope=0.2,
                    optimized=False)
        fit3 = Holt(self.air_ausair, damped=True).fit(smoothing_level=0.8,
                                                      smoothing_slope=0.2)
        assert_almost_equal(fit1.forecast(5), [43.76,45.59,47.43,49.27,51.10], 2)
        assert_almost_equal(fit1.slope,
                            [3.617628  ,3.59006512,3.33438212,3.23657639,2.69263502,
                             2.46388914,2.2229097 ,1.95959226,1.47054601,1.3604894 ,
                             1.28045881,1.20355193,1.88267152,2.09564416,1.83655482], 4)
        assert_almost_equal(fit1.fittedfcast,
                           [21.8601    ,22.032368  ,25.48461872,27.54058587,
                            30.28813356,30.26106173,31.58122149,32.599234  ,
                            33.24223906,32.26755382,33.07776017,33.95806605,
                            34.77708354,40.05535303,43.21586036,43.75696849], 4)
        assert_almost_equal(fit2.forecast(5),
                            [44.60,47.24,50.04,53.01,56.15], 2)
        assert_almost_equal(fit3.forecast(5),
                            [42.85,43.81,44.66,45.41,46.06], 2)

    def test_holt_damp(self):
        fit1 = SimpleExpSmoothing(self.livestock2_livestock).fit()
        mod4 = Holt(self.livestock2_livestock,damped=True)
        fit4 = mod4.fit(damping_slope=0.98)
        mod5 = Holt(self.livestock2_livestock,exponential=True,damped=True)
        fit5 = mod5.fit()
        #We accept the below values as we getting a better SSE than text book
        assert_almost_equal(fit1.params['smoothing_level'],1.00, 2)
        assert_almost_equal(fit1.params['smoothing_slope'],np.NaN, 2)
        assert_almost_equal(fit1.params['damping_slope'],np.NaN, 2)
        assert_almost_equal(fit1.params['initial_level'],263.92, 2)
        assert_almost_equal(fit1.params['initial_slope'],np.NaN, 2)
        assert_almost_equal(fit1.sse,6761.35, 2) #6080.26

        assert_almost_equal(fit4.params['smoothing_level'],0.98, 2)
        assert_almost_equal(fit4.params['smoothing_slope'],0.00, 2)
        assert_almost_equal(fit4.params['damping_slope'],0.98, 2)
        assert_almost_equal(fit4.params['initial_level'],257.36, 2)
        assert_almost_equal(fit4.params['initial_slope'],6.51, 2)
        assert_almost_equal(fit4.sse,6036.56, 2) #6080.26
        assert_almost_equal(fit5.params['smoothing_level'],0.97, 2)
        assert_almost_equal(fit5.params['smoothing_slope'],0.00, 2)
        assert_almost_equal(fit5.params['damping_slope'],0.98, 2)
        assert_almost_equal(fit5.params['initial_level'],258.95, 2)
        assert_almost_equal(fit5.params['initial_slope'],1.02, 2)
        assert_almost_equal(fit5.sse,6082.00, 2) #6100.11

    def test_hw_seasonal(self):
        fit1 = ExponentialSmoothing(self.aust, seasonal_periods=4,
                                    trend='additive',
                                    seasonal='additive').fit(use_boxcox=True)
        fit2 = ExponentialSmoothing(self.aust, seasonal_periods=4, trend='add',
                                    seasonal='mul').fit(use_boxcox=True)
        fit3 = ExponentialSmoothing(self.aust, seasonal_periods=4,
                                    seasonal='add').fit(use_boxcox=True)
        fit4 = ExponentialSmoothing(self.aust, seasonal_periods=4,
                                    seasonal='mul').fit(use_boxcox=True)
        fit5 = ExponentialSmoothing(self.aust, seasonal_periods=4,
                                    trend='mul', seasonal='add'
                                    ).fit(use_boxcox='log')
        fit6 = ExponentialSmoothing(self.aust, seasonal_periods=4,
                                    trend='multiplicative',
                                    seasonal='multiplicative'
                                    ).fit(use_boxcox='log')
        assert_almost_equal(fit1.forecast(8),
                            [61.34,37.24,46.84,51.01,64.47,39.78,49.64,53.90],
                            2)
        assert_almost_equal(fit2.forecast(8),
                            [60.97,36.99,46.71,51.48,64.46,39.02,49.29,54.32],
                            2)
        assert_almost_equal(fit3.forecast(8),
                            [59.91,35.71,44.64,47.62,59.91,35.71,44.64,47.62],
                            2)
        assert_almost_equal(fit4.forecast(8),
                            [60.71,35.70,44.63,47.55,60.71,35.70,44.63,47.55],
                            2)
        assert_almost_equal(fit5.forecast(1), [78.53], 2)
        assert_almost_equal(fit6.forecast(1), [54.82], 2)

    def test_raises(self):
        pass
