#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 12 09:44:01 2017

@author: tvzyl
"""

import numpy as np
import pandas as pd
from numpy.testing import assert_almost_equal, assert_equal, assert_raises
from statsmodels.tsa.holtwinters import holt_winters, ses, holt
from pandas import DataFrame, DatetimeIndex

class TestHoltWinters(object):
    @classmethod
    def setupClass(cls):
        oildata_oil_json = '{"851990400000":446.6565229,"883526400000":454.4733065,"915062400000":455.662974,"946598400000":423.6322388,"978220800000":456.2713279,"1009756800000":440.5880501,"1041292800000":425.3325201,"1072828800000":485.1494479,"1104451200000":506.0481621,"1135987200000":526.7919833,"1167523200000":514.268889,"1199059200000":494.2110193}'
        oildata_oil = pd.read_json(oildata_oil_json, typ='Series').sort_index()
        oildata_oil = oildata_oil.resample(pd.infer_freq(oildata_oil.index)).last()
        cls.oildata_oil = oildata_oil
        
        air_ausair_json = '{"662601600000":17.5534,"694137600000":21.8601,"725760000000":23.8866,"757296000000":26.9293,"788832000000":26.8885,"820368000000":28.8314,"851990400000":30.0751,"883526400000":30.9535,"915062400000":30.1857,"946598400000":31.5797,"978220800000":32.577569,"1009756800000":33.477398,"1041292800000":39.021581,"1072828800000":41.386432,"1104451200000":41.596552}'
        air_ausair = pd.read_json(air_ausair_json, typ='Series').sort_index()
        air_ausair = air_ausair.resample(pd.infer_freq(air_ausair.index)).last()
        cls.air_ausair = air_ausair
        
        livestock2_livestock_json = '{"31449600000":263.917747,"62985600000":268.307222,"94608000000":260.662556,"126144000000":266.639419,"157680000000":277.515778,"189216000000":283.834045,"220838400000":290.309028,"252374400000":292.474198,"283910400000":300.830694,"315446400000":309.286657,"347068800000":318.331081,"378604800000":329.37239,"410140800000":338.883998,"441676800000":339.244126,"473299200000":328.600632,"504835200000":314.255385,"536371200000":314.459695,"567907200000":321.413779,"599529600000":329.789292,"631065600000":346.385165,"662601600000":352.297882,"694137600000":348.370515,"725760000000":417.562922,"757296000000":417.12357,"788832000000":417.749459,"820368000000":412.233904,"851990400000":411.946817,"883526400000":394.697075,"915062400000":401.49927,"946598400000":408.270468,"978220800000":414.2428}'
        livestock2_livestock = pd.read_json(livestock2_livestock_json, typ='Series').sort_index()
        livestock2_livestock = livestock2_livestock.resample(pd.infer_freq(livestock2_livestock.index)).last()
        cls.livestock2_livestock = livestock2_livestock

        aust_json = '{"1104537600000":41.727458,"1112313600000":24.04185,"1120176000000":32.328103,"1128124800000":37.328708,"1136073600000":46.213153,"1143849600000":29.346326,"1151712000000":36.48291,"1159660800000":42.977719,"1167609600000":48.901525,"1175385600000":31.180221,"1183248000000":37.717881,"1191196800000":40.420211,"1199145600000":51.206863,"1207008000000":31.887228,"1214870400000":40.978263,"1222819200000":43.772491,"1230768000000":55.558567,"1238544000000":33.850915,"1246406400000":42.076383,"1254355200000":45.642292,"1262304000000":59.76678,"1270080000000":35.191877,"1277942400000":44.319737,"1285891200000":47.913736}'
        aust = pd.read_json(aust_json, typ='Series').sort_index()
        aust = aust.resample(pd.infer_freq(aust.index)).last()
        cls.aust = aust

        
    def test_ndarray(self):
        fit1 = holt_winters(self.aust.values, h=4, m=4, trend='add', seasonal='mul')
        assert_almost_equal(fit1.fcast, [61.3083,37.3730,46.9652,51.5578], 3)
        
    def test_pandas(self):
        fit1 = holt_winters(self.aust, h=4, m=4, trend='add', seasonal='add')
        assert_almost_equal(fit1.fcast.values, [60.9542,36.8505,46.1628,50.1272], 3)
        
    def test_ses(self):
        fit1 = ses(self.oildata_oil,0.2,optimised=False)
        fit2 = ses(self.oildata_oil,0.6,optimised=False)
        fit3 = ses(self.oildata_oil)
        assert_almost_equal(fit1.fcast.values, [484.802468], 4)
        assert_almost_equal(fit2.fcast.values, [501.837461], 4)
        assert_almost_equal(fit3.fcast.values, [496.493543], 4)
        assert_almost_equal(fit3.alpha, 0.891998, 4)
    
    def test_holt(self):
        fit1 = holt(self.air_ausair, alpha=0.8, beta=0.2, optimised=False, h=5)
        fit2 = holt(self.air_ausair, alpha=0.8, beta=0.2, exponential=True, optimised=False, h=5)
        fit3 = holt(self.air_ausair, alpha=0.8, beta=0.2, damped=True, h=5)
        assert_almost_equal(fit1.fcast.values, [43.76,45.59,47.43,49.27,51.10], 2)
        assert_almost_equal(fit2.fcast.values, [44.60,47.24,50.04,53.01,56.15], 2)
        assert_almost_equal(fit3.fcast.values, [42.85,43.81,44.66,45.41,46.06], 2)
        
    def test_holt_damp(self):        
        fit4 = holt(self.livestock2_livestock,damped=True, phi=0.98)
        fit5 = holt(self.livestock2_livestock,exponential=True,damped=True)
        assert_almost_equal(fit4.alpha,0.98, 2)
        assert_almost_equal(fit4.beta,0.00, 2)
        assert_almost_equal(fit4.phi,0.98, 2)
        assert_almost_equal(fit4.l0,257.36, 2)
        assert_almost_equal(fit4.b0,6.64, 2)
        assert_almost_equal(fit4.SSE,6036.56, 2)
        assert_almost_equal(fit5.alpha,0.97, 2)
        assert_almost_equal(fit5.beta,0.00, 2)
        assert_almost_equal(fit5.phi,0.98, 2)
        assert_almost_equal(fit5.l0,258.95, 2)
        assert_almost_equal(fit5.b0,1.02, 2)
        assert_almost_equal(fit5.SSE,6082.00, 2)        
        
    def test_hw_seasonal(self):
        fit1 = holt_winters(self.aust, h=8, m=4, trend='add', seasonal='add', boxcoxed=True)
        fit2 = holt_winters(self.aust, h=8, m=4, trend='add', seasonal='mul', boxcoxed=True)        
        assert_almost_equal(fit1.fcast.values, [61.34,37.24,46.84,51.01,64.47,39.78,49.64,53.90], 2)
        assert_almost_equal(fit2.fcast.values, [60.97,36.99,46.71,51.48,64.46,39.02,49.29,54.32], 2)
    
    def test_raises(self):
        pass
    