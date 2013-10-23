# -*- coding: utf-8 -*-
"""

Created on Thu Jul 04 23:44:33 2013

Author: Josef Perktold
"""

import numpy as np
from numpy.testing import assert_allclose

from statsmodels.tsa.arima_process import arma_generate_sample
from statsmodels.miscmodels.tmodel import TArma
from statsmodels.tsa.arma_mle import Arma
#from statsmodels.tsa.arima_model import ARMA



class CheckTArmaMixin(object):

    def test_params(self):
        attrs =  ['params', 'bse', 'tvalues', 'pvalues', 'bsejhj', 'bsejac']
        for row, attr in enumerate(attrs):
            assert_allclose(getattr(self.res, attr), self.res1_table[row], atol=1e-4,
                             rtol=1e-3)

        assert_allclose(self.res.conf_int(), self.res1_conf_int, atol=1e-4, rtol=1e-3)

    def test_smoke(self):
        self.res.summary()
        rmat = np.eye(len(self.res.params))
        self.res.t_test(rmat)
        self.res.f_test(rmat)

    def test_fit_ls(self):

        assert_allclose(self.res_ls[0], self.ls_params, atol=1e-4, rtol=1e-3)
        self.res_ls[1]
        # this is bse unscaled I think
        bse = np.sqrt(np.diag(self.res_ls[1]))
        assert_allclose(bse, self.ls_bse, atol=1e-4, rtol=1e-3)


class TestTArma(CheckTArmaMixin):
    #regression test for TArma

    @classmethod
    def setup_class(cls):

        nobs = 500
        ar = [1, -0.5, 0.1]
        ma = [1, 0.7]
        dist = lambda n: np.random.standard_t(3, size=n)
        np.random.seed(8659567)
        x = arma_generate_sample(ar, ma, nobs, sigma=1, distrvs=dist,
                                 burnin=500)

        mod = TArma(x)
        order = (2, 1)
        cls.res_ls = mod.fit(order=order)
        cls.res = mod.fit_mle(order=order,
                              start_params=np.r_[cls.res_ls[0], 5, 1],
                              method='nm', disp=False)

        cls.res1_table = np.array(
          [[  0.46157133,  -0.07694534,   0.70051876,  2.88693312,  0.97283396],
           [  0.04957594,   0.04345499,   0.03492473,  0.40854823,  0.05568439],
           [  9.31038915,  -1.7706905 ,  20.05795605,  7.06632146, 17.47049812],
           [  0.        ,   0.07661218,   0.        ,  0.        ,  0.        ],
           [  0.05487968,   0.04213054,   0.03102404,  0.37860956,  0.05228474],
           [  0.04649728,   0.04569133,   0.03990779,  0.44315449,  0.05996759]])

        cls.res1_conf_int = np.array([[ 0.36440426,  0.55873839],
                                   [-0.16211556,  0.00822488],
                                   [ 0.63206754,  0.76896998],
                                   [ 2.08619331,  3.68767294],
                                   [ 0.86369457,  1.08197335]])


        cls.ls_params = np.array([ 0.43393123, -0.08402678,  0.73293058])
        cls.ls_bse = np.array([ 0.0377741 ,  0.03567847,  0.02744488])

class TestArma(CheckTArmaMixin):
    #regression test for TArma

    @classmethod
    def setup_class(cls):

        nobs = 500
        ar = [1, -0.5, 0.1]
        ma = [1, 0.7]
        dist = lambda n: np.random.standard_t(3, size=n)
        np.random.seed(8659567)
        x = arma_generate_sample(ar, ma, nobs, sigma=1, distrvs=dist,
                                 burnin=500)

        mod = Arma(x)
        order = (2, 1)
        cls.res_ls = mod.fit(order=order)
        cls.res = mod.fit_mle(order=order,
                              start_params=np.r_[cls.res_ls[0], 1],
                              method='nm', disp=False)

        cls.res1_table = np.array(
          [[  0.4339072 ,  -0.08402653,   0.73292344,   1.61661128],
           [  0.05854268,   0.05562941,   0.04034178,   0.0511207 ],
           [  7.4118102 ,  -1.51046975,  18.16785075,  31.62341666],
           [  0.        ,   0.1309236 ,   0.        ,   0.        ],
           [  0.06713617,   0.05469138,   0.03785006,   0.1071093 ],
           [  0.05504093,   0.0574849 ,   0.04350945,   0.02510928]])

        cls.res1_conf_int = np.array([[ 0.31916567,  0.54864874],
                               [-0.19305817,  0.0250051 ],
                               [ 0.65385501,  0.81199188],
                               [ 1.51641655,  1.71680602]])

        cls.ls_params = np.array([ 0.43393123, -0.08402678,  0.73293058])
        cls.ls_bse = np.array([ 0.0377741 ,  0.03567847,  0.02744488])
