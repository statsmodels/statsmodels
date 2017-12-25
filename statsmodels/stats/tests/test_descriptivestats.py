#!/usr/bin/env python
# -*- coding: utf-8 -*-
import sys

import pytest

import numpy as np
from numpy.testing import assert_almost_equal, assert_equal

from statsmodels.stats.descriptivestats import sign_test, Describe



data1 = np.array([(1,2,'a','aa'),
                  (2,3,'b','bb'),
                  (2,4,'b','cc')],
                 dtype = [('alpha',float), ('beta', int),
                          ('gamma', '|S1'), ('delta', '|S2')])
data2 = np.array([(1,2),
                  (2,3),
                  (2,4)],
                 dtype = [('alpha',float), ('beta', float)])

data3 = np.array([[1,2,4,4],
                  [2,3,3,3],
                  [2,4,4,3]], dtype=float)

data4 = np.array([[1,2,3,4,5,6],
                  [6,5,4,3,2,1],
                  [9,9,9,9,9,9]])


def test_sign_test():
    x = [7.8, 6.6, 6.5, 7.4, 7.3, 7., 6.4, 7.1, 6.7, 7.6, 6.8]
    M, p = sign_test(x, mu0=6.5)
    # from R SIGN.test(x, md=6.5)
    # from R
    assert_almost_equal(p, 0.02148, 5)
    # not from R, we use a different convention
    assert_equal(M, 4)


@pytest.mark.smoke
class TestSimpleTable(object):
    #from statsmodels.iolib.table import SimpleTable, default_txt_fmt

    def test_noperc(self):
        t1 = Describe(data4)
        noperc = ['obs', 'mean', 'std', 'min', 'max', 'ptp', #'mode',  #'var',
                                'median', 'skew', 'uss', 'kurtosis']
        #TODO: mode var raise exception,
        #TODO: percentile writes list in cell (?), huge wide format
        t1.summary(stats=noperc)
        t1.summary()
        t1.summary(orientation='varcols')
        t1.summary(stats=['mean', 'median', 'min', 'max'],
                   orientation=('varcols'))
        t1.summary(stats='all')

    @pytest.mark.xfail(sys.version_info.major >= 3, reason="Not sure of dtype")
    def test_basic_1(self):
        t1 = Describe(data1)
        t1.summary()

    def test_basic_2(self):
        t2 = Describe(data2)
        t2.summary()

    def test_basic_3(self):
        t1 = Describe(data3)
        t1.summary()

    def test_basic_4(self):
        t1 = Describe(data4)
        t1.summary()

    def test_basic_1a(self):
        t1 = Describe(data1)
        t1.summary(stats='basic', columns=['alpha'])

    @pytest.mark.xfail(sys.version_info.major >= 3, reason="Not sure of dtype")
    def test_basic_1b(self):
        t1 = Describe(data1)
        t1.summary(stats='basic', columns='all')

    def test_basic_2a(self):
        t2 = Describe(data2)
        t2.summary(stats='all')

    def test_basic_3(aself):
        t1 = Describe(data3)
        t1.summary(stats='all')

    def test_basic_4a(self):
        t1 = Describe(data4)
        t1.summary(stats='all')
