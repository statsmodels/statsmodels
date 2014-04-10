# -*- coding: utf-8 -*-
"""
Created on Mon Oct 31 15:26:06 2011

Author: Chris Jordan Squire

extracted from test suite by josef-pktd
"""

from __future__ import print_function
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm

lowess = sm.nonparametric.lowess

# this is just to check direct import
import statsmodels.nonparametric.smoothers_lowess
statsmodels.nonparametric.smoothers_lowess.lowess

x = np.arange(20.)

#standard normal noise
noise = np.array([-0.76741118, -0.30754369,
                    0.39950921, -0.46352422, -1.67081778,
                    0.6595567 ,  0.66367639, -2.04388585,
                    0.8123281 ,  1.45977518,
                    1.21428038,  1.29296866,  0.78028477,
                    -0.2402853 , -0.21721302,
                    0.24549405,  0.25987014, -0.90709034,
                    -1.45688216, -0.31780505])
y = x + noise

expected_lowess = np.array([[  0.        ,  -0.58337912],
                           [  1.        ,   0.61951246],
                           [  2.        ,   1.82221628],
                           [  3.        ,   3.02536876],
                           [  4.        ,   4.22667951],
                           [  5.        ,   5.42387723],
                           [  6.        ,   6.60834945],
                           [  7.        ,   7.7797691 ],
                           [  8.        ,   8.91824348],
                           [  9.        ,   9.94997506],
                           [ 10.        ,  10.89697569],
                           [ 11.        ,  11.78746276],
                           [ 12.        ,  12.62356492],
                           [ 13.        ,  13.41538492],
                           [ 14.        ,  14.15745254],
                           [ 15.        ,  14.92343948],
                           [ 16.        ,  15.70019862],
                           [ 17.        ,  16.48167846],
                           [ 18.        ,  17.26380699],
                           [ 19.        ,  18.0466769 ]])

actual_lowess = lowess(y, x)
print(actual_lowess)
print(np.max(np.abs(actual_lowess-expected_lowess)))

plt.plot(y, 'o')
plt.plot(actual_lowess[:,1])
plt.plot(expected_lowess[:,1])

import os.path
import statsmodels.nonparametric.tests.results
rpath = os.path.split(statsmodels.nonparametric.tests.results.__file__)[0]
rfile = os.path.join(rpath, 'test_lowess_frac.csv')
test_data = np.genfromtxt(open(rfile, 'rb'),
                                  delimiter = ',', names = True)
expected_lowess_23 = np.array([test_data['x'], test_data['out_2_3']]).T
expected_lowess_15 = np.array([test_data['x'], test_data['out_1_5']]).T

actual_lowess_23 = lowess(test_data['y'], test_data['x'] ,frac = 2./3)
actual_lowess_15 = lowess(test_data['y'], test_data['x'] ,frac = 1./5)

#plt.show()
