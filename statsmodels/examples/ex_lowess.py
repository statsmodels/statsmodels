# -*- coding: utf-8 -*-
"""
Created on Mon Oct 31 15:26:06 2011

Author: Chris Jordan Squire

extracted from test suite by josef-pktd
"""

import numpy as np
from numpy.testing import assert_almost_equal
from statsmodels.nonparametric import lowess as lo

        
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

actual_lowess = lo.lowess(y,x)
print actual_lowess
print np.max(np.abs(actual_lowess-expected_lowess))
res0 = lo._lowess_initial_fit(x, y, 4, len(x))

doplot = 1
if doplot:
    import matplotlib.pyplot as plt
    plt.plot(y, 'o')
    plt.plot(actual_lowess[:,1])
    plt.plot(expected_lowess[:,1])
    plt.plot(res0[0])
    plt.show()
