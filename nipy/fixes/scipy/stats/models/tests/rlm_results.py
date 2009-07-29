"""
Covariance matrices for RLM tests.

Obtained from SAS.
"""
import numpy as np
### Based on stackloss data ###

### Covariances
def _shift_intercept(arr):
    """
    A convenience function to make the SAS covariance matrix
    compatible with stats.models.rlm covariance
    """
    side = np.sqrt(len(arr))
    arr = np.array(arr).reshape(side,side)
    tmp = np.zeros((side,side))
    tmp[:-1,:-1] = arr[1:,1:]
    tmp[-1,-1] = arr[0,0]
    tmp[-1,:-1] = arr[0,1:]
    tmp[:-1,-1] = arr[1:,0]
    return tmp

huber_h1 = [95.8813, 0.19485, -0.44161, -1.13577, 0.1949, 0.01232, -0.02474,
        -0.00484, -0.4416, -0.02474, 0.09177, 0.00001, -1.1358, -0.00484,
        0.00001, 0.01655]
huber_h1 = _shift_intercept(huber_h1)

huber_h2 = [82.6191, 0.07942, -0.23915, -0.95604, 0.0794, 0.01427, -0.03013,
        -0.00344, -0.2392, -0.03013, 0.10391, -0.00166, -0.9560, -0.00344,
        -0.00166, 0.01392]
huber_h2 = _shift_intercept(huber_h2)

huber_h3 = [70.1633, -0.04533, -0.00790, -0.78618, -0.0453, 0.01656, -0.03608,
        -0.00203, -0.0079, -0.03608,  0.11610, -0.00333, -0.7862, -0.00203,
        -0.00333,  0.01138]
huber_h3 = _shift_intercept(huber_h3)

hampel_h1 = [141.309,  0.28717, -0.65085, -1.67388, 0.287,  0.01816, -0.03646,
        -0.00713, -0.651, -0.03646,  0.13524,  0.00001, -1.674, -0.00713,
        0.00001,  0.02439]
hampel_h1 = _shift_intercept(hampel_h1)

hampel_h2 = [135.248,  0.18207, -0.36884, -1.60217, 0.182, 0.02120, -0.04563,
        -0.00567, -0.369, -0.04563,  0.15860, -0.00290, -1.602, -0.00567,
        -0.00290, 0.02329]
hampel_h2 = _shift_intercept(hampel_h2)

hampel_h3 = [128.921,  0.05409, -0.02445, -1.52732, 0.054,  0.02514, -0.05732,
        -0.00392, -0.024, -0.05732,  0.18871, -0.00652, -1.527, -0.00392,
        -0.00652,  0.02212]
hampel_h3 = _shift_intercept(hampel_h3)

bisquare_h1 = [90.3354,  0.18358, -0.41607, -1.07007, 0.1836, 0.01161,
        -0.02331, -0.00456, -0.4161, -0.02331,  0.08646, 0.00001, -1.0701,
        -0.00456, 0.00001,  0.01559]
bisquare_h1 = _shift_intercept(bisquare_h1)

bisquare_h2 = [67.82521, 0.091288, -0.29038, -0.78124, 0.091288, 0.013849,
        -0.02914, -0.00352, -0.29038, -0.02914, 0.101088, -0.001, -0.78124,
        -0.00352,   -0.001, 0.011766]
bisquare_h2 = _shift_intercept(bisquare_h2)

bisquare_h3 = [48.8983, 0.000442, -0.15919, -0.53523, 0.000442, 0.016113,
        -0.03461, -0.00259, -0.15919, -0.03461, 0.112728, -0.00164, -0.53523,
        -0.00259, -0.00164, 0.008414]
bisquare_h3 = _shift_intercept(bisquare_h3)

class andrews(object):
    andrews_h1 = [91.80527, -0.09171, 0.171716, -1.05244, -0.09171, 0.027999,
            -0.06493, -0.00223, 0.171716, -0.06493, 0.203254,  -0.0071,
            -1.05244, -0.00223,  -0.0071, 0.015584]
    andrews_h1 = _shift_intercept(andrews_h1)

    andrews_h2 = [87.5357, 0.177891, -0.40318, -1.03691, 0.177891,  0.01125,
            -0.02258, -0.00442, -0.40318, -0.02258, 0.083779, 6.481E-6,
            -1.03691, -0.00442, 6.481E-6,  0.01511]
    andrews_h2 = _shift_intercept(andrews_h2)

    andrews_h3 = [66.50472,  0.10489,  -0.3246, -0.76664, 0.10489, 0.012786,
            -0.02651,  -0.0036, -0.3246, -0.02651,  0.09406, -0.00065,
            -0.76664,  -0.0036, -0.00065, 0.011567]
    andrews_h3 = _shift_intercept(andrews_h3)


    def __init__(self):
        self.params = [0.9282, 0.6492, -.1123,-42.2930]
        self.bse = [.1061, .2894, .1229, 9.3561]
        self.scale = 2.2801
        self.weights = None
        self.resid = None
        self.df_model = 3.
        self.df_resid = 17.
        self.bcov_unscaled = []
        self.h1 = self.andrews_h1
        self.h2 = self.andrews_h2
        self.h3 = self.andrews_h3

### Huber's Proposal 2

