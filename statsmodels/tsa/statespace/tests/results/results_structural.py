"""
Results for SARIMAX tests

Results from R, KFAS library using script `test_ucm.R`.
See also Stata time series documentation.

Author: Chad Fulton
License: Simplified-BSD
"""
from numpy import pi

ntrend = {
    'model': {'irregular': True},
    'alt_model': {'level': 'ntrend'},
    'params': [36.74687342],
    'llf': -653.8562525,
    'kwargs': {}
}

dconstant = {
    'model': {'irregular': True, 'level': True},
    'alt_model': {'level': 'dconstant'},
    'params': [2.127438969],
    'llf': -365.5289923,
    'kwargs': {}
}

llevel = {
    'model': {'irregular': True, 'level': True, 'stochastic_level': True},
    'alt_model': {'level': 'llevel'},
    'params': [4.256647886e-06, 1.182078808e-01],
    'llf': -70.97242557,
    'kwargs': {}
}

rwalk = {
    'model': {'level': True, 'stochastic_level': True},
    'alt_model': {'level': 'rwalk'},
    'params': [0.1182174646],
    'llf': -70.96771641,
    'kwargs': {}
}

dtrend = {
    'model': {'irregular': True, 'level': True, 'trend': True},
    'alt_model': {'level': 'dtrend'},
    'params': [2.134137554],
    'llf': -370.7758666,
    'kwargs': {}
}

lldtrend = {
    'model': {'irregular': True, 'level': True, 'stochastic_level': True,
              'trend': True},
    'alt_model': {'level': 'lldtrend'},
    'params': [4.457592057e-06, 1.184455029e-01],
    'llf': -73.47291031,
    'kwargs': {}
}

rwdrift = {
    'model': {'level': True, 'stochastic_level': True, 'trend': True},
    'alt_model': {'level': 'rwdrift'},
    'params': [0.1184499547],
    'llf': -73.46798576,
    'kwargs': {}
}

lltrend = {
    'model': {'irregular': True, 'level': True, 'stochastic_level': True,
              'trend': True, 'stochastic_trend': True},
    'alt_model': {'level': 'lltrend'},
    'params': [1.339852549e-06, 1.008704925e-02, 6.091760810e-02],
    'llf': -31.15640107,
    'kwargs': {}
}

strend = {
    'model': {'irregular': True, 'level': True, 'trend': True,
              'stochastic_trend': True},
    'alt_model': {'level': 'strend'},
    'params': [0.0008824099119, 0.0753064234342],
    'llf': -31.92261408,
    'kwargs': {}
}

rtrend = {
    'model': {'level': True, 'trend': True, 'stochastic_trend': True},
    'alt_model': {'level': 'rtrend'},
    'params': [0.08054724989],
    'llf': -32.05607557,
    'kwargs': {}
}

cycle = {
    'model': {'irregular': True, 'cycle': True, 'stochastic_cycle': True},
    'params': [37.57197224, 0.1, 2*pi/10],
    'llf': -672.3102588,
    'kwargs': {
        # Required due to the way KFAS estimated loglikelihood which P1inf is
        # set in the R code
        'loglikelihood_burn': 0
    }
}

seasonal = {
    'model': {'irregular': True, 'seasonal': 4},
    'params': [38.1704278, 0.1],
    'llf': -655.3337155,
    'kwargs': {}
}

reg = {
    # Note: The test needs to fill in exog=np.log(dta['realgdp'])
    'model': {'irregular': True, 'exog': True, 'mle_regression': False},
    'alt_model': {'level': 'ntrend', 'exog': True, 'mle_regression': False},
    'params': [2.215447924],
    'llf': -379.6233483,
    'kwargs': {
        # Required due to the way KFAS estimated loglikelihood which P1inf is
        # set in the R code
        'loglikelihood_burn': 0
    }
}

rtrend_ar1 = {
    'model': {'level': True, 'trend': True, 'stochastic_trend': True,
              'autoregressive': 1},
    'alt_model': {'level': 'rtrend', 'autoregressive': 1},
    'params': [0.0609, 0.0097, 0.9592],
    'llf': -31.15629379,
    'kwargs': {}
}

lltrend_cycle_seasonal_reg_ar1 = {
    # Note: The test needs to fill in exog=np.log(dta['realgdp'])
    'model': {'irregular': True, 'level': True, 'stochastic_level': True,
              'trend': True, 'stochastic_trend': True, 'cycle': True,
              'stochastic_cycle': True, 'seasonal': 4, 'autoregressive': 1,
              'exog': True, 'mle_regression': False},
    'alt_model': {'level': 'lltrend', 'autoregressive': 1, 'cycle': True,
                  'stochastic_cycle': True, 'seasonal': 4, 'autoregressive': 1,
                  'exog': True, 'mle_regression': False},
    'params': [0.0001, 0.01, 0.06, 0.0001, 0.0001, 0.1, 2*pi / 10, 0.2],
    'llf': -168.5258709,
    'kwargs': {
        # Required due to the way KFAS estimated loglikelihood which P1inf is
        # set in the R code
        'loglikelihood_burn': 0
    }
}
