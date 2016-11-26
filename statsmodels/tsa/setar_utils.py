"""
Self-Exciting Threshold Autoregression Utilities

Author: Chad Fulton
License: BSD

Notes
-----

Joblib requires functions to be defined in a separate module.

"""

import setar_model

def _order_test_bootstrap(sample, delay, threshold_grid_size,
                          null_order, null_ar_order, alt_order, alt_ar_order):
    # Estimate a SETAR model on the simulated sample
    simul_res = setar_model.SETAR(
        sample,
        order=alt_order,
        ar_order=alt_ar_order,
        delay=delay,
        threshold_grid_size=threshold_grid_size
    ).fit()
    # Estimate the null SETAR model on the simulated sample
    simul_null = simul_res._get_model(null_order, null_ar_order)
    
    return simul_res.f_stat(simul_null)