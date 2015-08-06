"""
Results for VARMAX tests

Results from Stata using script `test_varmax_stata.do`.
See also Stata time series documentation, in particular `dfactor`.

Data from:

http://www.jmulti.de/download/datasets/e1.dat

Author: Chad Fulton
License: Simplified-BSD
"""

lutkepohl_dfm = {
    'params': [
        .0063728, .00660177, .00636009,   # Factor loadings
        .00203899, .00009016, .00005348,    # Idiosyncratic variances
        .33101874, .63927819,             # Factor transitions
    ],
    'bse_oim': [
        .002006,  .0012514, .0012128,   # Factor loadings
        .0003359, .0000184, .0000141,   # Idiosyncratic variances
        .1196637, .1218577,             # Factor transitions
    ],
    'loglike': 594.0902026190786,
    'aic': -1172.18,
    'bic': -1153.641,
}
