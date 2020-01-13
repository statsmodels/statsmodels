import warnings

from statsmodels.stats.diagnostic import (
    CompareCox, CompareJ, HetGoldfeldQuandt, OLS, ResultsStore, acf, acorr_breusch_godfrey,
    acorr_ljungbox, acorr_lm, breaks_cusumolsresid, breaks_hansen, compare_cox, compare_j,
    het_arch, het_breuschpagan, het_goldfeldquandt, het_white, linear_harvey_collier, linear_lm,
    linear_rainbow, recursive_olsresiduals, spec_white
)
from statsmodels.tsa.stattools import adfuller

__all__ = ['CompareCox', 'CompareJ', 'HetGoldfeldQuandt', 'OLS',
           'ResultsStore', 'acf', 'acorr_breusch_godfrey', 'acorr_ljungbox',
           'acorr_lm', 'adfuller', 'breaks_cusumolsresid', 'breaks_hansen',
           'compare_cox', 'compare_j', 'het_arch', 'het_breuschpagan',
           'het_goldfeldquandt', 'het_white', 'linear_harvey_collier',
           'linear_lm', 'linear_rainbow', 'recursive_olsresiduals',
           'spec_white', 'unitroot_adf']


# get the old signature back so the examples work
def unitroot_adf(x, maxlag=None, trendorder=0, autolag='AIC', store=False):
    warnings.warn("unitroot_adf is deprecated and will be removed after 0.11.",
                  FutureWarning)
    trendorder = {0: 'nc', 1: 'c', 2: 'ct', 3: 'ctt'}[trendorder]
    return adfuller(x, maxlag=maxlag, regression=trendorder, autolag=autolag,
                    store=store, regresults=False)


warnings.warn("The statsmodels.sandbox.stats.diagnostic module is deprecated. "
              "Use statsmodels.stats.diagnostic.", DeprecationWarning,
              stacklevel=2)
