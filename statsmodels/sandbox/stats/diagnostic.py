# flake8: noqa
import warnings

warnings.warn("statsmodels.sandbox.stats.diagnostic is deprecated. Import from"
              " statsmdoels.stats.diagnostic.", FutureWarning, stacklevel=2)

# collect some imports of verified (at least one example) functions
from statsmodels.stats.diagnostic import (
    acorr_ljungbox, breaks_cusumolsresid, breaks_hansen, CompareCox,
    CompareJ, compare_cox, compare_j, het_breuschpagan,
    HetGoldfeldQuandt, het_goldfeldquandt, het_arch,
    het_white, recursive_olsresiduals, acorr_breusch_godfrey,
    linear_harvey_collier, linear_rainbow, linear_lm,
    spec_white, unitroot_adf)

from statsmodels.stats._lilliefors import (kstest_fit, lilliefors,
                                           kstest_normal,
                                           kstest_exponential)
from statsmodels.stats._adnorm import normal_ad

__all__ = ['acorr_ljungbox', 'breaks_cusumolsresid', 'breaks_hansen',
           'CompareCox', 'CompareJ', 'compare_cox', 'compare_j',
           'het_breuschpagan', 'HetGoldfeldQuandt', 'het_goldfeldquandt',
           'het_arch', 'het_white', 'recursive_olsresiduals',
           'acorr_breusch_godfrey', 'linear_harvey_collier', 'linear_rainbow',
           'linear_lm', 'spec_white', 'unitroot_adf', 'kstest_fit',
           'lilliefors', 'kstest_normal', 'kstest_exponential', 'normal_ad']
