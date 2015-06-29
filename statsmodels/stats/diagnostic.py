#collect some imports of verified (at least one example) functions
from statsmodels.sandbox.stats.diagnostic import (
    acorr_ljungbox, breaks_cusumolsresid, breaks_hansen, CompareCox, CompareJ,
    compare_cox, compare_j, het_breuschpagan, HetGoldfeldQuandt,
    het_goldfeldquandt, het_arch,
    het_white, recursive_olsresiduals, acorr_breusch_godfrey,
    linear_harvey_collier, linear_rainbow, linear_lm,
    unitroot_adf,
    het_breushpagan, acorr_breush_godfrey  # deprecated because of misspelling
    )

from ._lilliefors import kstest_normal, lilliefors, lillifors # lillifors is deprecated
from ._adnorm import normal_ad
