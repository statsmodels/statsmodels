


import numpy as np
from scipy import stats, optimize
import statsmodels.stats.proportion as prop
from statsmodels.stats.proportion import (proportion_confint_baker,
                                          accept_binom, _delta_blaker)



alpha = 0.05

ex = 1
if ex == 1:
    n = 10
    x = int(n * 0.4)

    ci_005 = prop.proportion_confint(x, n, method='beta')
    ci_01 = prop.proportion_confint(x, n, alpha=0.1, method='beta')
    print(ci_005)
    print(ci_01)

    low, upp = ci_005
    print(stats.binom.sf(x - 1, n, upp))
    print(stats.binom.cdf(x, n, upp))
    low_exact, upp_exact = prop.proportion_confint(x, n, alpha=0.05, method='beta')
    low_exact2, upp_exact2 = prop.proportion_confint(x, n, alpha=2 * 0.05,
                                                     method='beta')

    pp = np.linspace(low_exact, low_exact2, 11)
    #print(np.column_stack((pp, np.column_stack(accept_binom(x, n, pp)))))

    pp = np.linspace(upp_exact2, upp_exact, 11)
    #print(np.column_stack((pp, np.column_stack(accept_binom(x, n, pp)))))

    low_b, upp_b = 0.150, 0.717
    print(accept_binom(x, n, low_b))
    print(accept_binom(x, n, upp_b))

    optimize.brentq(lambda p_: accept_binom(x, n, p_)[0] - 0.05, upp_exact2, upp_exact)

    print(proportion_confint_baker(4, 10, alpha=0.05)[0])

    print(proportion_confint_baker(2, 10, alpha=0.05)[:2])
    print(proportion_confint_baker(2, 123, alpha=0.05)[:2])


x=5;n=20
print(proportion_confint_baker(5, 20, alpha=0.05)[:2])
pl, pu = prop.proportion_confint(5, 20, alpha=0.05, method='beta')
pl2, pu2 = prop.proportion_confint(5, 20, alpha=2*0.05, method='beta')
print(_delta_blaker(x, n, pu))
