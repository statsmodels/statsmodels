
import numpy as np
from scipy import stats
import scikits.statsmodels.sandbox.stats.stats_extras as smstats

poissrvs = stats.poisson.rvs(0.6, size = 200)

freq, expfreq, histsupp = smstats.gof_binning_discrete(poissrvs, stats.poisson, (0.6,), nsupp=20)
(chi2val, pval) = stats.chisquare(freq, expfreq)
print chi2val, pval

print smstats.gof_chisquare_discrete(stats.poisson, (0.6,), poissrvs, 0.05,
                                     'Poisson')
