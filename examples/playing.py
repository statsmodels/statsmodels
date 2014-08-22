import numpy as np
from statsmodels.sandbox.stats.multicomp import MultiComparison

# data = range(10)
# groups = [1]*10
# mod = MultiComparison(np.array(data), groups, group_order=sorted(set(groups)))
# a = mod.tukeyhsd(alpha=0.01)


# data = [1]*1
# groups = [1]*1
# mod = MultiComparison(np.array(data), groups, group_order=sorted(set(groups)))
# a = mod.tukeyhsd(alpha=0.01)

data = [1]*10
groups = [1, 2]*4
mod = MultiComparison(np.array(data), groups, group_order=sorted(set(groups)))
a = mod.tukeyhsd(alpha=0.01)