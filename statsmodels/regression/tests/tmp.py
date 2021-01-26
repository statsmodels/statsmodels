import sys
sys.path.insert(0, "/afs/umich.edu/user/k/s/kshedden/statsmodels_fork/statsmodels")

import statsmodels.api as sm
import pandas as pd
import numpy as np
from statsmodels.regression.mixed_linear_model import MixedLM

def test_vcomp_2():
    """
    Simulated data comparison to R
    """

    np.random.seed(6241)
    n = 1600
    exog = np.random.normal(size=(n, 2))
    ex_vc = []
    groups = np.kron(np.arange(n / 16), np.ones(16))

    # Build up the random error vector
    errors = 0

    # The random effects
    exog_re = np.random.normal(size=(n, 2))
    slopes = np.random.normal(size=(n / 16, 2))
    slopes = np.kron(slopes, np.ones((16, 1))) * exog_re
    errors += slopes.sum(1)

    # First variance component
    subgroups1 = np.kron(np.arange(n / 4), np.ones(4))
    errors += np.kron(2*np.random.normal(size=n/4), np.ones(4))

    # Second variance component
    subgroups2 = np.kron(np.arange(n / 2), np.ones(2))
    errors += np.kron(2*np.random.normal(size=n/2), np.ones(2))

    # iid errors
    errors += np.random.normal(size=n)

    endog = exog.sum(1) + errors

    df = pd.DataFrame(index=range(n))
    df["y"] = endog
    df["groups"] = groups
    df["x1"] = exog[:, 0]
    df["x2"] = exog[:, 1]
    df["z1"] = exog_re[:, 0]
    df["z2"] = exog_re[:, 1]
    df["v1"] = subgroups1
    df["v2"] = subgroups2

    # Equivalent model in R:
    # df.to_csv("tst.csv")
    # model = lmer(y ~ x1 + x2 + (0 + z1 + z2 | groups) + (1 | v1) + (1 | v2), df)

    vcf = {"a": "0 + C(v1)", "b": "0 + C(v2)"}
    model1 = MixedLM.from_formula("y ~ x1 + x2", groups=groups, re_formula="0+z1+z2",
                                  vc_formula=vcf, data=df)
    result1 = model1.fit()


import cProfile, pstats
from io import StringIO
pr = cProfile.Profile()
pr.enable()
test_vcomp_2()
pr.disable()
s = StringIO()
sortby = 'cumulative'
ps = pstats.Stats(pr, stream=s).strip_dirs().sort_stats(sortby)
ps.print_stats()
print(s.getvalue())


