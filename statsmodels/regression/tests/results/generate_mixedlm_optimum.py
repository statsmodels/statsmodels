"""
Generate the data sets of the MixedLM fit tests against the lme4 optimum.

The data are simulated with a random intercept and a random slope for x, and
two further covariates.  After running this script, run mixedlm_optimum.R with R
to update the reference values in mixedlm_optimum_r_results.py.
"""
from pathlib import Path

import numpy as np
import pandas as pd

# case: (seed, sd of the random intercept, sd of the random slope)
CASES = {
    "slope_ml": (10, 0.8, 0.5),
    "slope_reml": (6, 0.2, 0.05),
    "vc_ml": (6, 0.8, 0.5),
    "vc_reml": (10, 0.8, 0.5),
}


def simulate(seed, sd_u, sd_b):
    rng = np.random.default_rng(seed)
    n_groups, n_per_group = 25, 6
    g = np.repeat(np.arange(n_groups), n_per_group)
    x = rng.normal(size=n_groups * n_per_group)
    z = rng.normal(size=n_groups * n_per_group)
    u = rng.normal(size=n_groups) * sd_u
    b = rng.normal(size=n_groups) * sd_b
    y = 1 + 0.5 * x + u[g] + b[g] * x + 0.3 * z + rng.normal(size=g.size)
    return pd.DataFrame({"y": y, "x": x, "z": z, "g": g})


if __name__ == "__main__":
    frames = []
    for case, args in CASES.items():
        frames.append(simulate(*args).assign(case=case))
    data = pd.concat(frames)[["case", "g", "y", "x", "z"]]
    fname = Path(__file__).resolve().parent.joinpath("mixedlm_optimum.csv")
    data.to_csv(fname, index=False, float_format="%.17g", lineterminator="\n")
