"""Reproduce the AIPW ATET/ATC point-estimate comparison with DoubleML 0.11.4.

Requires DoubleML and scikit-learn in addition to statsmodels. Run from the
repository root with::

    python -m statsmodels.treatment.tests.results.teffects_doubleml

This is a reference-generation script, not a test dependency. The WLS comparison
uses statsmodels outcome predictions and therefore checks only the AIPW score.
"""

from pathlib import Path

import doubleml
import numpy as np
from numpy.testing import assert_allclose
import pandas as pd
import sklearn
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from statsmodels.discrete.discrete_model import Logit
from statsmodels.regression.linear_model import OLS
from statsmodels.treatment.treatment_effects import TreatmentEffect

from .results_teffects import results_aipw_atet_dml


def main():
    data = pd.read_csv(Path(__file__).with_name("cataneo2.csv"))
    names = "prenatal1_ mmarried_ mage mage2 fbaby_ medu".split()
    x = data[names].to_numpy()
    y = data["bweight"].to_numpy()
    d = data["mbsmoke_"].to_numpy()
    formula = " + ".join(names)
    selection = Logit.from_formula("mbsmoke_ ~ " + formula, data).fit(disp=0)
    model = OLS.from_formula("bweight ~ " + formula, data)
    teff = TreatmentEffect(model, d, results_select=selection, ps_bounds=(0.01, 0.99))
    sample = np.arange(len(d))
    print(f"DoubleML {doubleml.__version__}; scikit-learn {sklearn.__version__}")
    print("method target DoubleML statsmodels relative_difference")
    for method in ["aipw", "aipw_wls"]:
        for group, target in [(1, "att"), (0, "atc")]:
            estimate = getattr(teff, method)(return_results=False, effect_group=group)[
                0
            ]
            treatment = d if group == 1 else 1 - d
            dml_data = doubleml.DoubleMLData.from_arrays(x, y, treatment)
            propensity = make_pipeline(
                StandardScaler(),
                LogisticRegression(C=np.inf, solver="lbfgs", tol=1e-12, max_iter=10000),
            )
            fit = doubleml.DoubleMLIRM(
                dml_data,
                LinearRegression(),
                propensity,
                score="ATTE",
                n_folds=1,
                draw_sample_splitting=False,
                normalize_ipw=False,
                trimming_rule="truncate",
                trimming_threshold=0.01,
            )
            fit.set_sample_splitting((sample, sample))
            external = None
            if method == "aipw_wls":
                g0 = teff.results_ipwwls0.predict(model.exog)
                g1 = teff.results_ipwwls1.predict(model.exog)
                p = selection.predict()
                if group == 0:
                    g0, g1, p = g1, g0, 1 - p
                external = {
                    dml_data.d_cols[0]: {
                        "ml_g0": g0[:, None],
                        "ml_g1": g1[:, None],
                        "ml_m": p[:, None],
                    }
                }
            fit.fit(external_predictions=external)
            reference = fit.coef[0] if group == 1 else -fit.coef[0]
            assert_allclose(
                reference, results_aipw_atet_dml[method + "_" + target], rtol=1e-7
            )
            assert_allclose(estimate, reference, rtol=1e-7)
            relative = abs((estimate - reference) / reference)
            print(f"{method} {target} {reference:.9f} {estimate:.9f} {relative:.2g}")


if __name__ == "__main__":
    main()
