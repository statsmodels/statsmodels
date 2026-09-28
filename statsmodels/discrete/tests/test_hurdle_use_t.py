import numpy as np
from numpy.testing import assert_allclose
import pytest
from scipy import stats

from statsmodels.discrete.truncated_model import HurdleCountModel
from statsmodels.sandbox.regression.tests.test_gmm_poisson import DATA


@pytest.mark.parametrize("use_t", [False, True])
def test_hurdle_fit_uses_requested_reference_distribution(use_t):
    endog = DATA["docvis"]
    exog = DATA[["const", "aget", "totchr"]]
    res = HurdleCountModel(endog, exog).fit(
        method="newton", maxiter=300, use_t=use_t
    )

    assert res.use_t is use_t
    dist = stats.t(res.df_resid) if use_t else stats.norm
    assert_allclose(res.pvalues, 2 * dist.sf(np.abs(res.tvalues)))
    crit = dist.ppf(0.975)
    expected_ci = np.column_stack(
        (res.params - crit * res.bse, res.params + crit * res.bse)
    )
    assert_allclose(res.conf_int(), expected_ci)
