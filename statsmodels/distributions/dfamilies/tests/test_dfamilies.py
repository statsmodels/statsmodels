# -*- coding: utf-8 -*-
"""
Created on Tue Oct 12 19:06:31 2021

Author: Josef Perktod
License: BSD-3
"""

import numpy as np
from numpy.testing import assert_allclose
import pytest

from statsmodels.distributions.dfamilies._continuous import (
    Gaussian, StudentT, JohnsonSU,
    )

from statsmodels.distributions.dfamilies._discrete import (
    BetaBinomialScale, BetaBinomialPrecision, BetaBinomialDispersion,
    BetaBinomialStd,
    )

from statsmodels.distributions.dfamilies._restricted import (
    BetaMP, WeibullMin,
    )


all_fam = [
    (Gaussian, (1, 0.5), {}),
    (StudentT, (1, 0.5, 5), {}),
    (JohnsonSU, (1, 0.5, 0.5, 2), {}),
    (BetaBinomialStd, (5, 7), {"n_trials": 10}),
    (BetaBinomialScale, (0.55, 0.1), {"n_trials": 10}),
    (BetaBinomialPrecision, (0.45, 2), {"n_trials": 10}),
    (BetaBinomialDispersion, (0.45, 0.5), {"n_trials": 10}),
    (BetaMP, (0.5, 5), {}),
    (WeibullMin, (0.75, 2), {}),
    ]


@pytest.mark.parametrize("case", all_fam)
def test_dfamilies_basic(case):
    fam, args, dkwds = case
    if fam.domain.startswith("real"):
        y = np.array([0.5, 1, 1.5])
    elif fam.domain == "ui":
        y = np.array([0.25, 0.5, 0.75])
    elif fam.domain.startswith("int"):
        y = np.arange(5)
    else:
        # pick something
        y = np.array([0.5, 1, 1.5])

    fam = fam()
    logpdf = fam.loglike_obs(y, *args, **dkwds)
    distr = fam.get_distribution(*args, **dkwds)
    if fam.domain.startswith("int"):
        logpdf2 = distr.logpmf(y)
    else:
        logpdf2 = distr.logpdf(y)
    assert_allclose(logpdf, logpdf2, rtol=1e-10)

    pdf = fam.pdf(y, *args, **dkwds)
    if fam.domain.startswith("int"):
        pdf2 = distr.pmf(y)
    else:
        pdf2 = distr.pdf(y)
    assert_allclose(pdf, pdf2, rtol=1e-10)
