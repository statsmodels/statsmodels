"""
Deprecation warnings for estimator classes with no test coverage and no
internal callers, found while auditing public-API test coverage. Each of
these emits FutureWarning on construction; see the individual warning
messages for rationale and the removal timeline.
"""
import numpy as np
import pytest


def test_nonlinearls_deprecated():
    from statsmodels.miscmodels.nonlinls import NonlinearLS

    with pytest.warns(FutureWarning, match="NonlinearLS is deprecated"):
        NonlinearLS()


def test_mlegls_deprecated():
    from statsmodels.miscmodels.try_mlecov import MLEGLS

    endog = np.arange(10.0)
    with pytest.warns(FutureWarning, match="MLEGLS is deprecated"):
        MLEGLS(endog)


def test_tsmlemodel_deprecated():
    from statsmodels.tsa.mlemodel import TSMLEModel

    endog = np.arange(10.0)
    with pytest.warns(FutureWarning, match="TSMLEModel is deprecated"):
        TSMLEModel(endog)


def test_glshet_deprecated():
    from statsmodels.regression.feasible_gls import GLSHet

    rs = np.random.RandomState(20260822)
    n = 30
    endog = rs.standard_normal(n)
    exog = np.column_stack([np.ones(n), rs.standard_normal(n)])
    exog_var = np.column_stack([np.ones(n), rs.standard_normal(n)])
    with pytest.warns(FutureWarning, match="GLSHet is deprecated"):
        GLSHet(endog, exog, exog_var=exog_var)


def test_glshet2_deprecated():
    from statsmodels.regression.feasible_gls import GLSHet2

    rs = np.random.RandomState(20260823)
    n = 30
    endog = rs.standard_normal(n)
    exog = np.column_stack([np.ones(n), rs.standard_normal(n)])
    exog_var = np.column_stack([np.ones(n), rs.standard_normal(n)])
    with pytest.warns(FutureWarning, match="GLSHet2 is deprecated"):
        GLSHet2(endog, exog, exog_var)


def test_tsadescriptive_deprecated():
    from statsmodels.tsa.descriptivestats import TsaDescriptive

    with pytest.warns(FutureWarning, match="TsaDescriptive is deprecated"):
        TsaDescriptive(np.arange(10.0))


def test_var_deprecated():
    from statsmodels.tsa.varma_process import _Var

    rs = np.random.RandomState(20260824)
    y = rs.standard_normal((50, 2))
    with pytest.warns(FutureWarning, match="_Var is deprecated"):
        _Var(y)


def test_smoothers_lowess_old_deprecated():
    from statsmodels.nonparametric.smoothers_lowess_old import lowess

    rs = np.random.RandomState(20260825)
    x = rs.standard_normal(30)
    y = x + rs.standard_normal(30) * 0.1
    with pytest.warns(FutureWarning, match="smoothers_lowess_old.lowess is deprecated"):
        lowess(y, x)
