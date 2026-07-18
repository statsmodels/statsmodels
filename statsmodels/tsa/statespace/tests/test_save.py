"""
Tests of save / load / remove_data state space functionality.
"""

import os
import pickle
import tempfile

from numpy.testing import assert_allclose
import pytest

from statsmodels import datasets
from statsmodels.tsa.statespace import (
    dynamic_factor,
    sarimax,
    structural,
    varmax,
)

current_path = os.path.dirname(os.path.abspath(__file__))
macrodata = datasets.macrodata.load_pandas().data


def temp_filename():
    _, filename = tempfile.mkstemp()
    return filename


def test_sarimax():
    mod = sarimax.SARIMAX(macrodata["realgdp"].values, order=(4, 1, 0))
    res = mod.smooth(mod.start_params)
    res.summary()
    file_name = temp_filename()
    res.save(file_name)
    res2 = sarimax.SARIMAXResults.load(file_name)
    assert_allclose(res.params, res2.params)
    assert_allclose(res.bse, res2.bse)
    assert_allclose(res.llf, res2.llf)


# GH7527
@pytest.mark.parametrize("order", [(4, 1, 0), (0, 1, 4), (0, 2, 0)])
def test_sarimax_save_remove_data(order):
    mod = sarimax.SARIMAX(macrodata["realgdp"].values, order=order)
    res = mod.smooth(mod.start_params)
    res.summary()
    file_name = temp_filename()
    res.save(file_name, remove_data=True)
    res2 = sarimax.SARIMAXResults.load(file_name)
    assert_allclose(res.params, res2.params)
    assert_allclose(res.bse, res2.bse)
    assert_allclose(res.llf, res2.llf)


def test_sarimax_pickle():
    mod = sarimax.SARIMAX(macrodata["realgdp"].values, order=(4, 1, 0))
    pkl_mod = pickle.loads(pickle.dumps(mod))

    res = mod.smooth(mod.start_params)
    pkl_res = pkl_mod.smooth(mod.start_params)

    assert_allclose(res.params, pkl_res.params)
    assert_allclose(res.bse, pkl_res.bse)
    assert_allclose(res.llf, pkl_res.llf)


def test_structural():
    mod = structural.UnobservedComponents(macrodata["realgdp"].values, "llevel")
    res = mod.smooth(mod.start_params)
    res.summary()
    file_name = temp_filename()
    res.save(file_name)
    res2 = structural.UnobservedComponentsResults.load(file_name)
    assert_allclose(res.params, res2.params)
    assert_allclose(res.bse, res2.bse)
    assert_allclose(res.llf, res2.llf)


def test_structural_pickle():
    mod = structural.UnobservedComponents(macrodata["realgdp"].values, "llevel")
    pkl_mod = pickle.loads(pickle.dumps(mod))

    res = mod.smooth(mod.start_params)
    pkl_res = pkl_mod.smooth(pkl_mod.start_params)

    assert_allclose(res.params, pkl_res.params)
    assert_allclose(res.bse, pkl_res.bse)
    assert_allclose(res.llf, pkl_res.llf)


def test_dynamic_factor():
    mod = dynamic_factor.DynamicFactor(
        macrodata[["realgdp", "realcons"]].diff().iloc[1:].values,
        k_factors=1,
        factor_order=1,
    )
    res = mod.smooth(mod.start_params)
    res.summary()
    file_name = temp_filename()
    res.save(file_name)
    res2 = dynamic_factor.DynamicFactorResults.load(file_name)
    assert_allclose(res.params, res2.params)
    assert_allclose(res.bse, res2.bse)
    assert_allclose(res.llf, res2.llf)


def test_dynamic_factor_pickle():
    mod = varmax.VARMAX(
        macrodata[["realgdp", "realcons"]].diff().iloc[1:].values, order=(1, 0)
    )
    pkl_mod = pickle.loads(pickle.dumps(mod))

    res = mod.smooth(mod.start_params)
    pkl_res = pkl_mod.smooth(mod.start_params)

    assert_allclose(res.params, pkl_res.params)
    assert_allclose(res.bse, pkl_res.bse)
    assert_allclose(res.llf, pkl_res.llf)

    res.summary()
    file_name = temp_filename()
    res.save(file_name)
    res2 = varmax.VARMAXResults.load(file_name)
    assert_allclose(res.params, res2.params)
    assert_allclose(res.bse, res2.bse)
    assert_allclose(res.llf, res2.llf)


def test_varmax():
    mod = varmax.VARMAX(
        macrodata[["realgdp", "realcons"]].diff().iloc[1:].values, order=(1, 0)
    )
    res = mod.smooth(mod.start_params)
    res.summary()
    file_name = temp_filename()
    res.save(file_name)
    res2 = varmax.VARMAXResults.load(file_name)
    assert_allclose(res.params, res2.params)
    assert_allclose(res.bse, res2.bse)
    assert_allclose(res.llf, res2.llf)


def test_varmax_pickle():
    mod = varmax.VARMAX(
        macrodata[["realgdp", "realcons"]].diff().iloc[1:].values, order=(1, 0)
    )
    res = mod.smooth(mod.start_params)

    res.summary()
    file_name = temp_filename()
    res.save(file_name)
    res2 = varmax.VARMAXResults.load(file_name)
    assert_allclose(res.params, res2.params)
    assert_allclose(res.bse, res2.bse)
    assert_allclose(res.llf, res2.llf)


def test_existing_pickle():
    pkl_file = os.path.join(current_path, "results", "sm-0.9-sarimax.pkl")
    loaded = sarimax.SARIMAXResults.load(pkl_file)
    assert isinstance(loaded, sarimax.SARIMAXResultsWrapper)
