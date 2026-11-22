import numpy as np
from numpy.testing import (
    assert_allclose,
    assert_array_equal,
)
import pandas as pd
import pytest
from scipy import linalg, special, stats

from statsmodels.datasets import danish_data
import statsmodels.datasets.interest_inflation.data as e6
from statsmodels.tools.sm_exceptions import MissingDataError
from statsmodels.tools.testing import assert_equal
from statsmodels.tsa.vector_ar.hypothesis_test_results import (
    CausalityTestResults,
)
from statsmodels.tsa.vector_ar.tests.JMulTi_results.parse_jmulti_vecm_output import (
    dt_s_tup_to_string,
    load_results_jmulti,
    sublists,
)
from statsmodels.tsa.vector_ar.tests.results import results_vecm_urca
from statsmodels.tsa.vector_ar.util import seasonal_dummies
from statsmodels.tsa.vector_ar.var_model import VAR, VARProcess
from statsmodels.tsa.vector_ar.vecm import (
    VECM,
    coint_johansen,
    select_coint_rank,
    select_order,
)

pytestmark = pytest.mark.filterwarnings("ignore:in the future np.array_split")


class DataSet:
    """
    A class for representing the data in a data module.

    Parameters
    ----------
    data_module : module
        A module contained in the statsmodels/datasets directory.
    n_seasons : list
        A list of integers. Each int represents a number of seasons to test the
        model with.
    first_season : list
        A list of integers. Each int corresponds to the int with the same index
        in the n_seasons list and declares to which season the first data entry
        belongs.
    variable_names : list
        A list of strings. Each string names a variable.
    """

    def __init__(self, data_module, n_seasons, first_season, variable_names):
        self.data_module = data_module
        self.seasons = n_seasons
        self.first_season = first_season
        self.dt_s_list = [
            (det, s, self.first_season[i])
            for det in deterministic_terms_list
            for i, s in enumerate(self.seasons)
        ]
        self.variable_names = variable_names

    def __str__(self):
        return self.data_module.__str__()


atol = 0.0005  # absolute tolerance
rtol = 0  # relative tolerance
datasets = []
data = {}
results_ref = {}
results_sm = {}
results_sm_exog = {}
results_sm_exog_coint = {}
coint_rank = 1

deterministic_terms_list = ["nc", "co", "colo", "ci", "cili"]


def load_data(dataset, data_dict):
    dtset = dataset.data_module.load_pandas()
    variables = dataset.variable_names
    loaded = dtset.data[variables].astype(float).values
    data_dict[dataset] = loaded.reshape((-1, len(variables)))


def load_results_statsmodels(dataset):
    results_per_deterministic_terms = dict.fromkeys(dataset.dt_s_list)

    for dt_s_tup in dataset.dt_s_list:
        model = VECM(
            data[dataset],
            k_ar_diff=3,
            coint_rank=coint_rank,
            deterministic=dt_s_tup[0],
            seasons=dt_s_tup[1],
            first_season=dt_s_tup[2],
        )
        results_per_deterministic_terms[dt_s_tup] = model.fit(method="ml")
    return results_per_deterministic_terms


def load_results_statsmodels_exog(dataset):
    """
    Load data with seasonal terms in `exog`.

    Same as load_results_statsmodels() except that the seasonal term is
    provided to :class:`VECM`'s `__init__()` method via the `eoxg` parameter.
    This is to check whether the same results are produced no matter whether
    `exog` or `seasons` is being used.

    Parameters
    ----------
    dataset : DataSet
    """
    results_per_deterministic_terms = dict.fromkeys(dataset.dt_s_list)
    endog = data[dataset]
    for dt_s_tup in dataset.dt_s_list:
        det_string = dt_s_tup[0]
        seasons = dt_s_tup[1]
        first_season = dt_s_tup[2]
        if seasons == 0:
            exog = None
        else:
            exog = seasonal_dummies(
                seasons, len(data[dataset]), first_season, centered=True
            )
            if "lo" in dt_s_tup[0]:
                exog = np.hstack((exog, 1 + np.arange(len(endog)).reshape(-1, 1)))
                # remove "lo" since it's now already in exog.
                det_string = det_string[:-2]
        model = VECM(
            endog,
            exog,
            k_ar_diff=3,
            coint_rank=coint_rank,
            deterministic=det_string,
        )
        results_per_deterministic_terms[dt_s_tup] = model.fit(method="ml")
    return results_per_deterministic_terms


def load_results_statsmodels_exog_coint(dataset):
    """
    Load data with deterministic terms in `exog_coint`.

    Same as load_results_statsmodels() except that deterministic terms inside
    the cointegration relation are provided to :class:`VECM`'s `__init__()`
    method via the `eoxg_coint` parameter. This is to check whether the same
    results are produced no matter whether `exog_coint` or the `deterministic`
    argument is being used.

    Parameters
    ----------
    dataset : DataSet
    """
    results_per_deterministic_terms = dict.fromkeys(dataset.dt_s_list)
    endog = data[dataset]
    for dt_s_tup in dataset.dt_s_list:
        det_string = dt_s_tup[0]

        if "ci" not in det_string and "li" not in det_string:
            exog_coint = None
        else:
            exog_coint = []
            if "li" in det_string:
                exog_coint.append(1 + np.arange(len(endog)).reshape(-1, 1))
                det_string = det_string[:-2]
            if "ci" in det_string:
                exog_coint.append(np.ones(len(endog)).reshape(-1, 1))
                det_string = det_string[:-2]
            # reversing (such that constant is first and linear is second)
            exog_coint = exog_coint[::-1]
            exog_coint = np.hstack(exog_coint)
        model = VECM(
            endog,
            exog=None,
            exog_coint=exog_coint,
            k_ar_diff=3,
            coint_rank=coint_rank,
            deterministic=det_string,
            seasons=dt_s_tup[1],
            first_season=dt_s_tup[2],
        )
        results_per_deterministic_terms[dt_s_tup] = model.fit(method="ml")
    return results_per_deterministic_terms


def build_err_msg(ds, dt_s, parameter_str):
    dt = dt_s_tup_to_string(dt_s)
    seasons = dt_s[1]
    err_msg = "Error in " + parameter_str + " for:\n"
    err_msg += "- Dataset: " + ds.__str__() + "\n"
    err_msg += "- Deterministic terms: "
    err_msg += dt_s[0] if dt != "n" else "no det. terms"
    if seasons > 0:
        err_msg += ", seasons: " + str(seasons)
    return err_msg


def setup():
    datasets.append(
        DataSet(e6, [0, 4], [0, 1], ["Dp", "R"]),
        # DataSet(...) TODO: append more data sets for more test cases.
    )

    for ds in datasets:
        load_data(ds, data)
        results_ref[ds] = load_results_jmulti(ds)
        results_sm[ds] = load_results_statsmodels(ds)
        results_sm_exog[ds] = load_results_statsmodels_exog(ds)
        results_sm_exog_coint[ds] = load_results_statsmodels_exog_coint(ds)


setup()


def test_ml_gamma():
    for ds in datasets:
        for dt in ds.dt_s_list:
            exog = results_sm_exog[ds][dt].exog is not None
            exog_coint = results_sm_exog_coint[ds][dt].exog_coint is not None

            # estimated parameter vector
            err_msg = build_err_msg(ds, dt, "Gamma")
            obtained = results_sm[ds][dt].gamma
            obtained_exog = results_sm_exog[ds][dt].gamma
            obtained_exog_coint = results_sm_exog_coint[ds][dt].gamma
            desired = results_ref[ds][dt]["est"]["Gamma"]
            assert_allclose(obtained, desired, rtol, atol, False, err_msg)
            if exog:
                assert_equal(obtained_exog, obtained, "WITH EXOG: " + err_msg)
            if exog_coint:
                assert_equal(
                    obtained_exog_coint,
                    obtained,
                    "WITH EXOG_COINT: " + err_msg,
                )

            # standard errors
            obt = results_sm[ds][dt].stderr_gamma
            obt_exog = results_sm_exog[ds][dt].stderr_gamma
            obt_exog_coint = results_sm_exog_coint[ds][dt].stderr_gamma
            des = results_ref[ds][dt]["se"]["Gamma"]
            assert_allclose(obt, des, rtol, atol, False, "STANDARD ERRORS\n" + err_msg)
            if exog:
                assert_equal(obt_exog, obt, "WITH EXOG: " + err_msg)
            if exog_coint:
                assert_equal(obt_exog_coint, obt, "WITH EXOG_COINT: " + err_msg)
            # t-values
            obt = results_sm[ds][dt].tvalues_gamma
            obt_exog = results_sm_exog[ds][dt].tvalues_gamma
            obt_exog_coint = results_sm_exog_coint[ds][dt].tvalues_gamma
            des = results_ref[ds][dt]["t"]["Gamma"]
            assert_allclose(obt, des, rtol, atol, False, "t-VALUES\n" + err_msg)
            if exog:
                assert_equal(obt_exog, obt, "WITH EXOG: " + err_msg)
            if exog_coint:
                assert_equal(obt_exog_coint, obt, "WITH EXOG_COINT: " + err_msg)
            # p-values
            obt = results_sm[ds][dt].pvalues_gamma
            obt_exog = results_sm_exog[ds][dt].pvalues_gamma
            obt_exog_coint = results_sm_exog_coint[ds][dt].pvalues_gamma
            des = results_ref[ds][dt]["p"]["Gamma"]
            assert_allclose(obt, des, rtol, atol, False, "p-VALUES\n" + err_msg)
            if exog:
                assert_equal(obt_exog, obt, "WITH EXOG: " + err_msg)
            if exog_coint:
                assert_equal(obt_exog_coint, obt, "WITH EXOG_COINT: " + err_msg)


def test_ml_alpha():
    for ds in datasets:
        for dt in ds.dt_s_list:
            exog = results_sm_exog[ds][dt].exog is not None
            exog_coint = results_sm_exog_coint[ds][dt].exog_coint is not None

            err_msg = build_err_msg(ds, dt, "alpha")
            obtained = results_sm[ds][dt].alpha
            obtained_exog = results_sm_exog[ds][dt].alpha
            obtained_exog_coint = results_sm_exog_coint[ds][dt].alpha
            desired = results_ref[ds][dt]["est"]["alpha"]
            assert_allclose(obtained, desired, rtol, atol, False, err_msg)
            if exog:
                assert_equal(obtained_exog, obtained, "WITH EXOG: " + err_msg)
            if exog_coint:
                assert_equal(
                    obtained_exog_coint,
                    obtained,
                    "WITH EXOG_COINT: " + err_msg,
                )

            # standard errors
            obt = results_sm[ds][dt].stderr_alpha
            obt_exog = results_sm_exog[ds][dt].stderr_alpha
            obt_exog_coint = results_sm_exog_coint[ds][dt].stderr_alpha
            des = results_ref[ds][dt]["se"]["alpha"]
            assert_allclose(obt, des, rtol, atol, False, "STANDARD ERRORS\n" + err_msg)
            if exog:
                assert_equal(obt_exog, obt, "WITH EXOG: " + err_msg)
            if exog_coint:
                assert_equal(obt_exog_coint, obt, "WITH EXOG_COINT: " + err_msg)
            # t-values
            obt = results_sm[ds][dt].tvalues_alpha
            obt_exog = results_sm_exog[ds][dt].tvalues_alpha
            obt_exog_coint = results_sm_exog_coint[ds][dt].tvalues_alpha
            des = results_ref[ds][dt]["t"]["alpha"]
            assert_allclose(obt, des, rtol, atol, False, "t-VALUES\n" + err_msg)
            if exog:
                assert_equal(obt_exog, obt, "WITH EXOG: " + err_msg)
            if exog_coint:
                assert_equal(obt_exog_coint, obt, "WITH EXOG_COINT: " + err_msg)
            # p-values
            obt = results_sm[ds][dt].pvalues_alpha
            obt_exog = results_sm_exog[ds][dt].pvalues_alpha
            obt_exog_coint = results_sm_exog_coint[ds][dt].pvalues_alpha
            des = results_ref[ds][dt]["p"]["alpha"]
            assert_allclose(obt, des, rtol, atol, False, "p-VALUES\n" + err_msg)
            if exog:
                assert_equal(obt_exog, obt, "WITH EXOG: " + err_msg)
            if exog_coint:
                assert_equal(obt_exog_coint, obt, "WITH EXOG_COINT: " + err_msg)


def test_ml_beta():
    for ds in datasets:
        for dt in ds.dt_s_list:
            exog = results_sm_exog[ds][dt].exog is not None
            exog_coint = results_sm_exog_coint[ds][dt].exog_coint is not None

            err_msg = build_err_msg(ds, dt, "beta")
            desired = results_ref[ds][dt]["est"]["beta"]
            rows = desired.shape[0]
            # - first coint_rank rows in JMulTi output have se=t_val=p_val=0
            # - beta includes deterministic terms in cointegration relation in
            #   sm, so we compare only the elements belonging to beta.
            obtained = results_sm[ds][dt].beta[coint_rank:rows]
            obtained_exog = results_sm_exog[ds][dt].beta[coint_rank:rows]
            obtained_exog_coint = results_sm_exog_coint[ds][dt].beta[coint_rank:rows]
            desired = desired[coint_rank:]
            assert_allclose(obtained, desired, rtol, atol, False, err_msg)
            if exog:
                assert_equal(obtained_exog, obtained, "WITH EXOG: " + err_msg)
            if exog_coint:
                assert_equal(
                    obtained_exog_coint,
                    obtained,
                    "WITH EXOG_COINT: " + err_msg,
                )

            # standard errors
            obt = results_sm[ds][dt].stderr_beta[coint_rank:rows]
            obt_exog = results_sm_exog[ds][dt].stderr_beta[coint_rank:rows]
            obt_exog_coint = results_sm_exog_coint[ds][dt].stderr_beta[coint_rank:rows]
            des = results_ref[ds][dt]["se"]["beta"][coint_rank:]
            assert_allclose(obt, des, rtol, atol, False, "STANDARD ERRORS\n" + err_msg)
            if exog:
                assert_equal(obt_exog, obt, "WITH EXOG: " + err_msg)
            if exog_coint:
                assert_equal(obt_exog_coint, obt, "WITH EXOG_COINT: " + err_msg)
            # t-values
            obt = results_sm[ds][dt].tvalues_beta[coint_rank:rows]
            obt_exog = results_sm_exog[ds][dt].tvalues_beta[coint_rank:rows]
            obt_exog_coint = results_sm_exog_coint[ds][dt].tvalues_beta[coint_rank:rows]
            des = results_ref[ds][dt]["t"]["beta"][coint_rank:]
            assert_allclose(obt, des, rtol, atol, False, "t-VALUES\n" + err_msg)
            if exog:
                assert_equal(obt_exog, obt, "WITH EXOG: " + err_msg)
            if exog_coint:
                assert_equal(obt_exog_coint, obt, "WITH EXOG_COINT: " + err_msg)
            # p-values
            obt = results_sm[ds][dt].pvalues_beta[coint_rank:rows]
            obt_exog = results_sm_exog[ds][dt].pvalues_beta[coint_rank:rows]
            obt_exog_coint = results_sm_exog_coint[ds][dt].pvalues_beta[coint_rank:rows]
            des = results_ref[ds][dt]["p"]["beta"][coint_rank:]
            assert_allclose(obt, des, rtol, atol, False, "p-VALUES\n" + err_msg)
            if exog:
                assert_equal(obt_exog, obt, "WITH EXOG: " + err_msg)
            if exog_coint:
                assert_equal(obt_exog_coint, obt, "WITH EXOG_COINT: " + err_msg)


def test_ml_c():  # test deterministic terms outside coint relation
    for ds in datasets:
        for dt in ds.dt_s_list:
            exog = results_sm_exog[ds][dt].exog is not None
            exog_coint = results_sm_exog_coint[ds][dt].exog_coint is not None

            C_obt = results_sm[ds][dt].det_coef
            C_obt_exog = results_sm_exog[ds][dt].det_coef
            C_obt_exog_coint = results_sm_exog_coint[ds][dt].det_coef
            se_C_obt = results_sm[ds][dt].stderr_det_coef
            se_C_obt_exog = results_sm_exog[ds][dt].stderr_det_coef
            se_C_obt_exog_coint = results_sm_exog_coint[ds][dt].stderr_det_coef
            t_C_obt = results_sm[ds][dt].tvalues_det_coef
            t_C_obt_exog = results_sm_exog[ds][dt].tvalues_det_coef
            t_C_obt_exog_coint = results_sm_exog_coint[ds][dt].tvalues_det_coef
            p_C_obt = results_sm[ds][dt].pvalues_det_coef
            p_C_obt_exog = results_sm_exog[ds][dt].pvalues_det_coef
            p_C_obt_exog_coint = results_sm_exog_coint[ds][dt].pvalues_det_coef

            if "C" not in results_ref[ds][dt]["est"].keys():
                # case: there are no deterministic terms
                if (
                    C_obt.size == 0
                    and se_C_obt.size == 0
                    and t_C_obt.size == 0
                    and p_C_obt.size == 0
                ):
                    assert (True)
                    continue

            desired = results_ref[ds][dt]["est"]["C"]
            dt_string = dt_s_tup_to_string(dt)
            if "co" in dt_string:
                err_msg = build_err_msg(ds, dt, "CONST")
                const_obt = C_obt[:, :1]
                const_obt_exog = C_obt_exog[:, :1]
                const_obt_exog_coint = C_obt_exog_coint[:, :1]
                const_des = desired[:, :1]
                C_obt = C_obt[:, 1:]
                C_obt_exog = C_obt_exog[:, 1:]
                C_obt_exog_coint = C_obt_exog_coint[:, 1:]
                desired = desired[:, 1:]
                assert_allclose(const_obt, const_des, rtol, atol, False, err_msg)
                if exog:
                    assert_equal(const_obt_exog, const_obt, "WITH EXOG: " + err_msg)
                if exog_coint:
                    assert_equal(
                        const_obt_exog_coint,
                        const_obt,
                        "WITH EXOG_COINT: " + err_msg,
                    )
            if "s" in dt_string:
                err_msg = build_err_msg(ds, dt, "SEASONAL")
                if "lo" in dt_string:
                    seas_obt = C_obt[:, :-1]
                    seas_obt_exog = C_obt_exog[:, :-1]
                    seas_obt_exog_coint = C_obt_exog_coint[:, :-1]
                    seas_des = desired[:, :-1]
                else:
                    seas_obt = C_obt
                    seas_obt_exog = C_obt_exog
                    seas_obt_exog_coint = C_obt_exog_coint
                    seas_des = desired
                assert_allclose(seas_obt, seas_des, rtol, atol, False, err_msg)
                if exog:
                    assert_equal(seas_obt_exog, seas_obt, "WITH EXOG: " + err_msg)
                if exog_coint:
                    assert_equal(
                        seas_obt_exog_coint,
                        seas_obt,
                        "WITH EXOG_COINT: " + err_msg,
                    )
            if "lo" in dt_string:
                err_msg = build_err_msg(ds, dt, "LINEAR TREND")
                lt_obt = C_obt[:, -1:]
                lt_obt_exog = C_obt_exog[:, -1:]
                lt_obt_exog_coint = C_obt_exog_coint[:, -1:]
                lt_des = desired[:, -1:]
                assert_allclose(lt_obt, lt_des, rtol, atol, False, err_msg)
                if exog:
                    assert_equal(lt_obt_exog, lt_obt, "WITH EXOG: " + err_msg)
                if exog_coint:
                    assert_equal(
                        lt_obt_exog_coint,
                        lt_obt,
                        "WITH EXOG_COINT: " + err_msg,
                    )
            # standard errors
            se_desired = results_ref[ds][dt]["se"]["C"]
            if "co" in dt_string:
                err_msg = build_err_msg(ds, dt, "SE CONST")
                se_const_obt = se_C_obt[:, 0][:, None]
                se_C_obt = se_C_obt[:, 1:]
                se_const_obt_exog = se_C_obt_exog[:, 0][:, None]
                se_C_obt_exog = se_C_obt_exog[:, 1:]
                se_const_obt_exog_coint = se_C_obt_exog_coint[:, 0][:, None]
                se_C_obt_exog_coint = se_C_obt_exog_coint[:, 1:]
                se_const_des = se_desired[:, 0][:, None]
                se_desired = se_desired[:, 1:]
                assert_allclose(se_const_obt, se_const_des, rtol, atol, False, err_msg)
                if exog:
                    assert_equal(
                        se_const_obt_exog,
                        se_const_obt,
                        "WITH EXOG: " + err_msg,
                    )
                if exog_coint:
                    assert_equal(
                        se_const_obt_exog_coint,
                        se_const_obt,
                        "WITH EXOG_COINT: " + err_msg,
                    )
            if "s" in dt_string:
                err_msg = build_err_msg(ds, dt, "SE SEASONAL")
                if "lo" in dt_string:
                    se_seas_obt = se_C_obt[:, :-1]
                    se_seas_obt_exog = se_C_obt_exog[:, :-1]
                    se_seas_obt_exog_coint = se_C_obt_exog_coint[:, :-1]
                    se_seas_des = se_desired[:, :-1]
                else:
                    se_seas_obt = se_C_obt
                    se_seas_obt_exog = se_C_obt_exog
                    se_seas_obt_exog_coint = se_C_obt_exog_coint
                    se_seas_des = se_desired
                assert_allclose(se_seas_obt, se_seas_des, rtol, atol, False, err_msg)
                if exog:
                    assert_equal(se_seas_obt_exog, se_seas_obt, "WITH EXOG: " + err_msg)
                if exog_coint:
                    assert_equal(
                        se_seas_obt_exog_coint,
                        se_seas_obt,
                        "WITH EXOG_COINT: " + err_msg,
                    )
                if "lo" in dt_string:
                    err_msg = build_err_msg(ds, dt, "SE LIN. TREND")
                    se_lt_obt = se_C_obt[:, -1:]
                    se_lt_obt_exog = se_C_obt_exog[:, -1:]
                    se_lt_obt_exog_coint = se_C_obt_exog_coint[:, -1:]
                    se_lt_des = se_desired[:, -1:]
                    assert_allclose(se_lt_obt, se_lt_des, rtol, atol, False, err_msg)
                    if exog:
                        assert_equal(se_lt_obt_exog, se_lt_obt, "WITH EXOG: " + err_msg)
                    if exog_coint:
                        assert_equal(
                            se_lt_obt_exog_coint,
                            se_lt_obt,
                            "WITH EXOG_COINT: " + err_msg,
                        )
            # t-values
            t_desired = results_ref[ds][dt]["t"]["C"]
            if "co" in dt_string:
                t_const_obt = t_C_obt[:, 0][:, None]
                t_C_obt = t_C_obt[:, 1:]
                t_const_obt_exog = t_C_obt_exog[:, 0][:, None]
                t_C_obt_exog = t_C_obt_exog[:, 1:]
                t_const_obt_exog_coint = t_C_obt_exog_coint[:, 0][:, None]
                t_C_obt_exog_coint = t_C_obt_exog_coint[:, 1:]
                t_const_des = t_desired[:, 0][:, None]
                t_desired = t_desired[:, 1:]
                assert_allclose(
                    t_const_obt,
                    t_const_des,
                    rtol,
                    atol,
                    False,
                    build_err_msg(ds, dt, "T CONST"),
                )
                if exog:
                    assert_equal(t_const_obt_exog, t_const_obt, "WITH EXOG: " + err_msg)
                if exog_coint:
                    assert_equal(
                        t_const_obt_exog_coint,
                        t_const_obt,
                        "WITH EXOG_COINT: " + err_msg,
                    )
            if "s" in dt_string:
                if "lo" in dt_string:
                    t_seas_obt = t_C_obt[:, :-1]
                    t_seas_obt_exog = t_C_obt_exog[:, :-1]
                    t_seas_obt_exog_coint = t_C_obt_exog_coint[:, :-1]
                    t_seas_des = t_desired[:, :-1]
                else:
                    t_seas_obt = t_C_obt
                    t_seas_obt_exog = t_C_obt_exog
                    t_seas_obt_exog_coint = t_C_obt_exog_coint
                    t_seas_des = t_desired
                assert_allclose(
                    t_seas_obt,
                    t_seas_des,
                    rtol,
                    atol,
                    False,
                    build_err_msg(ds, dt, "T SEASONAL"),
                )
                if exog:
                    assert_equal(t_seas_obt_exog, t_seas_obt, "WITH EXOG" + err_msg)
                if exog_coint:
                    assert_equal(
                        t_seas_obt_exog_coint,
                        t_seas_obt,
                        "WITH EXOG_COINT" + err_msg,
                    )
            if "lo" in dt_string:
                t_lt_obt = t_C_obt[:, -1:]
                t_lt_obt_exog = t_C_obt_exog[:, -1:]
                t_lt_obt_exog_coint = t_C_obt_exog_coint[:, -1:]
                t_lt_des = t_desired[:, -1:]
                assert_allclose(
                    t_lt_obt,
                    t_lt_des,
                    rtol,
                    atol,
                    False,
                    build_err_msg(ds, dt, "T LIN. TREND"),
                )
                if exog:
                    assert_equal(t_lt_obt_exog, t_lt_obt, "WITH EXOG" + err_msg)
                if exog_coint:
                    assert_equal(
                        t_lt_obt_exog_coint,
                        t_lt_obt,
                        "WITH EXOG_COINT" + err_msg,
                    )
            # p-values
            p_desired = results_ref[ds][dt]["p"]["C"]
            if "co" in dt_string:
                p_const_obt = p_C_obt[:, 0][:, None]
                p_C_obt = p_C_obt[:, 1:]
                p_const_obt_exog = p_C_obt_exog[:, 0][:, None]
                p_C_obt_exog = p_C_obt_exog[:, 1:]
                p_const_obt_exog_coint = p_C_obt_exog_coint[:, 0][:, None]
                p_const_des = p_desired[:, 0][:, None]
                p_desired = p_desired[:, 1:]
                assert_allclose(
                    p_const_obt,
                    p_const_des,
                    rtol,
                    atol,
                    False,
                    build_err_msg(ds, dt, "P CONST"),
                )
                if exog:
                    assert_equal(p_const_obt, p_const_obt_exog, "WITH EXOG" + err_msg)
                if exog_coint:
                    assert_equal(
                        p_const_obt,
                        p_const_obt_exog_coint,
                        "WITH EXOG_COINT" + err_msg,
                    )
            if "s" in dt_string:
                if "lo" in dt_string:
                    p_seas_obt = p_C_obt[:, :-1]
                    p_seas_obt_exog = p_C_obt_exog[:, :-1]
                    p_seas_obt_exog_coint = p_C_obt_exog_coint[:, :-1]
                    p_seas_des = p_desired[:, :-1]
                else:
                    p_seas_obt = p_C_obt
                    p_seas_obt_exog = p_C_obt_exog
                    p_seas_obt_exog_coint = p_C_obt_exog_coint
                    p_seas_des = p_desired
                assert_allclose(
                    p_seas_obt,
                    p_seas_des,
                    rtol,
                    atol,
                    False,
                    build_err_msg(ds, dt, "P SEASONAL"),
                )
                if exog:
                    assert_equal(p_seas_obt_exog, p_seas_obt, "WITH EXOG" + err_msg)
                if exog_coint:
                    assert_equal(
                        p_seas_obt_exog_coint,
                        p_seas_obt,
                        "WITH EXOG_COINT" + err_msg,
                    )
            if "lo" in dt_string:
                p_lt_obt = p_C_obt[:, -1:]
                p_lt_obt_exog = p_C_obt_exog[:, -1:]
                p_lt_obt_exog_coint = p_C_obt_exog_coint[:, -1:]
                p_lt_des = p_desired[:, -1:]
                assert_allclose(
                    p_lt_obt,
                    p_lt_des,
                    rtol,
                    atol,
                    False,
                    build_err_msg(ds, dt, "P LIN. TREND"),
                )
                if exog:
                    assert_equal(p_lt_obt_exog, p_lt_obt, "WITH EXOG" + err_msg)
                if exog_coint:
                    assert_equal(
                        p_lt_obt_exog_coint,
                        p_lt_obt,
                        "WITH EXOG_COINT" + err_msg,
                    )


def test_ml_det_terms_in_coint_relation():
    for ds in datasets:
        for dt in ds.dt_s_list:
            exog = results_sm_exog[ds][dt].exog is not None
            exog_coint = results_sm_exog_coint[ds][dt].exog_coint is not None

            err_msg = build_err_msg(ds, dt, "det terms in coint relation")
            dt_string = dt_s_tup_to_string(dt)
            obtained = results_sm[ds][dt].det_coef_coint
            obtained_exog = results_sm_exog[ds][dt].det_coef_coint
            obtained_exog_coint = results_sm_exog_coint[ds][dt].det_coef_coint
            if "ci" not in dt_string and "li" not in dt_string:
                if obtained.size > 0:
                    pytest.fail(
                        build_err_msg(
                            ds,
                            dt,
                            (
                                "There should not be any det terms in cointegration "
                                "for deterministic terms" + dt_string
                            ),
                        )
                    )

                else:
                    assert (True)
                continue
            desired = results_ref[ds][dt]["est"]["det_coint"]
            assert_allclose(obtained, desired, rtol, atol, False, err_msg)
            if exog:
                assert_equal(obtained_exog, obtained, "WITH EXOG" + err_msg)
            if exog_coint:
                assert_equal(obtained_exog_coint, obtained, "WITH EXOG_COINT" + err_msg)
            # standard errors
            se_obtained = results_sm[ds][dt].stderr_det_coef_coint
            se_obtained_exog = results_sm_exog[ds][dt].stderr_det_coef_coint
            se_obtained_exog_coint = results_sm_exog_coint[ds][dt].stderr_det_coef_coint
            se_desired = results_ref[ds][dt]["se"]["det_coint"]
            assert_allclose(
                se_obtained,
                se_desired,
                rtol,
                atol,
                False,
                "STANDARD ERRORS\n" + err_msg,
            )
            if exog:
                assert_equal(se_obtained_exog, se_obtained, "WITH EXOG" + err_msg)
            if exog_coint:
                assert_equal(
                    se_obtained_exog_coint,
                    se_obtained,
                    "WITH EXOG_COINT" + err_msg,
                )
            # t-values
            t_obtained = results_sm[ds][dt].tvalues_det_coef_coint
            t_obtained_exog = results_sm_exog[ds][dt].tvalues_det_coef_coint
            t_obtained_exog_coint = results_sm_exog_coint[ds][dt].tvalues_det_coef_coint
            t_desired = results_ref[ds][dt]["t"]["det_coint"]
            assert_allclose(
                t_obtained,
                t_desired,
                rtol,
                atol,
                False,
                "t-VALUES\n" + err_msg,
            )
            if exog:
                assert_equal(t_obtained_exog, t_obtained, "WITH EXOG" + err_msg)
            if exog_coint:
                assert_equal(
                    t_obtained_exog_coint,
                    t_obtained,
                    "WITH EXOG_COINT" + err_msg,
                )
            # p-values
            p_obtained = results_sm[ds][dt].pvalues_det_coef_coint
            p_obtained_exog = results_sm_exog[ds][dt].pvalues_det_coef_coint
            p_obtained_exog_coint = results_sm_exog_coint[ds][dt].pvalues_det_coef_coint
            p_desired = results_ref[ds][dt]["p"]["det_coint"]
            assert_allclose(
                p_obtained,
                p_desired,
                rtol,
                atol,
                False,
                "p-VALUES\n" + err_msg,
            )
            if exog:
                assert_equal(p_obtained_exog, p_obtained, "WITH EXOG" + err_msg)
            if exog_coint:
                assert_equal(
                    p_obtained_exog_coint,
                    p_obtained,
                    "WITH EXOG_COINT" + err_msg,
                )


def test_ml_sigma():
    for ds in datasets:
        for dt in ds.dt_s_list:
            exog = results_sm_exog[ds][dt].exog is not None
            exog_coint = results_sm_exog_coint[ds][dt].exog_coint is not None

            err_msg = build_err_msg(ds, dt, "Sigma_u")
            obtained = results_sm[ds][dt].sigma_u
            obtained_exog = results_sm_exog[ds][dt].sigma_u
            obtained_exog_coint = results_sm_exog_coint[ds][dt].sigma_u
            desired = results_ref[ds][dt]["est"]["Sigma_u"]
            assert_allclose(obtained, desired, rtol, atol, False, err_msg)
            if exog:
                assert_equal(obtained_exog, obtained, "WITH EXOG" + err_msg)
            if exog_coint:
                assert_equal(obtained_exog_coint, obtained, "WITH EXOG_COINT" + err_msg)


def test_var_rep():
    for ds in datasets:
        for dt in ds.dt_s_list:
            exog = results_sm_exog[ds][dt].exog is not None
            exog_coint = results_sm_exog_coint[ds][dt].exog_coint is not None

            err_msg = build_err_msg(ds, dt, "VAR repr. A")
            obtained = results_sm[ds][dt].var_rep
            obtained_exog = results_sm_exog[ds][dt].var_rep
            obtained_exog_coint = results_sm_exog_coint[ds][dt].var_rep
            p = obtained.shape[0]
            desired = np.hsplit(results_ref[ds][dt]["est"]["VAR A"], p)
            assert_allclose(obtained, desired, rtol, atol, False, err_msg)
            if exog:
                assert_equal(obtained_exog, obtained, "WITH EXOG" + err_msg)
            if exog_coint:
                assert_equal(obtained_exog_coint, obtained, "WITH EXOG_COINT" + err_msg)


def test_var_to_vecm():
    for ds in datasets:
        for dt in ds.dt_s_list:
            err_msg = build_err_msg(ds, dt, "VAR to VEC representation")
            sigma_u = results_sm[ds][dt].sigma_u
            coefs = results_sm[ds][dt].var_rep
            intercept = np.zeros(len(sigma_u))
            # Note: _params_info k_trend, k_exog, ... is inferred with defaults
            var = VARProcess(coefs, intercept, sigma_u)
            vecm_results = var.to_vecm()
            obtained_pi = vecm_results["Pi"]
            obtained_gamma = vecm_results["Gamma"]

            desired_pi = np.dot(results_sm[ds][dt].alpha, results_sm[ds][dt].beta.T)
            desired_gamma = results_sm[ds][dt].gamma
            assert_allclose(obtained_pi, desired_pi, rtol, atol, False, err_msg + " Pi")
            assert_allclose(
                obtained_gamma,
                desired_gamma,
                rtol,
                atol,
                False,
                err_msg + " Gamma",
            )


# Commented out since JMulTi shows the same det. terms for both VEC & VAR repr.
# def test_var_rep_det():
#     for ds in datasets:
#         for dt in dt_s_list:
#             if dt != "n":
#                 err_msg = build_err_msg(ds, dt, "VAR repr. deterministic")
#                 obtained = 0  # not implemented since the same values as VECM
#                 desired = results_ref[ds][dt]["est"]["VAR deterministic"]
#                 assert_allclose(obtained, desired, rtol, atol, False, err_msg)


def test_log_like():
    for ds in datasets:
        for dt in ds.dt_s_list:
            exog = results_sm_exog[ds][dt].exog is not None
            exog_coint = results_sm_exog_coint[ds][dt].exog_coint is not None

            err_msg = build_err_msg(ds, dt, "Log Likelihood")
            obtained = results_sm[ds][dt].llf
            obtained_exog = results_sm_exog[ds][dt].llf
            obtained_exog_coint = results_sm_exog_coint[ds][dt].llf
            # JMulTi's llf seems to have a bug (Stata and tsdyn suggest that
            # our code is correct). We use nobs to correct this inconsistency.
            nobs = results_sm[ds][dt].nobs
            desired = results_ref[ds][dt]["log_like"] * nobs / (nobs - 1)
            assert_allclose(obtained, desired, rtol, atol, False, err_msg)
            if exog:
                assert_equal(obtained_exog, obtained, "WITH EXOG" + err_msg)
            if exog_coint:
                assert_equal(obtained_exog_coint, obtained, "WITH EXOG_COINT" + err_msg)


def test_fc():
    for ds in datasets:
        for dt in ds.dt_s_list:
            STEPS = 5
            ALPHA = 0.05
            err_msg = build_err_msg(ds, dt, "FORECAST")
            # test point forecast functionality of predict method
            obtained = results_sm[ds][dt].predict(steps=STEPS)
            desired = results_ref[ds][dt]["fc"]["fc"]
            assert_allclose(obtained, desired, rtol, atol, False, err_msg)

            # -----------------------------------------------------------------
            # with exog:
            exog = results_sm_exog[ds][dt].exog is not None
            exog_fc = None
            if exog:  # build future values of exog and test:
                seasons = dt[1]
                exog_model = results_sm_exog[ds][dt].exog
                exog_seasons_fc = exog_model[-seasons:, : seasons - 1]
                exog_seasons_fc = np.pad(
                    exog_seasons_fc,
                    ((0, STEPS - exog_seasons_fc.shape[0]), (0, 0)),
                    "wrap",
                )
                # if linear trend in exog
                if exog_seasons_fc.shape[1] + 1 == exog_model.shape[1]:
                    exog_lt_fc = exog_model[-1, -1] + 1 + np.arange(STEPS)
                    exog_fc = np.column_stack((exog_seasons_fc, exog_lt_fc))
                else:
                    exog_fc = exog_seasons_fc
                obtained_exog = results_sm_exog[ds][dt].predict(
                    steps=STEPS, exog_fc=exog_fc
                )
                assert_allclose(
                    obtained_exog,
                    obtained,
                    1e-07,
                    0,
                    False,
                    "WITH EXOG" + err_msg,
                )
            # test predict method with confidence interval calculation
            err_msg = build_err_msg(ds, dt, "FORECAST WITH INTERVALS")
            obtained_w_intervals = results_sm[ds][dt].predict(steps=STEPS, alpha=ALPHA)
            obtained_w_intervals_exog = results_sm_exog[ds][dt].predict(
                steps=STEPS, alpha=ALPHA, exog_fc=exog_fc
            )
            obt = obtained_w_intervals.point_forecast  # forecast
            obt_l = obtained_w_intervals.forc_lower  # lower bound
            obt_u = obtained_w_intervals.forc_upper  # upper bound
            obt_exog = obtained_w_intervals_exog.point_forecast
            obt_exog_l = obtained_w_intervals_exog.forc_lower
            obt_exog_u = obtained_w_intervals_exog.forc_upper
            des = results_ref[ds][dt]["fc"]["fc"]
            des_l = results_ref[ds][dt]["fc"]["lower"]
            des_u = results_ref[ds][dt]["fc"]["upper"]
            assert_allclose(obt, des, rtol, atol, False, err_msg)
            assert_allclose(obt_l, des_l, rtol, atol, False, err_msg)
            assert_allclose(obt_u, des_u, rtol, atol, False, err_msg)
            if exog:
                assert_allclose(obt_exog, obt, 1e-07, 0, False, "WITH EXOG" + err_msg)
                assert_allclose(
                    obt_exog_l, obt_l, 1e-07, 0, False, "WITH EXOG" + err_msg
                )
                assert_allclose(
                    obt_exog_u, obt_u, 1e-07, 0, False, "WITH EXOG" + err_msg
                )

            # -----------------------------------------------------------------
            # with exog_coint:
            exog_coint_model = results_sm_exog_coint[ds][dt].exog_coint
            exog_coint = exog_coint_model is not None
            exog_coint_fc = None
            if exog_coint:  # build future values of exog_coint and test:
                # const is in exog_coint in all tests that have exog_coint
                exog_coint_fc = np.ones(STEPS - 1)
                # if linear trend in exog_coint
                if exog_coint_model.shape[1] == 2:
                    exog_coint_fc = np.column_stack(
                        (
                            exog_coint_fc,
                            exog_coint_model[-1, -1] + 1 + np.arange(STEPS - 1),
                        )
                    )
                obtained_exog_coint = results_sm_exog_coint[ds][dt].predict(
                    steps=STEPS, exog_coint_fc=exog_coint_fc
                )

                assert_allclose(
                    obtained_exog_coint,
                    obtained,
                    1e-07,
                    0,
                    False,
                    "WITH EXOG_COINT" + err_msg,
                )
            # test predict method with confidence interval calculation
            err_msg = build_err_msg(ds, dt, "FORECAST WITH INTERVALS")
            obtained_w_intervals = results_sm[ds][dt].predict(steps=STEPS, alpha=ALPHA)
            obtained_w_intervals_exog_coint = results_sm_exog_coint[ds][dt].predict(
                steps=STEPS, alpha=ALPHA, exog_coint_fc=exog_coint_fc
            )
            obt = obtained_w_intervals.point_forecast  # forecast
            obt_l = obtained_w_intervals.forc_lower  # lower bound
            obt_u = obtained_w_intervals.forc_upper  # upper bound
            obt_exog_coint = obtained_w_intervals_exog_coint.point_forecast
            obt_exog_coint_l = obtained_w_intervals_exog_coint.forc_lower
            obt_exog_coint_u = obtained_w_intervals_exog_coint.forc_upper
            des = results_ref[ds][dt]["fc"]["fc"]
            des_l = results_ref[ds][dt]["fc"]["lower"]
            des_u = results_ref[ds][dt]["fc"]["upper"]
            assert_allclose(obt, des, rtol, atol, False, err_msg)
            assert_allclose(obt_l, des_l, rtol, atol, False, err_msg)
            assert_allclose(obt_u, des_u, rtol, atol, False, err_msg)
            if exog_coint:
                assert_allclose(
                    obt_exog_coint,
                    obt,
                    1e-07,
                    0,
                    False,
                    "WITH EXOG_COINT" + err_msg,
                )
                assert_allclose(
                    obt_exog_coint_l,
                    obt_l,
                    1e-07,
                    0,
                    False,
                    "WITH EXOG_COINT" + err_msg,
                )
                assert_allclose(
                    obt_exog_coint_u,
                    obt_u,
                    1e-07,
                    0,
                    False,
                    "WITH EXOG_COINT" + err_msg,
                )


def test_granger_causality():
    for ds in datasets:
        for dt in ds.dt_s_list:
            exog = results_sm_exog[ds][dt].exog is not None
            exog_coint = results_sm_exog_coint[ds][dt].exog_coint is not None

            err_msg_g_p = build_err_msg(ds, dt, "GRANGER CAUS. - p-VALUE")
            err_msg_g_t = build_err_msg(ds, dt, "GRANGER CAUS. - TEST STAT.")

            v_ind = range(len(ds.variable_names))
            for causing_ind in sublists(v_ind, 1, len(v_ind) - 1):
                causing_names = ["y" + str(i + 1) for i in causing_ind]
                causing_key = tuple(ds.variable_names[i] for i in causing_ind)

                caused_ind = [i for i in v_ind if i not in causing_ind]
                caused_names = ["y" + str(i + 1) for i in caused_ind]
                caused_key = tuple(ds.variable_names[i] for i in caused_ind)

                granger_sm_ind = results_sm[ds][dt].test_granger_causality(
                    caused_ind, causing_ind
                )
                granger_sm_ind_exog = results_sm_exog[ds][dt].test_granger_causality(
                    caused_ind, causing_ind
                )
                granger_sm_ind_exog_coint = results_sm_exog_coint[ds][
                    dt
                ].test_granger_causality(caused_ind, causing_ind)
                granger_sm_str = results_sm[ds][dt].test_granger_causality(
                    caused_names, causing_names
                )

                # call methods to assure they do not raise exceptions
                granger_sm_ind.summary()
                str(granger_sm_ind)  # __str__()
                assert (granger_sm_ind == granger_sm_str)  # __eq__()

                # test test-statistic for Granger non-causality:
                g_t_obt = granger_sm_ind.test_statistic
                g_t_obt_exog = granger_sm_ind_exog.test_statistic
                g_t_obt_exog_coint = granger_sm_ind_exog_coint.test_statistic
                g_t_des = results_ref[ds][dt]["granger_caus"]["test_stat"][
                    (causing_key, caused_key)
                ]
                assert_allclose(g_t_obt, g_t_des, rtol, atol, False, err_msg_g_t)
                if exog:
                    assert_allclose(
                        g_t_obt_exog,
                        g_t_obt,
                        1e-07,
                        0,
                        False,
                        "WITH EXOG" + err_msg_g_t,
                    )
                if exog_coint:
                    assert_allclose(
                        g_t_obt_exog_coint,
                        g_t_obt,
                        1e-07,
                        0,
                        False,
                        "WITH EXOG_COINT" + err_msg_g_t,
                    )
                # check whether string sequences as args work in the same way:
                g_t_obt_str = granger_sm_str.test_statistic
                assert_allclose(
                    g_t_obt_str,
                    g_t_obt,
                    1e-07,
                    0,
                    False,
                    err_msg_g_t
                    + " - sequences of integers and ".upper()
                    + "strings as arguments do not yield the same result!".upper(),
                )
                # check if int (e.g., 0) as index and list of int ([0]) yield
                # the same result:
                if len(causing_ind) == 1 or len(caused_ind) == 1:
                    ci = causing_ind[0] if len(causing_ind) == 1 else causing_ind
                    ce = caused_ind[0] if len(caused_ind) == 1 else caused_ind
                    granger_sm_single_ind = results_sm[ds][dt].test_granger_causality(
                        ce, ci
                    )
                    g_t_obt_single = granger_sm_single_ind.test_statistic
                    assert_allclose(
                        g_t_obt_single,
                        g_t_obt,
                        1e-07,
                        0,
                        False,
                        err_msg_g_t
                        + " - list of int and int as ".upper()
                        + "argument do not yield the same result!".upper(),
                    )

                # test p-value for Granger non-causality:
                g_p_obt = granger_sm_ind.pvalue
                g_p_des = results_ref[ds][dt]["granger_caus"]["p"][
                    (causing_key, caused_key)
                ]
                assert_allclose(g_p_obt, g_p_des, rtol, atol, False, err_msg_g_p)
                # check whether string sequences as args work in the same way:
                g_p_obt_str = granger_sm_str.pvalue
                assert_allclose(
                    g_p_obt_str,
                    g_p_obt,
                    1e-07,
                    0,
                    False,
                    err_msg_g_t
                    + " - sequences of integers and ".upper()
                    + "strings as arguments do not yield the same result!".upper(),
                )
                # check if int (e.g., 0) as index and list of int ([0]) yield
                # the same result:
                if len(causing_ind) == 1:
                    g_p_obt_single = granger_sm_single_ind.pvalue
                    assert_allclose(
                        g_p_obt_single,
                        g_p_obt,
                        1e-07,
                        0,
                        False,
                        err_msg_g_t
                        + " - list of int and int as ".upper()
                        + "argument do not yield the same result!".upper(),
                    )


def test_inst_causality():  # test instantaneous causality
    for ds in datasets:
        for dt in ds.dt_s_list:
            exog = results_sm_exog[ds][dt].exog is not None
            exog_coint = results_sm_exog_coint[ds][dt].exog_coint is not None

            err_msg_i_p = build_err_msg(ds, dt, "INSTANT. CAUS. - p-VALUE")
            err_msg_i_t = build_err_msg(ds, dt, "INSTANT. CAUS. - TEST STAT.")

            v_ind = range(len(ds.variable_names))
            for causing_ind in sublists(v_ind, 1, len(v_ind) - 1):
                causing_names = ["y" + str(i + 1) for i in causing_ind]
                causing_key = tuple(ds.variable_names[i] for i in causing_ind)

                caused_ind = [i for i in v_ind if i not in causing_ind]
                caused_key = tuple(ds.variable_names[i] for i in caused_ind)
                inst_sm_ind = results_sm[ds][dt].test_inst_causality(causing_ind)
                inst_sm_ind_exog = results_sm_exog[ds][dt].test_inst_causality(
                    causing_ind
                )
                inst_sm_ind_exog_coint = results_sm_exog_coint[ds][
                    dt
                ].test_inst_causality(causing_ind)
                inst_sm_str = results_sm[ds][dt].test_inst_causality(causing_names)
                # call methods to assure they do not raise exceptions
                inst_sm_ind.summary()
                str(inst_sm_ind)  # __str__()
                assert (inst_sm_ind == inst_sm_str)  # __eq__()
                # test test-statistic for instantaneous non-causality
                t_obt = inst_sm_ind.test_statistic
                t_obt_exog = inst_sm_ind_exog.test_statistic
                t_obt_exog_coint = inst_sm_ind_exog_coint.test_statistic
                t_des = results_ref[ds][dt]["inst_caus"]["test_stat"][
                    (causing_key, caused_key)
                ]
                assert_allclose(t_obt, t_des, rtol, atol, False, err_msg_i_t)
                if exog:
                    assert_allclose(
                        t_obt_exog,
                        t_obt,
                        1e-07,
                        0,
                        False,
                        "WITH EXOG" + err_msg_i_t,
                    )
                if exog_coint:
                    assert_allclose(
                        t_obt_exog_coint,
                        t_obt,
                        1e-07,
                        0,
                        False,
                        "WITH EXOG_COINT" + err_msg_i_t,
                    )
                # check whether string sequences as args work in the same way:
                t_obt_str = inst_sm_str.test_statistic
                assert_allclose(
                    t_obt_str,
                    t_obt,
                    1e-07,
                    0,
                    False,
                    err_msg_i_t
                    + " - sequences of integers and ".upper()
                    + "strings as arguments do not yield the same result!".upper(),
                )
                # check if int (e.g., 0) as index and list of int ([0]) yield
                # the same result:
                if len(causing_ind) == 1:
                    inst_sm_single_ind = results_sm[ds][dt].test_inst_causality(
                        causing_ind[0]
                    )
                    t_obt_single = inst_sm_single_ind.test_statistic
                    assert_allclose(
                        t_obt_single,
                        t_obt,
                        1e-07,
                        0,
                        False,
                        err_msg_i_t
                        + " - list of int and int as ".upper()
                        + "argument do not yield the same result!".upper(),
                    )

                # test p-value for instantaneous non-causality
                p_obt = results_sm[ds][dt].test_inst_causality(causing_ind).pvalue
                p_des = results_ref[ds][dt]["inst_caus"]["p"][(causing_key, caused_key)]
                assert_allclose(p_obt, p_des, rtol, atol, False, err_msg_i_p)
                # check whether string sequences as args work in the same way:
                p_obt_str = inst_sm_str.pvalue
                assert_allclose(
                    p_obt_str,
                    p_obt,
                    1e-07,
                    0,
                    False,
                    err_msg_i_p
                    + " - sequences of integers and ".upper()
                    + "strings as arguments do not yield the same result!".upper(),
                )
                # check if int (e.g., 0) as index and list of int ([0]) yield
                # the same result:
                if len(causing_ind) == 1:
                    inst_sm_single_ind = results_sm[ds][dt].test_inst_causality(
                        causing_ind[0]
                    )
                    p_obt_single = inst_sm_single_ind.pvalue
                    assert_allclose(
                        p_obt_single,
                        p_obt,
                        1e-07,
                        0,
                        False,
                        err_msg_i_p
                        + " - list of int and int as ".upper()
                        + "argument do not yield the same result!".upper(),
                    )


def test_impulse_response():
    for ds in datasets:
        for dt in ds.dt_s_list:
            exog = results_sm_exog[ds][dt].exog is not None
            exog_coint = results_sm_exog_coint[ds][dt].exog_coint is not None

            err_msg = build_err_msg(ds, dt, "IMULSE-RESPONSE")
            periods = 20
            obtained_all = results_sm[ds][dt].irf(periods=periods).irfs
            obtained_all_exog = results_sm_exog[ds][dt].irf(periods=periods).irfs
            obtained_all_exog_coint = (
                results_sm_exog_coint[ds][dt].irf(periods=periods).irfs
            )
            # flatten inner arrays to make them comparable to parsed results:
            obtained_all = obtained_all.reshape(periods + 1, -1)
            obtained_all_exog = obtained_all_exog.reshape(periods + 1, -1)
            obtained_all_exog_coint = obtained_all_exog_coint.reshape(periods + 1, -1)
            desired_all = results_ref[ds][dt]["ir"]
            assert_allclose(obtained_all, desired_all, rtol, atol, False, err_msg)
            if exog:
                assert_equal(obtained_all_exog, obtained_all, "WITH EXOG" + err_msg)
            if exog_coint:
                assert_equal(
                    obtained_all_exog_coint,
                    obtained_all,
                    "WITH EXOG_COINT" + err_msg,
                )


def test_lag_order_selection():
    for ds in datasets:
        for dt in ds.dt_s_list:
            deterministic = dt[0]
            endog_tot = data[ds]

            obtained_all = select_order(
                endog_tot, 10, deterministic=dt[0], seasons=dt[1]
            )
            deterministic_outside_exog = ""
            # "co" is not in exog in any test case
            if "co" in deterministic:
                deterministic_outside_exog += "co"
            # "lo" is in exog only in test cases with seasons>0
            if "lo" in deterministic and dt[1] == 0:
                deterministic_outside_exog += "lo"

            exog_model = results_sm_exog[ds][dt].exog
            exog = exog_model is not None
            exog_coint_model = results_sm_exog_coint[ds][dt].exog_coint
            exog_coint = exog_coint_model is not None

            obtained_all_exog = select_order(
                endog_tot,
                10,
                deterministic_outside_exog,
                seasons=0,
                exog=exog_model,
            )
            # "ci" and "li" are always in exog_coint, so pass "n" as det. term
            obtained_all_exog_coint = select_order(
                endog_tot, 10, "n", seasons=dt[1], exog_coint=exog_coint_model
            )
            for ic in ["aic", "fpe", "hqic", "bic"]:
                err_msg = build_err_msg(ds, dt, "LAG ORDER SELECTION - " + ic.upper())
                obtained = getattr(obtained_all, ic)
                desired = results_ref[ds][dt]["lagorder"][ic]
                assert_allclose(obtained, desired, rtol, atol, False, err_msg)
                if exog:
                    assert_equal(
                        getattr(obtained_all_exog, ic),
                        getattr(obtained_all, ic),
                        "WITH EXOG" + err_msg,
                    )
                if exog_coint:
                    assert_equal(
                        getattr(obtained_all_exog_coint, ic),
                        getattr(obtained_all, ic),
                        "WITH EXOG_COINT" + err_msg,
                    )
            # call methods to assure they do not raise exceptions
            obtained_all.summary()
            str(obtained_all)  # __str__()


def test_normality():
    for ds in datasets:
        for dt in ds.dt_s_list:
            exog = results_sm_exog[ds][dt].exog is not None
            exog_coint = results_sm_exog_coint[ds][dt].exog_coint is not None

            obtained = results_sm[ds][dt].test_normality(signif=0.05)
            obtained_exog = results_sm_exog[ds][dt].test_normality(signif=0.05)
            obtained_exog_coint = results_sm_exog_coint[ds][dt].test_normality(
                signif=0.05
            )
            err_msg = build_err_msg(ds, dt, "TEST NON-NORMALITY - STATISTIC")
            obt_statistic = obtained.test_statistic
            obt_statistic_exog = obtained_exog.test_statistic
            obt_statistic_exog_coint = obtained_exog_coint.test_statistic
            des_statistic = results_ref[ds][dt]["test_norm"]["joint_test_statistic"]
            assert_allclose(obt_statistic, des_statistic, rtol, atol, False, err_msg)
            if exog:
                assert_equal(obt_statistic_exog, obt_statistic, "WITH EXOG" + err_msg)
            if exog_coint:
                assert_equal(
                    obt_statistic_exog_coint,
                    obt_statistic,
                    "WITH EXOG_COINT" + err_msg,
                )
            err_msg = build_err_msg(ds, dt, "TEST NON-NORMALITY - P-VALUE")
            obt_pvalue = obtained.pvalue
            des_pvalue = results_ref[ds][dt]["test_norm"]["joint_pvalue"]
            assert_allclose(obt_pvalue, des_pvalue, rtol, atol, False, err_msg)
            # call methods to assure they do not raise exceptions
            obtained.summary()
            str(obtained)  # __str__()
            assert (obtained == obtained_exog)  # __eq__()


def test_whiteness():
    for ds in datasets:
        for dt in ds.dt_s_list:
            exog = results_sm_exog[ds][dt].exog is not None
            exog_coint = results_sm_exog_coint[ds][dt].exog_coint is not None

            lags = results_ref[ds][dt]["whiteness"]["tested order"]

            obtained = results_sm[ds][dt].test_whiteness(nlags=lags)
            obtained_exog = results_sm_exog[ds][dt].test_whiteness(nlags=lags)
            obtained_exog_coint = results_sm_exog_coint[ds][dt].test_whiteness(
                nlags=lags
            )
            # test statistic
            err_msg = build_err_msg(ds, dt, "WHITENESS OF RESIDUALS - TEST STATISTIC")
            desired = results_ref[ds][dt]["whiteness"]["test statistic"]
            assert_allclose(
                obtained.test_statistic, desired, rtol, atol, False, err_msg
            )
            if exog:
                assert_equal(
                    obtained_exog.test_statistic,
                    obtained.test_statistic,
                    "WITH EXOG" + err_msg,
                )
            if exog_coint:
                assert_equal(
                    obtained_exog_coint.test_statistic,
                    obtained.test_statistic,
                    "WITH EXOG_COINT" + err_msg,
                )
            # p-value
            err_msg = build_err_msg(ds, dt, "WHITENESS OF RESIDUALS - P-VALUE")
            desired = results_ref[ds][dt]["whiteness"]["p-value"]
            assert_allclose(obtained.pvalue, desired, rtol, atol, False, err_msg)

            obtained = results_sm[ds][dt].test_whiteness(nlags=lags, adjusted=True)
            obtained_exog = results_sm_exog[ds][dt].test_whiteness(
                nlags=lags, adjusted=True
            )
            obtained_exog_coint = results_sm_exog_coint[ds][dt].test_whiteness(
                nlags=lags, adjusted=True
            )
            # test statistic (adjusted Portmanteau test)
            err_msg = build_err_msg(
                ds,
                dt,
                "WHITENESS OF RESIDUALS - TEST STATISTIC (ADJUSTED TEST)",
            )
            desired = results_ref[ds][dt]["whiteness"]["test statistic adj."]
            assert_allclose(
                obtained.test_statistic, desired, rtol, atol, False, err_msg
            )
            if exog:
                assert_equal(
                    obtained_exog.test_statistic,
                    obtained.test_statistic,
                    "WITH EXOG" + err_msg,
                )
            if exog_coint:
                assert_equal(
                    obtained_exog_coint.test_statistic,
                    obtained.test_statistic,
                    "WITH EXOG_COINT" + err_msg,
                )
            # p-value (adjusted Portmanteau test)
            err_msg = build_err_msg(
                ds, dt, "WHITENESS OF RESIDUALS - P-VALUE (ADJUSTED TEST)"
            )
            desired = results_ref[ds][dt]["whiteness"]["p-value adjusted"]
            assert_allclose(obtained.pvalue, desired, rtol, atol, False, err_msg)
            # call methods to assure they do not raise exceptions
            obtained.summary()
            str(obtained)  # __str__()
            assert (obtained == obtained_exog)  # __eq__()


def test_summary():
    for ds in datasets:
        for dt in ds.dt_s_list:
            # see if summary gets printed
            results_sm[ds][dt].summary(alpha=0.05)

            exog = results_sm_exog[ds][dt].exog is not None
            if exog is not None:
                results_sm_exog[ds][dt].summary(alpha=0.05)

            exog_coint = results_sm_exog_coint[ds][dt].exog_coint is not None
            if exog_coint is not None:
                results_sm_exog_coint[ds][dt].summary(alpha=0.05)


def test_exceptions():
    ds = datasets[0]
    dt = ds.dt_s_list[0]
    endog = data[datasets[0]]

    # select_coint_rank:
    # method argument cannot be "my_method"
    with pytest.raises(ValueError):
        select_coint_rank(endog, 0, 3, "my_method", 0.05)
    # det_order has to be -1, 0, or 1:
    with pytest.raises(ValueError):
        select_coint_rank(endog, 2, 3)
    with pytest.raises(ValueError):
        select_coint_rank(endog, 0.5, 3)
    # 0.025 is not possible (must be 0.1, 0.05, or 0.01)
    with pytest.raises(ValueError):
        select_coint_rank(endog, 0, 3, "trace", 0.025)

    # VECM.fit: method argument cannot be anything other than "ml"
    with pytest.raises(ValueError, match="method"):
        VECM(endog).fit(method="not-ml")

    # Granger_causality:
    # # 0<signif<1
    # this means signif=0
    with pytest.raises(ValueError):
        results_sm[ds][dt].test_granger_causality(0, None, 0)
    # # caused must be int, str or iterable of int or str
    # 0.5 not int
    with pytest.raises(TypeError):
        results_sm[ds][dt].test_granger_causality([0.5])
    # # causing must be None, int, str or iterable of int or str
    # .5 not int
    with pytest.raises(TypeError):
        results_sm[ds][dt].test_granger_causality(0, 0.5)

    # exceptions in VECM class
    # # choose only one of the two: "co" and "ci"
    with pytest.raises(ValueError, match="deterministic must be one "):
        model = VECM(endog, k_ar_diff=1, deterministic="cico")
    # # we analyze multiple time series
    univariate_data = endog[0]
    with pytest.raises(ValueError):
        VECM(univariate_data)
    # # fit only allowed with known method
    model = VECM(endog, k_ar_diff=1, deterministic="n")
    with pytest.raises(ValueError):
        model.fit("abc")  # no "abc" estim.-method

    # pass a shorter array than endog as exog_coint argument
    with pytest.raises(ValueError):
        VECM(endog, None, np.ones(len(endog) - 1))
    # forecasting: argument checks
    STEPS = 5
    # # with exog
    exog = seasonal_dummies(4, len(endog), 2, centered=True)  # seasonal...
    exog = np.hstack((exog, 1 + np.arange(len(endog)).reshape(-1, 1)))  # & lin
    vecm_res = VECM(
        endog, exog, k_ar_diff=3, coint_rank=coint_rank, deterministic="co"
    ).fit()
    # ##### exog_fc not passed as argument:
    with pytest.raises(ValueError, match="exog_fc is None"):
        vecm_res.predict()
    # ##### exog_fc is too short:
    exog_fc = np.ones(STEPS)
    with pytest.raises(
        ValueError, match=r".*exog_fc must have at least steps elements.*"
    ):
        vecm_res.predict(
            5,
            None,
            exog_fc[:2],
        )
        # [:2] shortens exog_fc
    # ##### exog_coint_fc (NOT exog_fc) is passed when there is no exog_coint
    with pytest.raises(ValueError, match=r".*exog_coint attribute is None.*"):
        vecm_res.predict(
            5,
            None,
            exog_fc,
            exog_fc,
        )  # passed as exog_coint_fc-argument!
    # # with exog_coint
    exog_coint = []
    exog_coint.append(np.ones(len(endog)).reshape(-1, 1))
    exog_coint.append(1 + np.arange(len(endog)).reshape(-1, 1))
    exog_coint = np.hstack(exog_coint)
    vecm_res = VECM(endog, k_ar_diff=1, deterministic="n", exog_coint=exog_coint).fit()
    # ##### exog_coint_fc not passed as argument:
    with pytest.raises(ValueError, match=r"exog_coint_fc is None.*"):
        vecm_res.predict()
    # ##### exog_coint_fc is too short:
    exog_coint_fc = np.ones(STEPS)
    with pytest.raises(
        ValueError, match=r".*exog_coint_fc must have at least steps elements.*"
    ):
        vecm_res.predict(
            5,
            None,
            None,
            exog_coint_fc[:2],
        )  # [:2] shortens
    # ##### exog_fc (NOT exog_coint_fc) is passed when there is no exog
    with pytest.raises(ValueError, match=r".*exog attribute is None.*"):
        vecm_res.predict(
            5,
            None,
            exog_coint_fc,
        )  # passed as exog_fc-argument!


def test_select_coint_rank():  # This is only a smoke test.
    endog = data[datasets[0]]
    neqs = endog.shape[1]

    trace_result = select_coint_rank(endog, 0, 3, method="trace")
    rank = trace_result.rank
    r_1 = trace_result.r_1
    test_stats = trace_result.test_stats
    crit_vals = trace_result.crit_vals
    if rank > 0:
        assert_equal(r_1[0], r_1[1])
        for i in range(rank):
            assert (test_stats[i] > crit_vals[i])
    if rank < neqs:
        assert (test_stats[rank] < crit_vals[rank])

    maxeig_result = select_coint_rank(endog, 0, 3, method="maxeig", signif=0.1)
    rank = maxeig_result.rank
    r_1 = maxeig_result.r_1
    test_stats = maxeig_result.test_stats
    crit_vals = maxeig_result.crit_vals
    if maxeig_result.rank > 0:
        assert_equal(r_1[0], r_1[1] - 1)
        for i in range(rank):
            assert (test_stats[i] > crit_vals[i])
    if rank < neqs:
        assert (test_stats[rank] < crit_vals[rank])


def test_coint_rank_results_summary():
    # CointRankResults.summary is exported (as the return type of
    # select_coint_rank) but had no direct test coverage: it renders a
    # SimpleTable from the same rank/test_stats/crit_vals data already
    # validated numerically in test_select_coint_rank above.
    endog = data[datasets[0]]

    trace_result = select_coint_rank(endog, 0, 3, method="trace", signif=0.05)
    table = trace_result.summary()
    text = str(table)
    assert "trace" in text.lower()
    assert "5%" in text
    num_tests = min(trace_result.rank, trace_result.neqs - 1)
    assert len(table.data) == num_tests + 2  # header row + one per test
    for i in range(num_tests + 1):
        assert f"{trace_result.test_stats[i]:#0.4g}" in text

    maxeig_result = select_coint_rank(
        endog, 0, 3, method="maxeig", signif=0.1
    )
    text_maxeig = str(maxeig_result.summary())
    assert "maximum eigenvalue" in text_maxeig.lower()
    assert "10%" in text_maxeig


def test_VECM_seasonal_forecast():
    # timing of seasonal dummies, VAR forecast horizon
    rs = np.random.RandomState(964255)
    nobs = 200
    seasons = 6
    fact = np.cumsum(0.1 + rs.randn(nobs, 2), 0)

    xx = rs.randn(nobs + 2, 3)
    xx = xx[2:] + 0.6 * xx[1:-1] + 0.25 * xx[:-2]
    xx[:, :2] += fact[:, 0][:, None]
    xx[:, 2:] += fact[:, 1][:, None]
    # add large seasonal effect
    xx += 3 * np.log(0.1 + (np.arange(nobs)[:, None] % seasons))

    res0 = VECM(
        xx,
        k_ar_diff=0,
        coint_rank=2,
        deterministic="co",
        seasons=seasons,
        first_season=0,
    ).fit()
    res2 = VECM(
        xx,
        k_ar_diff=2,
        coint_rank=2,
        deterministic="co",
        seasons=seasons,
        first_season=0,
    ).fit()
    res4 = VECM(
        xx,
        k_ar_diff=4,
        coint_rank=2,
        deterministic="co",
        seasons=seasons,
        first_season=0,
    ).fit()

    # check that seasonal dummy are independent of number of lags
    assert_allclose(
        res2._delta_x.T[-2 * seasons :, -seasons:],
        res0._delta_x.T[-2 * seasons :, -seasons:],
        rtol=1e-10,
    )
    assert_allclose(
        res4._delta_x.T[-2 * seasons :, -seasons:],
        res0._delta_x.T[-2 * seasons :, -seasons:],
        rtol=1e-10,
    )

    # check location of smallest seasonal coefficient
    assert_array_equal(np.argmin(res0.det_coef, axis=1), [1, 1, 1])
    assert_array_equal(np.argmin(res2.det_coef, axis=1), [1, 1, 1])
    assert_array_equal(np.argmin(res4.det_coef, axis=1), [1, 1, 1])

    # predict 3 cycles, check location of dips
    dips_true = np.array([[4, 4, 4], [10, 10, 10], [16, 16, 16]])
    for res in [res0, res2, res4]:
        forecast = res.predict(steps=3 * seasons)
        dips = np.sort(np.argsort(forecast, axis=0)[:3], axis=0)
        assert_array_equal(dips, dips_true)
    # res2.plot_forecast(steps=18, alpha=0.1, n_last_obs=4*seasons)


@pytest.mark.thread_unsafe(reason="Uses matplotlib")
@pytest.mark.matplotlib
def test_plot_forecast_and_plot_data(close_figures):
    rs = np.random.RandomState(20260821)
    n = 100
    common_trend = np.cumsum(rs.standard_normal(n))
    endog = np.column_stack([
        common_trend + rs.standard_normal(n) * 0.3,
        common_trend + rs.standard_normal(n) * 0.3,
        common_trend + rs.standard_normal(n) * 0.3,
    ])

    res = VECM(endog, k_ar_diff=1, coint_rank=1, deterministic="ci").fit()

    fig = res.plot_forecast(steps=10)
    assert len(fig.axes) == endog.shape[1]

    fig_short = res.plot_forecast(steps=10, plot_conf_int=False, n_last_obs=20)
    assert len(fig_short.axes) == endog.shape[1]

    fig_data = res.plot_data()
    assert len(fig_data.axes) == endog.shape[1]
    for ax in fig_data.axes:
        assert len(ax.get_lines()) >= 1
        assert len(ax.get_lines()[0].get_ydata()) == n - res.k_ar

    fig_full = res.plot_data(with_presample=True)
    for ax in fig_full.axes:
        assert len(ax.get_lines()[0].get_ydata()) == n

    # plot_data must also work when the model was fit with a date index
    # (previously crashed: self.dates[self.k_ar:] on a None self.dates
    # was only reached through the *array* path above)
    import pandas as pd
    dated = pd.DataFrame(endog, index=pd.date_range("2000-01-01", periods=n, freq="D"))
    res_dated = VECM(dated, k_ar_diff=1, coint_rank=1, deterministic="ci").fit()
    fig_dated = res_dated.plot_data()
    assert len(fig_dated.axes) == endog.shape[1]


def test_select_order_negative_maxlags():
    # a negative maxlags used to produce an empty criteria dict and crash
    # with a bare KeyError('aic')
    rs = np.random.RandomState(12345)
    data = rs.standard_normal((60, 2))
    with pytest.raises(ValueError, match="non-negative"):
        select_order(data, maxlags=-2)
    # maxlags=0 (a single comparison) remains valid
    res = select_order(data, maxlags=0)
    assert res.aic in (0, 1)


def _cointegrated_data(n, alpha, drift, c_coint, seed):
    # Delta y_t = alpha (beta' y_{t-1} - c) + 0.4 Delta y_{t-1} + drift + e_t
    # with beta = (1, -1)'
    rng = np.random.default_rng(seed)
    alpha = np.asarray(alpha)
    burn = 200
    y = np.zeros((n + burn, 2))
    for t in range(2, n + burn):
        z = y[t - 1, 0] - y[t - 1, 1] - c_coint
        dy_prev = y[t - 1] - y[t - 2]
        y[t] = y[t - 1] + alpha * z + 0.4 * dy_prev + drift + rng.standard_normal(2)
    return y[burn:]


@pytest.mark.parametrize(
    "n, alpha, drift, c_coint, deterministic, names",
    [
        (1500, [-0.3, 0.2], [0.35, 0.3], 0.0, "co", ("alpha", "gamma", "det_coef")),
        (1500, [-0.3, 0.2], [0.0, 0.0], 1.5, "ci", ("alpha", "gamma", "det_coef_coint")),
        (200, [-0.03, 0.018], [0.0, 0.0], 0.0, "n", ("beta",)),
    ],
    ids=["det_coef", "det_coef_coint", "beta"],
)
def test_pvalues_large_tvalues(n, alpha, drift, c_coint, deterministic, names):
    # 2 * (1 - norm.cdf(|t|)) is exactly 0 for |t| above 8.3 and the p-values
    # of the significant parameters were reported as 0.0
    y = _cointegrated_data(n, alpha, drift, c_coint, 20261004)
    res = VECM(y, k_ar_diff=1, coint_rank=1, deterministic=deterministic).fit()
    for name in names:
        tvalues = np.abs(getattr(res, f"tvalues_{name}"))
        pvalues = getattr(res, f"pvalues_{name}")
        if name == "beta":
            # the first coint_rank rows are normalized, not estimated
            tvalues = tvalues[res.coint_rank :]
            pvalues = pvalues[res.coint_rank :]
        # the data have t-values in the range where the tail was lost
        lost = (tvalues > 8.5) & (tvalues < 30)
        assert np.any(lost)
        # 2 * norm.sf(t) = erfc(t / sqrt(2))
        assert_allclose(pvalues, special.erfc(tvalues / np.sqrt(2)), rtol=1e-10)
        assert np.all(pvalues[lost] > 0)


DETERMINISTIC = ["n", "co", "ci", "lo", "li", "coli", "cili", "colo", "cilo"]


def _danish():
    return danish_data.load_pandas().data[["lrm", "lry", "ibo", "ide"]]


def _det_terms(deterministic, nobs_tot, seasons=0, first_season=0):
    # The deterministic terms D_t of the docstring of VECM at the observations
    # t = 1, ..., nobs_tot: a constant, the linear trend t and centered
    # seasonal dummies. The terms inside the relation enter as D_{t-1}.
    one = np.ones((nobs_tot, 1))
    trend = np.arange(1, nobs_tot + 1, dtype=float)[:, None]
    coint, outside = [np.empty((nobs_tot, 0))], [np.empty((nobs_tot, 0))]
    if "ci" in deterministic:
        coint.append(one)
    if "li" in deterministic:
        coint.append(trend)
    if "co" in deterministic:
        outside.append(one)
    if seasons > 0:
        season = (np.arange(nobs_tot) + first_season) % seasons
        outside.append((season[:, None] == np.arange(seasons - 1)) - 1 / seasons)
    if "lo" in deterministic:
        outside.append(trend)
    return np.column_stack(coint), np.column_stack(outside)


def _regressors(y, k_ar_diff, coint_det, det):
    # Delta y_t, (y_{t-1}, D^co_{t-1}) and (Delta y_{t-1}, ...,
    # Delta y_{t-k_ar_diff}, D_t) for t = k_ar_diff + 1, ..., nobs_tot - 1
    # (Luetkepohl 2005, p. 286 and p. 299)
    p = k_ar_diff + 1
    dy = np.diff(y, axis=0)  # dy[t - 1] = y[t] - y[t - 1]
    z0 = dy[p - 1 :]
    z1 = np.column_stack([y[p - 1 : -1], coint_det[p - 1 : -1]])
    lags = [dy[p - 1 - i : -i] for i in range(1, p)]
    z2 = np.column_stack([*lags, det[p:]])
    return z0, z1, z2


def _johansen_rrr(y, k_ar_diff, coint_rank, coint_det, det):
    # The reduced rank regression of Johansen as the generalized symmetric
    # eigenvalue problem |lambda S11 - S10 S00^{-1} S01| = 0 (Luetkepohl 2005,
    # Section 7.2.3), a second method next to the matrix square root and the
    # non-symmetric eigenvalue problem of VECM
    z0, z1, z2 = _regressors(y, k_ar_diff, coint_det, det)
    nobs, neqs = z0.shape

    def partial_out(z):
        if z2.shape[1] == 0:
            return z
        return z - z2 @ np.linalg.lstsq(z2, z, rcond=None)[0]

    r0, r1 = partial_out(z0), partial_out(z1)
    s00 = r0.T @ r0 / nobs
    s01 = r0.T @ r1 / nobs
    s11 = r1.T @ r1 / nobs
    lam, vec = linalg.eigh(s01.T @ np.linalg.solve(s00, s01), s11)
    lam, vec = lam[::-1], vec[:, ::-1]
    beta = vec[:, :coint_rank] @ np.linalg.inv(vec[:coint_rank, :coint_rank])
    x = np.column_stack([z1 @ beta, z2])
    coef = np.linalg.lstsq(x, z0, rcond=None)[0].T
    resid = z0 - x @ coef.T
    log_det = np.linalg.slogdet(s00)[1] + np.sum(np.log(1 - lam[:coint_rank]))
    llf = -nobs / 2 * (neqs * np.log(2 * np.pi) + log_det + neqs)
    alpha, gamma = coef[:, :coint_rank], coef[:, coint_rank:]
    return beta, alpha, gamma, resid.T @ resid / nobs, llf


@pytest.mark.parametrize(
    "case, deterministic, seasons",
    [
        ("co", "co", 0),
        ("ci", "ci", 0),
        ("coli", "coli", 0),
        ("co_seasons", "co", 4),
    ],
)
def test_vecm_urca_vars(case, deterministic, seasons):
    # References from tests/results/results_vecm_urca.R (urca::ca.jo,
    # urca::cajorls and vars::vec2var). R solves the eigenvalue problem with
    # a pivoted Cholesky factor and eigen(), VECM with a matrix square root
    # and eig(), so the estimates agree to about 1e-8, not to the precision
    # of the machine.
    ref = results_vecm_urca.results[case]
    rank = ref["coint_rank"]
    res = VECM(
        _danish(),
        k_ar_diff=ref["k_ar_diff"],
        coint_rank=rank,
        deterministic=deterministic,
        seasons=seasons,
    ).fit()
    rtol = 1e-6
    # both normalize the first rank rows of beta to the identity; atol
    # because the off-diagonal zeros are zero up to rounding
    assert_allclose(res.beta[:rank], np.eye(rank), atol=1e-12)
    beta = np.vstack((res.beta, res.det_coef_coint))
    assert_allclose(beta[rank:], ref["beta"][rank:], rtol=rtol)
    assert_allclose(res.alpha, ref["alpha"], rtol=rtol)
    assert_allclose(res.gamma, ref["gamma"], rtol=rtol)
    if "const" in ref:
        assert_allclose(res.const[:, 0], ref["const"], rtol=rtol)
    else:
        assert res.const.shape == (4, 0)
    if "seasonal" in ref:
        assert_allclose(res.seasonal, ref["seasonal"], rtol=rtol)
    else:
        assert res.seasonal.shape == (4, 0)
    assert_allclose(res.var_rep, ref["var_rep"], rtol=rtol)
    assert_allclose(res.sigma_u, ref["sigma_u"], rtol=rtol)
    assert_allclose(res.llf, ref["llf"], rtol=1e-10)
    assert_allclose(res.ma_rep(5), ref["ma_rep"], rtol=rtol)
    # P defaults to the Cholesky factor of sigma_u, as in vars::Psi
    assert_allclose(res.orth_ma_rep(5), ref["orth_ma_rep"], rtol=rtol)
    chol = np.linalg.cholesky(res.sigma_u)
    assert_allclose(res.orth_ma_rep(5, P=chol), ref["orth_ma_rep"], rtol=rtol)
    irf = res.irf(5)
    assert_allclose(irf.irfs, ref["ma_rep"], rtol=rtol)
    assert_allclose(irf.orth_irfs, ref["orth_ma_rep"], rtol=rtol)
    assert_allclose(res.predict(5), ref["fcst"], rtol=rtol)
    fcst = res.predict(5, alpha=0.05)
    assert_allclose(fcst.point_forecast, ref["fcst"], rtol=rtol)
    assert_allclose(fcst.forc_lower, ref["lower"], rtol=rtol)
    assert_allclose(fcst.forc_upper, ref["upper"], rtol=rtol)


@pytest.mark.parametrize("deterministic", DETERMINISTIC)
@pytest.mark.parametrize(
    "k_ar_diff, coint_rank, seasons, first_season",
    [(0, 1, 0, 0), (1, 1, 0, 0), (2, 2, 4, 1)],
)
def test_vecm_deterministic_rrr(
    deterministic, k_ar_diff, coint_rank, seasons, first_season
):
    # Every deterministic string against the reduced rank regression with
    # the deterministic terms D_t (constant 1, trend t) built here
    y = _danish().to_numpy()
    res = VECM(
        y,
        k_ar_diff=k_ar_diff,
        coint_rank=coint_rank,
        deterministic=deterministic,
        seasons=seasons,
        first_season=first_season,
    ).fit()
    coint_det, det = _det_terms(deterministic, len(y), seasons, first_season)
    beta, alpha, gamma, sigma_u, llf = _johansen_rrr(
        y, k_ar_diff, coint_rank, coint_det, det
    )
    # the two eigenvalue methods agree to about 1e-8 (relative)
    rtol = 1e-6
    res_beta = np.vstack((res.beta, res.det_coef_coint))
    assert res_beta.shape == beta.shape
    assert_allclose(res_beta[coint_rank:], beta[coint_rank:], rtol=rtol)
    assert_allclose(res.alpha, alpha, rtol=rtol)
    res_gamma = np.hstack((res.gamma, res.det_coef))
    assert res_gamma.shape == gamma.shape
    assert_allclose(res_gamma, gamma, rtol=rtol)
    assert_allclose(res.sigma_u, sigma_u, rtol=rtol)
    assert_allclose(res.llf, llf, rtol=1e-9)
    # The log-likelihood from the eigenvalues (Luetkepohl 2005, eq. 7.2.20) is
    # the Gaussian log-likelihood at the estimates
    nobs, neqs = res.resid.shape
    sigma_resid = res.resid.T @ res.resid / nobs
    log_det = np.linalg.slogdet(sigma_resid)[1]
    gauss = -nobs / 2 * (neqs * np.log(2 * np.pi) + log_det + neqs)
    assert_allclose(res.llf, gauss, rtol=1e-9)
    assert_allclose(res.resid + res.fittedvalues, y[k_ar_diff + 1 :], rtol=1e-12)


@pytest.mark.parametrize(
    "alias, deterministic",
    [("nc", "n"), ("", "n"), ("CO", "co"), ("CiLi", "cili")],
)
def test_vecm_deterministic_alias(alias, deterministic):
    # "nc" and "" are accepted for "n", and the string is not case sensitive
    y = _danish().to_numpy()
    res = VECM(y, k_ar_diff=1, deterministic=alias).fit()
    expected = VECM(y, k_ar_diff=1, deterministic=deterministic).fit()
    assert_allclose(res.llf, expected.llf, rtol=1e-12)
    assert_allclose(res.det_coef_coint, expected.det_coef_coint, rtol=1e-12)
    assert_allclose(res.det_coef, expected.det_coef, rtol=1e-12)


@pytest.mark.parametrize(
    "deterministic, trend", [("n", "n"), ("co", "c"), ("colo", "ct")]
)
def test_vecm_full_rank_is_var_in_levels(deterministic, trend):
    # With coint_rank equal to neqs the rank restriction is void and the VECM
    # is the VAR in levels estimated by OLS
    y = _danish().to_numpy()
    neqs = y.shape[1]
    res = VECM(y, k_ar_diff=1, coint_rank=neqs, deterministic=deterministic).fit()
    var = VAR(y).fit(2, trend=trend)
    assert_allclose(res.var_rep, var.coefs, rtol=1e-8)
    assert_allclose(res.sigma_u, var.sigma_u_mle, rtol=1e-8)
    assert_allclose(res.llf, var.llf, rtol=1e-10)
    if trend == "c":
        assert_allclose(res.const[:, 0], var.params[0], rtol=1e-8)
    # beta is the identity (atol: zero up to rounding), so it has no estimated
    # element and no standard error
    assert_allclose(res.beta, np.eye(neqs), atol=1e-12)
    assert_array_equal(res.stderr_beta, np.zeros((neqs, neqs)))
    assert_array_equal(res.tvalues_beta, np.zeros((neqs, neqs)))
    assert_array_equal(res.pvalues_beta, np.zeros((neqs, neqs)))


@pytest.mark.parametrize("deterministic", DETERMINISTIC)
def test_vecm_inference_arrays(deterministic):
    # t-values, p-values and confidence intervals of every parameter group,
    # including the empty groups of deterministic terms
    res = VECM(
        _danish(), k_ar_diff=1, coint_rank=2, deterministic=deterministic
    ).fit()
    neqs, rank = 4, 2
    n_coint = ("ci" in deterministic) + ("li" in deterministic)
    n_outside = ("co" in deterministic) + ("lo" in deterministic)
    shapes = {
        "alpha": (neqs, rank),
        "gamma": (neqs, neqs),
        "det_coef_coint": (n_coint, rank),
        "det_coef": (neqs, n_outside),
    }
    signif = 0.1
    crit = stats.norm.ppf(1 - signif / 2)
    for name, shape in shapes.items():
        est = getattr(res, name)
        stderr = getattr(res, f"stderr_{name}")
        tvalues = getattr(res, f"tvalues_{name}")
        pvalues = getattr(res, f"pvalues_{name}")
        conf_int = getattr(res, f"conf_int_{name}")(alpha=signif)
        for arr in (est, stderr, tvalues, pvalues, conf_int):
            assert arr.shape == shape
        assert np.all(stderr > 0)
        assert_allclose(tvalues, est / stderr, rtol=1e-12)
        erfc = special.erfc(np.abs(tvalues) / np.sqrt(2))
        assert_allclose(pvalues, erfc, rtol=1e-10)
        assert_allclose(conf_int["lower"], est - crit * stderr, rtol=1e-12)
        assert_allclose(conf_int["upper"], est + crit * stderr, rtol=1e-12)
    # the first rank rows of beta are normalized, not estimated
    assert_array_equal(res.stderr_beta[:rank], np.zeros((rank, rank)))
    assert_array_equal(res.tvalues_beta[:rank], np.zeros((rank, rank)))
    assert_array_equal(res.pvalues_beta[:rank], np.zeros((rank, rank)))
    beta, stderr = res.beta[rank:], res.stderr_beta[rank:]
    assert_allclose(res.tvalues_beta[rank:], beta / stderr, rtol=1e-12)
    conf_int = res.conf_int_beta(alpha=signif)
    assert_allclose(conf_int["lower"][rank:], beta - crit * stderr, rtol=1e-12)


@pytest.mark.parametrize("deterministic", ["n", "ci", "colo"])
@pytest.mark.parametrize("k_ar_diff", [0, 2])
def test_vecm_cov_var_repr(deterministic, k_ar_diff):
    # cov_params_default is eq. (7.2.21) of Luetkepohl (2005) at the
    # estimated beta and cov_var_repr maps it to A_1 = I + Pi + Gamma_1,
    # A_i = Gamma_i - Gamma_{i-1} and A_p = -Gamma_{p-1}
    y = _danish().to_numpy()
    neqs, rank = y.shape[1], 1
    res = VECM(
        y, k_ar_diff=k_ar_diff, coint_rank=rank, deterministic=deterministic
    ).fit()
    coint_det, det = _det_terms(deterministic, len(y))
    _, z1, z2 = _regressors(y, k_ar_diff, coint_det, det)
    beta = np.vstack((res.beta, res.det_coef_coint))
    x = np.column_stack([z1 @ beta, z2])
    b = linalg.block_diag(beta, np.eye(z2.shape[1]))
    cov = np.kron(b @ np.linalg.inv(x.T @ x) @ b.T, res.sigma_u)
    # x'x is ill-conditioned (levels, trend and differences), so another order
    # of the sums changes the small elements at about 1e-10 (relative)
    assert_allclose(res.cov_params_default, cov, rtol=1e-7)
    # vec(Pi) and vec(Gamma_1), ..., vec(Gamma_{k_ar_diff}); the coefficients
    # of the deterministic terms are not part of the VAR coefficients
    k2 = neqs**2
    start = neqs * z1.shape[1]
    keep = np.r_[0:k2, start : start + k2 * k_ar_diff]
    cov_pi_gamma = cov[np.ix_(keep, keep)]
    assert_allclose(res.cov_params_wo_det, cov_pi_gamma, rtol=1e-7)
    p = k_ar_diff + 1
    jac = np.zeros((k2 * p, k2 * p))
    eye = np.eye(k2)
    jac[:k2, :k2] = eye
    for i in range(p):
        if i < p - 1:
            jac[i * k2 : (i + 1) * k2, (i + 1) * k2 : (i + 2) * k2] = eye
        if i > 0:
            jac[i * k2 : (i + 1) * k2, i * k2 : (i + 1) * k2] = -eye
    assert_allclose(res.cov_var_repr, jac @ cov_pi_gamma @ jac.T, rtol=1e-7)


def test_granger_causality_causing_default():
    # causing=None tests the remaining variables of the system
    res = VECM(_danish(), k_ar_diff=1, coint_rank=1, deterministic="co").fit()
    for caused, causing in [(0, [1, 2, 3]), ("ibo", ["lrm", "lry", "ide"])]:
        default = res.test_granger_causality(caused)
        explicit = res.test_granger_causality(caused, causing)
        names = [res.names[c] if isinstance(c, int) else c for c in causing]
        assert default.causing == names
        assert_allclose(default.test_statistic, explicit.test_statistic, rtol=1e-12)
        assert_allclose(default.pvalue, explicit.pvalue, rtol=1e-12)
        assert default.df == explicit.df
        assert default == explicit
    two = res.test_granger_causality([0, 3])
    assert two.causing == ["lry", "ibo"]
    assert two.caused == ["lrm", "ide"]


def test_granger_causality_invalid_types():
    res = VECM(_danish(), k_ar_diff=1, coint_rank=1).fit()
    with pytest.raises(TypeError, match="caused has to be"):
        res.test_granger_causality([0.5])
    with pytest.raises(TypeError, match="causing has to be"):
        res.test_granger_causality(0, [0.5])
    with pytest.raises(TypeError, match="causing has to be"):
        res.test_granger_causality(0, [1, None])


def test_causality_results_eq():
    res = VECM(_danish(), k_ar_diff=1, coint_rank=1, deterministic="co").fit()
    # instantaneous causality is symmetric: the statistic is the same with
    # causing and caused swapped, and so are the results
    inst = res.test_inst_causality(0)
    swapped = res.test_inst_causality([1, 2, 3])
    assert inst.causing == swapped.caused
    assert inst.caused == swapped.causing
    assert_allclose(inst.test_statistic, swapped.test_statistic, rtol=1e-10)
    assert inst == swapped
    # Granger causality is not
    granger = res.test_granger_causality([1, 2, 3], 0)
    granger_swapped = res.test_granger_causality(0, [1, 2, 3])
    assert granger != granger_swapped
    assert res.test_granger_causality(0, 1) != res.test_granger_causality(0, 2)
    # results of a different class are never equal
    assert inst != res.test_normality()
    assert res.test_normality() != res.test_whiteness(5)
    assert res.test_whiteness(5) != 1.0


@pytest.mark.parametrize("method", [None, "lr"])
def test_causality_results_method(method):
    with pytest.raises(ValueError, match="must not be None"):
        CausalityTestResults(
            ["y1"], ["y2"], 3.0, 3.84, 0.08, 1, 0.05, method=method
        )


@pytest.mark.parametrize("signif", [0.01, 0.05, 0.1])
def test_hypothesis_test_results_conclusion(signif):
    # reject if and only if the p-value is below the significance level
    res = VECM(_danish(), k_ar_diff=1, coint_rank=1, deterministic="co").fit()
    results = [
        res.test_granger_causality(0, signif=signif),
        res.test_granger_causality(2, signif=signif),
        res.test_inst_causality(0, signif=signif),
        res.test_normality(signif=signif),
        res.test_whiteness(5, signif=signif),
        res.test_whiteness(8, signif=signif, adjusted=True),
    ]
    for result in results:
        expected = "reject" if result.pvalue < signif else "fail to reject"
        assert result.conclusion == expected
        assert result.signif == signif
        # the summary prints four significant digits
        row = result.summary().data[1]
        assert_allclose(float(row[0]), result.test_statistic, rtol=1e-3)
        assert_allclose(float(row[1]), result.crit_value, rtol=1e-3)
        assert row[3] == str(result.df)


@pytest.mark.parametrize("deterministic", ["n", "co", "ci", "colo"])
@pytest.mark.parametrize("k_ar_diff", [0, 1])
def test_summary_tables(deterministic, k_ar_diff):
    res = VECM(
        _danish(), k_ar_diff=k_ar_diff, coint_rank=1, deterministic=deterministic
    ).fit()
    neqs = 4
    tables = res.summary(alpha=0.1).tables
    n_outside = ("co" in deterministic) + ("lo" in deterministic)
    lagged = neqs if n_outside + k_ar_diff > 0 else 0
    assert len(tables) == lagged + neqs + 1
    # coefficients are printed with four decimals (atol 5e-5)
    for i in range(lagged):
        rows = tables[i].data[1:]
        names = [row[0].strip() for row in rows]
        expected = ["const"] * ("co" in deterministic)
        expected += ["lin_trend"] * ("lo" in deterministic)
        expected += [f"L1.{n}" for n in res.names] * k_ar_diff
        assert names == expected
        params = np.r_[res.det_coef[i], res.gamma[i]]
        assert_allclose([float(row[1]) for row in rows], params, atol=5e-5)
    for i in range(neqs):
        rows = tables[lagged + i].data[1:]
        assert [row[0].strip() for row in rows] == ["ec1"]
        assert_allclose(float(rows[0][1]), res.alpha[i, 0], atol=5e-5)
    rows = tables[-1].data[1:]
    coint = np.r_[res.beta[:, 0], res.det_coef_coint[:, 0]]
    assert_allclose([float(row[1]) for row in rows], coint, atol=5e-5)


def test_coint_rank_results_str():
    endog = _danish()
    johansen = coint_johansen(endog, 0, 1)
    for method, stat, crit in [
        ("trace", johansen.lr1, johansen.cvt),
        ("maxeig", johansen.lr2, johansen.cvm),
    ]:
        result = select_coint_rank(endog, 0, 1, method=method, signif=0.05)
        rank = result.rank
        # the first test that is not rejected at 5% gives the rank
        assert np.all(stat[:rank] > crit[:rank, 1])
        assert stat[rank] < crit[rank, 1]
        assert_allclose(result.test_stats, stat[: rank + 1], rtol=1e-12)
        assert_allclose(result.crit_vals, crit[: rank + 1, 1], rtol=1e-12)
        text = str(result)
        assert text == result.summary().as_text()
        for value in result.test_stats:
            assert f"{value:#0.4g}" in text


@pytest.mark.parametrize(
    "form", ["list", "tuple", "datetime", "period", "range", "int64", "int32"]
)
def test_vecm_endog_forms(form):
    # endog is array_like: every form gives the estimates of the float ndarray
    data = _danish()
    endog = data.to_numpy()
    expected_names = ["y1", "y2", "y3", "y4"]
    if form == "list":
        data = endog.tolist()
    elif form == "tuple":
        data = tuple(map(tuple, endog))
    elif form == "period":
        index = pd.period_range("1974Q1", periods=len(data), freq="Q")
        data = data.set_axis(index)
        expected_names = list(data.columns)
    elif form == "range":
        data = data.reset_index(drop=True)
        expected_names = list(data.columns)
    elif form == "datetime":
        expected_names = list(data.columns)
    else:
        # integer data: the series in units of 0.01
        endog = np.round(100 * endog)
        data = endog.astype(form)
    res = VECM(data, k_ar_diff=1, coint_rank=1, deterministic="co").fit()
    expected = VECM(endog, k_ar_diff=1, coint_rank=1, deterministic="co").fit()
    assert res.names == expected_names
    for name in ("alpha", "beta", "gamma", "det_coef", "sigma_u"):
        assert getattr(res, name).dtype == np.float64
        assert_allclose(getattr(res, name), getattr(expected, name), rtol=1e-8)
    assert_allclose(res.llf, expected.llf, rtol=1e-12)
    assert_allclose(res.predict(3), expected.predict(3), rtol=1e-10)
    if form == "datetime":
        assert res.dates.equals(data.index)
        assert res.model.data.freq == data.index.freqstr
    elif form == "period":
        assert res.dates.equals(data.index)
    else:
        assert res.dates is None


def test_vecm_dates_freq():
    endog = _danish().to_numpy()
    dates = pd.date_range("1974-01-01", periods=len(endog), freq="QS")
    res = VECM(endog, dates=dates, freq="QS", k_ar_diff=1).fit()
    expected = VECM(endog, k_ar_diff=1).fit()
    assert res.dates.equals(dates)
    assert res.model.data.freq == dates.freqstr
    assert_allclose(res.alpha, expected.alpha, rtol=1e-12)


@pytest.mark.parametrize("form", ["list", "1d", "dataframe"])
def test_vecm_exog_forms(form):
    # exog is array_like: a list, a 1-d array and a DataFrame for one column
    data = _danish()
    endog = data.to_numpy()
    exog = np.sqrt(np.arange(1.0, len(endog) + 1))[:, None]
    if form == "list":
        exog_arg = exog.tolist()
    elif form == "1d":
        exog_arg = exog[:, 0]
    else:
        exog_arg = pd.DataFrame(exog, index=data.index, columns=["sqrt_t"])
    res = VECM(endog, exog=exog_arg, k_ar_diff=1, deterministic="co").fit()
    expected = VECM(endog, exog=exog, k_ar_diff=1, deterministic="co").fit()
    assert_allclose(res.llf, expected.llf, rtol=1e-12)
    assert_allclose(res.exog_coefs, expected.exog_coefs, rtol=1e-10)
    # exog is a deterministic term outside the relation, as D_t
    coint_det, det = _det_terms("co", len(endog))
    det = np.column_stack([det, exog])
    _, alpha, gamma, _, llf = _johansen_rrr(endog, 1, 1, coint_det, det)
    assert_allclose(res.alpha, alpha, rtol=1e-7)
    assert_allclose(np.hstack((res.gamma, res.det_coef)), gamma, rtol=1e-7)
    assert_allclose(res.llf, llf, rtol=1e-9)
    exog_fc = np.sqrt(np.arange(len(endog) + 1.0, len(endog) + 4))[:, None]
    fcst = res.predict(3, exog_fc=exog_fc)
    assert_allclose(fcst, expected.predict(3, exog_fc=exog_fc), rtol=1e-10)
    assert len(res.summary().tables) == 4 + 4 + 1


@pytest.mark.parametrize("first_season", [1, 2, 3, 5, -1])
def test_vecm_first_season(first_season):
    # With centered dummies the seasonal terms span the same space for any
    # first season, so only the labels of the seasonal coefficients change:
    # the other estimates and the forecasts are the same. first_season is
    # taken modulo seasons.
    endog = _danish().to_numpy()
    res = VECM(
        endog, k_ar_diff=1, deterministic="co", seasons=4, first_season=first_season
    ).fit()
    expected = VECM(endog, k_ar_diff=1, deterministic="co", seasons=4).fit()
    assert_allclose(res.llf, expected.llf, rtol=1e-12)
    assert_allclose(res.alpha, expected.alpha, rtol=1e-8)
    assert_allclose(res.predict(9), expected.predict(9), rtol=1e-10)
    coint_det, det = _det_terms("co", len(endog), 4, first_season)
    _, _, gamma, _, _ = _johansen_rrr(endog, 1, 1, coint_det, det)
    assert_allclose(np.hstack((res.gamma, res.det_coef)), gamma, rtol=1e-7)


def test_vecm_missing():
    endog = _danish().to_numpy()
    with_nan = endog.copy()
    with_nan[20, 1] = np.nan
    with pytest.raises(MissingDataError):
        VECM(with_nan, k_ar_diff=1, missing="raise")
    res = VECM(with_nan, k_ar_diff=1, missing="drop").fit()
    keep = np.isfinite(with_nan).all(axis=1)
    expected = VECM(endog[keep], k_ar_diff=1).fit()
    assert res.nobs == expected.nobs == len(endog) - 1 - 2
    assert_allclose(res.alpha, expected.alpha, rtol=1e-12)
    assert_allclose(res.llf, expected.llf, rtol=1e-12)


@pytest.mark.parametrize("form", ["list", "dataframe"])
def test_select_order_data_forms(form):
    data = _danish()
    endog = data.to_numpy()
    data = endog.tolist() if form == "list" else data
    res = select_order(data, 3, deterministic="co", seasons=4)
    expected = select_order(endog, 3, deterministic="co", seasons=4)
    assert res.selected_orders == expected.selected_orders
    for ic in ("aic", "bic", "hqic", "fpe"):
        assert_allclose(res.ics[ic], expected.ics[ic], rtol=1e-12)


@pytest.mark.parametrize(
    "deterministic, k_ar_diff, coint_rank, seasons, expected",
    [
        ("co", 1, 1, 0, 6.4475421068456012e02),
        ("ci", 1, 1, 0, 6.4385197559552034e02),
        ("coli", 2, 2, 0, 6.5056586141151206e02),
        ("co", 1, 1, 4, 6.7010675372892763e02),
    ],
)
def test_llf_real(deterministic, k_ar_diff, coint_rank, seasons, expected):
    # llf was complex on NumPy 2.5 and later, where np.linalg.eig returns
    # complex eigenvalues also when they are real. The references are
    # logLik(vars::vec2var(ca.jo, r = coint_rank)) for
    # urca::ca.jo(y, ecdet = "none", "const" or "trend", K = k_ar_diff + 1,
    # spec = "transitory", season = NULL or 4) on the same data (R 4.6.2,
    # urca 1.3.4, vars 1.6.1).
    from statsmodels.datasets import danish_data

    data = danish_data.load_pandas().data[["lrm", "lry", "ibo", "ide"]]
    res = VECM(
        data,
        k_ar_diff=k_ar_diff,
        coint_rank=coint_rank,
        deterministic=deterministic,
        seasons=seasons,
    ).fit()
    assert isinstance(res.llf, float)
    assert_allclose(res.llf, expected, rtol=1e-10)
    # eq. (7.2.20) of Luetkepohl (2005) is the Gaussian log-likelihood at
    # the estimates
    nobs, neqs = res.resid.shape
    log_det = np.linalg.slogdet(res.resid.T @ res.resid / nobs)[1]
    gauss = -nobs / 2 * (neqs * np.log(2 * np.pi) + log_det + neqs)
    assert_allclose(res.llf, gauss, rtol=1e-9)


@pytest.mark.parametrize("coint_rank", [5, 7])
def test_vecm_coint_rank_above_neqs(coint_rank):
    # A coint_rank above neqs was accepted silently: the model is the one
    # with rank neqs, the VAR in levels estimated by OLS, but the given rank
    # is kept and used in the degrees of freedom of test_whiteness. This is
    # deprecated.
    from statsmodels.datasets import danish_data
    from statsmodels.tsa.vector_ar.var_model import VAR

    y = danish_data.load_pandas().data[["lrm", "lry", "ibo", "ide"]].to_numpy()
    neqs = y.shape[1]
    with pytest.warns(FutureWarning, match="coint_rank must be between 0 and neqs"):
        model = VECM(y, k_ar_diff=1, coint_rank=coint_rank)
    res = model.fit()
    var = VAR(y).fit(2, trend="n")
    assert_allclose(res.var_rep, var.coefs, rtol=1e-8)
    assert_allclose(res.llf, var.llf, rtol=1e-10)
    assert res.coint_rank == coint_rank
    nlags = 6
    assert res.test_whiteness(nlags).df == neqs**2 * (nlags - 1) - neqs * coint_rank


@pytest.mark.parametrize("coint_rank", [0, 4])
def test_vecm_coint_rank_bounds(coint_rank):
    # 0 and neqs are valid ranks
    import warnings

    from statsmodels.datasets import danish_data

    y = danish_data.load_pandas().data[["lrm", "lry", "ibo", "ide"]].to_numpy()
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        res = VECM(y, k_ar_diff=1, coint_rank=coint_rank).fit()
    assert res.alpha.shape == (4, coint_rank)


@pytest.mark.parametrize("coint_rank", [-1, -4])
def test_vecm_coint_rank_negative(coint_rank):
    # A negative coint_rank raised "negative dimensions are not allowed" in fit
    from statsmodels.datasets import danish_data

    y = danish_data.load_pandas().data[["lrm", "lry", "ibo", "ide"]].to_numpy()
    with pytest.raises(ValueError, match="coint_rank must be between 0 and neqs"):
        VECM(y, k_ar_diff=1, coint_rank=coint_rank)


@pytest.mark.parametrize(
    "deterministic, equivalent",
    [("cico", "co"), ("xyz", "n"), ("c", "n"), ("ct", "n"), ("lilo", "lo")],
)
def test_select_order_deterministic_deprecated(deterministic, equivalent):
    # select_order did not check deterministic: a value that VECM does not
    # accept used the terms whose names it contains. This is deprecated, and
    # the orders are unchanged.
    from statsmodels.datasets import danish_data

    y = danish_data.load_pandas().data[["lrm", "lry", "ibo", "ide"]].to_numpy()
    with pytest.warns(FutureWarning, match="deterministic must be one of"):
        res = select_order(y, 3, deterministic=deterministic)
    expected = select_order(y, 3, deterministic=equivalent)
    assert res.selected_orders == expected.selected_orders
    for ic in ("aic", "bic", "hqic", "fpe"):
        assert_allclose(res.ics[ic], expected.ics[ic], rtol=1e-12)


@pytest.mark.parametrize(
    "deterministic, equivalent",
    [("nc", "n"), ("", "n"), ("CO", "co"), ("CiLi", "cili"), ("colo", "colo")],
)
def test_select_order_deterministic_valid(deterministic, equivalent):
    # the values of VECM, its aliases of "n" and any case do not warn
    import warnings

    from statsmodels.datasets import danish_data

    y = danish_data.load_pandas().data[["lrm", "lry", "ibo", "ide"]].to_numpy()
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        res = select_order(y, 3, deterministic=deterministic)
        expected = select_order(y, 3, deterministic=equivalent)
    assert res.selected_orders == expected.selected_orders
    for ic in ("aic", "bic", "hqic", "fpe"):
        assert_allclose(res.ics[ic], expected.ics[ic], rtol=1e-12)


@pytest.mark.parametrize(
    "caused, causing",
    [
        (np.int64(0), np.int32(2)),
        (np.array([0, 1], dtype=np.int32), np.int64(3)),
        ([np.intp(3)], [np.int16(0), np.uint8(1), np.intc(2)]),
    ],
)
def test_causality_numpy_integer_indices(caused, causing):
    # numpy integers are indices of the variables as int is: a scalar raised
    # "object is not iterable" and an element of a sequence raised "has to be
    # of type string or int"
    from statsmodels.datasets import danish_data

    data = danish_data.load_pandas().data[["lrm", "lry", "ibo", "ide"]]
    res = VECM(data, k_ar_diff=1, coint_rank=1, deterministic="co").fit()

    def names(value):
        return [data.columns[int(v)] for v in np.atleast_1d(value)]

    granger = res.test_granger_causality(caused, causing)
    expected = res.test_granger_causality(names(caused), names(causing))
    assert granger.caused == expected.caused == names(caused)
    assert granger.causing == expected.causing == names(causing)
    assert_allclose(granger.test_statistic, expected.test_statistic, rtol=1e-12)
    assert granger.df == expected.df
    inst = res.test_inst_causality(causing)
    expected = res.test_inst_causality(names(causing))
    assert inst.causing == expected.causing == names(causing)
    assert_allclose(inst.test_statistic, expected.test_statistic, rtol=1e-12)


def test_causality_float_index():
    # A float is not an index: a scalar raised "object is not iterable"
    from statsmodels.datasets import danish_data

    data = danish_data.load_pandas().data[["lrm", "lry", "ibo", "ide"]]
    res = VECM(data, k_ar_diff=1, coint_rank=1, deterministic="co").fit()
    with pytest.raises(TypeError, match="caused has to be of type string or int"):
        res.test_granger_causality(0.5)
    with pytest.raises(TypeError, match="causing has to be of type string or int"):
        res.test_granger_causality(0, 1.0)
    with pytest.raises(TypeError, match="causing has to be of type string or int"):
        res.test_inst_causality(0.5)
    with pytest.raises(TypeError, match="causing has to be of type string or int"):
        res.test_inst_causality([0, 1.0])
