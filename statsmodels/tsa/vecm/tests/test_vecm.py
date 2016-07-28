from __future__ import absolute_import, print_function

import numpy as np
from numpy.testing import assert_, assert_allclose
import pandas

import statsmodels.datasets.interest_inflation.data as e6
from statsmodels.tsa.base.datetools import dates_from_str
from .results_JMulTi.parse_jmulti_output import load_results_jmulti
from .results_JMulTi.parse_jmulti_output import dt_s_tup_to_string
from statsmodels.tsa.vecm.vecm import VECM
from statsmodels.tsa.vector_ar.var_model import VARProcess

atol = 0.003  # absolute tolerance
rtol = 0.01  # relative tolerance
datasets = []
data = {}
results_ref = {}
results_sm = {}
coint_rank = 1

debug_mode = True
dont_test_se_t_p = False
deterministic_terms_list = ["nc", "co", "colo", "ci", "cili"]
seasonal_list = [0, 4]
dt_s_list = [(dt, s) for dt in deterministic_terms_list for s in seasonal_list]
all_tests = ["Gamma", "alpha", "beta", "C", "det_coint", "Sigma_u",
             "VAR repr. A", "VAR to VEC representation", "log_like"]
to_test = all_tests  # ["beta"]


def load_data(dataset, data_dict):  # TODO: make function compatible with other data_dict sets.
    # ...todo... e.g. by making the local variable "variables" part of data_dict.py.
    variables = ["Dp", "R"]
    loaded = dataset.load().data[variables].view(float, type=np.ndarray)
    data_dict[dataset] = loaded.reshape((-1, len(variables)))


def load_results_statsmodels(dataset):
    results_per_deterministic_terms = dict.fromkeys(dt_s_list)
    for dt_s_tup in dt_s_list:
        model = VECM(data[dataset])
        # print("\n\n\nDETERMINISTIC TERMS: " + deterministic_terms)
        results_per_deterministic_terms[dt_s_tup] = model.fit(
                                        diff_lags=3, method="ml",
                                        deterministic=dt_s_tup[0],
                                        seasons=dt_s_tup[1],
                                        coint_rank=coint_rank)
    return results_per_deterministic_terms


def build_err_msg(ds, dt_s, parameter_str):
    dt = dt_s_tup_to_string(dt_s)
    seasons = dt_s[1]
    err_msg = "Error in " + parameter_str + " for:\n"
    err_msg += "- Dataset: " + ds.__str__() + "\n"
    err_msg += "- Deterministic terms: "
    err_msg += (dt_s[0] if dt != "nc" else "no det. terms")
    if seasons > 0:
        err_msg += ", seasons: " + str(seasons)
    return err_msg


def test_ml_gamma():
    if debug_mode:
        if "Gamma" not in to_test:
            return
        print("\n\nGAMMA", end="")
    for ds in datasets:
        for dt_s in dt_s_list:
            if debug_mode:
                print("\n" + dt_s_tup_to_string(dt_s) + ": ", end="")

            # estimated parameter vector
            err_msg = build_err_msg(ds, dt_s, "Gamma")
            obtained = results_sm[ds][dt_s].gamma
            desired = results_ref[ds][dt_s]["est"]["Gamma"]
            # cols = desired.shape[1]
            # if obtained.shape[1] > cols:
            #     obtained = obtained[:, :cols]
            yield assert_allclose, obtained, desired, rtol, atol, False, \
                err_msg
            if debug_mode and dont_test_se_t_p:
                continue
            # standard errors
            obt = results_sm[ds][dt_s].stderr_gamma#[:, :cols]
            des = results_ref[ds][dt_s]["se"]["Gamma"]
            yield assert_allclose, obt, des, rtol, atol, False, \
                "STANDARD ERRORS\n"+err_msg
            # t-values
            obt = results_sm[ds][dt_s].tvalues_gamma#[:, :cols]
            des = results_ref[ds][dt_s]["t"]["Gamma"]
            yield assert_allclose, obt, des, rtol, atol, False, \
                "t-VALUES\n"+err_msg
            # p-values
            obt = results_sm[ds][dt_s].pvalues_gamma#[:, :cols]
            des = results_ref[ds][dt_s]["p"]["Gamma"]
            yield assert_allclose, obt, des, rtol, atol, False, \
                "p-VALUES\n"+err_msg


def test_ml_alpha():
    if debug_mode:
        if "alpha" not in to_test:
            return
        print("\n\nALPHA", end="")
    for ds in datasets:
        for dt in dt_s_list:
            if debug_mode:
                print("\n" + dt_s_tup_to_string(dt) + ": ", end="")

            err_msg = build_err_msg(ds, dt, "alpha")
            obtained = results_sm[ds][dt].alpha
            desired = results_ref[ds][dt]["est"]["alpha"]
            yield assert_allclose, obtained, desired, rtol, atol, False, \
                err_msg

            if debug_mode and dont_test_se_t_p:
                continue
            # standard errors
            obt = results_sm[ds][dt].stderr_alpha
            des = results_ref[ds][dt]["se"]["alpha"]
            yield assert_allclose, obt, des, rtol, atol, False, \
                "STANDARD ERRORS\n"+err_msg
            # t-values
            obt = results_sm[ds][dt].tvalues_alpha
            des = results_ref[ds][dt]["t"]["alpha"]
            yield assert_allclose, obt, des, rtol, atol, False, \
                "t-VALUES\n"+err_msg
            # p-values
            obt = results_sm[ds][dt].pvalues_alpha
            des = results_ref[ds][dt]["p"]["alpha"]
            yield assert_allclose, obt, des, rtol, atol, False, \
                "p-VALUES\n"+err_msg


def test_ml_beta():
    if debug_mode:
        if "beta" not in to_test:
            return
        print("\n\nBETA", end="")
    for ds in datasets:
        for dt in dt_s_list:
            if debug_mode:
                print("\n" + dt_s_tup_to_string(dt) + ": ", end="")

            err_msg = build_err_msg(ds, dt, "beta")
            desired = results_ref[ds][dt]["est"]["beta"]
            rows = desired.shape[0]
            # - first coint_rank rows in JMulTi output have se=t_val=p_val=0
            # - beta includes deterministic terms in cointegration relation in
            #   sm, so we compare only the elements belonging to beta.
            obtained = results_sm[ds][dt].beta[coint_rank:rows]
            desired = desired[coint_rank:]
            # print("\ndeterministic terms: " + dt)
            # print("tested elements of beta: ")
            # print(str(slice(coint_rank,rows)))
            yield assert_allclose, obtained, desired, rtol, atol, False, \
                err_msg

            if debug_mode and dont_test_se_t_p:
                continue
            # standard errors
            obt = results_sm[ds][dt].stderr_beta[coint_rank:rows]
            des = results_ref[ds][dt]["se"]["beta"][coint_rank:]
            yield assert_allclose, obt, des, rtol, atol, False, \
                "STANDARD ERRORS\n"+err_msg
            # t-values
            obt = results_sm[ds][dt].tvalues_beta[coint_rank:rows]
            des = results_ref[ds][dt]["t"]["beta"][coint_rank:]
            yield assert_allclose, obt, des, rtol, atol, False, \
                "t-VALUES\n"+err_msg
            # p-values
            obt = results_sm[ds][dt].pvalues_beta[coint_rank:rows]
            des = results_ref[ds][dt]["p"]["beta"][coint_rank:]
            yield assert_allclose, obt, des, rtol, atol, False, \
                "p-VALUES\n"+err_msg


def test_ml_c():  # test deterministic terms outside coint relation
    if debug_mode:
        if "C" not in to_test:
            return
        print("\n\nDET_COEF", end="")
    for ds in datasets:
        for dt in dt_s_list:
            if debug_mode:
                print("\n" + dt_s_tup_to_string(dt) + ": ", end="")

            C_obt = results_sm[ds][dt].det_coef
            se_C_obt = results_sm[ds][dt].stderr_det_coef
            t_C_obt = results_sm[ds][dt].tvalues_det_coef
            p_C_obt = results_sm[ds][dt].pvalues_det_coef

            if "C" not in results_ref[ds][dt]["est"].keys():
                # case: there are no deterministic terms
                if C_obt.size == 0 and se_C_obt.size == 0  \
                        and t_C_obt.size == 0 and p_C_obt.size == 0:
                    yield assert_, True, "no const & seasonal terms"
                    continue

            desired = results_ref[ds][dt]["est"]["C"]
            dt_string = dt_s_tup_to_string(dt)
            if "co" in dt_string:
                const_obt = C_obt[:, :1]
                const_des = desired[:, :1]
                C_obt = C_obt[:, 1:]
                desired = desired[:, 1:]
                yield assert_allclose, const_obt, const_des,  \
                      rtol, atol, False, build_err_msg(ds, dt, "CONST")
            if "s" in dt_string:
                if "lo" in dt_string:
                    seas_obt = C_obt[:, :-1]
                    seas_des = desired[:, :-1]
                else:
                    seas_obt = C_obt
                    seas_des = desired
                yield assert_allclose, seas_obt, seas_des,  \
                      rtol, atol, False, build_err_msg(ds, dt, "SEASONAL")
            if "lo" in dt_string:
                lt_obt = C_obt[:, -1:]
                lt_des = desired[:, -1:]
                yield assert_allclose, lt_obt, lt_des,  \
                      rtol, atol, False, build_err_msg(ds, dt, "LINEAR TREND")
            if debug_mode and dont_test_se_t_p:
                continue
            # standard errors
            se_desired = results_ref[ds][dt]["se"]["C"]
            if "co" in dt_string:
                se_const_obt = se_C_obt[:, 0][:, None]
                se_C_obt = se_C_obt[:, 1:]
                se_const_des = se_desired[:, 0][:, None]
                se_desired = se_desired[:, 1:]
                yield assert_allclose, se_const_obt, se_const_des,  \
                      rtol, atol, False, build_err_msg(ds, dt, "SE CONST")
            if "s" in dt_string:
                if "lo" in dt_string:
                    se_seas_obt = se_C_obt[:, :-1]
                    se_seas_des = se_desired[:, :-1]
                else:
                    se_seas_obt = se_C_obt
                    se_seas_des = se_desired
                yield assert_allclose, se_seas_obt, se_seas_des,  \
                      rtol, atol, False, build_err_msg(ds, dt, "SE SEASONAL")
                if "lo" in dt_string:
                    se_lt_obt = se_C_obt[:, -1:]
                    se_lt_des = se_desired[:, -1:]
                    yield assert_allclose, se_lt_obt, se_lt_des,  \
                          rtol, atol, False,  \
                          build_err_msg(ds, dt, "SE LIN. TREND")
            # t-values
            t_desired = results_ref[ds][dt]["t"]["C"]
            if "co" in dt_string:
                t_const_obt = t_C_obt[:, 0][:, None]
                t_C_obt = t_C_obt[:, 1:]
                t_const_des = t_desired[:, 0][:, None]
                t_desired = t_desired[:, 1:]
                yield assert_allclose, t_const_obt, t_const_des,  \
                      rtol, atol, False, build_err_msg(ds, dt, "T CONST")
            if "s" in dt_string:
                if "lo" in dt_string:
                    t_seas_obt = t_C_obt[:, :-1]
                    t_seas_des = t_desired[:, :-1]
                else:
                    t_seas_obt = t_C_obt
                    t_seas_des = t_desired
                yield assert_allclose, t_seas_obt, t_seas_des,  \
                      rtol, atol, False, build_err_msg(ds, dt, "T SEASONAL")
            if "lo" in dt_string:
                t_lt_obt = t_C_obt[:, -1:]
                t_lt_des = t_desired[:, -1:]
                yield assert_allclose, t_lt_obt, t_lt_des,  \
                      rtol, atol, False,  \
                      build_err_msg(ds, dt, "T LIN. TREND")
            # p-values
            p_desired = results_ref[ds][dt]["p"]["C"]
            if "co" in dt_string:
                p_const_obt = p_C_obt[:, 0][:, None]
                p_C_obt = p_C_obt[:, 1:]
                p_const_des = p_desired[:, 0][:, None]
                p_desired = p_desired[:, 1:]
                yield assert_allclose, p_const_obt, p_const_des,  \
                      rtol, atol, False, build_err_msg(ds, dt, "P CONST")
            if "s" in dt_string:
                if "lo" in dt_string:
                    p_seas_obt = p_C_obt[:, :-1]
                    p_seas_des = p_desired[:, :-1]
                else:
                    p_seas_obt = p_C_obt
                    p_seas_des = p_desired
                yield assert_allclose, p_seas_obt, p_seas_des,  \
                      rtol, atol, False, build_err_msg(ds, dt, "P SEASONAL")
            if "lo" in dt_string:
                p_lt_obt = p_C_obt[:, -1:]
                p_lt_des = p_desired[:, -1:]
                yield assert_allclose, p_lt_obt, p_lt_des,  \
                      rtol, atol, False,  \
                      build_err_msg(ds, dt, "P LIN. TREND")



def test_ml_det_terms_in_coint_relation():
    if debug_mode:
        if "det_coint" not in to_test:
            return
        print("\n\nDET_COEF_COINT", end="")
    for ds in datasets:
        for dt in dt_s_list:
            if debug_mode:
                print("\n" + dt_s_tup_to_string(dt) + ": ", end="")

            err_msg = build_err_msg(ds, dt, "det terms in coint relation")

            dt_string = dt_s_tup_to_string(dt)
            obtained = results_sm[ds][dt].det_coef_coint
            if "ci" not in dt_string and "li" not in dt_string:
                if obtained.size > 0:
                    yield assert_, False, build_err_msg(ds, dt,
                        "There should not be any det terms in cointegration "
                        + "for deterministic terms " + dt_string)
                else:
                    yield assert_, True
                continue
            desired = results_ref[ds][dt]["est"]["det_coint"]
            yield assert_allclose, obtained, desired, rtol, atol, False, \
                err_msg
            # TODO: test se, t, p!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!


def test_ml_sigma():
    if debug_mode:
        if "Sigma_u" not in to_test:
            return
        print("\n\nSIGMA_U", end="")
    for ds in datasets:
        for dt in dt_s_list:
            if debug_mode:
                print("\n" + dt_s_tup_to_string(dt) + ": ", end="")

            err_msg = build_err_msg(ds, dt, "Sigma_u")
            obtained = results_sm[ds][dt].sigma_u
            desired = results_ref[ds][dt]["est"]["Sigma_u"]
            yield assert_allclose, obtained, desired, rtol, atol, False, err_msg


def test_var_rep():
    if debug_mode:
        if "VAR repr. A" not in to_test:
            return
        print("\n\nVAR REPRESENTATION", end="")
    for ds in datasets:
        for dt in dt_s_list:
            if debug_mode:
                print("\n" + dt_s_tup_to_string(dt) + ": ", end="")

            err_msg = build_err_msg(ds, dt, "VAR repr. A")
            obtained = results_sm[ds][dt].var_repr
            p = obtained.shape[0]
            desired = np.hsplit(results_ref[ds][dt]["est"]["VAR A"], p)
            yield assert_allclose, obtained, desired, rtol, atol, False, err_msg


def test_var_to_vecm():
    if debug_mode:
        if "VAR to VEC representation" not in to_test:
            return
        print("\n\nVAR TO VEC", end="")
    for ds in datasets:
        for dt in dt_s_list:
            if debug_mode:
                print("\n" + dt_s_tup_to_string(dt) + ": ", end="")

            err_msg = build_err_msg(ds, dt, "VAR to VEC representation")
            sigma_u = results_sm[ds][dt].sigma_u
            coefs = results_sm[ds][dt].var_repr
            intercept = np.zeros(len(sigma_u))
            var = VARProcess(coefs, intercept, sigma_u)
            vecm_results = var.to_vecm()
            obtained_pi = vecm_results["Pi"]
            obtained_gamma = vecm_results["Gamma"]

            desired_pi = np.dot(results_sm[ds][dt].alpha,
                                results_sm[ds][dt].beta.T)
            desired_gamma = results_sm[ds][dt].gamma
            yield assert_allclose, obtained_pi, desired_pi, \
                  rtol, atol, False, err_msg+" Pi"
            yield assert_allclose, obtained_gamma, desired_gamma, \
                  rtol, atol, False, err_msg+" Gamma"

# Commented out since JMulTi shows the same det. terms for both VEC & VAR repr.
# def test_var_rep_det():
#     for ds in datasets:
#         for dt in dt_s_list:
#             if dt != "nc":
#                 err_msg = build_err_msg(ds, dt, "VAR repr. deterministic")
#                 obtained = 0  # not implemented since the same values as VECM
#                 desired = results_ref[ds][dt]["est"]["VAR deterministic"]
#                 yield assert_allclose, obtained, desired, rtol, atol, False, err_msg


def test_log_like():
    if debug_mode:
        if "log_like" not in to_test:
            return
        else:
            print("\n\nLOG LIKELIHOOD", end="")
    for ds in datasets:
        for dt in dt_s_list:
            if debug_mode:
                print("\n" + dt_s_tup_to_string(dt) + ": ", end="")

            err_msg = build_err_msg(ds, dt, "Log Likelihood")
            obtained = results_sm[ds][dt].llf
            desired = results_ref[ds][dt]["log_like"]
            yield assert_allclose, obtained, desired, rtol, atol, False, err_msg


def setup():
    datasets.append(e6)  # TODO: append more data sets for more test cases.
    
    for ds in datasets:
        load_data(ds, data)
        results_ref[ds] = load_results_jmulti(ds, dt_s_list)
        results_sm[ds] = load_results_statsmodels(ds)
        return results_sm[ds], results_ref[ds]


