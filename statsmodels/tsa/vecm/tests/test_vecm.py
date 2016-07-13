from __future__ import absolute_import

import numpy as np
from numpy.testing import assert_, assert_allclose
import pandas

import statsmodels.datasets.interest_inflation.data as e6
from statsmodels.tsa.base.datetools import dates_from_str
from .results.parse_jmulti_output import load_results_jmulti
from statsmodels.tsa.vecm.vecm import VECM
from statsmodels.tsa.vector_ar.var_model import VARProcess

atol = 0.005  # absolute tolerance
rtol = 0.01  # relative tolerance
datasets = []
data = {}
results_ref = {}
results_sm = {}
coint_rank = 1

debug_mode = False
deterministic_terms_list = ["", "c", "cs", "clt"]  # TODO: add combinations
all_tests = ["Gamma", "alpha", "beta", "C", "lin_trend", "Sigma_u",
             "VAR repr. A", "VAR to VEC representation", "log_like"]
to_test = ["C"]



def load_data(dataset):  # TODO: make this function compatible with other
    # datasets by passing "year", "quarter", ..., "R" as parameter ("year" and
    # "quarter" only necessary if other datasets not quaterly.
    iidata = dataset.load_pandas()
    mdata = iidata.data
    dates = mdata[["year", "quarter"]].astype(int).astype(str)
    quarterly = dates["year"] + "Q" + dates["quarter"]
    quarterly = dates_from_str(quarterly)
    mdata = mdata[["Dp", "R"]]
    mdata.index = pandas.DatetimeIndex(quarterly)
    data[dataset] = mdata


def load_results_statsmodels(dataset):
    results_per_deterministic_terms = dict.fromkeys(deterministic_terms_list)
    for deterministic_terms in deterministic_terms_list:
        model = VECM(data[dataset])
        # print("\n\n\nDETERMINISTIC TERMS: " + deterministic_terms)
        results_per_deterministic_terms[deterministic_terms] = model.fit(
                                        max_diff_lags=3, method="ml", 
                                        deterministic=deterministic_terms,
                                        coint_rank=coint_rank)
    return results_per_deterministic_terms


def build_err_msg(ds, dt, parameter_str):
    err_msg = "Error in " + parameter_str + " for:\n"
    err_msg += "- Dataset: " + ds.__str__() + "\n"
    err_msg += "- Deterministic terms: "
    err_msg += (dt if dt != "" else "no det. terms")
    return err_msg


def test_ml_gamma():
    if debug_mode and "Gamma" not in to_test:
        return
    for ds in datasets:
        for dt in deterministic_terms_list:
            # estimated parameter vector
            err_msg = build_err_msg(ds, dt, "Gamma")
            obtained = results_sm[ds][dt].gamma
            desired = results_ref[ds][dt]["est"]["Gamma"]
            cols = desired.shape[1]
            if obtained.shape[1] > cols:
                obtained = obtained[:, :cols]
            yield assert_allclose, obtained, desired, rtol, atol, False, err_msg
            # standard errors
            obt = results_sm[ds][dt].stderr_gamma[:, :cols]
            des = results_ref[ds][dt]["se"]["Gamma"]
            yield assert_allclose, obt, des, rtol, atol, False, \
                  "STANDARD ERRORS\n"+err_msg
            # t-values
            obt = results_sm[ds][dt].tvalues_gamma[:, :cols]
            des = results_ref[ds][dt]["t"]["Gamma"]
            yield assert_allclose, obt, des, rtol, atol, False, \
                  "t-VALUES\n"+err_msg
            # p-values
            obt = results_sm[ds][dt].pvalues_gamma[:, :cols]
            des = results_ref[ds][dt]["p"]["Gamma"]
            yield assert_allclose, obt, des, rtol, atol, False, \
                  "p-VALUES\n"+err_msg


def test_ml_alpha():
    if debug_mode and "alpha" not in to_test:
        return
    for ds in datasets:
        for dt in deterministic_terms_list:
            err_msg = build_err_msg(ds, dt, "alpha")
            obtained = results_sm[ds][dt].alpha
            desired = results_ref[ds][dt]["est"]["alpha"]
            yield assert_allclose, obtained, desired, rtol, atol, False, err_msg
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
    if debug_mode and "beta" not in to_test:
        return
    for ds in datasets:
        for dt in deterministic_terms_list:
            err_msg = build_err_msg(ds, dt, "beta")
            desired = results_ref[ds][dt]["est"]["beta"].T  # JMulTi: beta transposed
            rows = desired.shape[0]
            # - first coint_rank rows in JMulTi output have se=t_val=p_val=0
            # - beta includes deterministic terms in cointegration relation in
            #   sm, so we compare only the elements belonging to beta.
            obtained = results_sm[ds][dt].beta[coint_rank:rows]
            desired = desired[coint_rank:]
            print("\ndeterministic terms: " + dt)
            print("tested elements of beta: ")
            print(str(slice(coint_rank,rows)))
            yield assert_allclose, obtained, desired, rtol, atol, False, err_msg
            # standard errors
            obt = results_sm[ds][dt].stderr_beta[coint_rank:rows]
            des = results_ref[ds][dt]["se"]["beta"].T[coint_rank:]
            yield assert_allclose, obt, des, rtol, atol, False, \
                  "STANDARD ERRORS\n"+err_msg
            # t-values
            obt = results_sm[ds][dt].tvalues_beta[coint_rank:rows]
            des = results_ref[ds][dt]["t"]["beta"].T[coint_rank:]
            yield assert_allclose, obt, des, rtol, atol, False, \
                  "t-VALUES\n"+err_msg
            # p-values
            obt = results_sm[ds][dt].pvalues_beta[coint_rank:rows]
            des = results_ref[ds][dt]["p"]["beta"].T[coint_rank:]
            yield assert_allclose, obt, des, rtol, atol, False, \
                  "p-VALUES\n"+err_msg


def test_ml_c():  # test const outside coint relation and seasonal terms
    if debug_mode and "C" not in to_test:
        return
    for ds in datasets:
        for dt in deterministic_terms_list:

            C_obt = results_sm[ds][dt].det_coef  # coefs for const & seasonal
            se_C_obt = results_sm[ds][dt].stderr_det_coef
            t_C_obt = results_sm[ds][dt].tvalues_det_coef
            p_C_obt = results_sm[ds][dt].stderr_det_coef # TODO!!! pval instead of se

            if "C" not in results_ref[ds][dt]["est"].keys():
                # case: there are no deterministic terms
                if C_obt.size == 0 and se_C_obt.size == 0  \
                        and t_C_obt.size == 0 and p_C_obt.size == 0:
                    yield assert_, True, "no const & seasonal terms"
                    continue

            desired = results_ref[ds][dt]["est"]["C"]
            if "c" in dt:
                const_obt = C_obt[:, 0][:, None]
                const_des = desired[:, 0][:, None]
                C_obt = C_obt[:, 1:]
                desired = desired[:, 1:]
                yield assert_allclose, const_obt, const_des,  \
                      rtol, atol, False, build_err_msg(ds, dt, "CONST")
            if "s" in dt:
                seas_obt = C_obt
                seas_des = desired if "lt" not in dt else desired[:, :-1]
                yield assert_allclose, seas_obt, seas_des,  \
                      rtol, atol, False, build_err_msg(ds, dt, "SEASONAL")
            # standard errors
            se_desired = results_ref[ds][dt]["se"]["C"]
            if "c" in dt:
                se_const_obt = se_C_obt[:, 0][:, None]
                se_C_obt = se_C_obt[:, 1:]
                se_const_des = se_desired[:, 0][:, None]
                se_desired = se_desired[:, 1:]
                yield assert_allclose, se_const_obt, se_const_des,  \
                      rtol, atol, False, build_err_msg(ds, dt, "SE CONST")
            if "s" in dt:
                se_seas_obt = se_C_obt
                se_seas_des = se_desired if "lt" not in dt else se_desired[:, :-1]
                yield assert_allclose, se_seas_obt, se_seas_des,  \
                      rtol, atol, False, build_err_msg(ds, dt, "SE SEASONAL")
            # t-values
            t_desired = results_ref[ds][dt]["t"]["C"]
            if "c" in dt:
                t_const_obt = t_C_obt[:, 0][:, None]
                t_C_obt = t_C_obt[:, 1:]
                t_const_des = t_desired[:, 0][:, None]
                t_desired = t_desired[:, 1:]
                yield assert_allclose, t_const_obt, t_const_des,  \
                      rtol, atol, False, build_err_msg(ds, dt, "T CONST")
            if "s" in dt:
                t_seas_obt = t_C_obt
                t_seas_des = t_desired if "lt" not in dt else t_desired[:, :-1]
                yield assert_allclose, t_seas_obt, t_seas_des,  \
                      rtol, atol, False, build_err_msg(ds, dt, "T SEASONAL")
            # p-values
            p_desired = results_ref[ds][dt]["p"]["C"]
            if "c" in dt:
                p_const_obt = p_C_obt[:, 0][:, None]
                p_C_obt = p_C_obt[:, 1:]
                p_const_des = p_desired[:, 0][:, None]
                p_desired = p_desired[:, 1:]
                yield assert_allclose, p_const_obt, p_const_des,  \
                      rtol, atol, False, build_err_msg(ds, dt, "T CONST")
            if "s" in dt:
                p_seas_obt = p_C_obt
                p_seas_des = p_desired if "lt" not in dt else p_desired[:, :-1]
                yield assert_allclose, p_seas_obt, p_seas_des,  \
                      rtol, atol, False, build_err_msg(ds, dt, "T SEASONAL")

def test_ml_lin_trend():
    if debug_mode and "lin_trend" not in to_test:
        return
    for ds in datasets:
        for dt in deterministic_terms_list:
            err_msg = build_err_msg(ds, dt, "linear trend coefficients")
            
            beta_sm = results_sm[ds][dt].beta
            beta_ref = results_ref[ds][dt]["est"]["beta"].T  # JMulTi: beta transposed
            if "lt" not in dt:
                if (beta_sm.shape[0] == beta_ref.shape[0] and  # sm: no trend
                        "lin_trend" not in results_ref[ds][dt]["est"]):  # JMulTi:n.t.
                    yield assert_, True
                else:
                    yield assert_, False, err_msg
                continue
            a = results_sm[ds][dt].alpha
            b = results_sm[ds][dt].beta
            # obtained = take last col of Pi and make it 2 dimensional:
            obtained = np.dot(a, b.T)[:, -1][:, None]
            desired = results_ref[ds][dt]["est"]["lin_trend"]
            yield assert_allclose, obtained, desired, rtol, atol, False, err_msg
            # TODO: test se, t, p


def test_ml_sigma():
    if debug_mode and "Sigma_u" not in to_test:
        return
    for ds in datasets:
        for dt in deterministic_terms_list:
            err_msg = build_err_msg(ds, dt, "Sigma_u")
            obtained = results_sm[ds][dt].sigma_u
            desired = results_ref[ds][dt]["est"]["Sigma_u"]
            yield assert_allclose, obtained, desired, rtol, atol, False, err_msg


def test_var_rep():
    if debug_mode and "VAR repr. A" not in to_test:
        return
    for ds in datasets:
        for dt in deterministic_terms_list:
            err_msg = build_err_msg(ds, dt, "VAR repr. A")
            obtained = results_sm[ds][dt].var_repr
            p = obtained.shape[0]
            desired = np.hsplit(results_ref[ds][dt]["est"]["VAR A"], p)
            yield assert_allclose, obtained, desired, rtol, atol, False, err_msg


def test_var_to_vecm():
    if debug_mode and "VAR to VEC representation" not in to_test:
        return
    for ds in datasets:
        for dt in deterministic_terms_list:
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
#         for dt in deterministic_terms_list:
#             if dt != "":
#                 err_msg = build_err_msg(ds, dt, "VAR repr. deterministic")
#                 obtained = 0  # not implemented since the same values as VECM
#                 desired = results_ref[ds][dt]["est"]["VAR deterministic"]
#                 yield assert_allclose, obtained, desired, rtol, atol, False, err_msg


def test_log_like():
    if debug_mode and "log_like" not in to_test:
        return
    for ds in datasets:
        for dt in deterministic_terms_list:
            err_msg = build_err_msg(ds, dt, "Log Likelihood")
            obtained = results_sm[ds][dt].llf
            desired = results_ref[ds][dt]["log_like"]
            yield assert_allclose, obtained, desired, rtol, atol, False, err_msg


def setup():
    datasets.append(e6)  # TODO: append more data sets for more test cases.
    
    for ds in datasets:
        load_data(ds)
        results_ref[ds] = load_results_jmulti(ds)
        results_sm[ds] = load_results_statsmodels(ds)
        return results_sm[ds], results_ref[ds]


if __name__ == "__main__":
    np.testing.run_module_suite()

