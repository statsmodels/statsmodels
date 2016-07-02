import numpy as np
from numpy.testing import assert_, assert_allclose
import pandas

import statsmodels.datasets.interest_inflation.data as e6
from statsmodels.tsa.base.datetools import dates_from_str
from results.parse_jmulti_output import load_results_jmulti
from statsmodels.tsa.vecm.vecm import VECM


atol = 0.005 # absolute tolerance
rtol = 0.01  # relative tolerance
datasets = []
data = {}
results_ref = {}
results_sm = {}
deterministic_terms_list = ["", "c", "cs", "clt"]  # TODO: add combinations


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
        results_per_deterministic_terms[deterministic_terms] = model.fit(
                                        max_diff_lags=3, method="ml", 
                                        deterministic=deterministic_terms)
        results_per_deterministic_terms[deterministic_terms]["VAR A"] = \
            model.to_var(max_diff_lags=3, method="ml",
                         deterministic=deterministic_terms)
    return results_per_deterministic_terms


def build_err_msg(ds, dt, parameter_str):
    err_msg = "Error in " + parameter_str + " for:\n"
    err_msg += "- Dataset: " + ds.__str__() + "\n"
    err_msg += "- Deterministic terms: "
    err_msg += (dt if dt != "" else "no det. terms")
    return err_msg


def test_ml_gamma():
    for ds in datasets:
        for dt in deterministic_terms_list:
            err_msg = build_err_msg(ds, dt, "Gamma")
            obtained = results_sm[ds][dt]["Gamma"]
            desired = results_ref[ds][dt]["Gamma"]
            cols = desired.shape[1]
            if obtained.shape[1] > cols:
                obtained = obtained[:, :cols]
            yield assert_allclose, obtained, desired, rtol, atol, False, err_msg


def test_ml_alpha():
    for ds in datasets:
        for dt in deterministic_terms_list:
            err_msg = build_err_msg(ds, dt, "alpha")
            obtained = results_sm[ds][dt]["alpha"]
            desired = results_ref[ds][dt]["alpha"]
            yield assert_allclose, obtained, desired, rtol, atol, False, err_msg


def test_ml_beta():
    for ds in datasets:
        for dt in deterministic_terms_list:
            err_msg = build_err_msg(ds, dt, "beta")
            obtained = results_sm[ds][dt]["beta"]
            desired = results_ref[ds][dt]["beta"].T  # JMulTi: beta transposed
            rows = desired.shape[0]
            if obtained.shape[0] > rows:
                obtained = obtained[:rows]
            yield assert_allclose, obtained, desired, rtol, atol, False, err_msg


def test_ml_c():
    for ds in datasets:
        for dt in deterministic_terms_list:
            err_msg = build_err_msg(ds, dt, "C")
            
            gamma_sm = results_sm[ds][dt]["Gamma"]
            gamma_ref = results_ref[ds][dt]["Gamma"]
            
            beta_sm = results_sm[ds][dt]["beta"]
            beta_ref = results_ref[ds][dt]["beta"].T  # JMulTi: beta transposed

            if "C" not in results_ref[ds][dt].keys():
                # case: there are no deterministic terms
                if (gamma_sm.shape[1] == gamma_ref.shape[1] and
                        beta_sm.shape[0] == beta_ref.shape[0]):
                    yield assert_, True
                    continue
            cols = gamma_ref.shape[1]
            if gamma_sm.shape[1] > cols:
                obtained = gamma_sm[:, cols:]
            desired = results_ref[ds][dt]["C"]
            yield assert_allclose, obtained, desired, rtol, atol, False, err_msg


def test_ml_lin_trend():
    for ds in datasets:
        for dt in deterministic_terms_list:
            err_msg = build_err_msg(ds, dt, "linear trend coefficients")
            
            beta_sm = results_sm[ds][dt]["beta"]
            beta_ref = results_ref[ds][dt]["beta"].T  # JMulTi: beta transposed
            if "lt" not in dt:
                if (beta_sm.shape[0] == beta_ref.shape[0] and  # sm: no trend
                        "lin_trend" not in results_ref[ds][dt]):  # JMulTi:n.t.
                    yield assert_, True
                else:
                    yield assert_, False, err_msg
                continue
            a = results_sm[ds][dt]["alpha"]
            b = results_sm[ds][dt]["beta"]
            # obtained = take last col of Pi and make it 2 dimensional:
            obtained = np.dot(a, b.T)[:, -1][:,None]
            desired = results_ref[ds][dt]["lin_trend"]
            yield assert_allclose, obtained, desired, rtol, atol, False, err_msg


def test_ml_sigma():
    for ds in datasets:
        for dt in deterministic_terms_list:
            err_msg = build_err_msg(ds, dt, "Sigma_u")
            obtained = results_sm[ds][dt]["Sigma_u"]
            desired = results_ref[ds][dt]["Sigma_u"]
            yield assert_allclose, obtained, desired, rtol, atol, False, err_msg


def test_var_rep_A():
    for ds in datasets:
        for dt in deterministic_terms_list:
            err_msg = build_err_msg(ds, dt, "VAR repr. A")
            obtained = results_sm[ds][dt]["VAR A"]
            desired = results_ref[ds][dt]["VAR A"]
            yield assert_allclose, obtained, desired, rtol, atol, False, err_msg


# Commented out since JMulTi shows the same det. terms for both VEC & VAR repr.
# def test_var_rep_det():
#     for ds in datasets:
#         for dt in deterministic_terms_list:
#             if dt != "":
#                 err_msg = build_err_msg(ds, dt, "VAR repr. deterministic")
#                 obtained = 0  # not implemented since the same values as VECM
#                 desired = results_ref[ds][dt]["VAR deterministic"]
#                 yield assert_allclose, obtained, desired, rtol, atol, False, err_msg


def setup():
    datasets.append(e6)  # TODO: append more data sets for more test cases.
    
    for ds in datasets:
        load_data(ds)
        results_ref[ds] = load_results_jmulti(ds)
        results_sm[ds] = load_results_statsmodels(ds)
        return results_sm[ds], results_ref[ds]


if __name__ == "__main__":
    np.testing.run_module_suite()

