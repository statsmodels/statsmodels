import numpy as np
from numpy.testing import assert_, assert_allclose
import pandas
import scipy
import statsmodels.datasets.interest_inflation.data as e6
from statsmodels.tsa.base.datetools import dates_from_str
import statsmodels.api as sm
from statsmodels.tsa.vecm.vecm import VECM # TODO: possible to use sm here to shorten path?
import re
import os


atol = 0.005 # absolute tolerance
rtol = 0.01  # relative tolerance
datasets = []
data        = {}
results_ref = {}
results_sm  = {}
deterministic_terms_list = ['', 'c', 'cs', 'clt']#['', 'c', 'clt'] TODO: add more combinations

def load_data(dataset): # TODO: make this function compatible with other datasets
                    #       by passing 'year', 'quarter', ..., 'R' as parameter
                    #       ('year' and 'quarter' only necessery if other datasets
                    #       not quaterly.
    iidata = dataset.load_pandas()
    mdata = iidata.data
    dates = mdata[['year', 'quarter']].astype(int).astype(str)
    quarterly = dates["year"] + "Q" + dates["quarter"]
    quarterly = dates_from_str(quarterly)
    mdata = mdata[['Dp','R']]
    mdata.index = pandas.DatetimeIndex(quarterly)
    data[dataset] = mdata

def load_results_jmulti(dataset):
    source = 'jmulti'

    results_per_deterministic_terms = dict.fromkeys(deterministic_terms_list)

    regex_Gamma = re.compile("Lagged endogenous term")
    regex_C     = re.compile("Deterministic term")
    regex_alpha = re.compile("Loading coefficients")
    regex_beta  = re.compile("Estimated cointegration relation")
    regex_VAR   = re.compile("VAR REPRESENTATION")
        
    for deterministic_terms in deterministic_terms_list:
        directory = "results"
        file = dataset.__str__()+'_'+source+'_'+deterministic_terms+'.txt'
        file = os.path.join(directory, file)
        section_regex = [regex_Gamma, regex_C, regex_alpha, regex_beta, regex_VAR]
        sections = ["Gamma", "C", "alpha", "beta"]
        if deterministic_terms in ['', 'lt']: # TODO: check if jmulti lacks Deterministic
            print("no section CONST")
            del(section_regex[1])             #       section if det. term == 'lt'
            del(sections[1])
        results = dict.fromkeys(sections)

        section = -1
        result = []
        col_len = 0
        for line in open(file):
            if re.search(section_regex[section+1], line):
                section += 1
                #print("result: "+str(result))
                #print("### section: "+str(section))
                if section == 0:
                    continue
                results[sections[section-1]] = np.array(result)
                result = []
                if section == len(sections):
                    break
            regex_numbers = re.compile("\s-?\d+\.\d+")
            matrix_col = re.findall(regex_numbers, line)
            if matrix_col == []:
                #print("No values found, continue.")
                continue
            if result == []:
                col_len = len(matrix_col)
                result = [[] for i in range(col_len)]
            for i in range(col_len):
                #print(i)
                result[i].append(float(matrix_col[i]))
        results_per_deterministic_terms[deterministic_terms] = results
    return results_per_deterministic_terms

def load_results_statsmodels(dataset):
    results_per_deterministic_terms = dict.fromkeys(deterministic_terms_list)
    for deterministic_terms in deterministic_terms_list:
        model = VECM(data[dataset])
        results_per_deterministic_terms[deterministic_terms] = model.fit(
                                        max_diff_lags=3, method='ml', 
                                        deterministic_terms=deterministic_terms)
    return results_per_deterministic_terms

def build_err_msg(ds, dt, parameter_str):
    err_msg = "Error in " + parameter_str + " for:\n"
    err_msg = err_msg + "- Dataset: " + ds.__str__() + "\n"
    err_msg = err_msg + "- Deterministic terms: " + (dt if dt!='' else 'no det. terms')
    return err_msg

def test_ml_Gamma():

    for ds in datasets:
        for dt in deterministic_terms_list:
            err_msg = build_err_msg(ds, dt, "Gamma")
            obtained = results_sm[ds][dt]["Gamma"]
            desired  = results_ref[ds][dt]["Gamma"]
            cols = desired.shape[1]
            if obtained.shape[1] > cols:
                obtained = obtained[:, :cols]
            yield assert_allclose, obtained, desired, rtol, atol, False, err_msg

def test_ml_alpha():
    for ds in datasets:
        for dt in deterministic_terms_list:
            err_msg = build_err_msg(ds, dt, "alpha")
            obtained = results_sm[ds][dt]["alpha"]
            desired  = results_ref[ds][dt]["alpha"]
            yield assert_allclose, obtained, desired, rtol, atol, False, err_msg

def test_ml_beta():
    for ds in datasets:
        for dt in deterministic_terms_list:
            err_msg = build_err_msg(ds, dt, "beta")
            obtained = results_sm[ds][dt]["beta"]
            desired  = results_ref[ds][dt]["beta"].T # beta transposed in JMulTi
            rows = desired.shape[0]
            if obtained.shape[0] > rows:
                obtained = obtained[:rows]
            yield assert_allclose, obtained, desired, rtol, atol, False, err_msg


def setup():
    datasets.append(e6) # append more data sets for more test cases.
    
    for ds in datasets:
        load_data(ds)
        results_ref[ds] = load_results_jmulti(ds)
        results_sm[ds] = load_results_statsmodels(ds)
        #print("JMulTi:")
        #print(results_ref[ds])
        #print("===============================================")
        #print("statsmodels:")
        #print(results_sm[ds])
       # TODO: yield test for each dataset ds and each value of deterministic terms

if __name__ == "__main__":
    np.testing.run_module_suite()

