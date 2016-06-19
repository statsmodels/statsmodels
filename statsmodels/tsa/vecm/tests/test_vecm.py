import numpy as np
from numpy.testing import assert_
import pandas
import scipy
import statsmodels.datasets.interest_inflation.data as e6
from statsmodels.tsa.base.datetools import dates_from_str
import statsmodels.api as sm
from statsmodels.tsa.vecm.vecm import VECM # TODO: possible to use sm here to shorten path?
import re
import os


datasets = []
data        = {}
results_ref = {}
results_sm  = {}

def load_data(data_set): # TODO: make this function compatible with other datasets
                    #       by passing 'year', 'quarter', ..., 'R' as parameter
                    #       ('year' and 'quarter' only necessery if other datasets
                    #       not quaterly.
    iidata = data_set.load_pandas()
    mdata = iidata.data
    dates = mdata[['year', 'quarter']].astype(int).astype(str)
    quarterly = dates["year"] + "Q" + dates["quarter"]
    quarterly = dates_from_str(quarterly)
    mdata = mdata[['Dp','R']]
    mdata.index = pandas.DatetimeIndex(quarterly)
    data[data_set] = mdata

def load_results_jmulti():
    source = 'jmulti'
    deterministic_terms_list = ['', 'c', 'cs', 'clt']

    results_per_deterministic_terms = dict.fromkeys(deterministic_terms_list)

    regex_Gamma = re.compile("Lagged endogenous term")
    regex_C     = re.compile("Deterministic term")
    regex_alpha = re.compile("Loading coefficients")
    regex_beta  = re.compile("Estimated cointegration relation")
    regex_VAR   = re.compile("VAR REPRESENTATION")
    
    section_regex = [regex_Gamma, regex_C, regex_alpha, regex_beta, regex_VAR]
    sections = ["Gamma", "C", "alpha", "beta"]
        
    for deterministic_terms in deterministic_terms_list:
        directory = "results"
        file = datasets[0].__str__()+'_'+source+'_'+deterministic_terms+'.txt' # TODO:
                                                          # loop over several datasets
        file = os.path.join(directory, file)
        if deterministic_terms in ['', 'lt']: # TODO: check if jmulti lacks Deterministic
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

def setup():
    datasets.append(e6)
    for ds in datasets:
        load_data(ds)
        results_ref[ds] = load_results_jmulti()
        results_sm[ds] = VECM(data[ds]) # TODO: call VECM more often 
                                        # (with different deterministic_terms
        # TODO: yield test for each dataset ds and each value of deterministic terms


if __name__ == "__main__":
    np.testing.run_module_suite()

