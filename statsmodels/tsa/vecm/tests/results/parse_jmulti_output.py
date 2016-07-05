import re
import os

import numpy as np


deterministic_terms_list = ['', 'c', 'cs', 'clt', 'lt']


def load_results_jmulti(dataset):
    source = 'jmulti'

    results_for_different_det_terms = dict.fromkeys(deterministic_terms_list)
        
    for deterministic_terms in deterministic_terms_list:
        file = dataset.__str__()+'_'+source+'_'+deterministic_terms+'.txt'
        file = os.path.join(os.path.dirname(os.path.realpath(__file__)), file)
        # sections in jmulti output:
        section_header = ["Lagged endogenous term",  # Gamma
                          "Deterministic term",      # c, s, lt
                          "Loading coefficients",    # alpha
                          "Estimated cointegration relation",  # beta
                          "Legend",
                          "Lagged endogenous term",  # VAR representation
                          "Deterministic term"]      # VAR representation
        # the following "sections" will serve as key for the corresponding
        # result values
        sections = ["Gamma", 
                    "C",     # Here C as well as linear trend coefficients are collected.
                             # Later, the linear trend coefficients are stripped from C.
                    "alpha", 
                    "beta",
                    "Legend",
                    "VAR A",  # VAR parameter matrices
                    "VAR deterministic"]  # VAR deterministic terms
        if deterministic_terms == '':
            del(section_header[1])
            del(sections[1])
            del(section_header[-1])
            del(sections[-1])
        results = dict.fromkeys(sections)

        section = -1
        result = []
        rows = 0
        started_reading_section = False
        start_end_mark = "-----"
        # parse information about \alpha, \beta, \Gamma, deterministic of VECM
        # and A_i and deterministic of corresponding VAR:
        for line in open(file):
            if section == -1 and section_header[section+1] not in line:
                continue
            if section < len(section_header)-1 \
                    and section_header[section+1] in line:  # new section
                section += 1
                continue
            if not started_reading_section:
                if line.startswith(start_end_mark):
                    started_reading_section = True
                continue
            if started_reading_section:
                if line.startswith(start_end_mark):
                    results[sections[section]] = np.array(result)
                    result = []
                    started_reading_section = False
                    continue
                regex_number = re.compile("\s-?\d+\.\d+")
                matrix_col = re.findall(regex_number, line)
                if matrix_col == []:  # No values found in this line==>continue
                    continue
                if result == []:
                    rows = len(matrix_col)
                    result = [[] for i in range(rows)]
                for i in range(rows):
                    result[i].append(float(matrix_col[i]))
        # delete "Legend"-section of JMulTi:
        del results["Legend"]

        # parse  information regarding \Sigma_u
        sigmau_file = dataset.__str__()+'_'+source+'_'+deterministic_terms+'_Sigmau'+'.txt'
        sigmau_file = os.path.join(os.path.dirname(os.path.realpath(__file__)), sigmau_file)
        rows_to_parse = 0
        # all numbers of Sigma_u in notation with e (e.g. 2.283862e-05)
        regex_number = re.compile("\s+\S+e\S+")
        sigmau_section_reached = False
        for line in open(sigmau_file):
            if not sigmau_section_reached and not 'Covariance:' in line:
                continue
            if 'Covariance:' in line:
                sigmau_section_reached = True
                row = re.findall(regex_number, line)
                rows_to_parse = len(row)  # Sigma_u quadratic ==> #rows==#cols
                Sigma_u = np.empty((rows_to_parse, rows_to_parse))
            row = re.findall(regex_number, line)
            rows_to_parse -= 1
            Sigma_u[rows_to_parse] = row # rows are added in reverse order
            if rows_to_parse == 0:
                break
        results["Sigma_u"] = Sigma_u[::-1]

        results_for_different_det_terms[deterministic_terms] = results
        if 'lt' in deterministic_terms:
            C = results_for_different_det_terms[deterministic_terms]["C"]
            lt = (C[:,-1])[:, None] # take last col and make it 2-dimensional
            C = C[:, :-1]
            if C.shape[1] == 0:
                del results_for_different_det_terms[deterministic_terms]["C"]
            else:
                results_for_different_det_terms[deterministic_terms]["C"] = C
            results_for_different_det_terms[deterministic_terms]["lin_trend"] = lt
    
    return results_for_different_det_terms

if __name__ == "__main__":
    print(load_results_jmulti("e6"))