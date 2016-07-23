import re
import os

import numpy as np

debug_mode = True


def print_debug_output(results, dt):
        print("\n\n\nDETERMINISTIC TERMS: " + dt)
        alpha = results["est"]["alpha"]
        print("alpha:")
        print(str(type(alpha)) + str(alpha.shape))
        print(alpha)
        print("se: ")
        print(results["se"]["alpha"])
        print("t: ")
        print(results["t"]["alpha"])
        print("p: ")
        print(results["p"]["alpha"])
        beta = results["est"]["beta"]
        print("beta:")
        print(str(type(beta)) + str(beta.shape))
        print(beta)
        Gamma = results["est"]["Gamma"]
        print("Gamma:")
        print(str(type(Gamma)) + str(Gamma.shape))
        print(Gamma)
        if dt:
            C = results["est"]["C"]
            print("C:")
            print(str(type(C)) + str(C.shape))
            print(C)
            print("se: ")
            print(results["se"]["C"])


def dt_s_tup_to_string(dt_s_tup):
    dt_string = dt_s_tup[0]  # string for identifying the file to parse.
    if dt_s_tup[1] > 0:  # if there are seasons in the model
        if "co" in dt_string or "cc" in dt_string:
            dt_string = dt_string[:2] + "s" + dt_string[2:]
        else:
            dt_string = "s" + dt_string
    return dt_string


def load_results_jmulti(dataset, dt_s_list):
    """

    Parameters
    ----------
    dataset : module
        A data module in the statsmodels/datasets directory that defines a
        __str__() method returning the dataset's name.
    dt_s_list : list
        A list of strings where each string represents a combination of
        deterministic terms.

    Returns
    -------
    result : dict
        A dict (keys: tuples of deterministic terms and seasonal terms)
        of dicts (keys: strings "est" (for estimators),
                              "se" (for standard errors),
                              "t" (for t-values),
                              "p" (for p-values))
        of dicts (keys: strings "alpha", "beta", "Gamma" and other results)
    """
    source = "jmulti"

    results_dict_per_det_terms = dict.fromkeys(dt_s_list)
        
    for dt_s in dt_s_list:
        dt_string = dt_s_tup_to_string(dt_s)
        file = dataset.__str__()+"_"+source+"_"+dt_string+".txt"
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
                    "C",     # Here all deterministic term coefficients are
                             # collected. (const and linear trend which belong
                             # to cointegration relation as well as seasonal
                             # components which are outside the cointegration
                             # relation. Later, we will strip the terms related
                             # to the cointegration relation from C.
                    "alpha",
                    "beta",
                    "Legend",
                    "VAR A",  # VAR parameter matrices
                    "VAR deterministic"]  # VAR deterministic terms
        if dt_string == "":
            del(section_header[1])
            del(sections[1])
            del(section_header[-1])
            del(sections[-1])
        results = dict()
        results["est"] = dict.fromkeys(sections)
        results["se"] = dict.fromkeys(sections)
        results["t"] = dict.fromkeys(sections)
        results["p"] = dict.fromkeys(sections)
        section = -1
        result = []
        result_se = []
        result_t = []
        result_p = []

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
                    if result == []:  # no values collected in section "Legend"
                        started_reading_section = False
                        continue
                    results["est"][sections[section]] = np.column_stack(
                                                                    result)
                    result = []
                    results["se"][sections[section]] = np.column_stack(
                                                                    result_se)
                    result_se = []
                    results["t"][sections[section]] = np.column_stack(
                                                                    result_t)
                    result_t = []
                    results["p"][sections[section]] = np.column_stack(
                                                                    result_p)
                    result_p = []
                    started_reading_section = False
                    continue
                str_number = "-?\d+\.\d{3}"
                regex_est = re.compile(str_number + "[^\)\]\}]")
                est_col = re.findall(regex_est, line)
                # standard errors in parantheses in JMulTi output:
                regex_se = re.compile("\(" + str_number + "\)")
                se_col = re.findall(regex_se, line)
                # t-values in brackets in JMulTi output:
                regex_t_value = re.compile("\[" + str_number + "\]")
                t_col = re.findall(regex_t_value, line)
                # p-values in braces in JMulTi output:
                regex_p_value = re.compile("\{" + str_number + "\}")
                p_col = re.findall(regex_p_value, line)
                if result == [] and est_col != []:
                    rows = len(est_col)
                if est_col != []:
                    est_col = [float(el) for el in est_col]
                    result.append(est_col)
                elif se_col != []:
                    for i in range(rows):
                        se_col[i] = se_col[i].replace("(", "").replace(")", "")
                    se_col = [float(el) for el in se_col]
                    result_se.append(se_col)
                elif t_col != []:
                    for i in range(rows):
                        t_col[i] = t_col[i].replace("[", "").replace("]", "")
                    t_col = [float(el) for el in t_col]
                    result_t.append(t_col)
                elif p_col != []:
                    for i in range(rows):
                        p_col[i] = p_col[i].replace("{", "").replace("}", "")
                    p_col = [float(el) for el in p_col]
                    result_p.append(p_col)
        # delete "Legend"-section of JMulTi:
        del results["est"]["Legend"]
        del results["se"]["Legend"]
        del results["t"]["Legend"]
        del results["p"]["Legend"]
        # parse  information regarding \Sigma_u
        sigmau_file = dataset.__str__() + "_" + source + "_" + dt_string \
            + "_Sigmau" + ".txt"
        sigmau_file = os.path.join(os.path.dirname(os.path.realpath(__file__)),
                                   sigmau_file)
        rows_to_parse = 0
        # all numbers of Sigma_u in notation with e (e.g. 2.283862e-05)
        regex_est = re.compile("\s+\S+e\S+")
        sigmau_section_reached = False
        for line in open(sigmau_file):
            if line.startswith("Log Likelihood:"):
                line = line.split("Log Likelihood:")[1]
                results["log_like"] = float(re.findall(regex_est, line)[0])
            if not sigmau_section_reached and "Covariance:" not in line:
                continue
            if "Covariance:" in line:
                sigmau_section_reached = True
                row = re.findall(regex_est, line)
                rows_to_parse = len(row)  # Sigma_u quadratic ==> #rows==#cols
                Sigma_u = np.empty((rows_to_parse, rows_to_parse))
            row = re.findall(regex_est, line)
            rows_to_parse -= 1
            Sigma_u[rows_to_parse] = row  # rows are added in reverse order
            if rows_to_parse == 0:
                break
        results["est"]["Sigma_u"] = Sigma_u[::-1]

        if debug_mode:
            print_debug_output(results, dt_string)

        results_dict_per_det_terms[dt_s] = results
        if "cc" in dt_string or "lt" in dt_string:
            C = results_dict_per_det_terms[dt_s]["est"]["C"]
            det_coef_coint = []
            if "cc" in dt_string:
                det_coef_coint.append(C[:, :1])
                C = C[:, 1:]
            # if 'lt' in dt:
            #     det_coef_coint.append(C[:, -1:])
            #     C = C[:, :-1]
            if det_coef_coint != []:
                det_coef_coint = np.column_stack(det_coef_coint)
                results_dict_per_det_terms[dt_s]["est"]["det_coint"] = \
                    det_coef_coint
                if C.size == 0:
                    del results_dict_per_det_terms[dt_s]["est"]["C"]
                else:
                    results_dict_per_det_terms[dt_s]["est"]["C"] = C

    return results_dict_per_det_terms
