import itertools
import os
from pathlib import Path
import re

import numpy as np

debug_mode = False
here = Path(os.path.realpath(__file__)).parent


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
    gamma = results["est"]["Gamma"]
    print("Gamma:")
    print(str(type(gamma)) + str(gamma.shape))
    print(gamma)
    if "co" in dt or "s" in dt or "lo" in dt:
        c = results["est"]["C"]
        print("C:")
        print(str(type(c)) + str(c.shape))
        print(c)
        print("se: ")
        print(results["se"]["C"])


def dt_s_tup_to_string(dt_s_tup):
    dt_string = dt_s_tup[0]
    if dt_s_tup[1] > 0:
        if "co" in dt_string or "ci" in dt_string or "nc" in dt_string:
            dt_string = dt_string[:2] + "s" + dt_string[2:]
        else:
            dt_string = "s" + dt_string
    return dt_string


def sublists(lst, min_elmts=0, max_elmts=None):
    """
    Build a list of all possible sublists of a given list

    Restrictions on the length of the sublists can be posed via the
    min_elmts and max_elmts parameters. All sublists will have at least
    min_elmts elements and not more than max_elmts elements.

    Parameters
    ----------
    lst : list
        Original list from which sublists are generated.
    min_elmts : int
        Lower bound for the length of sublists.
    max_elmts : int or None
        If int, then max_elmts are the upper bound for the length of sublists.
        If None, sublists' length is not restricted. In this case the longest
        sublist will be of the same length as the original list lst.

    Returns
    -------
    result : list
        A list of all sublists of lst fulfilling the length restrictions.
    """
    if max_elmts is None:
        max_elmts = len(lst)
    result = itertools.chain.from_iterable(
        (
            itertools.combinations(lst, sublist_len)
            for sublist_len in range(min_elmts, max_elmts + 1)
        )
    )
    if type(result) is not list:
        result = list(result)
    return result


def stringify_var_names(var_list, delimiter=""):
    """
    Join variable names into a single, lowercased string

    Parameters
    ----------
    var_list : list[str]
        Each list element is the name of a variable.
    delimiter : str
        String used to join consecutive variable names. Default is the
        empty string.

    Returns
    -------
    result : str
        Concatenated variable names.
    """
    result = var_list[0]
    for var_name in var_list[1:]:
        result += delimiter + var_name
    return result.lower()


def load_results_jmulti(dataset):
    """
    Load and parse JMulTi VECM output results for a dataset

    Parameters
    ----------
    dataset : module
        A data module in the statsmodels/datasets directory that defines a
        __str__() method returning the dataset's name.

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
    results_dict_per_det_terms = dict.fromkeys(dataset.dt_s_list)
    for dt_s in dataset.dt_s_list:
        dt_string = dt_s_tup_to_string(dt_s)
        params_file = (
            "vecm_" + dataset.__str__() + "_" + source + "_" + dt_string + ".txt"
        )
        params_file = Path(here).joinpath(params_file)
        section_header = [
            "Lagged endogenous term",
            "Deterministic term",
            "Loading coefficients",
            "Estimated cointegration relation",
            "Legend",
            "Lagged endogenous term",
            "Deterministic term",
        ]
        sections = [
            "Gamma",
            "C",
            "alpha",
            "beta",
            "Legend",
            "VAR A",
            "VAR deterministic",
        ]
        if "co" not in dt_string and "lo" not in dt_string and ("s" not in dt_string):
            del section_header[1]
            del sections[1]
            if "ci" not in dt_string and "li" not in dt_string:
                del section_header[-1]
                del sections[-1]
        results = {}
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
        params_file = Path(params_file).open(encoding="latin_1")
        for line in params_file:
            if section == -1 and section_header[section + 1] not in line:
                continue
            if (
                section < len(section_header) - 1
                and section_header[section + 1] in line
            ):
                section += 1
                continue
            if not started_reading_section:
                if line.startswith(start_end_mark):
                    started_reading_section = True
                continue
            if started_reading_section:
                if line.startswith(start_end_mark):
                    if result == []:
                        started_reading_section = False
                        continue
                    results["est"][sections[section]] = np.column_stack(result)
                    result = []
                    results["se"][sections[section]] = np.column_stack(result_se)
                    result_se = []
                    results["t"][sections[section]] = np.column_stack(result_t)
                    result_t = []
                    results["p"][sections[section]] = np.column_stack(result_p)
                    result_p = []
                    started_reading_section = False
                    continue
                str_number = "-?\\d+\\.\\d{3}"
                regex_est = re.compile(str_number + "[^\\)\\]\\}]")
                est_col = re.findall(regex_est, line)
                regex_se = re.compile("\\(" + str_number + "\\)")
                se_col = re.findall(regex_se, line)
                regex_t_value = re.compile("\\[" + str_number + "\\]")
                t_col = re.findall(regex_t_value, line)
                regex_p_value = re.compile("\\{" + str_number + "\\}")
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
        params_file.close()
        del results["est"]["Legend"]
        del results["se"]["Legend"]
        del results["t"]["Legend"]
        del results["p"]["Legend"]
        results["est"]["beta"] = results["est"]["beta"].T
        results["se"]["beta"] = results["se"]["beta"].T
        results["t"]["beta"] = results["t"]["beta"].T
        results["p"]["beta"] = results["p"]["beta"].T
        alpha = results["est"]["alpha"]
        beta = results["est"]["beta"]
        alpha_rows = alpha.shape[0]
        if beta.shape[0] > alpha_rows:
            results["est"]["beta"], results["est"]["det_coint"] = np.vsplit(
                results["est"]["beta"], [alpha_rows]
            )
            results["se"]["beta"], results["se"]["det_coint"] = np.vsplit(
                results["se"]["beta"], [alpha_rows]
            )
            results["t"]["beta"], results["t"]["det_coint"] = np.vsplit(
                results["t"]["beta"], [alpha_rows]
            )
            results["p"]["beta"], results["p"]["det_coint"] = np.vsplit(
                results["p"]["beta"], [alpha_rows]
            )
        sigmau_file = (
            "vecm_"
            + dataset.__str__()
            + "_"
            + source
            + "_"
            + dt_string
            + "_Sigmau"
            + ".txt"
        )
        sigmau_file = Path(here).joinpath(sigmau_file)
        rows_to_parse = 0
        regex_est = re.compile("\\s+\\S+e\\S+")
        sigmau_section_reached = False
        sigmau_file = Path(sigmau_file).open(encoding="latin_1")
        for line in sigmau_file:
            if line.startswith("Log Likelihood:"):
                line = line.split("Log Likelihood:")[1]
                results["log_like"] = float(re.findall(regex_est, line)[0])
            if not sigmau_section_reached and "Covariance:" not in line:
                continue
            if "Covariance:" in line:
                sigmau_section_reached = True
                row = re.findall(regex_est, line)
                rows_to_parse = len(row)
                sigma_u = np.empty((rows_to_parse, rows_to_parse))
            row = re.findall(regex_est, line)
            rows_to_parse -= 1
            sigma_u[rows_to_parse] = row
            if rows_to_parse == 0:
                break
        sigmau_file.close()
        results["est"]["Sigma_u"] = sigma_u[::-1]
        fc_file = (
            "vecm_"
            + dataset.__str__()
            + "_"
            + source
            + "_"
            + dt_string
            + "_fc5"
            + ".txt"
        )
        fc_file = Path(here).joinpath(fc_file)
        fc, lower, upper, plu_min = ([], [], [], [])
        fc_file = Path(fc_file).open(encoding="latin_1")
        for line in fc_file:
            str_number = "(\\s+-?\\d+\\.\\d{4}\\s*?)"
            regex_number = re.compile(str_number)
            numbers = re.findall(regex_number, line)
            if numbers == []:
                continue
            fc.append(float(numbers[0]))
            lower.append(float(numbers[1]))
            upper.append(float(numbers[2]))
            plu_min.append(float(numbers[3]))
        fc_file.close()
        variables = alpha.shape[0]
        fc = np.hstack(np.vsplit(np.array(fc)[:, None], variables))
        lower = np.hstack(np.vsplit(np.array(lower)[:, None], variables))
        upper = np.hstack(np.vsplit(np.array(upper)[:, None], variables))
        results["fc"] = dict.fromkeys(["fc", "lower", "upper"])
        results["fc"]["fc"] = fc
        results["fc"]["lower"] = lower
        results["fc"]["upper"] = upper
        results["granger_caus"] = dict.fromkeys(["p", "test_stat"])
        results["granger_caus"]["p"] = {}
        results["granger_caus"]["test_stat"] = {}
        vn = dataset.variable_names
        var_combs = sublists(vn, 1, len(vn) - 1)
        for causing in var_combs:
            caused = tuple((el for el in vn if el not in causing))
            granger_file = (
                "vecm_"
                + dataset.__str__()
                + "_"
                + source
                + "_"
                + dt_string
                + "_granger_causality_"
                + stringify_var_names(causing)
                + "_"
                + stringify_var_names(caused)
                + ".txt"
            )
            granger_file = Path(here).joinpath(granger_file)
            granger_file = Path(granger_file).open(encoding="latin_1")
            granger_results = []
            for line in granger_file:
                str_number = "\\d+\\.\\d{4}"
                regex_number = re.compile(str_number)
                number = re.search(regex_number, line)
                if number is None:
                    continue
                number = float(number.group(0))
                granger_results.append(number)
            granger_file.close()
            results["granger_caus"]["test_stat"][causing, caused] = granger_results[0]
            results["granger_caus"]["p"][causing, caused] = granger_results[1]
        results["inst_caus"] = dict.fromkeys(["p", "test_stat"])
        results["inst_caus"]["p"] = {}
        results["inst_caus"]["test_stat"] = {}
        vn = dataset.variable_names
        var_combs = sublists(vn, 1, len(vn) - 1)
        for causing in var_combs:
            caused = tuple((el for el in vn if el not in causing))
            inst_file = (
                "vecm_"
                + dataset.__str__()
                + "_"
                + source
                + "_"
                + dt_string
                + "_inst_causality_"
                + stringify_var_names(causing)
                + "_"
                + stringify_var_names(caused)
                + ".txt"
            )
            inst_file = Path(here).joinpath(inst_file)
            inst_file = Path(inst_file).open(encoding="latin_1")
            inst_results = []
            for line in inst_file:
                str_number = "\\d+\\.\\d{4}"
                regex_number = re.compile(str_number)
                number = re.search(regex_number, line)
                if number is None:
                    continue
                number = float(number.group(0))
                inst_results.append(number)
            inst_file.close()
            results["inst_caus"]["test_stat"][causing, caused] = inst_results[2]
            results["inst_caus"]["p"][causing, caused] = inst_results[3]
        ir_file = (
            "vecm_"
            + dataset.__str__()
            + "_"
            + source
            + "_"
            + dt_string
            + "_ir"
            + ".txt"
        )
        ir_file = Path(here).joinpath(ir_file)
        ir_file = Path(ir_file).open(encoding="latin_1")
        causing = None
        caused = None
        data = None
        regex_vars = re.compile("\\w+")
        regex_vals = re.compile("-?\\d+\\.\\d{4}")
        line_start_causing = "time"
        data_line_indicator = "point estimate"
        data_rows_read = 0
        for line in ir_file:
            if causing is None and (not line.startswith(line_start_causing)):
                continue
            if line.startswith(line_start_causing):
                line = line[4:]
                causing = re.findall(regex_vars, line)
                data = np.empty((21, len(causing)))
                continue
            if caused is None:
                caused = re.findall(regex_vars, line)
                continue
            if data_line_indicator not in line:
                continue
            start = line.find(data_line_indicator) + len(data_line_indicator)
            line = line[start:]
            data[data_rows_read] = re.findall(regex_vals, line)
            data_rows_read += 1
        ir_file.close()
        results["ir"] = data
        lagorder_file = (
            "vecm_"
            + dataset.__str__()
            + "_"
            + source
            + "_"
            + dt_string
            + "_lagorder"
            + ".txt"
        )
        lagorder_file = Path(here).joinpath(lagorder_file)
        lagorder_file = Path(lagorder_file).open(encoding="latin_1")
        results["lagorder"] = {}
        aic_start = "Akaike Info Criterion:"
        fpe_start = "Final Prediction Error:"
        hqic_start = "Hannan-Quinn Criterion:"
        bic_start = "Schwarz Criterion:"
        for line in lagorder_file:
            if line.startswith(aic_start):
                results["lagorder"]["aic"] = int(line[len(aic_start) :])
            elif line.startswith(fpe_start):
                results["lagorder"]["fpe"] = int(line[len(fpe_start) :])
            elif line.startswith(hqic_start):
                results["lagorder"]["hqic"] = int(line[len(hqic_start) :])
            elif line.startswith(bic_start):
                results["lagorder"]["bic"] = int(line[len(bic_start) :])
        lagorder_file.close()
        test_norm_file = (
            "vecm_"
            + dataset.__str__()
            + "_"
            + source
            + "_"
            + dt_string
            + "_diag"
            + ".txt"
        )
        test_norm_file = Path(here).joinpath(test_norm_file)
        test_norm_file = Path(test_norm_file).open(encoding="latin_1")
        results["test_norm"] = {}
        reading_values = False
        line_start_statistic = "joint test statistic:"
        line_start_pvalue = " p-value:"
        for line in test_norm_file:
            if not reading_values:
                if "Introduction to Multiple Time Series Analysis" in line:
                    reading_values = True
                continue
            if "joint_pvalue" in results["test_norm"].keys():
                break
            if line.startswith(line_start_statistic):
                line_end = line[len(line_start_statistic) :]
                results["test_norm"]["joint_test_statistic"] = float(line_end)
            if line.startswith(line_start_pvalue):
                line_end = line[len(line_start_pvalue) :]
                results["test_norm"]["joint_pvalue"] = float(line_end)
        test_norm_file.close()
        whiteness_file = (
            "vecm_"
            + dataset.__str__()
            + "_"
            + source
            + "_"
            + dt_string
            + "_diag"
            + ".txt"
        )
        whiteness_file = Path(here).joinpath(whiteness_file)
        whiteness_file = Path(whiteness_file).open(encoding="latin_1")
        results["whiteness"] = {}
        section_start_marker = "PORTMANTEAU TEST"
        order_start = "tested order:"
        statistic_start = "test statistic:"
        p_start = " p-value:"
        adj_statistic_start = "adjusted test statistic:"
        unadjusted_finished = False
        in_section = False
        for line in whiteness_file:
            if not in_section and section_start_marker not in line:
                continue
            if not in_section and section_start_marker in line:
                in_section = True
                continue
            if line.startswith(order_start):
                results["whiteness"]["tested order"] = int(line[len(order_start) :])
                continue
            if line.startswith(statistic_start):
                results["whiteness"]["test statistic"] = float(
                    line[len(statistic_start) :]
                )
                continue
            if line.startswith(adj_statistic_start):
                results["whiteness"]["test statistic adj."] = float(
                    line[len(adj_statistic_start) :]
                )
                continue
            if line.startswith(p_start):
                if not unadjusted_finished:
                    results["whiteness"]["p-value"] = float(line[len(p_start) :])
                    unadjusted_finished = True
                else:
                    results["whiteness"]["p-value adjusted"] = float(
                        line[len(p_start) :]
                    )
                    break
        whiteness_file.close()
        if debug_mode:
            print_debug_output(results, dt_string)
        results_dict_per_det_terms[dt_s] = results
    return results_dict_per_det_terms
