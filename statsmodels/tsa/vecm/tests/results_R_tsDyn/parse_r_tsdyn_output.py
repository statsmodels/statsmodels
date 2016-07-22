import os

import numpy as np


def _num_of_det_terms(dt): # todo: make local variable of this (in: for dt...:)
    """

    Parameters
    ----------
    dt - string
        String with the deterministic terms. No seasonal terms possible since
        tsDyn doesn't support seasonal terms.

    Returns
    -------
    number - int
        Number of deterministic terms outside the cointegration relation.
    """
    return ("co" in dt) + ("lt" in dt)


def _reorder_gamma(gamma, neqs, diff_lags, det_terms):
    """

    Parameters
    ----------
    gamma - ndarray (neqs*diff_lags + det_terms x neqs)
        Matrix Gamma (or corresponding standard errors, t- or p-values) that
        needs to be transposed and reordered after reading in the columns.
    neqs - int
        Called K in Lutkepohl
    diff_lags - int
        Number of lags in the VEC representation. Called p-1 in Lutkepohl.
    det_terms - int
        Number of deterministic terms outside the cointegration relation.

    Returns
    -------
    gamma - ndarray (neqs x neqs*diff_lags + det_terms)
        Matrix Gamma transposed (that's why the shape is now different from
        that of the parameter gamma) and reordered such that it can be compared
        with statsmodels output.
    """  # TODO: remove last 2 parameters if they are not needed.

    C, gamma = np.hsplit(gamma, [det_terms])
    gamma = gamma.T
    gamma = np.hstack(np.vsplit(gamma, diff_lags))

    return C, gamma


def load_results_r_tsdyn(dataset, deterministic_terms_list, neqs, coint_rank=1,
                         diff_lags=3):
    """Read the output produced by the R package tsDyn and return it in form
    of a nested dict.

    Parameters
    ----------
    dataset - module
        A data module in the statsmodels/datasets directory that defines a
        __str__() method returning the dataset's name.
    deterministic_terms_list - list
        A list of strings where each string represents a combination of
        deterministic terms.
    coint_rank - int
        The cointegration rank; called r in Lutkepohl.
    neqs - int
        This is called K in Lutkepohl.
    diff_lags - int
        The number of lags in the VEC representation; called p-1 in Lutkepohl.

    Returns
    -------

    """  # todo: "Returns section"
    source = "r_tsDyn"

    results_dict_per_det_terms = dict.fromkeys(deterministic_terms_list)

    for dt in deterministic_terms_list:
        results_dict_per_det_terms[dt] = dict.fromkeys(("est", "se", "t", "p"))
        parameters = ["alpha", "gamma"]
        if _num_of_det_terms(dt) > 0:
            parameters.append("C")
        results_dict_per_det_terms[dt]["est"] = dict.fromkeys(parameters)
        results_dict_per_det_terms[dt]["se"] = dict.fromkeys(parameters)
        results_dict_per_det_terms[dt]["t"] = dict.fromkeys(parameters)
        results_dict_per_det_terms[dt]["p"] = dict.fromkeys(parameters)

        file_name_start = dataset.__str__()+"_"+source+"_"+dt
        file_name_end = ".txt"
        file_alpha_gamma = file_name_start + "_alpha_gamma" + file_name_end
        file_beta = file_name_start + "_beta" + file_name_end
        file_df_resid = file_name_start + "_df_resid" + file_name_end


        # read in alpha and gamma
        file_alpha_gamma = os.path.join(
                os.path.dirname(os.path.realpath(__file__)), file_alpha_gamma)
        alpha, alpha_row = [], []
        se_alpha, se_alpha_row = [], []  # standard errors for alpha
        t_alpha, t_alpha_row = [], []  # t-values for alpha
        p_alpha, p_alpha_row = [], []  # p-values for alpha
        gamma, gamma_col = [], []
        se_gamma, se_gamma_col = [], [] # standard errors for gamma
        t_gamma, t_gamma_col = [], []  # t-values for gamma
        p_gamma, p_gamma_col = [], []  # p-values for gamma
        C = None  # estimation for det. terms outside cointegration relation

        curr_comp = 0
        for line in open(file_alpha_gamma):
            if not line.startswith("V"):  # line is the header line
                continue

            if "ECT" in line:  # line has values for alpha
                _, est, se, t, p = line.split()
                alpha_row.append(float(est))
                se_alpha_row.append(float(se))
                t_alpha_row.append(float(t))
                p_alpha_row.append(float(p))

                # if a row is read completely:
                if len(alpha_row) == coint_rank:
                    alpha.append(alpha_row)
                    alpha_row = []
                    se_alpha.append(se_alpha_row)
                    se_alpha_row = []
                    t_alpha.append(t_alpha_row)
                    t_alpha_row = []
                    p_alpha.append(p_alpha_row)
                    p_alpha_row = []

                    # if all rows are read completely:
                    if len(alpha) == neqs:
                        alpha, se_alpha, t_alpha, p_alpha = \
                            map(np.array, (alpha, se_alpha, t_alpha, p_alpha))

            else:  # line has values for Gamma
                _, est, se, t, p = line.split()
                gamma_col.append(float(est))
                se_gamma_col.append(float(se))
                t_gamma_col.append(float(t))
                p_gamma_col.append(float(p))

                # if a col is read completely:
                if len(gamma_col) == neqs*diff_lags + _num_of_det_terms(dt):
                    gamma.append(gamma_col)
                    gamma_col = []
                    se_gamma.append(se_gamma_col)
                    se_gamma_col = []
                    t_gamma.append(t_gamma_col)
                    t_gamma_col = []
                    p_gamma.append(p_gamma_col)
                    p_gamma_col = []

                    # if all cols are read completely:
                    if len(gamma) == neqs:
                        gamma, se_gamma, t_gamma, p_gamma = \
                            map(np.array, (gamma, se_gamma, t_gamma, p_gamma))

                        print("\n\n\n\n"+dt+"\nGAMMA: " + str(gamma.shape))
                        print(gamma)

                        det_terms = _num_of_det_terms(dt)
                        C, gamma = _reorder_gamma(gamma, neqs,
                                                  diff_lags, det_terms)
                        se_C, se_gamma = _reorder_gamma(se_gamma, neqs,
                                                        diff_lags, det_terms)
                        t_C, t_gamma = _reorder_gamma(t_gamma, neqs,
                                                      diff_lags, det_terms)
                        p_C, p_gamma = _reorder_gamma(p_gamma, neqs,
                                                      diff_lags, det_terms)

                        print("\n\nGAMMA: " + str(gamma.shape))
                        print(gamma)
                        print("\n\nC: " + str(C.shape))
                        print(C)

        results_dict_per_det_terms[dt]["est"]["alpha"] = alpha
        results_dict_per_det_terms[dt]["se"]["alpha"] = se_alpha
        results_dict_per_det_terms[dt]["t"]["alpha"] = t_alpha
        results_dict_per_det_terms[dt]["p"]["alpha"] = p_alpha
        results_dict_per_det_terms[dt]["est"]["Gamma"] = gamma
        results_dict_per_det_terms[dt]["se"]["Gamma"] = se_gamma
        results_dict_per_det_terms[dt]["t"]["Gamma"] = t_gamma
        results_dict_per_det_terms[dt]["p"]["Gamma"] = p_gamma
        if C is not None and C.size > 0:
            results_dict_per_det_terms[dt]["est"]["C"] = C
            results_dict_per_det_terms[dt]["se"]["C"] = se_C
            results_dict_per_det_terms[dt]["t"]["C"] = t_C
            results_dict_per_det_terms[dt]["p"]["C"] = p_C
    return results_dict_per_det_terms





        # file_beta = os.path.join(
        #         os.path.dirname(os.path.realpath(__file__)), file_beta)
        # read in beta


        # file_df_resid  = os.path.join(
        #         os.path.dirname(os.path.realpath(__file__)), file_df_resid)
        # read in df resid