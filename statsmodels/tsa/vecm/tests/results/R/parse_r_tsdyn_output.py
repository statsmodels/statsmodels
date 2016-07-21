import os

def load_results_r_tsdyn(dataset, deterministic_terms_list, coint_rank,
                        neqs, diff_lags):
    """

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

    """ # todo: "Returns section"
    source = "r_tsDyn"

    results_dict_per_det_terms = dict.fromkeys(deterministic_terms_list)

    for dt in deterministic_terms_list:
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

                if len(alpha_row) == coint_rank:
                    alpha.append(alpha_row)
                    alpha_row = []
                    se_alpha.append(se_alpha_row)
                    se_alpha_row = []
                    t_alpha.append(t_alpha_row)
                    t_alpha_row = []
                    p_alpha.append(p_alpha_row)
                    p_alpha_row = []

            else:  # line has values for Gamma
                _, est, se, t, p = line.split()
                gamma_col.append(float(est))
                se_gamma_col.append(float(se))
                t_gamma_col.append(float(t))
                p_gamma_col.append(float(p))

                if len(gamma_col) == neqs:
                    gamma.append(gamma_col)
                    gamma_col = []
                    se_gamma.append(se_gamma_col)
                    se_gamma_col = []
                    t_gamma.append(t_gamma_col)
                    t_gamma_col = []
                    p_gamma.append(p_gamma_col)
                    p_gamma_col = []

        # TODO: reorder cols of gamma
        # TODO: return alpha & gamma for all combinations of det. terms (e.g.
        #       as dict with dt as key)
        return (alpha, se_alpha, t_alpha, p_alpha,
                gamma, se_gamma, t_gamma, p_gamma)





        # file_beta = os.path.join(
        #         os.path.dirname(os.path.realpath(__file__)), file_beta)
        # read in beta


        # file_df_resid  = os.path.join(
        #         os.path.dirname(os.path.realpath(__file__)), file_df_resid)
        # read in df resid