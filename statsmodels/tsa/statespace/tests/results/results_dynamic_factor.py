"""
Results for VARMAX tests

Results from Stata using script `test_varmax_stata.do`.
See also Stata time series documentation, in particular `dfactor`.

Data from:

http://www.jmulti.de/download/datasets/e1.dat

Author: Chad Fulton
License: Simplified-BSD
"""

lutkepohl_dfm = {
    "params": [
        0.0063728,
        0.00660177,
        0.00636009,  # Factor loadings
        0.00203899,
        0.00009016,
        0.00005348,  # Idiosyncratic variances
        0.33101874,
        0.63927819,  # Factor transitions
    ],
    "bse_oim": [
        0.002006,
        0.0012514,
        0.0012128,  # Factor loadings
        0.0003359,
        0.0000184,
        0.0000141,  # Idiosyncratic variances
        0.1196637,
        0.1218577,  # Factor transitions
    ],
    "loglike": 594.0902026190786,
    "aic": -1172.18,
    "bic": -1153.641,
}

lutkepohl_dfm2 = {
    "params": [
        0.03411188,
        0.03478764,  # Factor loadings: y1
        0.03553366,
        0.0344871,  # Factor loadings: y2
        0.03536757,
        0.03433391,  # Factor loadings: y3
        0.00224401,
        0.00014678,
        0.00010922,  # Idiosyncratic variances
        0.08845946,
        0.08862982,  # Factor transitions: Phi, row 1
        0.08754759,
        0.08758589,  # Phi, row 2
    ],
    "bse_oim": None,
    "loglike": 496.379832917306,
    "aic": -974.7597,
    "bic": -953.9023,
}

lutkepohl_dfm_exog1 = {
    "params": [
        -0.01254697,
        -0.00734604,
        -0.00671296,  # Factor loadings
        0.01803325,
        0.02066737,
        0.01983089,  # Beta.constant
        0.00198667,
        0.00008426,
        0.00005684,  # Idiosyncratic variances
        0.31140829,  # Factor transition
    ],
    "var_oim": [
        0.00004224,
        2.730e-06,
        3.625e-06,
        0.00003087,
        2.626e-06,
        2.013e-06,
        1.170e-07,
        5.133e-10,
        3.929e-10,
        0.07412117,
    ],
    "loglike": 596.9781590009525,
    "aic": -1173.956,
    "bic": -1150.781,
}

lutkepohl_dfm_exog2 = {
    "params": [
        0.01249096,
        0.00731147,
        0.00680776,  # Factor loadings
        0.02187812,
        -0.00009851,  # Betas, y1
        0.02302646,
        -0.00006045,  # Betas, y2
        0.02009233,
        -6.683e-06,  # Betas, y3
        0.0019856,
        0.00008378,
        0.00005581,  # Idiosyncratic variances
        0.2995768,  # Factor transition
    ],
    "var_oim": [
        0.00004278,
        2.659e-06,
        3.766e-06,
        0.00013003,
        6.536e-08,
        0.00001079,
        5.424e-09,
        8.393e-06,
        4.217e-09,
        1.168e-07,
        5.140e-10,
        4.181e-10,
        0.07578382,
    ],
    "loglike": 597.4550537198315,
    "aic": -1168.91,
    "bic": -1138.783,
}

lutkepohl_dfm_gen = {
    "params": [
        0.00312295,
        0.00332555,
        0.00318837,  # Factor loadings
        # .00195462,                        # Covariance, lower triangle
        #  3.642e-06, .00010047,
        # .00007018,  .00002565, .00006118
        # Note: the following are the Cholesky of the covariance
        # matrix defined just above
        0.04421108,  # Cholesky, lower triangle
        0.00008238,
        0.01002313,
        0.00158738,
        0.00254603,
        0.00722343,
        0.987374,  # Factor transition
        -0.25613562,
        0.00392166,
        0.44859028,  # Error transition parameters
        0.01635544,
        -0.249141,
        0.08170863,
        -0.02280001,
        0.02059063,
        -0.41808254,
    ],
    "var_oim": [
        1.418e-06,
        1.030e-06,
        9.314e-07,  # Factor loadings
        None,  # Cholesky, lower triangle
        None,
        None,
        None,
        None,
        None,
        0.00021421,  # Factor transition
        0.01307587,
        0.29167522,
        0.43204063,  # Error transition parameters
        0.00076899,
        0.01742173,
        0.0220161,
        0.00055435,
        0.01456365,
        0.01707167,
    ],
    "loglike": 607.7715711926285,
    "aic": -1177.543,
    "bic": -1133.511,
}

lutkepohl_dfm_ar2 = {
    "params": [
        0.00419132,
        0.0044007,
        0.00422976,  # Factor loadings
        0.00188101,
        0.0000786,
        0.0000418,  # Idiosyncratic variance
        0.97855802,  # Factor transition
        -0.28856258,
        -0.14910552,  # Error transition parameters
        -0.41544832,
        -0.26706536,
        -0.72661178,
        -0.27278821,
    ],
    "var_oim": [
        1.176e-06,
        7.304e-07,
        6.726e-07,  # Factor loadings
        9.517e-08,
        2.300e-10,
        1.389e-10,  # Idiosyncratic variance
        0.00041159,  # Factor transition
        0.0131511,
        0.01296008,  # Error transition parameters
        0.01748435,
        0.01616862,
        0.03262051,
        0.02546648,
    ],
    "loglike": 607.4203109232711,
    "aic": -1188.841,
    "bic": -1158.713,
}

lutkepohl_dfm_scalar = {
    "params": [
        0.04424851,
        0.00114077,
        0.00275081,  # Factor loadings
        0.01812298,
        0.02071169,
        0.01987196,  # Beta.constant
        0.00012067,  # Idiosyncratic variance
        -0.19915198,  # Factor transition
    ],
    "var_oim": [
        0.00001479,
        1.664e-06,
        1.671e-06,
        0.00001985,
        1.621e-06,
        1.679e-06,
        1.941e-10,
        0.01409482,
    ],
    "loglike": 588.7677809701966,
    "aic": -1161.536,
    "bic": -1142.996,
}

lutkepohl_sfm = {
    "params": [
        0.02177607,
        0.02089956,
        0.02239669,  # Factor loadings
        0.00201477,
        0.00013623,
        7.452e-16,  # Idiosyncratic variance
    ],
    "var_oim": [0.00003003, 4.729e-06, 3.344e-06, 1.083e-07, 4.950e-10, 0],
    "loglike": 532.2215594949788,
    "aic": -1054.443,
    "bic": -1042.856,
}

lutkepohl_sur = {
    "params": [
        0.02169026,
        -0.00009184,  # Betas, y1
        0.0229165,
        -0.00005654,  # Betas, y2
        0.01998994,
        -3.049e-06,  # Betas, y3
        # .00215703,                      # Covariance, lower triangle
        # .0000484,  .00014252,
        # .00012772, .00005642, .00010673,
        # Note: the following are the Cholesky of the covariance
        # matrix defined just above
        0.04644384,  # Cholesky, lower triangle
        0.00104212,
        0.0118926,
        0.00274999,
        0.00450315,
        0.00888196,
    ],
    "var_oim": [
        0.0001221,
        6.137e-08,
        8.067e-06,
        4.055e-09,
        6.042e-06,
        3.036e-09,
        None,
        None,
        None,
        None,
        None,
        None,
    ],
    "loglike": 597.6181259116113,
    "aic": -1171.236,
    "bic": -1143.426,
}


lutkepohl_sur_auto = {
    "params": [
        0.02243063,
        -0.00011112,  # Betas, y1
        0.02286952,
        -0.0000554,  # Betas, y2
        0.0020338,
        0.00013843,  # Idiosyncratic variance
        -0.21127833,
        0.50884609,  # Error transition parameters
        0.04292935,
        0.00855789,
    ],
    "var_oim": [
        0.00008357,
        4.209e-08,
        8.402e-06,
        4.222e-09,
        1.103e-07,
        5.110e-10,
        0.01259537,
        0.19382105,
        0.00085936,
        0.01321035,
    ],
    "loglike": 352.7250284160132,
    "aic": -685.4501,
    "bic": -662.2752,
}
