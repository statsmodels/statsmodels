"""Reference results for Theil textile regression tests."""

import numpy as np

from statsmodels.tools.testing import ParamsTableTestBunch

est = dict(
    rmse=0.0136097497582343,
    r2=0.9741055881598619,
    N=17,
    df_r=14,
    compat=0.860625753607033,
    vrank=2,
    pvalue=0.6503055973535645,
    frac_sample=0.7935370014985163,
    frac_prior=0.2064629985014838,
    cmd="tgmixed",
    predict="regres_p",
    depvar="lconsump",
    marginsok="XB default",
    cmdline="tgmixed lconsump lincome lprice, prior(lprice -0.7 0.15 lincome 1 0.15) cov(lprice lincome -0.01)",
    prior="lprice -0.7 0.15 lincome 1 0.15",
    properties="b V",
)

params_table = np.array(
    [
        1.0893571039001,
        0.10338923727975,
        10.53646523141,
        4.871483239e-08,
        0.86760924410848,
        1.3111049636916,
        14,
        2.1447866879178,
        0,
        -0.82054628653043,
        0.03496499383295,
        -23.467651401591,
        1.218701708e-12,
        -0.89553873984647,
        -0.74555383321439,
        14,
        2.1447866879178,
        0,
        1.4666439879147,
        0.20347802665937,
        7.2078740490733,
        4.509300573e-06,
        1.0302270250519,
        1.9030609507775,
        14,
        2.1447866879178,
        0,
    ]
).reshape(3, 9)

params_table_colnames = "b se t pvalue ll ul df crit eform".split()

params_table_rownames = "lincome lprice _cons".split()

cov = np.array(
    [
        0.01068933438529,
        -0.00081953185523,
        -0.0199747086722,
        -0.00081953185523,
        0.00122255079374,
        -0.00064024357954,
        -0.0199747086722,
        -0.00064024357954,
        0.04140330733319,
    ]
).reshape(3, 3)

cov_colnames = "lincome lprice _cons".split()

cov_rownames = "lincome lprice _cons".split()

cov_prior = np.array([0.0225, -0.01, 0, -0.01, 0.0225, 0, 0, 0, 0]).reshape(3, 3)

cov_prior_colnames = "lincome lprice _cons".split()

cov_prior_rownames = "lincome lprice _cons".split()


results_theil_textile = ParamsTableTestBunch(
    params_table=params_table,
    params_table_colnames=params_table_colnames,
    params_table_rownames=params_table_rownames,
    cov=cov,
    cov_colnames=cov_colnames,
    cov_rownames=cov_rownames,
    cov_prior=cov_prior,
    cov_prior_colnames=cov_prior_colnames,
    cov_prior_rownames=cov_prior_rownames,
    **est,
)
