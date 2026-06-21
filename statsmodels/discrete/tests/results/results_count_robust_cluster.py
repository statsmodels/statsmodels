import numpy as np

from statsmodels.tools.testing import ParamsTableTestBunch

est = dict(
    rank=3,
    N=34,
    ic=1,
    k=3,
    k_eq=1,
    k_dv=1,
    converged=1,
    rc=0,
    k_autoCns=0,
    N_clust=5,
    ll=-354.2436413025559,
    k_eq_model=1,
    ll_0=-356.2029100704882,
    df_m=2,
    chi2=5.204189583786304,
    p=0.0741181533729996,
    r2_p=0.0055004288638308,
    cmdline="poisson accident yr_con op_75_79, vce(cluster ship)",
    cmd="poisson",
    predict="poisso_p",
    estat_cmd="poisson_estat",
    gof="poiss_g",
    chi2type="Wald",
    opt="moptimize",
    vcetype="Robust",
    clustvar="ship",
    vce="cluster",
    title="Poisson regression",
    user="poiss_lf",
    crittype="log pseudolikelihood",
    ml_method="e2",
    singularHmethod="m-marquardt",
    technique="nr",
    which="max",
    depvar="accident",
    properties="b V",
)

params_table = np.array(
    [
        -0.02172061893549,
        0.19933709357097,
        -0.10896426022065,
        0.91323083771076,
        -0.41241414311748,
        0.36897290524649,
        np.nan,
        1.9599639845401,
        0,
        0.22148585072024,
        0.11093628220713,
        1.9965140918162,
        0.04587799343723,
        0.00405473301549,
        0.43891696842499,
        np.nan,
        1.9599639845401,
        0,
        2.2697077143215,
        1.1048569901548,
        2.054299999499,
        0.03994666479943,
        0.10422780555076,
        4.4351876230922,
        np.nan,
        1.9599639845401,
        0,
    ]
).reshape(3, 9)

params_table_colnames = "b se z pvalue ll ul df crit eform".split()

params_table_rownames = "yr_con op_75_79 _cons".split()

cov = np.array(
    [
        0.03973527687332,
        0.00976206273414,
        -0.21171095768584,
        0.00976206273414,
        0.01230685870994,
        -0.06297293767114,
        -0.21171095768584,
        -0.06297293767114,
        1.2207089686939,
    ]
).reshape(3, 3)

cov_colnames = "yr_con op_75_79 _cons".split()

cov_rownames = "yr_con op_75_79 _cons".split()


results_poisson_clu = ParamsTableTestBunch(
    params_table=params_table,
    params_table_colnames=params_table_colnames,
    params_table_rownames=params_table_rownames,
    cov=cov,
    cov_colnames=cov_colnames,
    cov_rownames=cov_rownames,
    **est,
)

est = dict(
    rank=3,
    N=34,
    ic=1,
    k=3,
    k_eq=1,
    k_dv=1,
    converged=1,
    rc=0,
    k_autoCns=0,
    ll=-354.2436413025559,
    k_eq_model=1,
    ll_0=-356.2029100704882,
    df_m=2,
    chi2=0.1635672212515404,
    p=0.9214713337295277,
    r2_p=0.0055004288638308,
    cmdline="poisson accident yr_con op_75_79, vce(robust)",
    cmd="poisson",
    predict="poisso_p",
    estat_cmd="poisson_estat",
    gof="poiss_g",
    chi2type="Wald",
    opt="moptimize",
    vcetype="Robust",
    vce="robust",
    title="Poisson regression",
    user="poiss_lf",
    crittype="log pseudolikelihood",
    ml_method="e2",
    singularHmethod="m-marquardt",
    technique="nr",
    which="max",
    depvar="accident",
    properties="b V",
)

params_table = np.array(
    [
        -0.02172061893549,
        0.19233713248134,
        -0.11292993014545,
        0.91008610728406,
        -0.39869447148862,
        0.35525323361764,
        np.nan,
        1.9599639845401,
        0,
        0.22148585072024,
        0.55301404772037,
        0.400506735106,
        0.68878332380143,
        -0.8624017657564,
        1.3053734671969,
        np.nan,
        1.9599639845401,
        0,
        2.2697077143215,
        0.66532523368388,
        3.4114258702533,
        0.00064624070669,
        0.96569421829539,
        3.5737212103476,
        np.nan,
        1.9599639845401,
        0,
    ]
).reshape(3, 9)

params_table_colnames = "b se z pvalue ll ul df crit eform".split()

params_table_rownames = "yr_con op_75_79 _cons".split()

cov = np.array(
    [
        0.03699357253114,
        -0.01521223175214,
        -0.09585501859714,
        -0.01521223175214,
        0.30582453697607,
        -0.1649339692102,
        -0.09585501859714,
        -0.1649339692102,
        0.44265766657651,
    ]
).reshape(3, 3)

cov_colnames = "yr_con op_75_79 _cons".split()

cov_rownames = "yr_con op_75_79 _cons".split()


results_poisson_hc1 = ParamsTableTestBunch(
    params_table=params_table,
    params_table_colnames=params_table_colnames,
    params_table_rownames=params_table_rownames,
    cov=cov,
    cov_colnames=cov_colnames,
    cov_rownames=cov_rownames,
    **est,
)

est = dict(
    rank=3,
    N=34,
    ic=4,
    k=3,
    k_eq=1,
    k_dv=1,
    converged=1,
    rc=0,
    k_autoCns=0,
    ll=-91.28727940081573,
    k_eq_model=1,
    ll_0=-122.0974139280415,
    df_m=2,
    chi2=61.62026905445154,
    p=4.16225408420e-14,
    r2_p=0.2523405986746273,
    cmdline="poisson accident yr_con op_75_79, exposure(service)",
    cmd="poisson",
    predict="poisso_p",
    estat_cmd="poisson_estat",
    offset="ln(service)",
    gof="poiss_g",
    chi2type="LR",
    opt="moptimize",
    vce="oim",
    title="Poisson regression",
    user="poiss_lf",
    crittype="log likelihood",
    ml_method="e2",
    singularHmethod="m-marquardt",
    technique="nr",
    which="max",
    depvar="accident",
    properties="b V",
)

params_table = np.array(
    [
        0.30633819450439,
        0.05790831365493,
        5.290055523458,
        1.222792336e-07,
        0.19283998533528,
        0.4198364036735,
        np.nan,
        1.9599639845401,
        0,
        0.35592229608495,
        0.12151759298719,
        2.9289775030556,
        0.00340079035234,
        0.11775219034206,
        0.59409240182785,
        np.nan,
        1.9599639845401,
        0,
        -6.974712802772,
        0.13252425018256,
        -52.629709605328,
        0,
        -7.234455560208,
        -6.714970045336,
        np.nan,
        1.9599639845401,
        0,
    ]
).reshape(3, 9)

params_table_colnames = "b se z pvalue ll ul df crit eform".split()

params_table_rownames = "yr_con op_75_79 _cons".split()

cov = np.array(
    [
        0.00335337279036,
        -0.00315267340017,
        -0.00589654294427,
        -0.00315267340017,
        0.0147665254054,
        -0.00165060980569,
        -0.00589654294427,
        -0.00165060980569,
        0.01756267688645,
    ]
).reshape(3, 3)

cov_colnames = "yr_con op_75_79 _cons".split()

cov_rownames = "yr_con op_75_79 _cons".split()


results_poisson_exposure_nonrobust = ParamsTableTestBunch(
    params_table=params_table,
    params_table_colnames=params_table_colnames,
    params_table_rownames=params_table_rownames,
    cov=cov,
    cov_colnames=cov_colnames,
    cov_rownames=cov_rownames,
    **est,
)

est = dict(
    rank=3,
    N=34,
    ic=4,
    k=3,
    k_eq=1,
    k_dv=1,
    converged=1,
    rc=0,
    k_autoCns=0,
    ll=-91.28727940081573,
    k_eq_model=1,
    ll_0=-122.0974139280415,
    df_m=2,
    chi2=15.1822804640621,
    p=0.0005049050167458,
    r2_p=0.2523405986746273,
    cmdline="poisson accident yr_con op_75_79, exposure(service) vce(robust)",
    cmd="poisson",
    predict="poisso_p",
    estat_cmd="poisson_estat",
    offset="ln(service)",
    gof="poiss_g",
    chi2type="Wald",
    opt="moptimize",
    vcetype="Robust",
    vce="robust",
    title="Poisson regression",
    user="poiss_lf",
    crittype="log pseudolikelihood",
    ml_method="e2",
    singularHmethod="m-marquardt",
    technique="nr",
    which="max",
    depvar="accident",
    properties="b V",
)

params_table = np.array(
    [
        0.30633819450439,
        0.09144457613957,
        3.3499875819514,
        0.00080815183366,
        0.12711011868929,
        0.48556627031949,
        np.nan,
        1.9599639845401,
        0,
        0.35592229608495,
        0.16103531267836,
        2.2102127177276,
        0.02709040275274,
        0.04029888299621,
        0.67154570917369,
        np.nan,
        1.9599639845401,
        0,
        -6.974712802772,
        0.2558675415017,
        -27.259076168227,
        1.29723387e-163,
        -7.4762039689282,
        -6.4732216366159,
        np.nan,
        1.9599639845401,
        0,
    ]
).reshape(3, 9)

params_table_colnames = "b se z pvalue ll ul df crit eform".split()

params_table_rownames = "yr_con op_75_79 _cons".split()

cov = np.array(
    [
        0.00836211050535,
        0.00098797681063,
        -0.01860743122756,
        0.00098797681063,
        0.02593237192942,
        -0.02395236210603,
        -0.01860743122756,
        -0.02395236210603,
        0.06546819879413,
    ]
).reshape(3, 3)

cov_colnames = "yr_con op_75_79 _cons".split()

cov_rownames = "yr_con op_75_79 _cons".split()


results_poisson_exposure_hc1 = ParamsTableTestBunch(
    params_table=params_table,
    params_table_colnames=params_table_colnames,
    params_table_rownames=params_table_rownames,
    cov=cov,
    cov_colnames=cov_colnames,
    cov_rownames=cov_rownames,
    **est,
)

est = dict(
    rank=3,
    N=34,
    ic=4,
    k=3,
    k_eq=1,
    k_dv=1,
    converged=1,
    rc=0,
    k_autoCns=0,
    N_clust=5,
    ll=-91.28727940081573,
    k_eq_model=1,
    ll_0=-122.0974139280415,
    df_m=2,
    chi2=340.7343047354823,
    p=1.02443835269e-74,
    r2_p=0.2523405986746273,
    cmdline="poisson accident yr_con op_75_79, exposure(service) vce(cluster ship)",
    cmd="poisson",
    predict="poisso_p",
    estat_cmd="poisson_estat",
    offset="ln(service)",
    gof="poiss_g",
    chi2type="Wald",
    opt="moptimize",
    vcetype="Robust",
    clustvar="ship",
    vce="cluster",
    title="Poisson regression",
    user="poiss_lf",
    crittype="log pseudolikelihood",
    ml_method="e2",
    singularHmethod="m-marquardt",
    technique="nr",
    which="max",
    depvar="accident",
    properties="b V",
)

params_table = np.array(
    [
        0.30633819450439,
        0.03817694295902,
        8.0241677504982,
        1.022165435e-15,
        0.23151276126487,
        0.38116362774391,
        np.nan,
        1.9599639845401,
        0,
        0.35592229608495,
        0.09213163536669,
        3.8631930787765,
        0.00011191448109,
        0.17534760892947,
        0.53649698324044,
        np.nan,
        1.9599639845401,
        0,
        -6.974712802772,
        0.0968656626603,
        -72.003975518463,
        0,
        -7.1645660129248,
        -6.7848595926192,
        np.nan,
        1.9599639845401,
        0,
    ]
).reshape(3, 9)

params_table_colnames = "b se z pvalue ll ul df crit eform".split()

params_table_rownames = "yr_con op_75_79 _cons".split()

cov = np.array(
    [
        0.0014574789737,
        -0.00277745275086,
        0.00108765624666,
        -0.00277745275086,
        0.00848823823534,
        -0.00469929607507,
        0.00108765624666,
        -0.00469929607507,
        0.00938295660262,
    ]
).reshape(3, 3)

cov_colnames = "yr_con op_75_79 _cons".split()

cov_rownames = "yr_con op_75_79 _cons".split()


results_poisson_exposure_clu = ParamsTableTestBunch(
    params_table=params_table,
    params_table_colnames=params_table_colnames,
    params_table_rownames=params_table_rownames,
    cov=cov,
    cov_colnames=cov_colnames,
    cov_rownames=cov_rownames,
    **est,
)

est = dict(
    rank=4,
    N=34,
    ic=2,
    k=4,
    k_eq=2,
    k_dv=1,
    converged=1,
    rc=0,
    k_autoCns=0,
    N_clust=5,
    ll=-109.0877965183258,
    k_eq_model=1,
    ll_0=-109.1684720604314,
    rank0=2,
    df_m=2,
    chi2=5.472439553195301,
    p=0.0648148991694882,
    k_aux=1,
    alpha=2.330298308905143,
    cmdline="nbreg accident yr_con op_75_79, vce(cluster ship)",
    cmd="nbreg",
    predict="nbreg_p",
    dispers="mean",
    diparm_opt2="noprob",
    chi2type="Wald",
    opt="moptimize",
    vcetype="Robust",
    clustvar="ship",
    vce="cluster",
    title="Negative binomial regression",
    diparm1="lnalpha, exp label(",
    user="nbreg_lf",
    crittype="log pseudolikelihood",
    ml_method="e2",
    singularHmethod="m-marquardt",
    technique="nr",
    which="max",
    depvar="accident",
    properties="b V",
)

params_table = np.array(
    [
        -0.03536709401845,
        0.27216090050938,
        -0.12994921001605,
        0.89660661037787,
        -0.56879265701682,
        0.49805846897992,
        np.nan,
        1.9599639845401,
        0,
        0.23211570238882,
        0.09972456245386,
        2.3275680201277,
        0.01993505322091,
        0.03665915160525,
        0.42757225317239,
        np.nan,
        1.9599639845401,
        0,
        2.2952623989519,
        1.2335785495143,
        1.8606536242509,
        0.06279310688494,
        -0.12250713019722,
        4.7130319281011,
        np.nan,
        1.9599639845401,
        0,
        0.84599628895555,
        0.22483100011931,
        np.nan,
        np.nan,
        0.40533562611357,
        1.2866569517975,
        np.nan,
        1.9599639845401,
        0,
        2.3302983089051,
        0.52392329936749,
        np.nan,
        np.nan,
        1.4998057895818,
        3.6206622525444,
        np.nan,
        1.9599639845401,
        0,
    ]
).reshape(5, 9)

params_table_colnames = "b se z pvalue ll ul df crit eform".split()

params_table_rownames = "yr_con op_75_79 _cons _cons alpha".split()

cov = np.array(
    [
        0.07407155576607,
        -0.00421355148283,
        -0.32663130963457,
        0.02015715724983,
        -0.00421355148283,
        0.00994498835661,
        0.00992613461881,
        -0.00714955450361,
        -0.32663130963457,
        0.00992613461881,
        1.5217160378218,
        -0.09288283512096,
        0.02015715724983,
        -0.00714955450361,
        -0.09288283512096,
        0.05054897861465,
    ]
).reshape(4, 4)

cov_colnames = "yr_con op_75_79 _cons _cons".split()

cov_rownames = "yr_con op_75_79 _cons _cons".split()


results_negbin_clu = ParamsTableTestBunch(
    params_table=params_table,
    params_table_colnames=params_table_colnames,
    params_table_rownames=params_table_rownames,
    cov=cov,
    cov_colnames=cov_colnames,
    cov_rownames=cov_rownames,
    **est,
)

est = dict(
    rank=4,
    N=34,
    ic=2,
    k=4,
    k_eq=2,
    k_dv=1,
    converged=1,
    rc=0,
    k_autoCns=0,
    ll=-109.0877965183258,
    k_eq_model=1,
    ll_0=-109.1684720604314,
    rank0=2,
    df_m=2,
    chi2=0.1711221347493475,
    p=0.9179970816706797,
    r2_p=0.0007390003778831,
    k_aux=1,
    alpha=2.330298308905143,
    cmdline="nbreg accident yr_con op_75_79, vce(robust)",
    cmd="nbreg",
    predict="nbreg_p",
    dispers="mean",
    diparm_opt2="noprob",
    chi2type="Wald",
    opt="moptimize",
    vcetype="Robust",
    vce="robust",
    title="Negative binomial regression",
    diparm1="lnalpha, exp label(",
    user="nbreg_lf",
    crittype="log pseudolikelihood",
    ml_method="e2",
    singularHmethod="m-marquardt",
    technique="nr",
    which="max",
    depvar="accident",
    properties="b V",
)

params_table = np.array(
    [
        -0.03536709401845,
        0.26106873337039,
        -0.13547043172065,
        0.89223994079058,
        -0.5470524089139,
        0.476318220877,
        np.nan,
        1.9599639845401,
        0,
        0.23211570238882,
        0.56245325203342,
        0.41268443475019,
        0.67983783029986,
        -0.87027241458412,
        1.3345038193618,
        np.nan,
        1.9599639845401,
        0,
        2.2952623989519,
        0.76040210713867,
        3.0184850586341,
        0.00254041928465,
        0.80490165519179,
        3.7856231427121,
        np.nan,
        1.9599639845401,
        0,
        0.84599628895555,
        0.24005700345444,
        np.nan,
        np.nan,
        0.37549320794823,
        1.3164993699629,
        np.nan,
        1.9599639845401,
        0,
        2.3302983089051,
        0.55940442919073,
        np.nan,
        np.nan,
        1.4557092049439,
        3.7303399539165,
        np.nan,
        1.9599639845401,
        0,
    ]
).reshape(5, 9)

params_table_colnames = "b se z pvalue ll ul df crit eform".split()

params_table_rownames = "yr_con op_75_79 _cons _cons alpha".split()

cov = np.array(
    [
        0.06815688354362,
        -0.03840590969835,
        -0.16217402790798,
        0.02098165591138,
        -0.03840590969835,
        0.31635366072297,
        -0.11049674936104,
        -0.02643483668568,
        -0.16217402790798,
        -0.11049674936104,
        0.57821136454093,
        -0.03915049342584,
        0.02098165591138,
        -0.02643483668568,
        -0.03915049342584,
        0.05762736490753,
    ]
).reshape(4, 4)

cov_colnames = "yr_con op_75_79 _cons _cons".split()

cov_rownames = "yr_con op_75_79 _cons _cons".split()


results_negbin_hc1 = ParamsTableTestBunch(
    params_table=params_table,
    params_table_colnames=params_table_colnames,
    params_table_rownames=params_table_rownames,
    cov=cov,
    cov_colnames=cov_colnames,
    cov_rownames=cov_rownames,
    **est,
)

est = dict(
    rank=4,
    N=34,
    ic=4,
    k=4,
    k_eq=2,
    k_dv=1,
    converged=1,
    rc=0,
    k_autoCns=0,
    ll=-82.49115612464289,
    k_eq_model=1,
    ll_0=-84.68893065247886,
    rank0=2,
    df_m=2,
    chi2=4.39554905567195,
    p=0.1110500222994781,
    ll_c=-91.28727940081573,
    chi2_c=17.5922465523457,
    r2_p=0.0259511427397111,
    k_aux=1,
    alpha=0.2457422083490335,
    cmdline="nbreg accident yr_con op_75_79, exposure(service)",
    cmd="nbreg",
    predict="nbreg_p",
    offset="ln(service)",
    dispers="mean",
    diparm_opt2="noprob",
    chi2_ct="LR",
    chi2type="LR",
    opt="moptimize",
    vce="oim",
    title="Negative binomial regression",
    diparm1="lnalpha, exp label(",
    user="nbreg_lf",
    crittype="log likelihood",
    ml_method="e2",
    singularHmethod="m-marquardt",
    technique="nr",
    which="max",
    depvar="accident",
    properties="b V",
)

params_table = np.array(
    [
        0.28503762550355,
        0.14983643534827,
        1.9023251910727,
        0.05712865433138,
        -0.00863639135093,
        0.57871164235802,
        np.nan,
        1.9599639845401,
        0,
        0.17127003537767,
        0.27580549562862,
        0.62098122804736,
        0.53461197443513,
        -0.36929880279264,
        0.71183887354798,
        np.nan,
        1.9599639845401,
        0,
        -6.5908639033905,
        0.40391814231008,
        -16.31732574748,
        7.432080344e-60,
        -7.3825289150206,
        -5.7991988917604,
        np.nan,
        1.9599639845401,
        0,
        -1.4034722260565,
        0.51305874839271,
        np.nan,
        np.nan,
        -2.4090488948595,
        -0.39789555725363,
        np.nan,
        1.9599639845401,
        0,
        0.24574220834903,
        0.12608018984282,
        np.nan,
        np.nan,
        0.089900758997,
        0.67173218155228,
        np.nan,
        1.9599639845401,
        0,
    ]
).reshape(5, 9)

params_table_colnames = "b se z pvalue ll ul df crit eform".split()

params_table_rownames = "yr_con op_75_79 _cons _cons alpha".split()

cov = np.array(
    [
        0.02245095735788,
        -0.01097939549632,
        -0.05127649084781,
        0.00045725833006,
        -0.01097939549632,
        0.07606867141895,
        -0.0197375670989,
        -0.00926008351523,
        -0.05127649084781,
        -0.0197375670989,
        0.16314986568722,
        0.02198323898312,
        0.00045725833006,
        -0.00926008351523,
        0.02198323898312,
        0.26322927930229,
    ]
).reshape(4, 4)

cov_colnames = "yr_con op_75_79 _cons _cons".split()

cov_rownames = "yr_con op_75_79 _cons _cons".split()


results_negbin_exposure_nonrobust = ParamsTableTestBunch(
    params_table=params_table,
    params_table_colnames=params_table_colnames,
    params_table_rownames=params_table_rownames,
    cov=cov,
    cov_colnames=cov_colnames,
    cov_rownames=cov_rownames,
    **est,
)

est = dict(
    rank=4,
    N=34,
    ic=4,
    k=4,
    k_eq=2,
    k_dv=1,
    converged=1,
    rc=0,
    k_autoCns=0,
    N_clust=5,
    ll=-82.49115612464289,
    k_eq_model=1,
    ll_0=-84.68893065247886,
    rank0=2,
    df_m=2,
    chi2=5.473741859983782,
    p=0.0647727084656973,
    k_aux=1,
    alpha=0.2457422083490335,
    cmdline="nbreg accident yr_con op_75_79, exposure(service) vce(cluster ship)",
    cmd="nbreg",
    predict="nbreg_p",
    offset="ln(service)",
    dispers="mean",
    diparm_opt2="noprob",
    chi2type="Wald",
    opt="moptimize",
    vcetype="Robust",
    clustvar="ship",
    vce="cluster",
    title="Negative binomial regression",
    diparm1="lnalpha, exp label(",
    user="nbreg_lf",
    crittype="log pseudolikelihood",
    ml_method="e2",
    singularHmethod="m-marquardt",
    technique="nr",
    which="max",
    depvar="accident",
    properties="b V",
)

params_table = np.array(
    [
        0.28503762550355,
        0.14270989695062,
        1.9973220610073,
        0.04579020833966,
        0.00533136724292,
        0.56474388376418,
        np.nan,
        1.9599639845401,
        0,
        0.17127003537767,
        0.17997186802799,
        0.95164892854829,
        0.34127505843023,
        -0.18146834418759,
        0.52400841494293,
        np.nan,
        1.9599639845401,
        0,
        -6.5908639033905,
        0.62542746996715,
        -10.538174640357,
        5.760612980e-26,
        -7.8166792194681,
        -5.3650485873129,
        np.nan,
        1.9599639845401,
        0,
        -1.4034722260565,
        0.86579403765571,
        np.nan,
        np.nan,
        -3.1003973578913,
        0.29345290577817,
        np.nan,
        1.9599639845401,
        0,
        0.24574220834903,
        0.21276213878894,
        np.nan,
        np.nan,
        0.0450313052935,
        1.3410500222158,
        np.nan,
        1.9599639845401,
        0,
    ]
).reshape(5, 9)

params_table_colnames = "b se z pvalue ll ul df crit eform".split()

params_table_rownames = "yr_con op_75_79 _cons _cons alpha".split()

cov = np.array(
    [
        0.02036611468766,
        -0.00330004038514,
        -0.08114367170947,
        -0.07133030733881,
        -0.00330004038514,
        0.03238987328148,
        -0.03020509748676,
        -0.09492663454187,
        -0.08114367170947,
        -0.03020509748676,
        0.39115952018952,
        0.43276143586693,
        -0.07133030733881,
        -0.09492663454187,
        0.43276143586693,
        0.74959931564018,
    ]
).reshape(4, 4)

cov_colnames = "yr_con op_75_79 _cons _cons".split()

cov_rownames = "yr_con op_75_79 _cons _cons".split()


results_negbin_exposure_clu = ParamsTableTestBunch(
    params_table=params_table,
    params_table_colnames=params_table_colnames,
    params_table_rownames=params_table_rownames,
    cov=cov,
    cov_colnames=cov_colnames,
    cov_rownames=cov_rownames,
    **est,
)
