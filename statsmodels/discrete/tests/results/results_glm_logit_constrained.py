import numpy as np

from statsmodels.tools.testing import ParamsTableTestBunch

est = dict(
    k_eq_model=0,
    phi=1,
    vf=1,
    df=28,
    df_m=3,
    power=0,
    canonical=1,
    rank=4,
    aic=1.055602138883215,
    rc=0,
    p=0.0388431588742135,
    chi2=8.376256383189103,
    ll=-12.88963422213144,
    k_autoCns=0,
    converged=1,
    k_dv=1,
    k_eq=1,
    k=4,
    ic=3,
    N=32,
    nbml=0,
    bic=-71.26133683412948,
    dispers_ps=0.9734684585933382,
    deviance_ps=27.25711684061347,
    dispers_p=0.9734684585933382,
    deviance_p=27.25711684061347,
    dispers_s=0.920688158723674,
    deviance_s=25.77926844426287,
    dispers=0.920688158723674,
    deviance=25.77926844426287,
    cmdline="glm grade  gpa tuce psi, family(binomial)",
    cmd="glm",
    predict="glim_p",
    marginsnotok="stdp Anscombe Cooksd Deviance Hat Likelihood Pearson Response Score Working ADJusted STAndardized STUdentized MODified",
    marginsok="default",
    hac_lag="30",
    vcetype="OIM",
    vce="oim",
    linkt="Logit",
    linkf="ln(u/(1-u))",
    varfunct="Bernoulli",
    varfuncf="u*(1-u)",
    opt1="ML",
    oim="oim",
    a="1",
    m="1",
    varfunc="glim_v2",
    link="glim_l02",
    chi2type="Wald",
    opt="moptimize",
    title="Generalized linear models",
    user="glim_lf",
    crittype="log likelihood",
    ml_method="e2",
    singularHmethod="m-marquardt",
    technique="nr",
    which="max",
    depvar="grade",
    properties="b V",
)

params_table = np.array(
    [
        2.8261124216999,
        1.2629410221647,
        2.2377231969675,
        0.02523911156938,
        0.35079350365878,
        5.301431339741,
        np.nan,
        1.9599639845401,
        0,
        0.09515765001172,
        0.141554201358,
        0.67223472774972,
        0.50143427587633,
        -0.18228348651028,
        0.37259878653373,
        np.nan,
        1.9599639845401,
        0,
        2.3786875040587,
        1.0645642078703,
        2.2344237073472,
        0.02545520725424,
        0.29217999740245,
        4.4651950107149,
        np.nan,
        1.9599639845401,
        0,
        -13.021345912635,
        4.931323890811,
        -2.6405375515688,
        0.00827746189686,
        -22.686563134726,
        -3.3561286905433,
        np.nan,
        1.9599639845401,
        0,
    ]
).reshape(4, 9)

params_table_colnames = "b se z pvalue ll ul df crit eform".split()

params_table_rownames = "gpa tuce psi _cons".split()

cov = np.array(
    [
        1.5950200254665,
        -0.03692058012179,
        0.42761557297075,
        -4.5734780841711,
        -0.03692058012179,
        0.0200375919221,
        0.01491263753083,
        -0.34625566662867,
        0.42761557297075,
        0.01491263753083,
        1.1332969526786,
        -2.3591604492672,
        -4.5734780841711,
        -0.34625566662867,
        -2.3591604492672,
        24.317955316083,
    ]
).reshape(4, 4)

cov_colnames = "gpa tuce psi _cons".split()

cov_rownames = "gpa tuce psi _cons".split()

infocrit = np.array([32, np.nan, -12.889634222131, 4, 33.779268444263, 39.642212055462])

infocrit_colnames = "N ll0 ll df AIC BIC".split()

infocrit_rownames = ".".split()


results_noconstraint = ParamsTableTestBunch(
    params_table=params_table,
    params_table_colnames=params_table_colnames,
    params_table_rownames=params_table_rownames,
    cov=cov,
    cov_colnames=cov_colnames,
    cov_rownames=cov_rownames,
    infocrit=infocrit,
    infocrit_colnames=infocrit_colnames,
    infocrit_rownames=infocrit_rownames,
    **est,
)

est = dict(
    k_eq_model=0,
    phi=1,
    vf=1,
    df=28,
    df_m=3,
    power=0,
    canonical=1,
    rank=4,
    aic=1.055602138883215,
    rc=0,
    p=0.0248623136764981,
    chi2=9.360530997638559,
    ll=-12.88963422213144,
    k_autoCns=0,
    converged=1,
    k_dv=1,
    k_eq=1,
    k=4,
    ic=3,
    N=32,
    nbml=0,
    bic=-71.26133683412948,
    dispers_ps=0.9734684585933382,
    deviance_ps=27.25711684061347,
    dispers_p=0.9734684585933382,
    deviance_p=27.25711684061347,
    dispers_s=0.920688158723674,
    deviance_s=25.77926844426287,
    dispers=0.920688158723674,
    deviance=25.77926844426287,
    cmdline="glm grade  gpa tuce psi, family(binomial) vce(robust)",
    cmd="glm",
    predict="glim_p",
    marginsnotok="stdp Anscombe Cooksd Deviance Hat Likelihood Pearson Response Score Working ADJusted STAndardized STUdentized MODified",
    marginsok="default",
    hac_lag="30",
    vcetype="Robust",
    vce="robust",
    linkt="Logit",
    linkf="ln(u/(1-u))",
    varfunct="Bernoulli",
    varfuncf="u*(1-u)",
    opt1="ML",
    oim="oim",
    a="1",
    m="1",
    varfunc="glim_v2",
    link="glim_l02",
    chi2type="Wald",
    opt="moptimize",
    title="Generalized linear models",
    user="glim_lf",
    crittype="log pseudolikelihood",
    ml_method="e2",
    singularHmethod="m-marquardt",
    technique="nr",
    which="max",
    depvar="grade",
    properties="b V",
)

params_table = np.array(
    [
        2.8261124216999,
        1.287827879216,
        2.1944799202672,
        0.02820092594159,
        0.30201616014984,
        5.3502086832499,
        np.nan,
        1.9599639845401,
        0,
        0.09515765001172,
        0.1198091371814,
        0.79424367999287,
        0.42705358424294,
        -0.13966394388263,
        0.32997924390608,
        np.nan,
        1.9599639845401,
        0,
        2.3786875040587,
        0.97985082470462,
        2.4276016757712,
        0.01519902587997,
        0.45821517741577,
        4.2991598307016,
        np.nan,
        1.9599639845401,
        0,
        -13.021345912635,
        5.2807513766642,
        -2.4658130981467,
        0.01367026437574,
        -23.371428422207,
        -2.6712634030626,
        np.nan,
        1.9599639845401,
        0,
    ]
).reshape(4, 9)

params_table_colnames = "b se z pvalue ll ul df crit eform".split()

params_table_rownames = "gpa tuce psi _cons".split()

cov = np.array(
    [
        1.6585006464861,
        0.00630184631279,
        0.20368998146717,
        -5.7738061195745,
        0.00630184631279,
        0.01435422935215,
        0.01997066738212,
        -0.34768562593344,
        0.20368998146717,
        0.01997066738212,
        0.96010763867432,
        -1.5315997267117,
        -5.7738061195745,
        -0.34768562593344,
        -1.5315997267117,
        27.886335102141,
    ]
).reshape(4, 4)

cov_colnames = "gpa tuce psi _cons".split()

cov_rownames = "gpa tuce psi _cons".split()

infocrit = np.array([32, np.nan, -12.889634222131, 4, 33.779268444263, 39.642212055462])

infocrit_colnames = "N ll0 ll df AIC BIC".split()

infocrit_rownames = ".".split()


results_noconstraint_robust = ParamsTableTestBunch(
    params_table=params_table,
    params_table_colnames=params_table_colnames,
    params_table_rownames=params_table_rownames,
    cov=cov,
    cov_colnames=cov_colnames,
    cov_rownames=cov_rownames,
    infocrit=infocrit,
    infocrit_colnames=infocrit_colnames,
    infocrit_rownames=infocrit_rownames,
    **est,
)

est = dict(
    k_eq_model=0,
    phi=1,
    vf=1,
    df=29,
    df_m=2,
    power=0,
    canonical=1,
    rank=3,
    aic=0.993115540206396,
    rc=0,
    p=0.0600760311411508,
    chi2=5.624288666552698,
    ll=-12.88984864330234,
    k_autoCns=0,
    converged=1,
    k_dv=1,
    k_eq=1,
    k=4,
    ic=3,
    N=32,
    nbml=0,
    bic=-74.7266438945874,
    dispers_ps=0.9340711710496038,
    deviance_ps=27.08806396043851,
    dispers_p=0.9340711710496038,
    deviance_p=27.08806396043851,
    dispers_s=0.8889550788484368,
    deviance_s=25.77969728660467,
    dispers=0.8889550788484368,
    deviance=25.77969728660467,
    cmdline="glm grade  gpa tuce psi, family(binomial) constraints(1)",
    cmd="glm",
    predict="glim_p",
    marginsnotok="stdp Anscombe Cooksd Deviance Hat Likelihood Pearson Response Score Working ADJusted STAndardized STUdentized MODified",
    marginsok="default",
    hac_lag="30",
    vcetype="OIM",
    vce="oim",
    linkt="Logit",
    linkf="ln(u/(1-u))",
    varfunct="Bernoulli",
    varfuncf="u*(1-u)",
    opt1="ML",
    oim="oim",
    a="1",
    m="1",
    varfunc="glim_v2",
    link="glim_l02",
    chi2type="Wald",
    opt="moptimize",
    title="Generalized linear models",
    user="glim_lf",
    crittype="log likelihood",
    ml_method="e2",
    singularHmethod="m-marquardt",
    technique="nr",
    which="max",
    depvar="grade",
    properties="b V",
)

params_table = np.array(
    [
        2.8,
        np.nan,
        np.nan,
        np.nan,
        np.nan,
        np.nan,
        np.nan,
        1.9599639845401,
        0,
        0.09576464077943,
        0.13824841412912,
        0.69269974185736,
        0.48849800113543,
        -0.17519727183342,
        0.36672655339228,
        np.nan,
        1.9599639845401,
        0,
        2.3717067235827,
        1.0071435928909,
        2.3548843882081,
        0.01852846934254,
        0.39774155425619,
        4.3456718929091,
        np.nan,
        1.9599639845401,
        0,
        -12.946549758905,
        3.3404275889275,
        -3.8757163309928,
        0.00010631147941,
        -19.493667526167,
        -6.3994319916434,
        np.nan,
        1.9599639845401,
        0,
    ]
).reshape(4, 9)

params_table_colnames = "b se z pvalue ll ul df crit eform".split()

params_table_rownames = "gpa tuce psi _cons".split()

cov = np.array(
    [
        0,
        0,
        0,
        0,
        0,
        0.01911262400922,
        0.02461998233256,
        -0.45036648979107,
        0,
        0.02461998233256,
        1.0143382167012,
        -1.126241119498,
        0,
        -0.45036648979107,
        -1.126241119498,
        11.158456476868,
    ]
).reshape(4, 4)

cov_colnames = "gpa tuce psi _cons".split()

cov_rownames = "gpa tuce psi _cons".split()

infocrit = np.array([32, np.nan, -12.889848643302, 3, 31.779697286605, 36.176904995004])

infocrit_colnames = "N ll0 ll df AIC BIC".split()

infocrit_rownames = ".".split()


results_constraint1 = ParamsTableTestBunch(
    params_table=params_table,
    params_table_colnames=params_table_colnames,
    params_table_rownames=params_table_rownames,
    cov=cov,
    cov_colnames=cov_colnames,
    cov_rownames=cov_rownames,
    infocrit=infocrit,
    infocrit_colnames=infocrit_colnames,
    infocrit_rownames=infocrit_rownames,
    **est,
)

est = dict(
    k_eq_model=0,
    phi=1,
    vf=1,
    df=29,
    df_m=2,
    power=0,
    canonical=1,
    rank=3,
    aic=0.9965088127779717,
    rc=0,
    p=0.0151376593316312,
    chi2=8.381139289068923,
    ll=-12.94414100444755,
    k_autoCns=0,
    converged=1,
    k_dv=1,
    k_eq=1,
    k=4,
    ic=3,
    N=32,
    nbml=0,
    bic=-74.61805917229698,
    dispers_ps=0.9101961406899989,
    deviance_ps=26.39568808000997,
    dispers_p=0.9101961406899989,
    deviance_p=26.39568808000997,
    dispers_s=0.892699379617072,
    deviance_s=25.88828200889509,
    dispers=0.892699379617072,
    deviance=25.88828200889509,
    cmdline="glm grade  gpa tuce psi, family(binomial) constraints(2)",
    cmd="glm",
    predict="glim_p",
    marginsnotok="stdp Anscombe Cooksd Deviance Hat Likelihood Pearson Response Score Working ADJusted STAndardized STUdentized MODified",
    marginsok="default",
    hac_lag="30",
    vcetype="OIM",
    vce="oim",
    linkt="Logit",
    linkf="ln(u/(1-u))",
    varfunct="Bernoulli",
    varfuncf="u*(1-u)",
    opt1="ML",
    oim="oim",
    a="1",
    m="1",
    varfunc="glim_v2",
    link="glim_l02",
    chi2type="Wald",
    opt="moptimize",
    title="Generalized linear models",
    user="glim_lf",
    crittype="log likelihood",
    ml_method="e2",
    singularHmethod="m-marquardt",
    technique="nr",
    which="max",
    depvar="grade",
    properties="b V",
)

params_table = np.array(
    [
        2.5537914524884,
        0.92662050289421,
        2.7560273537138,
        0.00585081038138,
        0.73764863947939,
        4.3699342654975,
        np.nan,
        1.9599639845401,
        0,
        0.10791139824293,
        0.13554656123081,
        0.79612051580696,
        0.42596199070477,
        -0.15775497999771,
        0.37357777648357,
        np.nan,
        1.9599639845401,
        0,
        2.5537914524884,
        0.92662050289421,
        2.7560273537138,
        0.00585081038138,
        0.73764863947939,
        4.3699342654975,
        np.nan,
        1.9599639845401,
        0,
        -12.527922070831,
        4.6393777844052,
        -2.7003453163357,
        0.00692675392223,
        -21.62093543894,
        -3.4349087027211,
        np.nan,
        1.9599639845401,
        0,
    ]
).reshape(4, 9)

params_table_colnames = "b se z pvalue ll ul df crit eform".split()

params_table_rownames = "gpa tuce psi _cons".split()

cov = np.array(
    [
        0.85862555638391,
        -0.00408642741742,
        0.85862555638391,
        -3.1725052764862,
        -0.00408642741742,
        0.0183728702615,
        -0.00408642741742,
        -0.40376368789892,
        0.85862555638391,
        -0.00408642741742,
        0.85862555638391,
        -3.1725052764862,
        -3.1725052764862,
        -0.40376368789892,
        -3.1725052764862,
        21.523826226433,
    ]
).reshape(4, 4)

cov_colnames = "gpa tuce psi _cons".split()

cov_rownames = "gpa tuce psi _cons".split()

infocrit = np.array([32, np.nan, -12.944141004448, 3, 31.888282008895, 36.285489717294])

infocrit_colnames = "N ll0 ll df AIC BIC".split()

infocrit_rownames = ".".split()

predict_mu = np.array(
    [
        0.02720933393726,
        0.05877785527304,
        0.17341537851768,
        0.02240274574181,
        0.48834788561471,
        0.03262255746648,
        0.02545734725406,
        0.05057489993471,
        0.10986224061161,
        0.64848146294279,
        0.02525325609066,
        0.17259131542841,
        0.28314297612096,
        0.18171413480391,
        0.33018645131295,
        0.02988039105483,
        0.05693576903037,
        0.03731338966779,
        0.61672273095571,
        0.68861137241716,
        0.08792248035539,
        0.90822178043053,
        0.25295501355621,
        0.85758484919326,
        0.83972248507748,
        0.54158048311843,
        0.62661357692624,
        0.36224489285202,
        0.83387563062407,
        0.93837010344092,
        0.55200183830167,
        0.13940358008872,
    ]
)

predict_mu_colnames = "predict_mu".split()

predict_mu_rownames = ["r" + str(n) for n in range(1, 33)]

predict_linpred_std = np.array(
    [
        1.2186852972383,
        0.98250143329647,
        0.71300625338041,
        1.7281112031272,
        0.58278126610648,
        1.2588643933597,
        1.323097466817,
        1.0187451680624,
        0.92583226681839,
        0.97445803529749,
        1.2426520057509,
        0.66674211884633,
        0.53877733839827,
        0.77006015103931,
        0.70670367147137,
        1.2036701125873,
        1.1407798755705,
        1.1376397495763,
        0.57331962577752,
        0.65764380198652,
        0.85122884445037,
        1.1282943138296,
        1.2981327331615,
        0.91561885084703,
        0.8524827403359,
        0.75030433039358,
        1.0902299962647,
        0.53350768600347,
        0.96511132361274,
        1.2127047415358,
        0.61923877005984,
        0.80300912367498,
    ]
)

predict_linpred_std_colnames = "predict_linpred_std".split()

predict_linpred_std_rownames = ["r" + str(n) for n in range(1, 33)]

predict_hat = np.array(
    [
        0.03931157544567,
        0.05340381182541,
        0.07287215399916,
        0.06540404284993,
        0.0848623883214,
        0.0500117280211,
        0.04343078449564,
        0.04983412818394,
        0.08382437063813,
        0.21645722203914,
        0.03801090644315,
        0.06348261316195,
        0.05891921860299,
        0.08817451110282,
        0.11045563375857,
        0.04199779738721,
        0.06987634275981,
        0.04648995770552,
        0.0776956378885,
        0.09273814423054,
        0.05810645039404,
        0.10611489289649,
        0.31844046474321,
        0.10239122636412,
        0.09780916971071,
        0.13976583081559,
        0.27809589396914,
        0.06575633167064,
        0.12902962938834,
        0.08505028097419,
        0.09482722348113,
        0.07735963673184,
    ]
)

predict_hat_colnames = "predict_hat".split()

predict_hat_rownames = ["r" + str(n) for n in range(1, 33)]


results_constraint2 = ParamsTableTestBunch(
    params_table=params_table,
    params_table_colnames=params_table_colnames,
    params_table_rownames=params_table_rownames,
    cov=cov,
    cov_colnames=cov_colnames,
    cov_rownames=cov_rownames,
    infocrit=infocrit,
    infocrit_colnames=infocrit_colnames,
    infocrit_rownames=infocrit_rownames,
    predict_mu=predict_mu,
    predict_mu_colnames=predict_mu_colnames,
    predict_mu_rownames=predict_mu_rownames,
    predict_linpred_std=predict_linpred_std,
    predict_linpred_std_colnames=predict_linpred_std_colnames,
    predict_linpred_std_rownames=predict_linpred_std_rownames,
    predict_hat=predict_hat,
    predict_hat_colnames=predict_hat_colnames,
    predict_hat_rownames=predict_hat_rownames,
    **est,
)

est = dict(
    k_eq_model=0,
    phi=1,
    vf=1,
    df=29,
    df_m=2,
    power=0,
    canonical=1,
    rank=3,
    aic=0.9965088127779717,
    rc=0,
    p=0.0085760854232441,
    chi2=9.517555427941099,
    ll=-12.94414100444755,
    k_autoCns=0,
    converged=1,
    k_dv=1,
    k_eq=1,
    k=4,
    ic=3,
    N=32,
    nbml=0,
    bic=-74.61805917229698,
    dispers_ps=0.9101961406899989,
    deviance_ps=26.39568808000997,
    dispers_p=0.9101961406899989,
    deviance_p=26.39568808000997,
    dispers_s=0.892699379617072,
    deviance_s=25.88828200889509,
    dispers=0.892699379617072,
    deviance=25.88828200889509,
    cmdline="glm grade  gpa tuce psi, family(binomial) constraints(2) vce(robust)",
    cmd="glm",
    predict="glim_p",
    marginsnotok="stdp Anscombe Cooksd Deviance Hat Likelihood Pearson Response Score Working ADJusted STAndardized STUdentized MODified",
    marginsok="default",
    hac_lag="30",
    vcetype="Robust",
    vce="robust",
    linkt="Logit",
    linkf="ln(u/(1-u))",
    varfunct="Bernoulli",
    varfuncf="u*(1-u)",
    opt1="ML",
    oim="oim",
    a="1",
    m="1",
    varfunc="glim_v2",
    link="glim_l02",
    chi2type="Wald",
    opt="moptimize",
    title="Generalized linear models",
    user="glim_lf",
    crittype="log pseudolikelihood",
    ml_method="e2",
    singularHmethod="m-marquardt",
    technique="nr",
    which="max",
    depvar="grade",
    properties="b V",
)

params_table = np.array(
    [
        2.5537914524884,
        0.83609404798719,
        3.0544308485827,
        0.00225487991353,
        0.91507723074524,
        4.1925056742316,
        np.nan,
        1.9599639845401,
        0,
        0.10791139824293,
        0.12275592600281,
        0.8790728216287,
        0.37936179287834,
        -0.13268579561143,
        0.3485085920973,
        np.nan,
        1.9599639845401,
        0,
        2.5537914524884,
        0.83609404798719,
        3.0544308485827,
        0.00225487991353,
        0.91507723074524,
        4.1925056742316,
        np.nan,
        1.9599639845401,
        0,
        -12.527922070831,
        4.510414281113,
        -2.7775546302454,
        0.00547696322683,
        -21.368171617167,
        -3.6876725244938,
        np.nan,
        1.9599639845401,
        0,
    ]
).reshape(4, 9)

params_table_colnames = "b se z pvalue ll ul df crit eform".split()

params_table_rownames = "gpa tuce psi _cons".split()

cov = np.array(
    [
        0.6990532570796,
        0.01512804251258,
        0.6990532570796,
        -2.9662622048441,
        0.01512804251258,
        0.01506901736881,
        0.01512804251258,
        -0.3968065659911,
        0.6990532570796,
        0.01512804251258,
        0.6990532570796,
        -2.9662622048441,
        -2.9662622048441,
        -0.3968065659911,
        -2.9662622048441,
        20.343836987269,
    ]
).reshape(4, 4)

cov_colnames = "gpa tuce psi _cons".split()

cov_rownames = "gpa tuce psi _cons".split()

infocrit = np.array([32, np.nan, -12.944141004448, 3, 31.888282008895, 36.285489717294])

infocrit_colnames = "N ll0 ll df AIC BIC".split()

infocrit_rownames = ".".split()


results_constraint2_robust = ParamsTableTestBunch(
    params_table=params_table,
    params_table_colnames=params_table_colnames,
    params_table_rownames=params_table_rownames,
    cov=cov,
    cov_colnames=cov_colnames,
    cov_rownames=cov_rownames,
    infocrit=infocrit,
    infocrit_colnames=infocrit_colnames,
    infocrit_rownames=infocrit_rownames,
    **est,
)

est = dict(
    N_cds=0,
    N_cdf=0,
    p=0.0151376589433054,
    chi2=8.381139340374848,
    df_m=2,
    k_eq_model=1,
    ll=-12.94414100444751,
    k_autoCns=0,
    rc=0,
    converged=1,
    k_dv=1,
    k_eq=1,
    k=4,
    ic=5,
    N=32,
    rank=3,
    cmdline="logit grade  gpa tuce psi, constraints(2)",
    cmd="logit",
    estat_cmd="logit_estat",
    predict="logit_p",
    marginsnotok="stdp DBeta DEviance DX2 DDeviance Hat Number Residuals RStandard SCore",
    title="Logistic regression",
    chi2type="Wald",
    opt="moptimize",
    vce="oim",
    user="mopt__logit_d2()",
    crittype="log likelihood",
    ml_method="d2",
    singularHmethod="m-marquardt",
    technique="nr",
    which="max",
    depvar="grade",
    properties="b V",
)

params_table = np.array(
    [
        2.5537916456996,
        0.92662056628814,
        2.7560273736742,
        0.00585081002433,
        0.73764870844071,
        4.3699345829585,
        np.nan,
        1.9599639845401,
        0,
        0.10791141442743,
        0.13554656655573,
        0.79612060393329,
        0.42596193948753,
        -0.15775497424986,
        0.37357780310472,
        np.nan,
        1.9599639845401,
        0,
        2.5537916456996,
        0.92662056628814,
        2.7560273736742,
        0.00585081002433,
        0.73764870844071,
        4.3699345829585,
        np.nan,
        1.9599639845401,
        0,
        -12.527923225554,
        4.6393781670436,
        -2.7003453425175,
        0.00692675337706,
        -21.62093734362,
        -3.4349091074867,
        np.nan,
        1.9599639845401,
        0,
    ]
).reshape(4, 9)

params_table_colnames = "b se z pvalue ll ul df crit eform".split()

params_table_rownames = "gpa tuce psi _cons".split()

cov = np.array(
    [
        0.85862567386816,
        -0.00408642236043,
        0.85862567386816,
        -3.172505858545,
        -0.00408642236043,
        0.01837287170505,
        -0.00408642236043,
        -0.40376374127778,
        0.85862567386816,
        -0.00408642236043,
        0.85862567386816,
        -3.172505858545,
        -3.172505858545,
        -0.40376374127778,
        -3.172505858545,
        21.523829776841,
    ]
).reshape(4, 4)

cov_colnames = "gpa tuce psi _cons".split()

cov_rownames = "gpa tuce psi _cons".split()

infocrit = np.array([32, np.nan, -12.944141004448, 3, 31.888282008895, 36.285489717294])

infocrit_colnames = "N ll0 ll df AIC BIC".split()

infocrit_rownames = ".".split()


results_logit_constraint2 = ParamsTableTestBunch(
    params_table=params_table,
    params_table_colnames=params_table_colnames,
    params_table_rownames=params_table_rownames,
    cov=cov,
    cov_colnames=cov_colnames,
    cov_rownames=cov_rownames,
    infocrit=infocrit,
    infocrit_colnames=infocrit_colnames,
    infocrit_rownames=infocrit_rownames,
    **est,
)
