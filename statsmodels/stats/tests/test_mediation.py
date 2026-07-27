from pathlib import Path

import numpy as np
from numpy.testing import assert_allclose
import pandas as pd
import pytest

import statsmodels.api as sm
from statsmodels.formula._manager import FormulaManager
from statsmodels.stats.mediation import Mediation

df = [
    ["index", "Estimate", "Lower CI bound", "Upper CI bound", "P-value"],
    ["ACME (control)", 0.085106, 0.029938, 0.141525, 0.0],
    ["ACME (treated)", 0.085674, 0.031089, 0.147762, 0.0],
    ["ADE (control)", 0.016938, -0.129157, 0.121945, 0.66],
    ["ADE (treated)", 0.017506, -0.139649, 0.13003, 0.66],
    ["Total effect", 0.102612, -0.036749, 0.227213, 0.2],
    ["Prop. mediated (control)", 0.69807, -6.901715, 2.725978, 0.2],
    ["Prop. mediated (treated)", 0.718648, -6.145419, 2.51075, 0.2],
    ["ACME (average)", 0.08539, 0.030272, 0.144768, 0.0],
    ["ADE (average)", 0.017222, -0.134465, 0.125987, 0.66],
    ["Prop. mediated (average)", 0.7109, -6.523567, 2.618364, 0.2],
]
framing_boot_4231 = pd.DataFrame(df[1:], columns=df[0]).set_index("index")
df = [
    ["index", "Estimate", "Lower CI bound", "Upper CI bound", "P-value"],
    ["ACME (control)", 0.075529, 0.024995, 0.132408, 0.0],
    ["ACME (treated)", 0.076348, 0.027475, 0.130138, 0.0],
    ["ADE (control)", 0.021389, -0.094323, 0.139148, 0.68],
    ["ADE (treated)", 0.022207, -0.101239, 0.14574, 0.68],
    ["Total effect", 0.097736, -0.025384, 0.225386, 0.16],
    ["Prop. mediated (control)", 0.65682, -3.664956, 4.845269, 0.16],
    ["Prop. mediated (treated)", 0.68769, -3.449415, 4.469289, 0.16],
    ["ACME (average)", 0.075938, 0.026109, 0.12945, 0.0],
    ["ADE (average)", 0.021798, -0.097781, 0.142444, 0.68],
    ["Prop. mediated (average)", 0.669659, -3.557185, 4.657279, 0.16],
]
framing_para_4231 = pd.DataFrame(df[1:], columns=df[0]).set_index("index")
df = [
    ["index", "Estimate", "Lower CI bound", "Upper CI bound", "P-value"],
    ["ACME (control)", 0.065989, 0.003366, 0.152261, 0.04],
    ["ACME (treated)", 0.081424, 0.008888, 0.199853, 0.04],
    ["ADE (control)", 0.240392, -0.026286, 0.470918, 0.08],
    ["ADE (treated)", 0.255827, -0.030681, 0.491535, 0.08],
    ["Total effect", 0.321816, 0.037238, 0.54953, 0.0],
    ["Prop. mediated (control)", 0.196935, 0.015232, 1.864804, 0.04],
    ["Prop. mediated (treated)", 0.248896, 0.032229, 1.738846, 0.04],
    ["ACME (average)", 0.073707, 0.006883, 0.169923, 0.04],
    ["ADE (average)", 0.248109, -0.028483, 0.478978, 0.08],
    ["Prop. mediated (average)", 0.226799, 0.028865, 1.801825, 0.04],
]
framing_moderated_4231 = pd.DataFrame(df[1:], columns=df[0]).set_index("index")


@pytest.mark.slow
def test_framing_example():
    cur_dir = Path(__file__).resolve().parent
    data = pd.read_csv(Path(cur_dir).joinpath("results", "framing.csv"))
    mgr = FormulaManager()
    outcome = np.asarray(data["cong_mesg"])
    outcome_exog = mgr.get_matrices("emo + treat + age + educ + gender + income", data)
    outcome_model = sm.GLM(
        outcome,
        outcome_exog,
        family=sm.families.Binomial(link=sm.families.links.Probit()),
    )
    mediator = np.asarray(data["emo"])
    mediator_exog = mgr.get_matrices("treat + age + educ + gender + income", data)
    mediator_model = sm.OLS(mediator, mediator_exog)
    tx_pos = [
        outcome_exog.columns.tolist().index("treat"),
        mediator_exog.columns.tolist().index("treat"),
    ]
    med_pos = outcome_exog.columns.tolist().index("emo")
    med = Mediation(
        outcome_model,
        mediator_model,
        tx_pos,
        med_pos,
        outcome_fit_kwargs={"atol": 1e-11},
    )
    rs = np.random.RandomState(4231)
    para_rslt = med.fit(method="parametric", n_rep=100, rng=rs)
    diff = np.asarray(para_rslt.summary() - framing_para_4231)
    assert_allclose(diff, 0, atol=1e-06)
    rs = np.random.RandomState(4231)
    boot_rslt = med.fit(method="boot", n_rep=100, rng=rs)
    diff = np.asarray(boot_rslt.summary() - framing_boot_4231)
    assert_allclose(diff, 0, atol=1e-06)


def test_framing_example_moderator():
    cur_dir = Path(__file__).resolve().parent
    data = pd.read_csv(Path(cur_dir).joinpath("results", "framing.csv"))
    mgr = FormulaManager()
    outcome = np.asarray(data["cong_mesg"])
    outcome_exog = mgr.get_matrices("emo + treat + age + educ + gender + income", data)
    outcome_model = sm.GLM(
        outcome,
        outcome_exog,
        family=sm.families.Binomial(link=sm.families.links.Probit()),
    )
    mediator = np.asarray(data["emo"])
    mediator_exog = mgr.get_matrices("treat + age + educ + gender + income", data)
    mediator_model = sm.OLS(mediator, mediator_exog)
    tx_pos = [
        outcome_exog.columns.tolist().index("treat"),
        mediator_exog.columns.tolist().index("treat"),
    ]
    med_pos = outcome_exog.columns.tolist().index("emo")
    ix = (
        outcome_exog.columns.tolist().index("age"),
        mediator_exog.columns.tolist().index("age"),
    )
    moderators = {ix: 20}
    med = Mediation(
        outcome_model, mediator_model, tx_pos, med_pos, moderators=moderators
    )
    rs = np.random.RandomState(4231)
    med.fit(method="parametric", n_rep=100, rng=rs)


@pytest.mark.slow
def test_framing_example_formula():
    cur_dir = Path(__file__).resolve().parent
    data = pd.read_csv(Path(cur_dir).joinpath("results", "framing.csv"))
    outcome_model = sm.GLM.from_formula(
        "cong_mesg ~ emo + treat + age + educ + gender + income",
        data,
        family=sm.families.Binomial(link=sm.families.links.Probit()),
    )
    mediator_model = sm.OLS.from_formula(
        "emo ~ treat + age + educ + gender + income", data
    )
    med = Mediation(
        outcome_model,
        mediator_model,
        "treat",
        "emo",
        outcome_fit_kwargs={"atol": 1e-11},
    )
    rs = np.random.RandomState(4231)
    med_rslt = med.fit(method="boot", n_rep=100, rng=rs)
    diff = np.asarray(med_rslt.summary() - framing_boot_4231)
    assert_allclose(diff, 0, atol=1e-06)
    rs = np.random.RandomState(4231)
    med_rslt = med.fit(method="parametric", n_rep=100, rng=rs)
    diff = np.asarray(med_rslt.summary() - framing_para_4231)
    assert_allclose(diff, 0, atol=1e-06)


@pytest.mark.slow
def test_framing_example_moderator_formula():
    cur_dir = Path(__file__).resolve().parent
    data = pd.read_csv(Path(cur_dir).joinpath("results", "framing.csv"))
    outcome_model = sm.GLM.from_formula(
        "cong_mesg ~ emo + treat*age + emo*age + educ + gender + income",
        data,
        family=sm.families.Binomial(link=sm.families.links.Probit()),
    )
    mediator_model = sm.OLS.from_formula(
        "emo ~ treat*age + educ + gender + income", data
    )
    moderators = {"age": 20}
    med = Mediation(
        outcome_model, mediator_model, "treat", "emo", moderators=moderators
    )
    rs = np.random.RandomState(4231)
    med_rslt = med.fit(method="parametric", n_rep=100, rng=rs)
    diff = np.asarray(med_rslt.summary() - framing_moderated_4231)
    assert_allclose(diff, 0, atol=1e-06)


def t_est_mixedlm():
    rs = np.random.RandomState(3424)
    mn = rs.randn(5)
    c = 0.0001 * (rs.rand(5, 5) - 0.5)
    cov = np.eye(5) + c + c.T
    rvs = rs.multivariate_normal(mn, cov)
    rvs1 = [0.3357151, 1.26183927, 1.22539916, 0.85838887, -0.0493799]
    assert_allclose(rvs, rvs1, atol=1e-07)
    rs = np.random.RandomState(3424)
    n = 200
    x = rs.normal(size=n)
    xv = np.outer(x, np.ones(3))
    mx = np.asarray([4.0, 4, 1])
    mx /= np.sqrt(np.sum(mx**2))
    med = mx[0] * np.outer(x, np.ones(3))
    med += mx[1] * np.outer(rs.normal(size=n), np.ones(3))
    med += mx[2] * rs.normal(size=(n, 3))
    ey = np.outer(x, np.r_[0, 0.5, 1]) + med
    ex = np.asarray([5.0, 2, 2])
    ex /= np.sqrt(np.sum(ex**2))
    e = ex[0] * np.outer(rs.normal(size=n), np.ones(3))
    e += ex[1] * np.outer(rs.normal(size=n), np.r_[-1, 0, 1])
    e += ex[2] * rs.normal(size=(n, 3))
    y = ey + e
    idx = np.outer(np.arange(n), np.ones(3))
    tim = np.outer(np.ones(n), np.r_[-1, 0, 1])
    df = pd.DataFrame(
        {
            "y": y.flatten(),
            "x": xv.flatten(),
            "id": idx.flatten(),
            "time": tim.flatten(),
            "med": med.flatten(),
        }
    )
    dmean = [-0.13643661, -0.14266871, 99.5, 0.0, -0.15102166]
    assert_allclose(np.asarray(df.mean()), dmean, atol=1e-07)
    mediator_model = sm.MixedLM.from_formula("med ~ x", groups="id", data=df)
    outcome_model = sm.MixedLM.from_formula("y ~ med + x", groups="id", data=df)
    me = Mediation(outcome_model, mediator_model, "x", "med")
    rs = np.random.RandomState(383628)
    mr = me.fit(n_rep=100, rng=rs)
    st = mr.summary()
    params_om = me.outcome_model.fit().params.to_numpy()
    p_om = [0.08118371, 0.96107436, 0.50801102, 1.22452252]
    assert_allclose(params_om, p_om, atol=1e-07)
    params_mm = me.mediator_model.fit().params.to_numpy()
    p_mm = [-0.0547506, 0.67478745, 17.03184275]
    assert_allclose(params_mm, p_mm, atol=1e-07)
    res_summ = np.array(
        [
            [0.64539794, 0.57652012, 0.71427576, 0.0],
            [0.64539794, 0.57652012, 0.71427576, 0.0],
            [0.59401941, 0.56963807, 0.61840074, 0.0],
            [0.59401941, 0.56963807, 0.61840074, 0.0],
            [1.23941735, 1.1461582, 1.33267651, 0.0],
            [0.51935169, 0.50285723, 0.53584615, 0.0],
            [0.51935169, 0.50285723, 0.53584615, 0.0],
            [0.64539794, 0.57652012, 0.71427576, 0.0],
            [0.59401941, 0.56963807, 0.61840074, 0.0],
            [0.51935169, 0.50285723, 0.53584615, 0.0],
        ]
    )
    assert_allclose(st.to_numpy(), res_summ, rtol=0.15)
    assert_allclose(st.iloc[-1, 0], 0.56, rtol=0.01, atol=0.01)
    pm = st.loc["Prop. mediated (average)", "Estimate"]
    assert_allclose(pm, 0.56, rtol=0.01, atol=0.01)


def test_surv():
    rs = np.random.RandomState(2341)
    n = 1000
    exp = rs.normal(size=n)
    mn = np.exp(exp)
    mtime0 = -mn * np.log(rs.uniform(size=n))
    ctime = -2 * mn * np.log(rs.uniform(size=n))
    mstatus = (ctime >= mtime0).astype(int)
    mtime = np.where(mtime0 <= ctime, mtime0, ctime)
    for mt in ("full", "partial", "no"):
        if mt == "full":
            lp = 0.5 * mtime0
        elif mt == "partial":
            lp = exp + mtime0
        else:
            lp = exp
        mn = np.exp(-lp)
        ytime0 = -mn * np.log(rs.uniform(size=n))
        ctime = -2 * mn * np.log(rs.uniform(size=n))
        ystatus = (ctime >= ytime0).astype(int)
        ytime = np.where(ytime0 <= ctime, ytime0, ctime)
        df = pd.DataFrame(
            {
                "ytime": ytime,
                "ystatus": ystatus,
                "mtime": mtime,
                "mstatus": mstatus,
                "exp": exp,
            }
        )
        fml = "ytime ~ exp + mtime"
        outcome_model = sm.PHReg.from_formula(fml, status="ystatus", data=df)
        fml = "mtime ~ exp"
        mediator_model = sm.PHReg.from_formula(fml, status="mstatus", data=df)
        med = Mediation(
            outcome_model,
            mediator_model,
            "exp",
            "mtime",
            outcome_predict_kwargs={"pred_only": True},
            outcome_fit_kwargs={"method": "lbfgs"},
            mediator_fit_kwargs={"method": "lbfgs"},
        )
        med_result = med.fit(n_rep=2, rng=rs)
        dr = med_result.summary()
        pm = dr.loc["Prop. mediated (average)", "Estimate"]
        if mt == "no":
            assert_allclose(pm, 0, atol=0.1, rtol=0.1)
        elif mt == "full":
            assert_allclose(pm, 1, atol=0.1, rtol=0.1)
        else:
            assert_allclose(pm, 0.5, atol=0.1, rtol=0.1)
