import numpy as np
import statsmodels.api as sm
import os
from statsmodels.stats.mediation import Mediation
import pandas as pd
from numpy.testing import assert_allclose
import patsy
import pytest

# Compare to mediation R package vignette
df = [['index', 'Estimate', 'Lower CI bound', 'Upper CI bound', 'P-value'],
      ['ACME (control)', 0.085106, 0.029938, 0.141525, 0.00],
      ['ACME (treated)', 0.085674, 0.031089, 0.147762, 0.00],
      ['ADE (control)', 0.016938, -0.129157, 0.121945, 0.66],
      ['ADE (treated)', 0.017506, -0.139649, 0.130030, 0.66],
      ['Total effect', 0.102612, -0.036749, 0.227213, 0.20],
      ['Prop. mediated (control)', 0.698070, -6.901715, 2.725978, 0.20],
      ['Prop. mediated (treated)', 0.718648, -6.145419, 2.510750, 0.20],
      ['ACME (average)', 0.085390, 0.030272, 0.144768, 0.00],
      ['ADE (average)', 0.017222, -0.134465, 0.125987, 0.66],
      ['Prop. mediated (average)', 0.710900, -6.523567, 2.618364, 0.20]]
framing_boot_4231 = pd.DataFrame(df[1:], columns=df[0]).set_index('index')

# Compare to mediation R package vignette
df = [['index', 'Estimate', 'Lower CI bound', 'Upper CI bound', 'P-value'],
      ['ACME (control)', 0.075529, 0.024995, 0.132408, 0.00],
      ['ACME (treated)', 0.076348, 0.027475, 0.130138, 0.00],
      ['ADE (control)', 0.021389, -0.094323, 0.139148, 0.68],
      ['ADE (treated)', 0.022207, -0.101239, 0.145740, 0.68],
      ['Total effect', 0.097736, -0.025384, 0.225386, 0.16],
      ['Prop. mediated (control)', 0.656820, -3.664956, 4.845269, 0.16],
      ['Prop. mediated (treated)', 0.687690, -3.449415, 4.469289, 0.16],
      ['ACME (average)', 0.075938, 0.026109, 0.129450, 0.00],
      ['ADE (average)', 0.021798, -0.097781, 0.142444, 0.68],
      ['Prop. mediated (average)', 0.669659, -3.557185, 4.657279, 0.16]]
framing_para_4231 = pd.DataFrame(df[1:], columns=df[0]).set_index('index')



df = [['index', 'Estimate', 'Lower CI bound', 'Upper CI bound', 'P-value'],
      ['ACME (control)', 0.065989, 0.003366, 0.152261, 0.04],
      ['ACME (treated)', 0.081424, 0.008888, 0.199853, 0.04],
      ['ADE (control)', 0.240392, -0.026286, 0.470918, 0.08],
      ['ADE (treated)', 0.255827, -0.030681, 0.491535, 0.08],
      ['Total effect', 0.321816, 0.037238, 0.549530, 0.00],
      ['Prop. mediated (control)', 0.196935, 0.015232, 1.864804, 0.04],
      ['Prop. mediated (treated)', 0.248896, 0.032229, 1.738846, 0.04],
      ['ACME (average)', 0.073707, 0.006883, 0.169923, 0.04],
      ['ADE (average)', 0.248109, -0.028483, 0.478978, 0.08],
      ['Prop. mediated (average)', 0.226799, 0.028865, 1.801825, 0.04]]
framing_moderated_4231 = pd.DataFrame(df[1:], columns=df[0]).set_index('index')


@pytest.mark.slow
def test_framing_example():

    cur_dir = os.path.dirname(os.path.abspath(__file__))
    data = pd.read_csv(os.path.join(cur_dir, 'results', "framing.csv"))

    outcome = np.asarray(data["cong_mesg"])
    outcome_exog = patsy.dmatrix("emo + treat + age + educ + gender + income", data,
                                  return_type='dataframe')
    probit = sm.families.links.probit
    outcome_model = sm.GLM(outcome, outcome_exog, family=sm.families.Binomial(link=probit()))

    mediator = np.asarray(data["emo"])
    mediator_exog = patsy.dmatrix("treat + age + educ + gender + income", data,
                                 return_type='dataframe')
    mediator_model = sm.OLS(mediator, mediator_exog)

    tx_pos = [outcome_exog.columns.tolist().index("treat"),
              mediator_exog.columns.tolist().index("treat")]
    med_pos = outcome_exog.columns.tolist().index("emo")

    med = Mediation(outcome_model, mediator_model, tx_pos, med_pos,
                    outcome_fit_kwargs={'atol':1e-11})

    np.random.seed(4231)
    para_rslt = med.fit(method='parametric', n_rep=100)
    diff = np.asarray(para_rslt.summary() - framing_para_4231)
    assert_allclose(diff, 0, atol=1e-6)

    np.random.seed(4231)
    boot_rslt = med.fit(method='boot', n_rep=100)
    diff = np.asarray(boot_rslt.summary() - framing_boot_4231)
    assert_allclose(diff, 0, atol=1e-6)



def test_framing_example_moderator():
    # moderation without formulas, generally not useful but test anyway

    cur_dir = os.path.dirname(os.path.abspath(__file__))
    data = pd.read_csv(os.path.join(cur_dir, 'results', "framing.csv"))

    outcome = np.asarray(data["cong_mesg"])
    outcome_exog = patsy.dmatrix("emo + treat + age + educ + gender + income", data,
                                  return_type='dataframe')
    probit = sm.families.links.probit
    outcome_model = sm.GLM(outcome, outcome_exog, family=sm.families.Binomial(link=probit()))

    mediator = np.asarray(data["emo"])
    mediator_exog = patsy.dmatrix("treat + age + educ + gender + income", data,
                                 return_type='dataframe')
    mediator_model = sm.OLS(mediator, mediator_exog)

    tx_pos = [outcome_exog.columns.tolist().index("treat"),
              mediator_exog.columns.tolist().index("treat")]
    med_pos = outcome_exog.columns.tolist().index("emo")

    ix = (outcome_exog.columns.tolist().index("age"),
          mediator_exog.columns.tolist().index("age"))
    moderators = {ix : 20}
    med = Mediation(outcome_model, mediator_model, tx_pos, med_pos,
                    moderators=moderators)

    # Just a smoke test
    np.random.seed(4231)
    med_rslt = med.fit(method='parametric', n_rep=100)


@pytest.mark.slow
def test_framing_example_formula():

    cur_dir = os.path.dirname(os.path.abspath(__file__))
    data = pd.read_csv(os.path.join(cur_dir, 'results', "framing.csv"))

    probit = sm.families.links.probit
    outcome_model = sm.GLM.from_formula("cong_mesg ~ emo + treat + age + educ + gender + income",
                                        data, family=sm.families.Binomial(link=probit()))

    mediator_model = sm.OLS.from_formula("emo ~ treat + age + educ + gender + income", data)

    med = Mediation(outcome_model, mediator_model, "treat", "emo",
                    outcome_fit_kwargs={'atol': 1e-11})

    np.random.seed(4231)
    med_rslt = med.fit(method='boot', n_rep=100)
    diff = np.asarray(med_rslt.summary() - framing_boot_4231)
    assert_allclose(diff, 0, atol=1e-6)

    np.random.seed(4231)
    med_rslt = med.fit(method='parametric', n_rep=100)
    diff = np.asarray(med_rslt.summary() - framing_para_4231)
    assert_allclose(diff, 0, atol=1e-6)


@pytest.mark.slow
def test_framing_example_moderator_formula():

    cur_dir = os.path.dirname(os.path.abspath(__file__))
    data = pd.read_csv(os.path.join(cur_dir, 'results', "framing.csv"))

    probit = sm.families.links.probit
    outcome_model = sm.GLM.from_formula("cong_mesg ~ emo + treat*age + emo*age + educ + gender + income",
                                        data, family=sm.families.Binomial(link=probit()))

    mediator_model = sm.OLS.from_formula("emo ~ treat*age + educ + gender + income", data)

    moderators = {"age" : 20}
    med = Mediation(outcome_model, mediator_model, "treat", "emo",
                    moderators=moderators)

    np.random.seed(4231)
    med_rslt = med.fit(method='parametric', n_rep=100)
    diff = np.asarray(med_rslt.summary() - framing_moderated_4231)
    assert_allclose(diff, 0, atol=1e-6)


def test_mixedlm():

    np.random.seed(3424)

    n = 200

    # The exposure (not time varying)
    x = np.random.normal(size=n)
    xv = np.outer(x, np.ones(3))

    # The mediator (with random intercept)
    mx = np.asarray([4., 4, 1])
    mx /= np.sqrt(np.sum(mx**2))
    med = mx[0] * np.outer(x, np.ones(3))
    med += mx[1] * np.outer(np.random.normal(size=n), np.ones(3))
    med += mx[2] * np.random.normal(size=(n, 3))

    # The outcome (exposure and mediator effects)
    ey = np.outer(x, np.r_[0, 0.5, 1]) + med

    # Random structure of the outcome (random intercept and slope)
    ex = np.asarray([5., 2, 2])
    ex /= np.sqrt(np.sum(ex**2))
    e = ex[0] * np.outer(np.random.normal(size=n), np.ones(3))
    e += ex[1] * np.outer(np.random.normal(size=n), np.r_[-1, 0, 1])
    e += ex[2] * np.random.normal(size=(n, 3))
    y = ey + e

    # Group membership
    idx = np.outer(np.arange(n), np.ones(3))

    # Time
    tim = np.outer(np.ones(n), np.r_[-1, 0, 1])

    df = pd.DataFrame({"y": y.flatten(), "x": xv.flatten(),
                       "id": idx.flatten(), "time": tim.flatten(),
                       "med": med.flatten()})

    mediator_model = sm.MixedLM.from_formula("med ~ x", groups="id", data=df)
    outcome_model = sm.MixedLM.from_formula("y ~ med + x", groups="id", data=df)
    me = Mediation(outcome_model, mediator_model, "x", "med")
    mr = me.fit(n_rep=2)
    st = mr.summary()
    pm = st.loc["Prop. mediated (average)", "Estimate"]
    assert_allclose(pm, 0.52, rtol=1e-2, atol=1e-2)
