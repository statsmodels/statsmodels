import numpy as np
import statsmodels.api as sm
import os
from statsmodels.sandbox.mediation import Mediation
import pandas as pd
from numpy.testing import assert_allclose
import patsy

df = [['index', 'Estimate', '95% lower bound', '95% upper bound'],
      ['ACME (control)', 0.08694201, 0.06480221, 0.1088247],
      ['ACME (treated)', 0.08801048, 0.06560949, 0.1100008],
      ['ADE (control)',  0.01266038, 0.01245537, 0.01295223],
      ['ADE (treated)',  0.01159191, 0.01133799, 0.01185508],
      ['Total effect',   0.09960239, 0.07729146, 0.1214335],
      ['Prop. mediated (control)', 0.8791767, 0.8381833, 0.8959312],
      ['Prop. mediated (treated)', 0.8896202, 0.8486033, 0.9056457],
      ['ACME (average)', 0.08747624, 0.06520585, 0.1094128],
      ['ADE (average)', 0.01212614, 0.011955, 0.01228722],
      ['Prop. mediated (average)', 0.8843984, 0.8433933, 0.9007885]]
framing_boot_4231 = pd.DataFrame(df[1:], columns=df[0]).set_index('index')

df = [['index', 'Estimate', '95% lower bound', '95% upper bound'],
      ['ACME (control)', 0.086423, 0.050459, 0.122682],
      ['ACME (treated)', 0.086180, 0.055951, 0.126854],
      ['ADE (control)', 0.016463, -0.105838, 0.135283],
      ['ADE (treated)', 0.016706, -0.100041, 0.131978],
      ['Total effect', 0.102886, -0.017573, 0.219822],
      ['Prop. mediated (control)', 0.568071, -97.797424, 1.903790],
      ['Prop. mediated (treated)', 0.594115, -87.763233, 1.809403],
      ['ACME (average)', 0.086301, 0.053205, 0.124768],
      ['ADE (average)', 0.016585, -0.102939, 0.133630],
      ['Prop. mediated (average)', 0.581093, -92.780329, 1.856597]]
framing_para_4231 = pd.DataFrame(df[1:], columns=df[0]).set_index('index')

df = [['index', 'Estimate', '95% lower bound', '95% upper bound'],
      ['ACME (control)', 0.065531, 0.013544, 0.135273],
      ['ACME (treated)', 0.079500, 0.021696, 0.162567],
      ['ADE (control)',  0.273361, 0.000750, 0.468304],
      ['ADE (treated)',  0.259392, 0.003435, 0.452858],
      ['Total effect',   0.338892, 0.062783, 0.528931],
      ['Prop. mediated (control)', 0.178878, 0.041276, 1.698771],
      ['Prop. mediated (treated)', 0.229056, 0.071959, 1.579572],
      ['ACME (average)', 0.072515, 0.017953, 0.148920],
      ['ADE (average)', 0.266377, 0.002093, 0.459290],
      ['Prop. mediated (average)', 0.203967, 0.057539, 1.637799]]
framing_moderated_4231 = pd.DataFrame(df[1:], columns=df[0]).set_index('index')


def test_framing_example():

    cur_dir = os.path.dirname(os.path.abspath(__file__))
    data = pd.read_csv(os.path.join(cur_dir, 'results', "framing.csv"))

    outcome = np.asarray(data["cong_mesg"])
    outcome_exog = patsy.dmatrix("emo + treat + age + educ + gender + income", data,
                                  return_type='dataframe')
    probit = sm.families.links.probit
    outcome_model = sm.GLM(outcome, outcome_exog, family=sm.families.Binomial(link=probit))

    mediator = np.asarray(data["emo"])
    mediator_exog = patsy.dmatrix("treat + age + educ + gender + income", data,
                                 return_type='dataframe')
    mediator_model = sm.OLS(mediator, mediator_exog)

    tx_pos = [outcome_exog.columns.tolist().index("treat"),
              mediator_exog.columns.tolist().index("treat")]
    med_pos = outcome_exog.columns.tolist().index("emo")

    med = Mediation(outcome_model, mediator_model, tx_pos, med_pos)

    np.random.seed(4231)
    med_rslt = med.fit(method='boot', n_rep=10)
    diff = np.asarray(med_rslt.summary() - framing_boot_4231)
    assert_allclose(diff, 0, atol=1e-7)

    np.random.seed(4231)
    med_rslt = med.fit(method='parametric', n_rep=10)
    diff = np.asarray(med_rslt.summary() - framing_para_4231)
    assert_allclose(diff, 0, atol=1e-6)


def test_framing_example_moderator():
    # moderation without formulas, generally not useful but test anyway

    cur_dir = os.path.dirname(os.path.abspath(__file__))
    data = pd.read_csv(os.path.join(cur_dir, 'results', "framing.csv"))

    outcome = np.asarray(data["cong_mesg"])
    outcome_exog = patsy.dmatrix("emo + treat + age + educ + gender + income", data,
                                  return_type='dataframe')
    probit = sm.families.links.probit
    outcome_model = sm.GLM(outcome, outcome_exog, family=sm.families.Binomial(link=probit))

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


def test_framing_example_formula():

    cur_dir = os.path.dirname(os.path.abspath(__file__))
    data = pd.read_csv(os.path.join(cur_dir, 'results', "framing.csv"))

    probit = sm.families.links.probit
    outcome_model = sm.GLM.from_formula("cong_mesg ~ emo + treat + age + educ + gender + income",
                                        data, family=sm.families.Binomial(link=probit))

    mediator_model = sm.OLS.from_formula("emo ~ treat + age + educ + gender + income", data)

    med = Mediation(outcome_model, mediator_model, "treat", "emo")

    np.random.seed(4231)
    med_rslt = med.fit(method='boot', n_rep=10)
    diff = np.asarray(med_rslt.summary() - framing_boot_4231)
    assert_allclose(diff, 0, atol=1e-7)

    np.random.seed(4231)
    med_rslt = med.fit(method='parametric', n_rep=10)
    diff = np.asarray(med_rslt.summary() - framing_para_4231)
    assert_allclose(diff, 0, atol=1e-6)


def test_framing_example_moderator_formula():

    cur_dir = os.path.dirname(os.path.abspath(__file__))
    data = pd.read_csv(os.path.join(cur_dir, 'results', "framing.csv"))

    probit = sm.families.links.probit
    outcome_model = sm.GLM.from_formula("cong_mesg ~ emo + treat*age + emo*age + educ + gender + income",
                                        data, family=sm.families.Binomial(link=probit))

    mediator_model = sm.OLS.from_formula("emo ~ treat*age + educ + gender + income", data)

    moderators = {"age" : 20}
    med = Mediation(outcome_model, mediator_model, "treat", "emo",
                    moderators=moderators)

    np.random.seed(4231)
    med_rslt = med.fit(method='parametric', n_rep=10)
    diff = np.asarray(med_rslt.summary() - framing_moderated_4231)
    assert_allclose(diff, 0, atol=1e-6)
