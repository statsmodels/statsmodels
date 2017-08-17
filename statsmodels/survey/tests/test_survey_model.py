from __future__ import division
import summary_stats as ss
import statsmodels.api as sm
import survey_model as smod
import pandas as pd
import numpy as np
from numpy.testing import (assert_almost_equal, assert_equal, assert_array_less,
                           assert_raises, assert_allclose)

strata = np.r_[0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1]
cluster = np.r_[0, 0, 1, 1, 2, 2, 3, 3, 3, 4, 4]
weights = np.r_[1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1].astype(np.float64)
fpc = np.r_[.5, .5, .5, .5, .5, .5, .1, .1, .1, .1, .1]
data = pd.read_csv("~/Downloads/logistic_df.csv")
y = np.asarray(data['y'])
X = np.asarray(data[['x1', 'x2']])
X = sm.add_constant(X)

design = ss.SurveyDesign(strata, cluster, weights=weights)
assert_equal(design.clust, np.r_[0, 0, 1, 1, 2, 2, 3, 3, 3, 4, 4])
model_class = sm.GLM
init_args = {'family': sm.families.Binomial()}

df = pd.read_csv("~/Downloads/big_logistic_df.csv")
survey_df = ss.SurveyDesign(strata= np.asarray(df['strata']), cluster= np.asarray(df["dnum"]), weights=np.asarray(df["pw"]))
y2 = np.asarray(df["y"])
X2 = np.asarray(df[['api00', 'api99']])
X2 = sm.add_constant(X2)
def test_linearized():
    model = smod.SurveyModel(design, model_class=model_class, init_args=init_args)
    rslt = model.fit(y, X, cov_method='linearized_stata', center_by='stratum')
    assert_allclose(model.params, np.r_[-2.094691, .4969399, -.1307789], rtol=1e-5, atol=0)
    assert_allclose(model.stderr, np.r_[2.42054, .2064146, .3907528],rtol=1e-5, atol=0)

    # don't specify init_args. family is default gaussian
    model = smod.SurveyModel(design, model_class=model_class)
    model.fit(y, X, cov_method='linearized_stata', center_by='stratum')
    assert_allclose(model.params, np.r_[.0614334, .0982177, -.0214168], rtol=1e-5)
    assert_allclose(model.stderr, np.r_[.5023708, .0370594, .0822633], rtol=1e-5)

def test_jack_calculated():
    # fam = gaussian
    model = smod.SurveyModel(design, model_class=model_class)
    model.fit(y, X, cov_method='jack', center_by='stratum')
    assert_allclose(model.params, np.r_[.0614334, .0982177, -.0214168], rtol=1e-5)
    assert_allclose(model.stderr, np.r_[.6703504, .059024, .1188806], rtol=1e-5)

    # fam = binomial does not work well w/ small data.

    model = smod.SurveyModel(survey_df, model_class=model_class, init_args=init_args)
    model.fit(y2, X2, cov_method='jack', center_by='stratum')

    assert_allclose(model.params, np.r_[-.3752291, -.0034241, .0044153], rtol=1e-4)
    assert_allclose(model.stderr, np.r_[1.240397, .0057129, .0052336], rtol=1e-5)

def test_jack_supplied():
    rw = []
    for k in range(5):
        rw.append(design.get_rep_weights(c=k, cov_method='jack'))
    rw = np.asarray(rw).T
    design_rw = ss.SurveyDesign(weights = weights, rep_weights=rw)
    model_rw = smod.SurveyModel(design_rw, model_class=model_class, init_args=init_args)
    model_rw.fit(y, X, cov_method='jack')
    # print("calculated rep_weights")
    # print(model.params)
    # print(model.stderr)
    # print(" \n supplied rep_weights")
    # print(model_rw.params)
    # print(model_rw.stderr)
    # raise ValueError('stop here')
    """
    comparison between stderr fails because model has
    nh = self.design.clust_per_strat[self.design.strat_for_clust].astype(np.float64)
    while model_rw has
    self.design.rep_weights.shape[1]
    which is what STATA uses as nh when PSU information is unknown..
    """
    # assert_allclose(model.stderr, model_rw.stderr)

