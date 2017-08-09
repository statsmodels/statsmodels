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
data = np.asarray([[1, 3, 2, 5, 4, 1, 2, 3, 4, 6, 9],
                   [5, 3, 2, 1, 4, 7, 8, 9, 5, 4, 3],
                   [3, 2, 1, 5, 6, 7, 4, 2, 1, 6, 4]], dtype=np.float64).T
# need to get stata results to compare
y = np.random.choice([0,1], 11)
X = data[:, [1,2]]
design = SurveyDesign(strata, cluster, weights)
# assert_equal(design.clust, np.r_[0, 0, 1, 1, 2, 2, 3, 3, 3, 4, 4])
model_class = sm.GLM(y, X, family = sm.families.Binomial())
model_class.fit()

init_args = {'family': sm.families.Binomial()}
model_class = sm.GLM
model = SurveyModel(design, model_class=model_class, init_args=init_args)
model.fit(y, X, cov_method='linearized_stata',center_by='est')

def test_jack_repw():
    design = ss.SurveyDesign(strata, cluster, weights)
    assert_equal(design.clust, np.r_[0, 0, 1, 1, 2, 2, 3, 3, 3, 4, 4])
    model_class = sm.GLM
    init_args = {'family': sm.families.Binomial()}
    model = smod.SurveyModel(design, model_class=model_class, init_args=init_args)
    rslt = model.fit(y, X, cov_method='linearized_sas', center_by='est')

    rw = []
    for k in range(5):
        rw.append(design.get_rep_weights(c=k, cov_method='jack'))
    rw = np.asarray(rw).T
    model = smod.SurveyModel(design, model_class=model_class)
    model.fit(y, X)
    design_rw = ss.SurveyDesign(weights = weights, rep_weights=rw)
    model_rw = smod.SurveyModel(design_rw, model_class=model_class)
    model_rw.fit(y, X, cov_method='jack')

    assert_allclose(model.params, model_rw.params)
    """
    comparison between stderr fails because model has
    nh = self.design.clust_per_strat[self.design.strat_for_clust].astype(np.float64)
    while model_rw has
    self.design.rep_weights.shape[1]
    which is what STATA uses as nh when PSU information is unknown..
    """
    # assert_allclose(model.stderr, model_rw.stderr)

