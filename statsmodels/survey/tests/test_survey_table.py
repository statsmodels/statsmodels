from __future__ import division
import statsmodels.survey.summary_stats as ss
import statsmodels.survey.survey_model as smod
import statsmodels.survey.survey_table as stab
import pandas as pd
import numpy as np
import os
from numpy.testing import (assert_almost_equal, assert_equal, assert_array_less,
                           assert_raises, assert_allclose)


strata = np.r_[0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1]
cluster = np.r_[0, 0, 1, 1, 2, 2, 3, 3, 3, 4, 4]
weights = np.r_[1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1].astype(np.float64)

design = ss.SurveyDesign(strata, cluster, weights)

cur_dir = os.path.dirname(os.path.abspath(__file__))
fname = "nhanes2b.dta"
fpath = os.path.join(cur_dir, 'data', fname)
data = pd.read_stata(fpath)
design = ss.SurveyDesign(data.stratid, data.psuid, data.finalwgt)
data = data[['race', 'diabetes']]
def test_pearson():
    tab = stab.SurveyTable(design, data)
    tab.test_pearson()
    tab.test_lrt()
    print(tab)
    print(tab.pearson, tab.pearson_chi, tab.pearson_f)
    print(tab.dof_F)
    # print(tab.delta_numer)
    # print(tab.delta_denom)
    # print(tab._delta_est)
    # print(tab.stderr)
    # print(tab.d.shape)
    # raise ValueError('stop here')

def test_contrast():
    tab = stab.SurveyTable(design, data)
    contrast = tab._contrast_matrix()
    mat = tab._main_effects_mat()
    R, C = tab.table.shape
    assert_equal(contrast.shape, (R*C, (R-1)*(C-1)))
    assert_almost_equal(np.dot(contrast.T, mat), 0)

    # raise ValueError('stop here')

# print(tab.table)
# print(tab) # does same but w/ cell proportions
