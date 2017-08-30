from __future__ import division
import summary_stats as ss
import statsmodels.api as sm
import survey_model as smod
import survey_table as stab
import pandas as pd
import numpy as np
from numpy.testing import (assert_almost_equal, assert_equal, assert_array_less,
                           assert_raises, assert_allclose)


strata = np.r_[0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1]
cluster = np.r_[0, 0, 1, 1, 2, 2, 3, 3, 3, 4, 4]
weights = np.r_[1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1].astype(np.float64)

design = ss.SurveyDesign(strata, cluster, weights)
data = np.array([np.random.choice([0,1], 11),
                 np.random.choice([3,4], 11)]).T

def test_pearson():
    tab = stab.SurveyTable(design, data)
    tab.test_pearson()
    print(tab.pearson, tab.dof)
# print(tab.table)
# print(tab) # does same but w/ cell proportions
