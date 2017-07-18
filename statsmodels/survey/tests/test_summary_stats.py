import summary_stats as ss
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

design = ss.SurveyDesign(strata, cluster, weights, cov_method='jack')
design_fpc = ss.SurveyDesign(strata, cluster, weights, fpc=fpc, cov_method='jack')

def test_design():
    assert_equal(design.clust_per_strat, np.array([3, 2]))
    assert_equal(design.strat_for_clust, np.array([0, 0, 0, 1, 1]))
    # make sure get_rep_weights works
    rep = [design.get_rep_weights(c=c) for c in range(design.nclust)]

def test_mean_jack():
    avg_fpc = ss.SurveyMean(design_fpc, data)
    assert_allclose(avg_fpc.est, np.r_[3.625, 4.6875, 3.9375])
    assert_allclose(avg_fpc.stderr, np.r_[0.7907643, 1.05731, .8268258],  rtol=1e-5, atol=0)

    # avg_fpc_mse = ss.SurveyMean(design_fpc, data, mse=True)
    # assert_allclose(avg_fpc_mse.est, np.r_[3.625, 4.6875, 3.9375])
    # assert_allclose(avg_fpc_mse.std, np.r_[0.7907643, 1.05731, .8268258],  rtol=1e-5, atol=0)

    avg = ss.SurveyMean(design, data)
    assert_allclose(avg.est, np.r_[3.625, 4.6875, 3.9375])
    assert_allclose(avg.stderr, np.r_[0.8652204, 1.223652, .9952406],  rtol=1e-5, atol=0)

    avg_mse = ss.SurveyMean(design, data, mse=True)
    assert_allclose(avg_mse.est, np.r_[3.625, 4.6875, 3.9375])
    assert_allclose(avg_mse.stderr, np.r_[.8666358, 1.225125, .9961644],  rtol=1e-5, atol=0)

# def test_mean_boot():
#     avg = ss.SurveyMean(design, data, 'boot')
#     assert_allclose(avg.est, np.r_[0, 0])
#     assert_allclose(avg.vcov, np.r_[0, 0], rtol=1e-5, atol=0)

def test_total_jack():
    tot_fpc = ss.SurveyTotal(design_fpc, data)
    assert_allclose(tot_fpc.est, np.r_[58, 75, 63])
    assert_allclose(tot_fpc.stderr, np.r_[9.402127, 20.82066, 10.51665],  rtol=1e-5, atol=0)

    # tot_fpc_mse = ss.SurveyMean(design_fpc, data, 'jack', mse=True)
    # assert_allclose(tot_fpc_mse.est, np.r_[3.625, 4.6875, 3.9375])
    # assert_allclose(tot_fpc_mse.vcov, np.r_[0.7907643, 1.05731, .8268258],  rtol=1e-5, atol=0)

    tot = ss.SurveyTotal(design, data)
    assert_allclose(tot.est, np.r_[58, 75, 63])
    assert_allclose(tot.stderr, np.r_[10.58301, 23.38803, 13.49074],  rtol=1e-5, atol=0)

    tot_mse = ss.SurveyTotal(design, data, mse=True)
    assert_allclose(tot_mse.est, np.r_[58, 75, 63])
    assert_allclose(tot_mse.stderr, np.r_[10.58301, 23.38803, 13.49074],  rtol=1e-5, atol=0)

# def test_total_boot():
#     tot = ss.SurveyTotal(design, data, 'boot')
#     assert_allclose(tot.est, np.r_[68, 85])
#     assert_allclose(tot.vcov, np.r_[19.79899, 15.71623],  rtol=1e-5, atol=0)

# def test_quantile_jack():
#     quant = ss.SurveyQuantile(design, data, [.1, .25, .33, .5, .75, .99], 'jack')
#     assert_allclose(quant.est[0], np.r_[1, 2, 2, 3.5, 5, 9])
#     ## change 7 to 6 to accommodate w/ stata
#     assert_allclose(quant.est[1], np.r_[2, 3, 3, 4.5, 7, 9])

#     # ensure that median yields same result
#     med = ss.SurveyMedian(design, data, 'jack')
#     assert_equal(quant.est[0][-3], med.est[0][0])
#     assert_equal(quant.est[1][-3], med.est[1][0])

# def test_quantile_boot():
#     quant = ss.SurveyQuantile(design, data, [.1, .25, .33, .5, .75, .99], 'boot')
#     assert_allclose(quant.est[0], np.r_[1, 2, 2, 3.5, 5, 9])
#     ## change 7 to 6 to accommodate w/ stata
#     assert_allclose(quant.est[1], np.r_[2, 3, 3, 4.5, 7, 9])

#     # ensure that median yields same result
#     med = ss.SurveyMedian(design, data, 'boot')
#     assert_equal(quant.est[0][-3], med.est[0][0])
#     assert_equal(quant.est[1][-3], med.est[1][0])
