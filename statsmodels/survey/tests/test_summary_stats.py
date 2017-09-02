import statsmodels.survey.summary_stats as ss
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

design = ss.SurveyDesign(strata, cluster, weights)
design_fpc = ss.SurveyDesign(strata, cluster, weights, fpc=fpc)
design_no_weights = ss.SurveyDesign(strata, cluster, fpc=fpc)
rw_list = [design_no_weights.get_rep_weights(c=c, cov_method='jack') for c in range(design.n_clust)]
rw = np.asarray(rw_list).T
design_rw = ss.SurveyDesign(rep_weights=rw)

def test_design():
    assert_equal(design.clust_per_strat, np.array([3, 2]))
    assert_equal(design.strat_for_clust, np.array([0, 0, 0, 1, 1]))

    # SurveyDesign works with lists
    design_list = ss.SurveyDesign(rep_weights=rw_list, weights=weights)
    assert_equal(design_list.rep_weights, design_rw.rep_weights)


def test_mean_jack():
    avg_fpc = ss.SurveyMean(design_fpc, data, cov_method='jack', center_by='stratum')
    assert_allclose(avg_fpc.est, np.r_[3.625, 4.6875, 3.9375])
    assert_allclose(avg_fpc.stderr, np.r_[0.7907643, 1.05731, .8268258],  rtol=1e-5, atol=0)

    # avg_fpc_mse = ss.SurveyMean(design_fpc, data, mse=True)
    # assert_allclose(avg_fpc_mse.est, np.r_[3.625, 4.6875, 3.9375])
    # assert_allclose(avg_fpc_mse.std, np.r_[0.7907643, 1.05731, .8268258],  rtol=1e-5, atol=0)

    avg = ss.SurveyMean(design, data, cov_method='jack', center_by='stratum')
    assert_allclose(avg.est, np.r_[3.625, 4.6875, 3.9375])
    assert_allclose(avg.stderr, np.r_[0.8652204, 1.223652, .9952406],  rtol=1e-5, atol=0)

    avg_mse = ss.SurveyMean(design, data, cov_method='jack', center_by='est')
    assert_allclose(avg_mse.est, np.r_[3.625, 4.6875, 3.9375])
    assert_allclose(avg_mse.stderr, np.r_[.8666358, 1.225125, .9961644],  rtol=1e-5, atol=0)

def test_mean_boot():
    # stderr for both should be the same. But they differ by the following
    # array([ 0.00089597,  0.00085691,  0.00046335])
    # This is why rtol=1e-2 for the first test. Note that STATA has them as the same
    # So this could be an issue

    avg = ss.SurveyMean(design_rw, data, cov_method='boot', center_by='global')
    assert_allclose(avg.est, np.r_[3.636364, 4.636364, 3.727273])
    assert_allclose(avg.stderr, np.r_[0.5762342, 0.6967576, 0.5428541], rtol=1e-2, atol=0)

    avg_mse = ss.SurveyMean(design_rw, data, cov_method='boot', center_by='est')
    assert_allclose(avg_mse.est, np.r_[3.636364, 4.636364, 3.727273])
    assert_allclose(avg_mse.stderr, np.r_[0.5762342, 0.6967576, 0.5428541], rtol=1e-5, atol=0)

    reps_boot = [design_no_weights.get_rep_weights(c=c, cov_method='boot') for c in range(design.n_clust)]
    reps_boot = np.asarray(reps_boot).T
    assert_equal(reps_boot.shape, rw.shape)

def test_mean_linearized():
    avg = ss.SurveyMean(design, data, cov_method='linearized', center_by='stratum')
    assert_allclose(avg.est, np.r_[3.625, 4.6875, 3.9375])
    assert_allclose(avg.stderr, np.r_[0.8623882, 1.220708, .993394], rtol=1e-2, atol=0)

    avg_fpc = ss.SurveyMean(design_fpc, data, cov_method='linearized', center_by='stratum')
    assert_allclose(avg_fpc.est, np.r_[3.625, 4.6875, 3.9375])
    assert_allclose(avg_fpc.stderr, np.r_[0.787975, 1.054243, .8248248],  rtol=1e-5, atol=0)


def test_total_jack():
    tot_fpc = ss.SurveyTotal(design_fpc, data, cov_method='jack', center_by='stratum')
    assert_allclose(tot_fpc.est, np.r_[58, 75, 63])
    assert_allclose(tot_fpc.stderr, np.r_[9.402127, 20.82066, 10.51665],  rtol=1e-5, atol=0)

    # tot_fpc_mse = ss.SurveyMean(design_fpc, data, 'jack', mse=True)
    # assert_allclose(tot_fpc_mse.est, np.r_[3.625, 4.6875, 3.9375])
    # assert_allclose(tot_fpc_mse.vcov, np.r_[0.7907643, 1.05731, .8268258],  rtol=1e-5, atol=0)

    tot = ss.SurveyTotal(design, data, cov_method='jack', center_by='stratum')
    assert_allclose(tot.est, np.r_[58, 75, 63])
    assert_allclose(tot.stderr, np.r_[10.58301, 23.38803, 13.49074],  rtol=1e-5, atol=0)

    tot_mse = ss.SurveyTotal(design, data, cov_method='jack', center_by='est')
    assert_allclose(tot_mse.est, np.r_[58, 75, 63])
    assert_allclose(tot_mse.stderr, np.r_[10.58301, 23.38803, 13.49074],  rtol=1e-5, atol=0)

def test_total_boot():
    tot_mse = ss.SurveyTotal(design_rw, data, cov_method='boot', center_by='global')
    assert_allclose(tot_mse.est, np.r_[40, 51, 41])
    assert_allclose(tot_mse.stderr, np.r_[4.062019 , 10.2323, 4.549725],  rtol=1e-5, atol=0)

    tot = ss.SurveyTotal(design_rw, data, cov_method='boot', center_by='est')
    assert_allclose(tot.est, np.r_[40, 51, 41])
    assert_allclose(tot.stderr, np.r_[4.062019 , 10.2323, 4.549725],  rtol=1e-5, atol=0)

def test_total_linearized():
    tot = ss.SurveyTotal(design, data, cov_method='linearized', center_by='stratum')
    assert_allclose(tot.est, np.r_[58, 75, 63])
    assert_allclose(tot.stderr, np.r_[10.58301, 23.38803, 13.49074], rtol=1e-2, atol=0)

    tot_fpc = ss.SurveyTotal(design_fpc, data, cov_method='linearized', center_by='stratum')
    assert_allclose(tot_fpc.est, np.r_[58, 75, 63])
    assert_allclose(tot_fpc.stderr, np.r_[9.402127, 20.82066, 10.51665],  rtol=1e-5, atol=0)


# For now, no SE is given by STATA. So for now, will only have
# test_quantile_jack() check the estimate, and comment out
# test_quantile_boot()
def test_quantile_jack():
    quant_list = [.1, .25, .33, .5,.75, .99]
    quant = [ss.SurveyQuantile(design, data, q, 'jack', center_by='est').est for q in quant_list]
    quant = np.asarray(quant).T
    # each row is a variable, the columns are the values for the quantile
    assert_allclose(quant[0,:], np.r_[1, 2, 3, 3, 5, 9])
    assert_allclose(quant[1,:], np.r_[1, 3, 3, 4, 7, 9])
    assert_allclose(quant[2,:], np.r_[1, 2, 2, 4, 6, 7])

    # ensure that median yields same result
    med = ss.SurveyMedian(design, data)
    assert_equal(quant[:,3], med.est)

def test_ratio_linearized():
    ratio_fpc = ss.SurveyRatio(design_fpc, data[:, :2], cov_method='linearized', center_by='stratum')
    assert_allclose(ratio_fpc.est, np.r_[.7733333])
    assert_allclose(ratio_fpc.stderr, np.r_[.3391315], rtol=1e-2, atol=0)

def test_ratio_jackknife():
    ratio_fpc = ss.SurveyRatio(design_fpc, data[:, :2], cov_method='jack', center_by='stratum')
    assert_allclose(ratio_fpc.est, np.r_[.7733333])
    assert_allclose(ratio_fpc.stderr, np.r_[.3609269], rtol=1e-2, atol=0)

# import pandas as pd
# df = pd.read_stata("/home/jarvis/Downloads/nhanes2jknife.dta")

# # # Jackknife replicate weights
# rw = [np.asarray(df[x]) for x in df.columns if x.startswith("jkw_")]
# rw = np.asarray(rw).T

# ds = SurveyDesign(rep_weights=rw, cov_method='jack')
# nm = SurveyMean(ds, np.asarray(df[["age", "height"]]), mse=False)
# nm_mse = SurveyMean(ds, np.asarray(df[["age", "height"]]), mse=True)
# stata_mse_std = np.array([ 29.66141 ,   4.506857])
# print(stata_mse_std / nm_mse.std) # .712
# strata = np.r_[0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1]
# cluster = np.r_[0, 0, 1, 1, 2, 2, 3, 3, 3, 4, 4]
# weights = np.r_[1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1].astype(np.float64)
# fpc = np.r_[.5, .5, .5, .5, .5, .5, .1, .1, .1, .1, .1]
# data = np.asarray([[1, 3, 2, 5, 4, 1, 2, 3, 4, 6, 9],
#                    [5, 3, 2, 1, 4, 7, 8, 9, 5, 4, 3],
#                    [3, 2, 1, 5, 6, 7, 4, 2, 1, 6, 4]], dtype=np.float64).T

# design = SurveyDesign(strata, cluster, weights, cov_method='jack')

# # Get the jackknife replicate weights

# rw = []
# for k in range(5):
#     rw.append(design.get_rep_weights(k))
# rw = np.asarray(rw).T

# sm1 = SurveyMean(design, data[:, 0:2], mse=True)

    # Get the means, using provided weights
# design_rw = SurveyDesign(rep_weights=rw)
# sm2 = SurveyMean(design_rw, data[:, 0:2], mse=True)

# assert_allclose(sm1.est, sm2.est)
# assert_allclose(sm1.std, sm2.std)