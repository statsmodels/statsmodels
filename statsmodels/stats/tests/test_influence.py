# -*- coding: utf-8 -*-
"""
Created on Tue Jun 12 13:18:12 2018

Author: Josef Perktold
"""
import json
import os

from statsmodels.compat.pandas import testing as pdt

import numpy as np
from numpy.testing import assert_allclose, assert_equal, assert_array_equal
import pandas as pd

import pytest

from statsmodels.tools.tools import add_constant
from statsmodels.datasets import macrodata
from statsmodels.regression.linear_model import OLS
from statsmodels.genmod.generalized_linear_model import GLM
from statsmodels.genmod import families

import statsmodels.stats.outliers_influence as oi
from statsmodels.stats.outliers_influence import MLEInfluence

# -------------------------------------------------------------------
# Helpers to load results

cur_dir = os.path.abspath(os.path.dirname(__file__))

file_name = 'binary_constrict.csv'
file_path = os.path.join(cur_dir, 'results', file_name)
data_bin = pd.read_csv(file_path, index_col=0)

file_name = 'results_influence_logit.csv'
file_path = os.path.join(cur_dir, 'results', file_name)
results_sas_df = pd.read_csv(file_path, index_col=0)


# previously in test_diagnostic
def get_duncan_data():
    # results from R with NA -> 1. Just testing interface here because
    # outlier_test is just a wrapper
    labels = ['accountant', 'pilot', 'architect', 'author', 'chemist',
              'minister', 'professor', 'dentist', 'reporter', 'engineer',
              'undertaker', 'lawyer', 'physician', 'welfare.worker', 'teacher',
              'conductor', 'contractor', 'factory.owner', 'store.manager',
              'banker', 'bookkeeper', 'mail.carrier', 'insurance.agent',
              'store.clerk', 'carpenter', 'electrician', 'RR.engineer',
              'machinist', 'auto.repairman', 'plumber', 'gas.stn.attendant',
              'coal.miner', 'streetcar.motorman', 'taxi.driver',
              'truck.driver', 'machine.operator', 'barber', 'bartender',
              'shoe.shiner', 'cook', 'soda.clerk', 'watchman', 'janitor',
              'policeman', 'waiter']
    # Duncan's prestige data from car
    exog = [[1.0, 62.0, 86.0], [1.0, 72.0, 76.0], [1.0, 75.0, 92.0],
            [1.0, 55.0, 90.0], [1.0, 64.0, 86.0], [1.0, 21.0, 84.0],
            [1.0, 64.0, 93.0], [1.0, 80.0, 100.0], [1.0, 67.0, 87.0],
            [1.0, 72.0, 86.0], [1.0, 42.0, 74.0], [1.0, 76.0, 98.0],
            [1.0, 76.0, 97.0], [1.0, 41.0, 84.0], [1.0, 48.0, 91.0],
            [1.0, 76.0, 34.0], [1.0, 53.0, 45.0], [1.0, 60.0, 56.0],
            [1.0, 42.0, 44.0], [1.0, 78.0, 82.0], [1.0, 29.0, 72.0],
            [1.0, 48.0, 55.0], [1.0, 55.0, 71.0], [1.0, 29.0, 50.0],
            [1.0, 21.0, 23.0], [1.0, 47.0, 39.0], [1.0, 81.0, 28.0],
            [1.0, 36.0, 32.0], [1.0, 22.0, 22.0], [1.0, 44.0, 25.0],
            [1.0, 15.0, 29.0], [1.0, 7.0, 7.0], [1.0, 42.0, 26.0],
            [1.0, 9.0, 19.0], [1.0, 21.0, 15.0], [1.0, 21.0, 20.0],
            [1.0, 16.0, 26.0], [1.0, 16.0, 28.0], [1.0, 9.0, 17.0],
            [1.0, 14.0, 22.0], [1.0, 12.0, 30.0], [1.0, 17.0, 25.0],
            [1.0, 7.0, 20.0], [1.0, 34.0, 47.0], [1.0, 8.0, 32.0]]
    endog = [82.,  83.,  90.,  76.,  90.,  87.,  93.,  90.,  52.,  88.,  57.,
             89.,  97.,  59.,  73.,  38.,  76.,  81.,  45.,  92.,  39.,  34.,
             41.,  16.,  33.,  53.,  67.,  57.,  26.,  29.,  10.,  15.,  19.,
             10.,  13.,  24.,  20.,   7.,   3.,  16.,   6.,  11.,   8.,  41.,
             10.]

    return endog, exog, labels


# -------------------------------------------------------------------
# Tests previously in test_diagnostic

def test_outlier_test():
    endog, exog, labels = get_duncan_data()
    ndarray_mod = OLS(endog, exog).fit()
    rstudent = [3.1345185839, -2.3970223990,  2.0438046359, -1.9309187757,
                1.8870465798, -1.7604905300, -1.7040324156,  1.6024285876,
                -1.4332485037, -1.1044851583,  1.0688582315,  1.0185271840,
                -0.9024219332, -0.9023876471, -0.8830953936,  0.8265782334,
                0.8089220547,  0.7682770197,  0.7319491074, -0.6665962829,
                0.5227352794, -0.5135016547,  0.5083881518,  0.4999224372,
                -0.4980818221, -0.4759717075, -0.4293565820, -0.4114056499,
                -0.3779540862,  0.3556874030,  0.3409200462,  0.3062248646,
                0.3038999429, -0.3030815773, -0.1873387893,  0.1738050251,
                0.1424246593, -0.1292266025,  0.1272066463, -0.0798902878,
                0.0788467222,  0.0722556991,  0.0505098280,  0.0233215136,
                0.0007112055]
    unadj_p = [0.003177202, 0.021170298, 0.047432955, 0.060427645, 0.066248120,
               0.085783008, 0.095943909, 0.116738318, 0.159368890, 0.275822623,
               0.291386358, 0.314400295, 0.372104049, 0.372122040, 0.382333561,
               0.413260793, 0.423229432, 0.446725370, 0.468363101, 0.508764039,
               0.603971990, 0.610356737, 0.613905871, 0.619802317, 0.621087703,
               0.636621083, 0.669911674, 0.682917818, 0.707414459, 0.723898263,
               0.734904667, 0.760983108, 0.762741124, 0.763360242, 0.852319039,
               0.862874018, 0.887442197, 0.897810225, 0.899398691, 0.936713197,
               0.937538115, 0.942749758, 0.959961394, 0.981506948, 0.999435989]
    bonf_p = [
        0.1429741, 0.9526634, 2.1344830, 2.7192440, 2.9811654, 3.8602354,
        4.3174759, 5.2532243, 7.1716001, 12.4120180, 13.1123861, 14.1480133,
        16.7446822, 16.7454918, 17.2050103, 18.5967357, 19.0453245,
        20.1026416, 21.0763395, 22.8943818, 27.1787396, 27.4660532,
        27.6257642, 27.8911043, 27.9489466, 28.6479487, 30.1460253,
        30.7313018, 31.8336506, 32.5754218, 33.0707100, 34.2442399,
        34.3233506, 34.3512109, 38.3543568, 38.8293308, 39.9348989,
        40.4014601, 40.4729411, 42.1520939, 42.1892152, 42.4237391,
        43.1982627, 44.1678127, 44.9746195]
    bonf_p = np.array(bonf_p)
    bonf_p[bonf_p > 1] = 1
    sorted_labels = [
        "minister", "reporter", "contractor", "insurance.agent",
        "machinist", "store.clerk", "conductor", "factory.owner",
        "mail.carrier", "streetcar.motorman", "carpenter", "coal.miner",
        "bartender", "bookkeeper", "soda.clerk", "chemist", "RR.engineer",
        "professor", "electrician", "gas.stn.attendant", "auto.repairman",
        "watchman", "banker", "machine.operator", "dentist", "waiter",
        "shoe.shiner", "welfare.worker", "plumber", "physician", "pilot",
        "engineer", "accountant", "lawyer", "undertaker", "barber",
        "store.manager", "truck.driver", "cook", "janitor", "policeman",
        "architect", "teacher", "taxi.driver", "author"]

    res2 = np.c_[rstudent, unadj_p, bonf_p]
    res = oi.outlier_test(ndarray_mod, method='b', labels=labels, order=True)
    assert_allclose(res.values, res2, atol=1.5e-7, rtol=0)
    assert_equal(res.index.tolist(), sorted_labels)

    data = pd.DataFrame(np.column_stack((endog, exog)),
                        columns='y const var1 var2'.split(),
                        index=labels)

    # check `order` with pandas bug in GH#3971
    res_pd = OLS.from_formula('y ~ const + var1 + var2 - 0', data).fit()

    res_outl2 = oi.outlier_test(res_pd, method='b', order=True)
    assert_allclose(res_outl2.values, res2, atol=1.5e-7, rtol=0)
    assert_equal(res_outl2.index.tolist(), sorted_labels)

    res_outl1 = res_pd.outlier_test(method='b')
    res_outl1 = res_outl1.sort_values(['unadj_p'], ascending=True)
    assert_allclose(res_outl1.values, res2, atol=1.5e-7, rtol=0)
    assert_equal(res_outl1.index.tolist(), sorted_labels)
    assert_array_equal(res_outl2.index, res_outl1.index)

    # additional keywords in method
    res_outl3 = res_pd.outlier_test(method='b', order=True)
    assert_equal(res_outl3.index.tolist(), sorted_labels)
    res_outl4 = res_pd.outlier_test(method='b', order=True, cutoff=0.15)
    assert_equal(res_outl4.index.tolist(), sorted_labels[:1])


@pytest.mark.smoke
def test_outlier_influence_funcs(reset_randomstate):
    x = add_constant(np.random.randn(10, 2))
    y = x.sum(1) + np.random.randn(10)
    res = OLS(y, x).fit()
    out_05 = oi.summary_table(res)
    # GH#3344 : Check alpha has an effect
    out_01 = oi.summary_table(res, alpha=0.01)
    assert np.all(out_01[1][:, 6] <= out_05[1][:, 6])
    assert np.all(out_01[1][:, 7] >= out_05[1][:, 7])

    res2 = OLS(y, x[:,0]).fit()
    oi.summary_table(res2, alpha=0.05)
    infl = res2.get_influence()
    infl.summary_table()


def test_influence_dtype():
    # see GH#2148  bug when endog is integer
    y = np.ones(20)
    np.random.seed(123)
    x = np.random.randn(20, 3)
    res1 = OLS(y, x).fit()

    res2 = OLS(y*1., x).fit()
    cr1 = res1.get_influence().cov_ratio
    cr2 = res2.get_influence().cov_ratio
    assert_allclose(cr1, cr2, rtol=1e-14)
    # regression test for values from R
    cr3 = np.array(
      [1.22239215,  1.31551021,  1.52671069,  1.05003921,  0.89099323,
       1.57405066,  1.03230092,  0.95844196,  1.15531836,  1.21963623,
       0.87699564,  1.16707748,  1.10481391,  0.98839447,  1.08999334,
       1.35680102,  1.46227715,  1.45966708,  1.13659521,  1.22799038])
    assert_allclose(cr1, cr3, atol=1.5e-8, rtol=0)


def test_influence_wrapped():
    d = macrodata.load_pandas().data
    # growth rates
    gs_l_realinv = 400 * np.log(d['realinv']).diff().dropna()
    gs_l_realgdp = 400 * np.log(d['realgdp']).diff().dropna()
    lint = d['realint'][:-1]

    # re-index these because they won't conform to lint
    gs_l_realgdp.index = lint.index
    gs_l_realinv.index = lint.index

    data = dict(const=np.ones_like(lint), lint=lint, lrealgdp=gs_l_realgdp)
    # Note: column order is important
    exog = pd.DataFrame(data, columns=['const', 'lrealgdp', 'lint'])

    res = OLS(gs_l_realinv, exog).fit()

    #basic
    # already tested
    #assert_allclose(lsdiag['cov.scaled'],
    #                res.cov_params().values.ravel(),
    #                atol=1.5e-14, rtol=0)
    #assert_allclose(lsdiag['cov.unscaled'],
    #                res.normalized_cov_params.values.ravel(),
    #                atol=1.5e-14, rtol=0)

    infl = oi.OLSInfluence(res)

    # smoke test just to make sure it works, results separately tested
    df = infl.summary_frame()
    assert isinstance(df, pd.DataFrame)

    # slow: TODO: separate and mark?
    path = os.path.join(cur_dir, "results", "influence_lsdiag_R.json")
    with open(path, "r") as fp:
        lsdiag = json.load(fp)

    c0, c1 = infl.cooks_distance  # TODO: what's c1, it's pvalues? -ss

    # NOTE: we get a hard-cored 5 decimals with pandas testing
    assert_allclose(c0, lsdiag['cooks'], atol=1.5e-14, rtol=0)
    assert_allclose(infl.hat_matrix_diag, (lsdiag['hat']),
                    atol=1.5e-14, rtol=0)
    assert_allclose(infl.resid_studentized_internal, lsdiag['std.res'],
                    atol=1.5e-14, rtol=0)

    # slow: TODO: separate and mark?
    dffits, dffth = infl.dffits
    assert_allclose(dffits, lsdiag['dfits'],
                    atol=1.5e-14, rtol=0)
    assert_allclose(infl.resid_studentized_external, lsdiag['stud.res'],
                    atol=1.5e-14, rtol=0)

    import pandas
    fn = os.path.join(cur_dir, "results", "influence_measures_R.csv")
    infl_r = pd.read_csv(fn, index_col=0)
    #not used yet:
    #conv = lambda s: 1 if s == 'TRUE' else 0
    #path = os.path.join(cur_dir, "results", "influence_measures_bool_R.csv")
    #infl_bool_r  = pd.read_csv(path, index_col=0,
    #                           converters=dict(zip(lrange(7),[conv]*7)))
    infl_r2 = np.asarray(infl_r)
    # TODO: finish wrapping this stuff
    assert_allclose(infl.dfbetas, infl_r2[:,:3], atol=1.5e-13, rtol=0)
    assert_allclose(infl.cov_ratio, infl_r2[:,4], atol=1.5e-14, rtol=0)


# -------------------------------------------------------------------

def test_influence_glm_bernoulli():
    # example uses Finney's data and is used in Pregibon 1981

    df = data_bin
    results_sas = np.asarray(results_sas_df)

    res = GLM(df['constrict'], df[['const', 'log_rate', 'log_volumne']],
              family=families.Binomial()).fit(attach_wls=True, atol=1e-10)

    infl = res.get_influence(observed=False)

    k_vars = 3
    assert_allclose(infl.dfbetas, results_sas[:, 5:8], atol=1e-4)
    assert_allclose(infl.d_params, results_sas[:, 5:8] * res.bse.values, atol=1e-4)
    assert_allclose(infl.cooks_distance[0] * k_vars, results_sas[:, 8], atol=6e-5)
    assert_allclose(infl.hat_matrix_diag, results_sas[:, 4], atol=6e-5)

    c_bar = infl.cooks_distance[0] * 3 * (1 - infl.hat_matrix_diag)
    assert_allclose(c_bar, results_sas[:, 9], atol=6e-5)


class InfluenceCompareExact(object):
    # Mixin to compare and test two Influence instances

    def test_basics(self):
        infl1 = self.infl1
        infl0 = self.infl0

        assert_allclose(infl0.hat_matrix_diag, infl1.hat_matrix_diag,
                        rtol=1e-12)

        assert_allclose(infl0.resid_studentized,
                        infl1.resid_studentized, rtol=1e-12, atol=1e-7)

        cd_rtol = getattr(self, 'cd_rtol', 1e-7)
        assert_allclose(infl0.cooks_distance[0], infl1.cooks_distance[0],
                        rtol=cd_rtol)
        assert_allclose(infl0.dfbetas, infl1.dfbetas, rtol=1e-9, atol=5e-9)
        assert_allclose(infl0.d_params, infl1.d_params, rtol=1e-9, atol=5e-9)
        assert_allclose(infl0.d_fittedvalues, infl1.d_fittedvalues, rtol=5e-9)
        assert_allclose(infl0.d_fittedvalues_scaled,
                        infl1.d_fittedvalues_scaled, rtol=5e-9)

    @pytest.mark.matplotlib
    def test_plots(self, close_figures):
        # SMOKE tests for plots
        infl1 = self.infl1
        infl0 = self.infl0

        fig = infl0.plot_influence(external=False)
        fig = infl1.plot_influence(external=False)

        fig = infl0.plot_index('resid', threshold=0.2, title='')
        fig = infl1.plot_index('resid', threshold=0.2, title='')

        fig = infl0.plot_index('dfbeta', idx=1, threshold=0.2, title='')
        fig = infl1.plot_index('dfbeta', idx=1, threshold=0.2, title='')

        fig = infl0.plot_index('cook', idx=1, threshold=0.2, title='')
        fig = infl1.plot_index('cook', idx=1, threshold=0.2, title='')

        fig = infl0.plot_index('hat', idx=1, threshold=0.2, title='')
        fig = infl1.plot_index('hat', idx=1, threshold=0.2, title='')


    def test_summary(self):
        infl1 = self.infl1
        infl0 = self.infl0

        df0 = infl0.summary_frame()
        df1 = infl1.summary_frame()
        assert_allclose(df0.values, df1.values, rtol=5e-5)
        pdt.assert_index_equal(df0.index, df1.index)


def _check_looo(self):
    infl = self.infl1
    # unwrap if needed
    results = getattr(infl.results, '_results', infl.results)

    res_looo = infl._res_looo
    mask_infl = infl.cooks_distance[0] > 2 * infl.cooks_distance[0].std()
    mask_low = ~mask_infl
    diff_params = results.params - res_looo['params']
    assert_allclose(infl.d_params[mask_low], diff_params[mask_low], atol=0.05)
    assert_allclose(infl.params_one[mask_low], res_looo['params'][mask_low], rtol=0.01)


class TestInfluenceLogitGLMMLE(InfluenceCompareExact):

    @classmethod
    def setup_class(cls):
        df = data_bin
        res = GLM(df['constrict'], df[['const', 'log_rate', 'log_volumne']],
              family=families.Binomial()).fit(attach_wls=True, atol=1e-10)

        cls.infl1 = res.get_influence()
        cls.infl0 = MLEInfluence(res)

    def test_looo(self):
        _check_looo(self)


class TestInfluenceBinomialGLMMLE(InfluenceCompareExact):
    # example based on Williams and R docs

    @classmethod
    def setup_class(cls):
        yi = np.array([0, 2, 14, 19, 30])
        ni = 40 * np.ones(len(yi))
        xi = np.arange(1, len(yi) + 1)
        exog = np.column_stack((np.ones(len(yi)), xi))
        endog = np.column_stack((yi, ni - yi))

        res = GLM(endog, exog, family=families.Binomial()).fit()

        cls.infl1 = res.get_influence()
        cls.infl0 = MLEInfluence(res)
        cls.cd_rtol = 5e-5

    def test_looo(self):
        _check_looo(self)

    def test_r(self):
        # values from R,
        # > xi <- 1:5
        # > yi <- c(0,2,14,19,30)    # number of mice responding to dose xi
        # > mi <- rep(40, 5)         # number of mice exposed
        # > glmI <- glm(cbind(yi, mi -yi) ~ xi, family = binomial)
        # > imI <- influence.measures(glmI)
        # > t(imI$infmat)

        # dfbeta/dfbetas and dffits don't make sense to me and are furthe away from
        # looo than mine
        # resid seem to be resid_deviance based and not resid_pearson
        # I didn't compare cov.r
        infl1 = self.infl1
        cooks_d = [0.25220202795934726, 0.26107981497746285, 1.28985614424132389,
                   0.08449722285516942, 0.36362110845918005]
        hat = [0.2594393406119333,  0.3696442663244837,  0.3535768402250521,
               0.389209198535791057,  0.6281303543027403]

        assert_allclose(infl1.hat_matrix_diag, hat, rtol=5e-6)
        assert_allclose(infl1.cooks_distance[0], cooks_d, rtol=1e-5)


class TestInfluenceGaussianGLMMLE(InfluenceCompareExact):

    @classmethod
    def setup_class(cls):
        endog, exog, labels = get_duncan_data()
        data = pd.DataFrame(np.column_stack((endog, exog)),
                        columns='y const var1 var2'.split(),
                        index=labels)

        res = GLM.from_formula('y ~ const + var1 + var2 - 1', data).fit()
        #res = GLM(endog, exog).fit()

        cls.infl1 = res.get_influence()
        cls.infl0 = MLEInfluence(res)

    def test_looo(self):
        _check_looo(self)


class TestInfluenceGaussianGLMOLS(InfluenceCompareExact):

    @classmethod
    def setup_class(cls):
        endog, exog, labels = get_duncan_data()
        data = pd.DataFrame(np.column_stack((endog, exog)),
                        columns='y const var1 var2'.split(),
                        index=labels)

        res0 = GLM.from_formula('y ~ const + var1 + var2 - 1', data).fit()
        res1 = OLS.from_formula('y ~ const + var1 + var2 - 1', data).fit()
        cls.infl1 = res1.get_influence()
        cls.infl0 = res0.get_influence()

    def test_basics(self):
        # needs to override attributes that are not equivalent,
        # i.e. not available or different definition like external vs internal
        infl1 = self.infl1
        infl0 = self.infl0

        assert_allclose(infl0.hat_matrix_diag, infl1.hat_matrix_diag,
                        rtol=1e-12)
        assert_allclose(infl0.resid_studentized,
                        infl1.resid_studentized, rtol=1e-12, atol=1e-7)
        assert_allclose(infl0.cooks_distance, infl1.cooks_distance, rtol=1e-7)
        assert_allclose(infl0.dfbetas, infl1.dfbetas, rtol=0.1) # changed
        # OLSInfluence only has looo dfbeta/d_params
        assert_allclose(infl0.d_params, infl1.dfbeta, rtol=1e-9, atol=1e-14)
        # d_fittedvalues is not available in OLSInfluence, i.e. only scaled dffits
        # assert_allclose(infl0.d_fittedvalues, infl1.d_fittedvalues, rtol=1e-9)
        assert_allclose(infl0.d_fittedvalues_scaled,
                        infl1.dffits_internal[0], rtol=1e-9)

        # specific to linear link
        assert_allclose(infl0.d_linpred,
                        infl0.d_fittedvalues, rtol=1e-12)
        assert_allclose(infl0.d_linpred_scaled,
                        infl0.d_fittedvalues_scaled, rtol=1e-12)

    def test_summary(self):
        infl1 = self.infl1
        infl0 = self.infl0

        df0 = infl0.summary_frame()
        df1 = infl1.summary_frame()
        # just some basic check on overlap except for dfbetas
        cols = ['cooks_d', 'standard_resid', 'hat_diag', 'dffits_internal']
        assert_allclose(df0[cols].values, df1[cols].values, rtol=1e-5)
        pdt.assert_index_equal(df0.index, df1.index)
