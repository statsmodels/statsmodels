import os.path

import numpy as np
from numpy.testing import (
    assert_allclose,
    assert_array_almost_equal,
    assert_raises,
)
import pandas as pd
import pytest

from statsmodels.formula._manager import FormulaManager
from statsmodels.multivariate.multivariate_ols import (
    MultivariateLS,
    _MultivariateOLS,
)
from statsmodels.regression.linear_model import OLS

dir_path = os.path.dirname(os.path.abspath(__file__))
csv_path = os.path.join(dir_path, 'results', 'mvreg.csv')
data_mvreg = pd.read_csv(csv_path)

data = pd.DataFrame([['Morphine', 'N', .04, .20, .10, .08],
                     ['Morphine', 'N', .02, .06, .02, .02],
                     ['Morphine', 'N', .07, 1.40, .48, .24],
                     ['Morphine', 'N', .17, .57, .35, .24],
                     ['Morphine', 'Y', .10, .09, .13, .14],
                     ['placebo', 'Y', .07, .07, .06, .07],
                     ['placebo', 'Y', .05, .07, .06, .07],
                     ['placebo', 'N', .03, .62, .31, .22],
                     ['placebo', 'N', .03, 1.05, .73, .60],
                     ['placebo', 'N', .07, .83, 1.07, .80],
                     ['Trimethaphan', 'N', .09, 3.13, 2.06, 1.23],
                     ['Trimethaphan', 'Y', .10, .09, .09, .08],
                     ['Trimethaphan', 'Y', .08, .09, .09, .10],
                     ['Trimethaphan', 'Y', .13, .10, .12, .12],
                     ['Trimethaphan', 'Y', .06, .05, .05, .05]],
                    columns=['Drug', 'Depleted',
                             'Histamine0', 'Histamine1',
                             'Histamine3', 'Histamine5'])

for i in range(2, 6):
    data.iloc[:, i] = np.log(data.iloc[:, i])

models = [_MultivariateOLS, MultivariateLS]


def compare_r_output_dogs_data(method, model):
    ''' Testing within-subject effect interact with 2 between-subject effect
    Compares with R car library Anova(, type=3) output

    Note: The test statistis Phillai, Wilks, Hotelling-Lawley
          and Roy are the same as R output but the approximate F and degree
          of freedoms can be different. This is due to the fact that this
          implementation is based on SAS formula [1]

    .. [*] https://support.sas.com/documentation/cdl/en/statug/63033/HTML/default/
           viewer.htm#statug_introreg_sect012.htm
    '''

    # Repeated measures with orthogonal polynomial contrasts coding
    mod = model.from_formula(
        'Histamine0 + Histamine1 + Histamine3 + Histamine5 ~ Drug * Depleted',
        data)
    r = mod.fit(method=method)
    r = r.mv_test()
    a = [[2.68607660e-02, 4, 6, 5.43435304e+01, 7.59585610e-05],
         [9.73139234e-01, 4, 6, 5.43435304e+01, 7.59585610e-05],
         [3.62290202e+01, 4, 6, 5.43435304e+01, 7.59585610e-05],
         [3.62290202e+01, 4, 6, 5.43435304e+01, 7.59585610e-05]]
    assert_array_almost_equal(r['Intercept']['stat'].values, a, decimal=6)
    a = [[8.39646619e-02, 8, 1.20000000e+01, 3.67658068e+00, 2.12614444e-02],
         [1.18605382e+00, 8, 1.40000000e+01, 2.55003861e+00, 6.01270701e-02],
         [7.69391362e+00, 8, 6.63157895e+00, 5.50814270e+00, 2.07392260e-02],
         [7.25036952e+00, 4, 7.00000000e+00, 1.26881467e+01, 2.52669877e-03]]
    assert_array_almost_equal(r['Drug']['stat'].values, a, decimal=6)
    a = [[0.32048892, 4., 6., 3.18034906, 0.10002373],
         [0.67951108, 4., 6., 3.18034906, 0.10002373],
         [2.12023271, 4., 6., 3.18034906, 0.10002373],
         [2.12023271, 4., 6., 3.18034906, 0.10002373]]
    assert_array_almost_equal(r['Depleted']['stat'].values, a, decimal=6)
    a = [[0.15234366, 8., 12.,        2.34307678, 0.08894239],
         [1.13013353, 8., 14.,        2.27360606, 0.08553213],
         [3.70989596, 8., 6.63157895, 2.65594824, 0.11370285],
         [3.1145597,  4., 7.,         5.45047947, 0.02582767]]
    assert_array_almost_equal(r['Drug:Depleted']['stat'].values, a, decimal=6)


@pytest.mark.parametrize("model", models)
def test_glm_dogs_example(model):
    compare_r_output_dogs_data(method='svd', model=model)
    compare_r_output_dogs_data(method='pinv', model=model)


@pytest.mark.parametrize("model", models)
def test_specify_L_M_by_string(model):
    mod = model.from_formula(
        'Histamine0 + Histamine1 + Histamine3 + Histamine5 ~ Drug * Depleted',
        data)
    r = mod.fit()
    r1 = r.mv_test(hypotheses=[['Intercept', ['Intercept'], None]])
    a = [[2.68607660e-02, 4, 6, 5.43435304e+01, 7.59585610e-05],
         [9.73139234e-01, 4, 6, 5.43435304e+01, 7.59585610e-05],
         [3.62290202e+01, 4, 6, 5.43435304e+01, 7.59585610e-05],
         [3.62290202e+01, 4, 6, 5.43435304e+01, 7.59585610e-05]]
    assert_array_almost_equal(r1['Intercept']['stat'].values, a, decimal=6)
    L = ['Intercept', 'Drug[T.Trimethaphan]', 'Drug[T.placebo]']
    M = ['Histamine1', 'Histamine3', 'Histamine5']
    r1 = r.mv_test(hypotheses=[['a', L, M]])
    a = [[1, 0, 0, 0, 0, 0],
         [0, 1, 0, 0, 0, 0],
         [0, 0, 1, 0, 0, 0]]
    assert_array_almost_equal(r1['a']['contrast_L'], a, decimal=10)
    a = [[0, 1, 0, 0],
         [0, 0, 1, 0],
         [0, 0, 0, 1]]
    assert_array_almost_equal(r1['a']['transform_M'].T, a, decimal=10)


@pytest.mark.parametrize("model", models)
def test_independent_variable_singular(model):
    data1 = data.copy()
    data1['dup'] = data1['Drug']
    mod = model.from_formula(
        'Histamine0 + Histamine1 + Histamine3 + Histamine5 ~ Drug * dup',
        data1)
    assert_raises(ValueError, mod.fit)
    mod = model.from_formula(
        'Histamine0 + Histamine1 + Histamine3 + Histamine5 ~ Drug * dup',
        data1)
    assert_raises(ValueError,  mod.fit)


@pytest.mark.parametrize("model", models)
def test_from_formula_vs_no_formula(model):
    mod = model.from_formula(
        'Histamine0 + Histamine1 + Histamine3 + Histamine5 ~ Drug * Depleted',
        data)
    r = mod.fit(method='svd')
    r0 = r.mv_test()
    mgr = FormulaManager()
    endog, exog = mgr.get_matrices(
        "Histamine0 + Histamine1 + Histamine3 + Histamine5 ~ Drug * Depleted", data
    )
    L = np.array([[1, 0, 0, 0, 0, 0]])
    # DataFrame input
    r = model(endog, exog).fit(method='svd')
    r1 = r.mv_test(hypotheses=[['Intercept', L, None]])
    assert_array_almost_equal(r1['Intercept']['stat'].values,
                              r0['Intercept']['stat'].values, decimal=6)
    # Numpy array input
    r = model(endog.values, exog.values).fit(method='svd')
    r1 = r.mv_test(hypotheses=[['Intercept', L, None]])
    assert_array_almost_equal(r1['Intercept']['stat'].values,
                              r0['Intercept']['stat'].values, decimal=6)
    L = np.array([[0, 1, 0, 0, 0, 0],
                  [0, 0, 1, 0, 0, 0],
                  ])
    r1 = r.mv_test(hypotheses=[['Drug', L, None]])
    # DataFrame input
    r = model(endog, exog).fit(method='svd')
    r1 = r.mv_test(hypotheses=[['Drug', L, None]])
    assert_array_almost_equal(r1['Drug']['stat'].values,
                              r0['Drug']['stat'].values, decimal=6)
    # Numpy array input
    r = model(endog.values, exog.values).fit(method='svd')
    r1 = r.mv_test(hypotheses=[['Drug', L, None]])
    assert_array_almost_equal(r1['Drug']['stat'].values,
                              r0['Drug']['stat'].values, decimal=6)


@pytest.mark.parametrize("model", models)
def test_L_M_matrices_1D_array(model):
    mod = model.from_formula(
        'Histamine0 + Histamine1 + Histamine3 + Histamine5 ~ Drug * Depleted',
        data)
    r = mod.fit(method='svd')
    L = np.array([1, 0, 0, 0, 0, 0])
    assert_raises(ValueError, r.mv_test, hypotheses=[['Drug', L, None]])
    L = np.array([[1, 0, 0, 0, 0, 0]])
    M = np.array([1, 0, 0, 0, 0, 0])
    assert_raises(ValueError, r.mv_test, hypotheses=[['Drug', L, M]])


@pytest.mark.parametrize("model", models)
def test_exog_1D_array(model):
    mod = model.from_formula(
        'Histamine0 + Histamine1 + Histamine3 + Histamine5 ~ 0 + Depleted',
        data)
    r = mod.fit(method='svd')
    r0 = r.mv_test()
    a = [[0.0019, 8.0000, 20.0000, 55.0013, 0.0000],
         [1.8112, 8.0000, 22.0000, 26.3796, 0.0000],
         [97.8858, 8.0000, 12.1818, 117.1133, 0.0000],
         [93.2742, 4.0000, 11.0000, 256.5041, 0.0000]]
    assert_array_almost_equal(r0['Depleted']['stat'].values, a, decimal=4)


def test_endog_1D_array():
    assert_raises(
        ValueError, _MultivariateOLS.from_formula, 'Histamine0 ~ 0 + Depleted', data
    )


@pytest.mark.parametrize("model", models)
def test_affine_hypothesis(model):
    # Testing affine hypothesis, compared with R car linearHypothesis
    # Note: The test statistis Phillai, Wilks, Hotelling-Lawley
    # and Roy are the same as R output but the approximate F and degree
    # of freedoms can be different. This is due to the fact that this
    # implementation is based on SAS formula [1]
    mod = model.from_formula(
        'Histamine0 + Histamine1 + Histamine3 + Histamine5 ~ Drug * Depleted',
        data)
    r = mod.fit(method='svd')
    L = np.array([[0, 1.2, 1.1, 1.3, 1.5, 1.4],
                  [0, 3.2, 2.1, 3.3, 5.5, 4.4]])
    M = None
    C = np.array([[1, 2, 3, 4],
                  [5, 6, 7, 8]])
    r0 = r.mv_test(hypotheses=[('test1', L, M, C)])
    a = [[0.0269, 8.0000, 12.0000, 7.6441, 0.0010],
         [1.4277, 8.0000, 14.0000, 4.3657, 0.0080],
         [19.2678, 8.0000, 6.6316, 13.7940, 0.0016],
         [18.3470, 4.0000, 7.0000, 32.1072, 0.0001]]
    assert_array_almost_equal(r0['test1']['stat'].values, a, decimal=4)
    r0.summary(show_contrast_L=True, show_transform_M=True,
               show_constant_C=True)


class CheckMVConsistent:

    def test_basic(self):
        res = self.res
        tt = res.t_test(np.eye(res.params.size))
        assert_allclose(tt.effect, res.params.to_numpy().ravel(order="F"),
                        rtol=1e-13)
        assert_allclose(tt.sd, res.bse.to_numpy().ravel(order="F"),
                        rtol=1e-13)
        assert_allclose(tt.tvalue, res.tvalues.to_numpy().ravel(order="F"),
                        rtol=1e-13)
        assert_allclose(tt.pvalue, res.pvalues.to_numpy().ravel(order="F"),
                        rtol=1e-13)
        assert_allclose(tt.conf_int(), res.conf_int().to_numpy(),
                        rtol=1e-13)

        exog_names = res.model.exog_names
        endog_names = res.model.endog_names

        # hypothesis test for one parameter:
        for xn in exog_names:
            cnx = []
            for yn in endog_names:
                cn = f"y{yn}_{xn}"
                tt = res.t_test(cn)
                mvt = res.mv_test(hypotheses=[("one", [xn], [yn])])
                assert_allclose(tt.pvalue, mvt.summary_frame["Pr > F"].iloc[0],
                                rtol=1e-10)
                cnx.append(cn)

            # wald test effect of an exog is zero across all endog
            # test methods are only asymptotically equivalent
            tt = res.wald_test(cnx, scalar=True)
            mvt = res.mv_test(hypotheses=[(xn, [xn], endog_names)])
            assert_allclose(tt.pvalue, mvt.summary_frame["Pr > F"].iloc[0],
                            rtol=0.1, atol=1e-20)

    def test_ols(self):
        res1 = self.res._results  # use numpy results, not pandas
        endog = res1.model.endog
        exog = res1.model.exog
        k_endog = endog.shape[1]
        k_exog = exog.shape[1]

        for k in range(k_endog):
            res_ols = OLS(endog[:, k], exog).fit()
            assert_allclose(res1.params[:, k], res_ols.params, rtol=1e-12)
            assert_allclose(res1.bse[:, k], res_ols.bse, rtol=1e-13)
            assert_allclose(res1.tvalues[:, k], res_ols.tvalues, rtol=1e-12)
            assert_allclose(res1.pvalues[:, k], res_ols.pvalues, rtol=1e-12,
                            atol=1e-15)
            # todo: why does conf_int have endog at axis=0
            assert_allclose(res1.conf_int()[k], res_ols.conf_int(),
                            rtol=1e-10)

            idx0 = k * k_exog
            idx1 = (k + 1) * k_exog
            assert_allclose(res1.cov_params()[idx0:idx1, idx0:idx1],
                            res_ols.cov_params(), rtol=1e-12)

            assert_allclose(res1.resid[:, k], res_ols.resid, rtol=1e-10)
            assert_allclose(res1.fittedvalues[:, k], res_ols.fittedvalues,
                            rtol=1e-10)


class TestMultivariateLS(CheckMVConsistent):

    @classmethod
    def setup_class(cls):
        formula2 = ("locus_of_control + self_concept + motivation ~ "
                    "read + write + science + prog")
        mod = MultivariateLS.from_formula(formula2, data=data_mvreg)
        cls.res = mod.fit()
        # ttn = cls.res.mv_test()
