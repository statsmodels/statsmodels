"""
Test functions for sm.rlm
"""

import numpy as np
from numpy.testing import assert_almost_equal, assert_allclose
from scipy import stats
import statsmodels.api as sm
from statsmodels.robust.robust_linear_model import RLM
from nose import SkipTest

DECIMAL_4 = 4
DECIMAL_3 = 3
DECIMAL_2 = 2
DECIMAL_1 = 1

class CheckRlmResultsMixin(object):
    '''
    res2 contains  results from Rmodelwrap or were obtained from a statistical
    packages such as R, Stata, or SAS and written to results.results_rlm

    Covariance matrices were obtained from SAS and are imported from
    results.results_rlm
    '''
    def test_params(self):
        assert_almost_equal(self.res1.params, self.res2.params, DECIMAL_4)

    decimal_standarderrors = DECIMAL_4
    def test_standarderrors(self):
        assert_almost_equal(self.res1.bse, self.res2.bse,
                self.decimal_standarderrors)

#TODO: get other results from SAS, though if it works for one...
    def test_confidenceintervals(self):
        if not hasattr(self.res2, 'conf_int'):
            raise SkipTest("Results from R")
        else:
            assert_almost_equal(self.res1.conf_int(), self.res2.conf_int(),
            DECIMAL_4)

    decimal_scale = DECIMAL_4
    def test_scale(self):
        assert_almost_equal(self.res1.scale, self.res2.scale,
                self.decimal_scale)

    def test_weights(self):
        assert_almost_equal(self.res1.weights, self.res2.weights, DECIMAL_4)

    def test_residuals(self):
        assert_almost_equal(self.res1.resid, self.res2.resid, DECIMAL_4)

    def test_degrees(self):
        assert_almost_equal(self.res1.model.df_model, self.res2.df_model,
                DECIMAL_4)
        assert_almost_equal(self.res1.model.df_resid, self.res2.df_resid,
                DECIMAL_4)

    def test_bcov_unscaled(self):
        if not hasattr(self.res2, 'bcov_unscaled'):
            raise SkipTest("No unscaled cov matrix from SAS")
        else:
            assert_almost_equal(self.res1.bcov_unscaled,
                    self.res2.bcov_unscaled, DECIMAL_4)

    decimal_bcov_scaled = DECIMAL_4
    def test_bcov_scaled(self):
        assert_almost_equal(self.res1.bcov_scaled, self.res2.h1,
                self.decimal_bcov_scaled)
        assert_almost_equal(self.res1.h2, self.res2.h2,
                self.decimal_bcov_scaled)
        assert_almost_equal(self.res1.h3, self.res2.h3,
                self.decimal_bcov_scaled)


    def test_tvalues(self):
        if not hasattr(self.res2, 'tvalues'):
            raise SkipTest("No tvalues in benchmark")
        else:
            assert_allclose(self.res1.tvalues, self.res2.tvalues, rtol=0.003)

    def test_tpvalues(self):
        # test comparing tvalues and pvalues with normal implementation
        # make sure they use normal distribution (inherited in results class)
        params = self.res1.params
        tvalues = params / self.res1.bse
        pvalues = stats.norm.sf(np.abs(tvalues)) * 2
        half_width = stats.norm.isf(0.025) * self.res1.bse
        conf_int = np.column_stack((params - half_width, params + half_width))

        assert_almost_equal(self.res1.tvalues, tvalues)
        assert_almost_equal(self.res1.pvalues, pvalues)
        assert_almost_equal(self.res1.conf_int(), conf_int)


class TestRlm(CheckRlmResultsMixin):
    from statsmodels.datasets.stackloss import load
    data = load()   # class attributes for subclasses
    data.exog = sm.add_constant(data.exog, prepend=False)
    def __init__(self):
        # Test precisions
        self.decimal_standarderrors = DECIMAL_1
        self.decimal_scale = DECIMAL_3

        results = RLM(self.data.endog, self.data.exog,\
                    M=sm.robust.norms.HuberT()).fit()   # default M
        h2 = RLM(self.data.endog, self.data.exog,\
                    M=sm.robust.norms.HuberT()).fit(cov="H2").bcov_scaled
        h3 = RLM(self.data.endog, self.data.exog,\
                    M=sm.robust.norms.HuberT()).fit(cov="H3").bcov_scaled
        self.res1 = results
        self.res1.h2 = h2
        self.res1.h3 = h3


    def setup(self):
#        r.library('MASS')
#        self.res2 = RModel(self.data.endog, self.data.exog,
#                        r.rlm, psi="psi.huber")
        from .results.results_rlm import Huber
        self.res2 = Huber()

    def test_summary(self):
        # smoke test that summary at least returns something
        self.res1.summary()

class TestHampel(TestRlm):
    def __init__(self):
        # Test precisions
        self.decimal_standarderrors = DECIMAL_2
        self.decimal_scale = DECIMAL_3
        self.decimal_bcov_scaled = DECIMAL_3

        results = RLM(self.data.endog, self.data.exog,
                    M=sm.robust.norms.Hampel()).fit()
        h2 = RLM(self.data.endog, self.data.exog,\
                    M=sm.robust.norms.Hampel()).fit(cov="H2").bcov_scaled
        h3 = RLM(self.data.endog, self.data.exog,\
                    M=sm.robust.norms.Hampel()).fit(cov="H3").bcov_scaled
        self.res1 = results
        self.res1.h2 = h2
        self.res1.h3 = h3

    def setup(self):
#        self.res2 = RModel(self.data.endog[:,None], self.data.exog,
#        r.rlm, psi="psi.hampel") #, init="lts")
        from .results.results_rlm import Hampel
        self.res2 = Hampel()



class TestRlmBisquare(TestRlm):
    def __init__(self):
        # Test precisions
        self.decimal_standarderrors = DECIMAL_1

        results = RLM(self.data.endog, self.data.exog,
                    M=sm.robust.norms.TukeyBiweight()).fit()
        h2 = RLM(self.data.endog, self.data.exog,\
                    M=sm.robust.norms.TukeyBiweight()).fit(cov=\
                    "H2").bcov_scaled
        h3 = RLM(self.data.endog, self.data.exog,\
                    M=sm.robust.norms.TukeyBiweight()).fit(cov=\
                    "H3").bcov_scaled
        self.res1 = results
        self.res1.h2 = h2
        self.res1.h3 = h3

    def setup(self):
#        self.res2 = RModel(self.data.endog, self.data.exog,
#                        r.rlm, psi="psi.bisquare")
        from .results.results_rlm import BiSquare
        self.res2 = BiSquare()


class TestRlmAndrews(TestRlm):
    def __init__(self):
        results = RLM(self.data.endog, self.data.exog,
                    M=sm.robust.norms.AndrewWave()).fit()
        h2 = RLM(self.data.endog, self.data.exog,
                    M=sm.robust.norms.AndrewWave()).fit(cov=\
                    "H2").bcov_scaled
        h3 = RLM(self.data.endog, self.data.exog,
                    M=sm.robust.norms.AndrewWave()).fit(cov=\
                    "H3").bcov_scaled
        self.res1 = results
        self.res1.h2 = h2
        self.res1.h3 = h3

    def setup(self):
        from .results.results_rlm import Andrews
        self.res2 = Andrews()

### tests with Huber scaling

class TestRlmHuber(CheckRlmResultsMixin):
    from statsmodels.datasets.stackloss import load
    data = load()
    data.exog = sm.add_constant(data.exog, prepend=False)
    def __init__(self):
        results = RLM(self.data.endog, self.data.exog,\
                    M=sm.robust.norms.HuberT()).fit(scale_est=\
                    sm.robust.scale.HuberScale())
        h2 = RLM(self.data.endog, self.data.exog,\
                    M=sm.robust.norms.HuberT()).fit(cov="H2",
                    scale_est=sm.robust.scale.HuberScale()).bcov_scaled
        h3 = RLM(self.data.endog, self.data.exog,\
                    M=sm.robust.norms.HuberT()).fit(cov="H3",
                    scale_est=sm.robust.scale.HuberScale()).bcov_scaled
        self.res1 = results
        self.res1.h2 = h2
        self.res1.h3 = h3

    def setup(self):
        from .results.results_rlm import HuberHuber
        self.res2 = HuberHuber()

class TestHampelHuber(TestRlm):
    def __init__(self):
        results = RLM(self.data.endog, self.data.exog,
                    M=sm.robust.norms.Hampel()).fit(scale_est=\
                    sm.robust.scale.HuberScale())
        h2 = RLM(self.data.endog, self.data.exog,\
                    M=sm.robust.norms.Hampel()).fit(cov="H2",
                    scale_est=\
                    sm.robust.scale.HuberScale()).bcov_scaled
        h3 = RLM(self.data.endog, self.data.exog,\
                    M=sm.robust.norms.Hampel()).fit(cov="H3",
                    scale_est=\
                    sm.robust.scale.HuberScale()).bcov_scaled
        self.res1 = results
        self.res1.h2 = h2
        self.res1.h3 = h3

    def setup(self):
        from .results.results_rlm import HampelHuber
        self.res2 = HampelHuber()

class TestRlmBisquareHuber(TestRlm):
    def __init__(self):
        results = RLM(self.data.endog, self.data.exog,
                    M=sm.robust.norms.TukeyBiweight()).fit(\
                    scale_est=\
                    sm.robust.scale.HuberScale())
        h2 = RLM(self.data.endog, self.data.exog,\
                    M=sm.robust.norms.TukeyBiweight()).fit(cov=\
                    "H2", scale_est=\
                    sm.robust.scale.HuberScale()).bcov_scaled
        h3 = RLM(self.data.endog, self.data.exog,\
                    M=sm.robust.norms.TukeyBiweight()).fit(cov=\
                    "H3", scale_est=\
                    sm.robust.scale.HuberScale()).bcov_scaled
        self.res1 = results
        self.res1.h2 = h2
        self.res1.h3 = h3

    def setup(self):
        from .results.results_rlm import BisquareHuber
        self.res2 = BisquareHuber()

class TestRlmAndrewsHuber(TestRlm):
    def __init__(self):
        results = RLM(self.data.endog, self.data.exog,
                    M=sm.robust.norms.AndrewWave()).fit(scale_est=\
                    sm.robust.scale.HuberScale())
        h2 = RLM(self.data.endog, self.data.exog,
                    M=sm.robust.norms.AndrewWave()).fit(cov=\
                    "H2", scale_est=\
                    sm.robust.scale.HuberScale()).bcov_scaled
        h3 = RLM(self.data.endog, self.data.exog,
                    M=sm.robust.norms.AndrewWave()).fit(cov=\
                    "H3", scale_est=\
                    sm.robust.scale.HuberScale()).bcov_scaled
        self.res1 = results
        self.res1.h2 = h2
        self.res1.h3 = h3

    def setup(self):
        from .results.results_rlm import AndrewsHuber
        self.res2 = AndrewsHuber()

class TestRlmSresid(CheckRlmResultsMixin):
    #Check GH:187
    from statsmodels.datasets.stackloss import load
    data = load()   # class attributes for subclasses
    data.exog = sm.add_constant(data.exog, prepend=False)
    def __init__(self):
        # Test precisions
        self.decimal_standarderrors = DECIMAL_1
        self.decimal_scale = DECIMAL_3

        results = RLM(self.data.endog, self.data.exog,\
                    M=sm.robust.norms.HuberT()).fit(conv='sresid') # default M
        h2 = RLM(self.data.endog, self.data.exog,\
                    M=sm.robust.norms.HuberT()).fit(cov="H2").bcov_scaled
        h3 = RLM(self.data.endog, self.data.exog,\
                    M=sm.robust.norms.HuberT()).fit(cov="H3").bcov_scaled
        self.res1 = results
        self.res1.h2 = h2
        self.res1.h3 = h3


    def setup(self):
#        r.library('MASS')
#        self.res2 = RModel(self.data.endog, self.data.exog,
#                        r.rlm, psi="psi.huber")
        from .results.results_rlm import Huber
        self.res2 = Huber()


def test_missing():
    # see 2083
    import statsmodels.formula.api as smf

    d = {'Foo': [1, 2, 10, 149], 'Bar': [1, 2, 3, np.nan]}
    mod = smf.rlm('Foo ~ Bar', data=d)
