import os
import numpy as np
from statsmodels.sandbox.phreg import PHreg
from numpy.testing import assert_almost_equal

# TODO: Include some corner cases: data sets with empty strata, strata
#      with no events, entry times after censoring times, etc.

# All the R results
from . import survival_r_results

"""
Tests of phreg against R coxph.

Tests include entry times and stratification.

phreg_gentests.py generates the test data sets and puts them into the
results folder.

survival.R runs R on all the test data sets and constructs the
survival_r_results module.
"""

# Arguments passed to the phreg fit method.
args = {"method": "bfgs", "disp": 0}

def get_results(n, p, ext, ties):
    if ext is None:
        coef_name = "coef_%d_%d_%s" % (n, p, ties)
        se_name = "se_%d_%d_%s" % (n, p, ties)
        time_name = "time_%d_%d_%s" % (n, p, ties)
        hazard_name = "hazard_%d_%d_%s" % (n, p, ties)
    else:
        coef_name = "coef_%d_%d_%s_%s" % (n, p, ext, ties)
        se_name = "se_%d_%d_%s_%s" % (n, p, ext, ties)
        time_name = "time_%d_%d_%s_%s" % (n, p, ext, ties)
        hazard_name = "hazard_%d_%d_%s_%s" % (n, p, ext, ties)
    coef = getattr(survival_r_results, coef_name)
    se = getattr(survival_r_results, se_name)
    time = getattr(survival_r_results, time_name)
    hazard = getattr(survival_r_results, hazard_name)
    return coef, se, time, hazard

class TestPHreg(object):

    # Load a data file from the results directory
    def load_file(self, fname):
        cur_dir = os.path.dirname(os.path.abspath(__file__))
        data = np.genfromtxt(os.path.join(cur_dir, 'results', fname),
                             delimiter=" ")
        time = data[:,0]
        status = data[:,1]
        entry = data[:,2]
        exog = data[:,3:]

        return time, status, entry, exog


    # Run a single test against R output
    def do1(self, fname, ties, entry_f, strata_f):

        # Read the test data.
        time, status, entry, exog = self.load_file(fname)
        n = len(time)

        vs = fname.split("_")
        n = int(vs[2])
        p = int(vs[3].split(".")[0])
        ties1 = ties[0:3]

        # Needs to match the kronecker statement in survival.R
        strata = np.kron(range(5), np.ones(n/5))

        # No stratification or entry times
        mod = PHreg(time, exog, status, ties=ties)
        phrb = mod.fit(**args)
        coef_r, se_r, time_r, hazard_r = get_results(n, p, None, ties1)
        assert_almost_equal(phrb.params, coef_r, decimal=4)
        assert_almost_equal(phrb.bse, se_r, decimal=4)
        #time_h, cumhaz, surv = phrb.baseline_hazard[0]

        # Entry times but no stratification
        phrb = PHreg(time, exog, status, entry=entry,
                     ties=ties).fit(**args)
        coef, se, time_r, hazard_r = get_results(n, p, "et", ties1)
        assert_almost_equal(phrb.params, coef, decimal=4)
        assert_almost_equal(phrb.bse, se, decimal=4)

        # Stratification but no entry times
        phrb = PHreg(time, exog, status, strata=strata,
                      ties=ties).fit(**args)
        coef, se, time_r, hazard_r = get_results(n, p, "st", ties1)
        assert_almost_equal(phrb.params, coef, decimal=4)
        assert_almost_equal(phrb.bse, se, decimal=4)

        # Stratification and entry times
        phrb = PHreg(time, exog, status, entry=entry,
                     strata=strata, ties=ties).fit(**args)
        coef, se, time_r, hazard_r = get_results(n, p, "et_st", ties1)
        assert_almost_equal(phrb.params, coef, decimal=4)
        assert_almost_equal(phrb.bse, se, decimal=4)


    # Run all the tests
    def test_r(self):

        cur_dir = os.path.dirname(os.path.abspath(__file__))
        rdir = os.path.join(cur_dir, 'results')
        fnames = os.listdir(rdir)
        fnames = [x for x in fnames if x.startswith("survival")
                  and x.endswith(".csv")]

        for fname in fnames:
            for ties in "breslow","efron":
                for entry_f in False,True:
                    for strata_f in False,True:
                        yield self.do1, fname, ties, entry_f, \
                            strata_f

    def test_missing(self):

        np.random.seed(34234)
        time = 50 * np.random.uniform(size=200)
        status = np.random.randint(0, 2, 200).astype(np.float64)
        exog = np.random.normal(size=(200,4))

        time[0:5] = np.nan
        status[5:10] = np.nan
        exog[10:15,:] = np.nan

        md = PHreg(time, exog, status, missing='drop')
        assert(len(md.endog) == 185)
        assert(len(md.status) == 185)
        assert(all(md.exog.shape == np.r_[185,4]))

    def test_offset(self):

        np.random.seed(34234)
        time = 50 * np.random.uniform(size=200)
        status = np.random.randint(0, 2, 200).astype(np.float64)
        exog = np.random.normal(size=(200,4))

        mod1 = PHreg(time, exog, status)
        rslt1 = mod1.fit()
        offset = exog[:,0] * rslt1.params[0]
        exog = exog[:, 1:]

        mod2 = PHreg(time, exog, status, offset=offset)
        rslt2 = mod2.fit()

        assert_almost_equal(rslt2.params, rslt1.params[1:])

    def test_post_estimation(self):
        # All regression tests
        np.random.seed(34234)
        time = 50 * np.random.uniform(size=200)
        status = np.random.randint(0, 2, 200).astype(np.float64)
        exog = np.random.normal(size=(200,4))

        mod = PHreg(time, exog, status)
        rslt = mod.fit()
        mart_resid = rslt.martingale_residuals
        assert_almost_equal(np.abs(mart_resid).sum(), 120.72475743348433)

        w_avg = rslt.weighted_covariate_averages
        assert_almost_equal(np.abs(w_avg[0]).sum(0),
               np.r_[7.31008415, 9.77608674,10.89515885, 13.1106801])

        bc_haz = rslt.baseline_cumulative_hazard
        v = [np.mean(np.abs(x)) for x in bc_haz[0]]
        w = np.r_[23.482841556421608, 0.44149255358417017,
                  0.68660114081275281]
        assert_almost_equal(v, w)

        score_resid = rslt.score_residuals
        v = np.r_[ 0.50924792, 0.4533952, 0.4876718, 0.5441128]
        w = np.abs(score_resid).mean(0)
        assert_almost_equal(v, w)

        groups = np.random.randint(0, 3, 200)
        mod = PHreg(time, exog, status)
        rslt = mod.fit(groups=groups)
        robust_cov = rslt.cov_params()
        v = [0.00513432, 0.01278423, 0.00810427, 0.00293147]
        w = np.abs(robust_cov).mean(0)
        assert_almost_equal(v, w)

        s_resid = rslt.schoenfeld_residuals
        ii = np.flatnonzero(np.isfinite(s_resid).all(1))
        s_resid = s_resid[ii, :]
        v = np.r_[0.85154336, 0.72993748, 0.73758071, 0.78599333]
        assert_almost_equal(np.abs(s_resid).mean(0), v)

    def test_summary(self):
        # smoke test
        np.random.seed(34234)
        time = 50 * np.random.uniform(size=200)
        status = np.random.randint(0, 2, 200).astype(np.float64)
        exog = np.random.normal(size=(200,4))

        mod = PHreg(time, exog, status)
        rslt = mod.fit()
        rslt.summary()

    def test_predict(self):
        # All smoke tests. We should be able to convert the lhr and hr
        # tests into real tests against R.  There are many options to
        # this function that may interact in complicated ways.  Only a
        # few key combinations are tested here.
        np.random.seed(34234)
        endog = 50 * np.random.uniform(size=200)
        status = np.random.randint(0, 2, 200).astype(np.float64)
        exog = np.random.normal(size=(200,4))

        mod = PHreg(endog, exog, status)
        rslt = mod.fit()
        rslt.predict()
        for pred_type in 'lhr', 'hr', 'cumhaz', 'surv':
            rslt.predict(pred_type=pred_type)
            rslt.predict(endog=endog[0:10], pred_type=pred_type)
            rslt.predict(endog=endog[0:10], exog=exog[0:10,:],
                         pred_type=pred_type)

    def test_get_distribution(self):
        # Smoke test
        np.random.seed(34234)
        exog = np.random.normal(size=(200, 2))
        lin_pred = exog.sum(1)
        elin_pred = np.exp(-lin_pred)
        time = -elin_pred * np.log(np.random.uniform(size=200))

        mod = PHreg(time, exog)
        rslt = mod.fit()

        dist = rslt.get_distribution()

        fitted_means = dist.mean()
        true_means = elin_pred
        fitted_var = dist.var()
        fitted_sd = dist.std()
        sample = dist.rvs()

    def test_fit_regularized(self):
        # Smoke test
        np.random.seed(34234)
        n = 100
        p = 4
        exog = np.random.normal(size=(n, p))
        params = np.zeros(p, dtype=np.float64)
        params[p/2:] = 1
        lin_pred = np.dot(exog, params)
        elin_pred = np.exp(-lin_pred)
        time = -elin_pred * np.log(np.random.uniform(size=n))

        mod = PHreg(time, exog)
        rslt = mod.fit_regularized(alpha=20)
        smry = rslt.summary()


if  __name__=="__main__":

    import nose

    nose.runmodule(argv=[__file__,'-vvs','-x','--pdb', '--pdb-failure'],
                   exit=False)
