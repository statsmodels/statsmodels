import itertools
from pathlib import Path

import numpy as np
from numpy.testing import assert_, assert_allclose, assert_equal
import pandas as pd
import pytest

from statsmodels.duration.hazard_regression import PHReg
from statsmodels.formula._manager import FormulaManager
from statsmodels.iolib.summary2 import Summary

# All the R results
from .results import (
    results_phreg_residuals as residual_results,
    survival_enet_r_results,
    survival_r_results,
)

# TODO: Include some corner cases: data sets with empty strata, strata
#      with no events, entry times after censoring times, etc.


"""
Tests of PHReg against R coxph.

Tests include entry times and stratification.

phreg_gentests.py generates the test data sets and puts them into the
results folder.

survival.R runs R on all the test data sets and constructs the
survival_r_results module.
"""

# Arguments passed to the PHReg fit method.
args = {"method": "bfgs", "disp": 0}


def get_results(n, p, ext, ties):
    if ext is None:
        coef_name = f"coef_{n:d}_{p:d}_{ties}"
        se_name = f"se_{n:d}_{p:d}_{ties}"
        time_name = f"time_{n:d}_{p:d}_{ties}"
        hazard_name = f"hazard_{n:d}_{p:d}_{ties}"
    else:
        coef_name = f"coef_{n:d}_{p:d}_{ext}_{ties}"
        se_name = f"se_{n:d}_{p:d}_{ext}_{ties}"
        time_name = f"time_{n:d}_{p:d}_{ext}_{ties}"
        hazard_name = f"hazard_{n:d}_{p:d}_{ext}_{ties}"
    coef = getattr(survival_r_results, coef_name)
    se = getattr(survival_r_results, se_name)
    time = getattr(survival_r_results, time_name)
    hazard = getattr(survival_r_results, hazard_name)
    return coef, se, time, hazard


class TestPHReg:

    # Load a data file from the results directory
    @staticmethod
    def load_file(fname):
        cur_dir = Path(__file__).resolve().parent
        df = pd.read_csv(Path(cur_dir).joinpath("results", fname), delimiter=" ", header=None)
        data = df.values
        time = data[:, 0]
        status = data[:, 1]
        entry = data[:, 2]
        exog = data[:, 3:]

        return time, status, entry, exog

    # Run a single test against R output
    @staticmethod
    def do1(fname, ties, entry_f, strata_f):

        # Read the test data.
        time, status, entry, exog = TestPHReg.load_file(fname)
        n = len(time)

        vs = fname.split("_")
        n = int(vs[2])
        p = int(vs[3].split(".")[0])
        ties1 = ties[0:3]

        # Needs to match the kronecker statement in survival.R
        strata = np.kron(range(5), np.ones(n // 5))

        # No stratification or entry times
        mod = PHReg(time, exog, status, ties=ties)
        phrb = mod.fit(**args)
        coef_r, se_r, time_r, hazard_r = get_results(n, p, None, ties1)
        assert_allclose(phrb.params, coef_r, rtol=1e-3)
        assert_allclose(phrb.bse, se_r, rtol=1e-4)
        time_h, cumhaz, surv = phrb.baseline_cumulative_hazard[0]

        # Entry times but no stratification
        phrb = PHReg(time, exog, status, entry=entry, ties=ties).fit(**args)
        coef, se, time_r, hazard_r = get_results(n, p, "et", ties1)
        assert_allclose(phrb.params, coef, rtol=1e-3)
        assert_allclose(phrb.bse, se, rtol=1e-3)

        # Stratification but no entry times
        phrb = PHReg(time, exog, status, strata=strata, ties=ties).fit(**args)
        coef, se, time_r, hazard_r = get_results(n, p, "st", ties1)
        assert_allclose(phrb.params, coef, rtol=1e-4)
        assert_allclose(phrb.bse, se, rtol=1e-4)

        # Stratification and entry times
        phrb = PHReg(time, exog, status, entry=entry, strata=strata, ties=ties).fit(
            **args
        )
        coef, se, time_r, hazard_r = get_results(n, p, "et_st", ties1)
        assert_allclose(phrb.params, coef, rtol=1e-3)
        assert_allclose(phrb.bse, se, rtol=1e-4)

        # smoke test
        time_h, cumhaz, surv = phrb.baseline_cumulative_hazard[0]

    def test_missing(self):

        rs = np.random.RandomState(34234)
        time = 50 * rs.uniform(size=200)
        status = rs.randint(0, 2, 200).astype(np.float64)
        exog = rs.normal(size=(200, 4))

        time[0:5] = np.nan
        status[5:10] = np.nan
        exog[10:15, :] = np.nan

        md = PHReg(time, exog, status, missing="drop")
        assert_allclose(len(md.endog), 185)
        assert_allclose(len(md.status), 185)
        assert_allclose(md.exog.shape, np.r_[185, 4])

    def test_formula(self):

        rs = np.random.RandomState(34234)
        time = 50 * rs.uniform(size=200)
        status = rs.randint(0, 2, 200).astype(np.float64)
        exog = rs.normal(size=(200, 4))
        entry = np.zeros_like(time)
        entry[0:10] = time[0:10] / 2

        df = pd.DataFrame(
            {
                "time": time,
                "status": status,
                "exog1": exog[:, 0],
                "exog2": exog[:, 1],
                "exog3": exog[:, 2],
                "exog4": exog[:, 3],
                "entry": entry,
            }
        )

        mod1 = PHReg(time, exog, status, entry=entry)
        rslt1 = mod1.fit()

        # works with "0 +" on RHS but issues warning
        fml = "time ~ exog1 + exog2 + exog3 + exog4"
        mod2 = PHReg.from_formula(fml, df, status=status, entry=entry)
        rslt2 = mod2.fit()

        mod3 = PHReg.from_formula(fml, df, status="status", entry="entry")
        rslt3 = mod3.fit()

        assert_allclose(rslt1.params, rslt2.params)
        assert_allclose(rslt1.params, rslt3.params)
        assert_allclose(rslt1.bse, rslt2.bse)
        assert_allclose(rslt1.bse, rslt3.bse)

    def test_formula_environment(self):
        """Test that PHReg uses the right environment for formulas."""

        def times_two(x):
            return 2 * x

        rng = np.random.default_rng(0)

        exog = rng.uniform(size=100)
        endog = np.exp(exog) * -np.log(rng.uniform(size=len(exog)))
        data = pd.DataFrame({"endog": endog, "exog": exog})

        result_direct = PHReg(endog, times_two(exog)).fit()

        result_formula = PHReg.from_formula("endog ~ times_two(exog)", data=data).fit()

        assert_allclose(result_direct.params, result_formula.params)
        assert_allclose(result_direct.bse, result_formula.bse)

    def test_formula_cat_interactions(self):

        time = np.r_[1, 2, 3, 4, 5, 6, 7, 8, 9]
        status = np.r_[1, 1, 0, 0, 1, 0, 1, 1, 1]
        x1 = np.r_[1, 1, 1, 2, 2, 2, 3, 3, 3]
        x2 = np.r_[1, 2, 3, 1, 2, 3, 1, 2, 3]
        df = pd.DataFrame({"time": time, "status": status, "x1": x1, "x2": x2})

        model1 = PHReg.from_formula(
            "time ~ C(x1) + C(x2) + C(x1)*C(x2)", status="status", data=df
        )
        assert_equal(model1.exog.shape, [9, 8])

    def test_predict_formula(self):

        n = 100
        rs = np.random.RandomState(34234)
        time = 50 * rs.uniform(size=n)
        status = rs.randint(0, 2, n).astype(np.float64)
        exog = rs.uniform(1, 2, size=(n, 2))

        df = pd.DataFrame(
            {"time": time, "status": status, "exog1": exog[:, 0], "exog2": exog[:, 1]}
        )

        # Works with "0 +" on RHS but issues warning
        fml = "time ~ 0 + exog1 + np.log(exog2) + exog1*exog2"
        model1 = PHReg.from_formula(fml, df, status=status)
        result1 = model1.fit()

        mgr = FormulaManager()
        dfp = mgr.get_matrices(model1.data.model_spec, df)

        pr1 = result1.predict()
        pr2 = result1.predict(exog=df)
        pr3 = model1.predict(result1.params, exog=dfp)  # No standard errors
        pr4 = model1.predict(result1.params, cov_params=result1.cov_params(), exog=dfp)

        prl = (pr1, pr2, pr3, pr4)
        for i in range(4):
            for j in range(i):
                assert_allclose(prl[i].predicted_values, prl[j].predicted_values)

        prl = (pr1, pr2, pr4)
        for i in range(3):
            for j in range(i):
                assert_allclose(prl[i].standard_errors, prl[j].standard_errors)

    def test_formula_args(self):

        rs = np.random.RandomState(34234)
        n = 200
        time = 50 * rs.uniform(size=n)
        status = rs.randint(0, 2, size=n).astype(np.float64)
        exog = rs.normal(size=(200, 2))
        offset = rs.uniform(size=n)
        entry = rs.uniform(0, 1, size=n) * time

        df = pd.DataFrame(
            {
                "time": time,
                "status": status,
                "x1": exog[:, 0],
                "x2": exog[:, 1],
                "offset": offset,
                "entry": entry,
            }
        )
        model1 = PHReg.from_formula(
            "time ~ x1 + x2", status="status", offset="offset", entry="entry", data=df
        )
        result1 = model1.fit()
        model2 = PHReg.from_formula(
            "time ~ x1 + x2",
            status=df.status,
            offset=df.offset,
            entry=df.entry,
            data=df,
        )
        result2 = model2.fit()
        assert_allclose(result1.params, result2.params)
        assert_allclose(result1.bse, result2.bse)

    def test_offset(self):

        rs = np.random.RandomState(34234)
        time = 50 * rs.uniform(size=200)
        status = rs.randint(0, 2, 200).astype(np.float64)
        exog = rs.normal(size=(200, 4))

        for ties in "breslow", "efron":
            mod1 = PHReg(time, exog, status)
            rslt1 = mod1.fit()
            offset = exog[:, 0] * rslt1.params[0]
            exog = exog[:, 1:]

            mod2 = PHReg(time, exog, status, offset=offset, ties=ties)
            rslt2 = mod2.fit()

            assert_allclose(rslt2.params, rslt1.params[1:])

    def test_post_estimation(self):
        # All regression tests
        rs = np.random.RandomState(34234)
        time = 50 * rs.uniform(size=200)
        status = rs.randint(0, 2, 200).astype(np.float64)
        exog = rs.normal(size=(200, 4))

        mod = PHReg(time, exog, status)
        rslt = mod.fit()
        mart_resid = rslt.martingale_residuals
        # R: sum(abs(resid(coxph(Surv(time, status) ~ ., ties="breslow"), "martingale")))
        assert_allclose(np.abs(mart_resid).sum(), 123.12721130671, rtol=1e-10)
        assert_allclose(mart_resid.sum(), 0, atol=1e-10)

        w_avg = rslt.weighted_covariate_averages
        assert_allclose(
            np.abs(w_avg[0]).sum(0),
            np.r_[7.31008415, 9.77608674, 10.89515885, 13.1106801],
        )

        bc_haz = rslt.baseline_cumulative_hazard
        v = [np.mean(np.abs(x)) for x in bc_haz[0]]
        w = np.r_[23.482841556421608, 0.44149255358417017, 0.68660114081275281]
        assert_allclose(v, w)

        score_resid = rslt.score_residuals
        v = np.r_[0.50924792, 0.4533952, 0.4876718, 0.5441128]
        w = np.abs(score_resid).mean(0)
        assert_allclose(v, w)

        groups = rs.randint(0, 3, 200)
        mod = PHReg(time, exog, status)
        rslt = mod.fit(groups=groups)
        robust_cov = rslt.cov_params()
        v = [0.00513432, 0.01278423, 0.00810427, 0.00293147]
        w = np.abs(robust_cov).mean(0)
        assert_allclose(v, w, rtol=1e-6)

        s_resid = rslt.schoenfeld_residuals
        ii = np.flatnonzero(np.isfinite(s_resid).all(1))
        s_resid = s_resid[ii, :]
        v = np.r_[0.85154336, 0.72993748, 0.73758071, 0.78599333]
        assert_allclose(np.abs(s_resid).mean(0), v)

    @pytest.mark.smoke
    def test_summary(self):
        rs = np.random.RandomState(34234)
        time = 50 * rs.uniform(size=200)
        status = rs.randint(0, 2, 200).astype(np.float64)
        exog = rs.normal(size=(200, 4))

        mod = PHReg(time, exog, status)
        rslt = mod.fit()
        smry = rslt.summary()

        strata = np.kron(np.arange(50), np.ones(4))
        mod = PHReg(time, exog, status, strata=strata)
        rslt = mod.fit()
        smry = rslt.summary()
        msg = "3 strata dropped for having no events"
        assert_(msg in str(smry))

        groups = np.kron(np.arange(25), np.ones(8))
        mod = PHReg(time, exog, status)
        rslt = mod.fit(groups=groups)
        smry = rslt.summary()

        entry = rs.uniform(0.1, 0.8, 200) * time
        mod = PHReg(time, exog, status, entry=entry)
        rslt = mod.fit()
        smry = rslt.summary()
        msg = "200 observations have positive entry times"
        assert_(msg in str(smry))

    def test_summary_after_remove_data(self):
        # summary() must still work after remove_data() has been called
        rs = np.random.RandomState(34234)
        time = 50 * rs.uniform(size=200)
        status = rs.randint(0, 2, 200).astype(np.float64)
        exog = rs.normal(size=(200, 4))

        mod = PHReg(time, exog, status)
        res = mod.fit()

        assert isinstance(res.summary(), Summary)
        res.remove_data()
        assert isinstance(res.summary(), Summary)

    @pytest.mark.smoke
    def test_predict(self):
        # All smoke tests. We should be able to convert the lhr and hr
        # tests into real tests against R.  There are many options to
        # this function that may interact in complicated ways.  Only a
        # few key combinations are tested here.
        rs = np.random.RandomState(34234)
        endog = 50 * rs.uniform(size=200)
        status = rs.randint(0, 2, 200).astype(np.float64)
        exog = rs.normal(size=(200, 4))

        mod = PHReg(endog, exog, status)
        rslt = mod.fit()
        rslt.predict()
        for pred_type in "lhr", "hr", "cumhaz", "surv":
            rslt.predict(pred_type=pred_type)
            rslt.predict(endog=endog[0:10], pred_type=pred_type)
            rslt.predict(endog=endog[0:10], exog=exog[0:10, :], pred_type=pred_type)

    @pytest.mark.smoke
    def test_get_distribution(self):
        rs = np.random.RandomState(34234)
        n = 200
        exog = rs.normal(size=(n, 2))
        lin_pred = exog.sum(1)
        elin_pred = np.exp(-lin_pred)
        time = -elin_pred * np.log(rs.uniform(size=n))
        status = np.ones(n)
        status[0:20] = 0
        strata = np.kron(range(5), np.ones(n // 5))

        mod = PHReg(time, exog, status=status, strata=strata)
        rslt = mod.fit()

        dist = rslt.get_distribution()

        # Smoke checks
        dist.mean()
        dist.var()
        dist.std()
        dist.rvs(rng=rs)

    def test_fit_regularized(self):

        # Data set sizes
        for n, p in (50, 2), (100, 5):

            # Penalty weights
            for js, s in enumerate([0, 0.1]):

                coef_name = f"coef_{n:d}_{p:d}_{js:d}"
                params = getattr(survival_enet_r_results, coef_name)

                fname = f"survival_data_{n:d}_{p:d}.csv"
                time, status, entry, exog = self.load_file(fname)

                exog -= exog.mean(0)
                exog /= exog.std(0, ddof=1)

                model = PHReg(time, exog, status=status, ties="breslow")
                sm_result = model.fit_regularized(alpha=s)

                # The agreement is not very high, the issue may be on
                # the R side.  See below for further checks.
                assert_allclose(sm_result.params, params, rtol=0.3)

                # The penalized log-likelihood that we are maximizing.
                def plf(params, model, time, s):
                    llf = model.loglike(params) / len(time)
                    L1_wt = 1
                    llf = llf - s * (
                        (1 - L1_wt) * np.sum(params**2) / 2
                        + L1_wt * np.sum(np.abs(params))
                    )
                    return llf

                # Confirm that we are doing better than glmnet.
                llf_r = plf(params, model, time, s)
                llf_sm = plf(sm_result.params, model, time, s)
                assert_equal(np.sign(llf_sm - llf_r), 1)

    def test_invalid_ties_raises(self):
        time, status, entry, exog = self.load_file("survival_data_50_2.csv")
        with pytest.raises(ValueError, match="ties"):
            PHReg(time, exog, status=status, ties="not-a-tie-method")

    def test_fit_regularized_invalid_method_raises(self):
        time, status, entry, exog = self.load_file("survival_data_50_2.csv")
        model = PHReg(time, exog, status=status, ties="breslow")
        with pytest.raises(ValueError, match="method"):
            model.fit_regularized(method="not-a-method")


cur_dir = Path(__file__).resolve().parent
rdir = Path(cur_dir).joinpath("results")
fnames = [p.name for p in rdir.iterdir()]
fnames = [x for x in fnames if x.startswith("survival") and x.endswith(".csv")]

ties = ("breslow", "efron")
entry_f = (False, True)
strata_f = (False, True)


@pytest.mark.parametrize(
    "fname,ties,entry_f,strata_f",
    list(itertools.product(fnames, ties, entry_f, strata_f)),
)
def test_r(fname, ties, entry_f, strata_f):
    TestPHReg.do1(fname, ties, entry_f, strata_f)


@pytest.mark.parametrize("stratified", [False, True])
def test_schoenfeld_residuals_efron_ties(stratified):
    # GH 10288.  Reference values from R 4.6.1, survival 3.8.6:
    #
    # d <- data.frame(
    #   time = c(1, 1, 2, 2, 3, 4, 4, 5, 6, 7, 8, 9),
    #   status = c(1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 0, 1),
    #   x = c(0.5, -1.2, 0.3, 1.1, -0.7, 0.8, -0.2, 1.5, -1.0, 0.4, 0.9, -0.3),
    #   z = c(1, 0, 0, 1, 1, 0, 1, 0, 0, 1, 1, 0),
    #   g = c(0, 0, 0, 0, 1, 0, 0, 1, 1, 1, 1, 1),
    #   entry = c(0, 0, 0, 1.5, 0, 2.5, 0, 3.5, 0, 0, 5.5, 0))
    # f <- coxph(Surv(time, status) ~ x, data = d, ties = "efron")
    # coef(f); resid(f, "schoenfeld")
    # f <- coxph(Surv(entry, time, status) ~ x + z + strata(g), data = d,
    #            ties = "efron")
    # coef(f); resid(f, "schoenfeld")
    #
    # R returns one row per event, ordered by stratum and then time.  The
    # rows below are in data order, with NaN for the censored subjects.
    time = np.array([1, 1, 2, 2, 3, 4, 4, 5, 6, 7, 8, 9.0])
    status = np.array([1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 0, 1])
    x = np.array([0.5, -1.2, 0.3, 1.1, -0.7, 0.8, -0.2, 1.5, -1.0, 0.4, 0.9, -0.3])
    z = np.array([1, 0, 0, 1, 1, 0, 1, 0, 0, 1, 1, 0.0])
    nan = np.nan
    if stratified:
        exog = np.column_stack([x, z])
        strata = np.array([0, 0, 0, 0, 1, 0, 0, 1, 1, 1, 1, 1])
        entry = np.array([0, 0, 0, 1.5, 0, 2.5, 0, 3.5, 0, 0, 5.5, 0])
        mod = PHReg(time, exog, status, entry=entry, strata=strata, ties="efron")
        params = [-0.69172174953, -0.01604387583]
        resid = [
            [0.91413479736, 0.5998424419],
            [-0.78586520264, -0.4001575581],
            [0.08207985669, -0.6617698174],
            [nan, nan],
            [-0.12815290248, 0.5793829822],
            [0.66277336193, -0.6627733619],
            [-0.33722663807, 0.3372266381],
            [nan, nan],
            [-0.64624439218, -0.2830406625],
            [0.23850111936, 0.4912893378],
            [nan, nan],
            [0.0, 0.0],
        ]
    else:
        mod = PHReg(time, x, status, ties="efron")
        params = [-0.6346743889]
        resid = [
            [0.71763481291],
            [-0.98236518709],
            [0.39462580605],
            [nan],
            [-0.49199674188],
            [0.90917493627],
            [-0.09082506373],
            [nan],
            [-0.67754711138],
            [0.22129854829],
            [nan],
            [0.0],
        ]
    res = mod.fit(disp=0)
    assert_allclose(res.params, params, rtol=1e-6)
    assert_allclose(res.schoenfeld_residuals, resid, rtol=1e-6, atol=1e-8)


@pytest.mark.parametrize("ties", ["efron", "breslow"])
def test_schoenfeld_residuals_multiple_ties(ties):
    # Tied failure times with 3 and 4 events, the ties in the test above have
    # 2 events. Reference values from R 4.6.1, survival 3.8.6:
    #
    # d <- data.frame(
    #   time = c(1, 2, 2, 2, 3, 4, 4, 4, 4, 5, 6, 7),
    #   status = c(1, 1, 1, 1, 0, 1, 1, 1, 1, 0, 1, 1),
    #   x = c(0.5, -1.2, 0.3, 1.1, -0.7, 0.8, -0.2, 1.5, -1.0, 0.4, 0.9, -0.3),
    #   z = c(1, 0, 0, 1, 1, 0, 1, 0, 0, 1, 1, 0))
    # f <- coxph(Surv(time, status) ~ x + z, data = d, ties = "efron")
    # coef(f); resid(f, "schoenfeld")
    #
    # The rows of the residuals are in data order, with NaN for the censored
    # subjects.
    time = np.array([1, 2, 2, 2, 3, 4, 4, 4, 4, 5, 6, 7.0])
    status = np.array([1, 1, 1, 1, 0, 1, 1, 1, 1, 0, 1, 1])
    x = np.array([0.5, -1.2, 0.3, 1.1, -0.7, 0.8, -0.2, 1.5, -1.0, 0.4, 0.9, -0.3])
    z = np.array([1, 0, 0, 1, 1, 0, 1, 0, 0, 1, 1, 0.0])
    nan = np.nan
    if ties == "efron":
        params = [0.126134856833, -0.199405602022]
        resid = [
            [0.253142098917, 0.540858401508],
            [-1.43787285568, -0.422328567227],
            [0.0621271443206, -0.422328567227],
            [0.862127144321, 0.577671432773],
            [nan, nan],
            [0.436516526926, -0.446470333377],
            [-0.563483473074, 0.553529666623],
            [1.13651652693, -0.446470333377],
            [-1.36348347307, -0.446470333377],
            [nan, nan],
            [0.614410360416, 0.51200863368],
            [0.0, 0.0],
        ]
    else:
        params = [0.110602036159, -0.0669040880161]
        resid = [
            [0.255356192995, 0.508801050419],
            [-1.42147107045, -0.445026923901],
            [0.0785289295504, -0.445026923901],
            [0.87852892955, 0.554973076099],
            [nan, nan],
            [0.432198849958, -0.414317906871],
            [-0.567801150042, 0.585682093129],
            [1.13219884996, -0.414317906871],
            [-1.36780115004, -0.414317906871],
            [nan, nan],
            [0.580261618521, 0.483551348767],
            [0.0, 0.0],
        ]
    res = PHReg(time, np.column_stack([x, z]), status, ties=ties).fit(disp=0)
    assert_allclose(res.params, params, rtol=1e-6)
    assert_allclose(res.schoenfeld_residuals, resid, rtol=1e-6, atol=1e-8)


def _residual_model(case):
    d = residual_results.data[case["data"]]
    exog = np.column_stack([d[c] for c in case["exog"]])
    kwds = {k: d[case[k]] for k in ("strata", "entry", "offset") if case[k]}
    mod = PHReg(d["time"], exog, d["status"], ties=case["ties"], **kwds)
    groups = d[case["groups"]] if case["groups"] else None
    return mod, groups


@pytest.mark.parametrize("name", list(residual_results.results))
def test_residuals_r(name):
    # GH 10288. Score and martingale residuals, and the naive and the robust
    # covariance of the coefficients for the Breslow and Efron approximation
    # of ties, with strata, delayed entry, an offset and clusters. The
    # reference values are from R survival 3.8.6, see results_phreg_residuals.
    # R has 0 for the residuals of observations that are not used, PHReg has NaN.
    case = residual_results.results[name]
    mod, groups = _residual_model(case)
    res = mod.fit(groups=groups, disp=0)
    assert_allclose(res.params, case["coef"], rtol=1e-6)

    used = np.isfinite(res.score_residuals).all(1)
    assert_equal(np.isfinite(res.martingale_residuals), used)
    assert_allclose(res.score_residuals[used], case["score"][used], rtol=1e-6, atol=1e-8)
    assert_allclose(
        res.martingale_residuals[used], case["martingale"][used], rtol=1e-6, atol=1e-8
    )
    assert_allclose(res.cov_params(), case["var"], rtol=1e-6)
    if "var_naive" in case:
        assert_allclose(mod.fit(disp=0).cov_params(), case["var_naive"], rtol=1e-6)


def test_residuals_not_used_observation():
    # an observation that is censored before the first event is not used, the
    # residuals are NaN and it does not contribute to the robust covariance
    case = residual_results.results["d4_efron_cluster"]
    mod, groups = _residual_model(case)
    res = mod.fit(groups=groups, disp=0)
    assert np.all(np.isnan(res.score_residuals[0]))
    assert np.isnan(res.martingale_residuals[0])
    assert np.all(np.isfinite(res.score_residuals[1:]))
    assert np.all(np.isfinite(res.cov_params()))


@pytest.mark.parametrize("name", list(residual_results.results))
def test_residuals_sums(name):
    # The score residuals add up to the score of the partial likelihood for
    # all parameters, and the martingale residuals of a stratum add up to zero.
    case = residual_results.results[name]
    mod, groups = _residual_model(case)
    res = mod.fit(groups=groups, disp=0)
    for params in (res.params, case["coef"] * 0.5 + 0.2):
        resid = mod.score_residuals(params)
        assert_allclose(np.nansum(resid, 0), mod.score(params), atol=1e-12)
    d = residual_results.data[case["data"]]
    strata = d[case["strata"]] if case["strata"] else np.zeros(len(d["time"]))
    for g in np.unique(strata):
        assert_allclose(np.nansum(res.martingale_residuals[strata == g]), 0, atol=1e-10)
