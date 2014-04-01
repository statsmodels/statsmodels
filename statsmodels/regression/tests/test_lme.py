import numpy as np
import pandas as pd
from statsmodels.regression.lme import MixedLM
from numpy.testing import assert_almost_equal
from lme_r_results import *
from scipy.misc import derivative
from statsmodels.base import penalties
import os
import csv


class TestMixedLM(object):

    # Test analytic scores using numeric differentiation
    # TODO: should also do this with the hessian
    def test_score(self):

        n = 200
        m = 5
        p = 3
        pr = 2

        for jl in 0,1:
            for reml in False,True:
                for cov_pen_wt in 0,10:

                    cov_pen = penalties.PSD(cov_pen_wt)

                    exog_fe = np.random.normal(size=(n*m, p))
                    exog_re = np.random.normal(size=(n*m, pr))
                    endog = exog_fe.sum(1) + np.random.normal(size=n*m)
                    groups = np.kron(range(n), np.ones(m))

                    md = MixedLM(endog, exog_fe, groups, exog_re)
                    md.reml = reml
                    md.cov_pen = cov_pen
                    if jl == 0:
                        like = lambda x: -md.loglike_L(x)
                        score = lambda x: -md.score_L(x)
                    else:
                        like = lambda x: -md.loglike(x)
                        score = lambda x: -md.score(x)

                    for kr in range(5):
                        fe_params = np.random.normal(size=p)
                        cov_re = np.random.normal(size=(pr,pr))
                        cov_re = np.dot(cov_re.T, cov_re)
                        params_prof = md._pack(fe_params, cov_re)
                        gr = score(params_prof)

                        ngr = np.zeros_like(gr)
                        for k in range(len(ngr)):
                            def f(x):
                                pp = params_prof.copy()
                                pp[k] = x
                                return like(pp)
                            ngr[k] = derivative(f, params_prof[k],
                                                dx=1e-6)

                        assert_almost_equal(gr / ngr, np.ones(len(gr)),
                                            decimal=3)


    def test_default_re(self):

        exog = np.random.normal(size=(300,4))
        groups = np.kron(np.arange(100), [1,1,1])
        g_errors = np.kron(np.random.normal(size=100), [1,1,1])
        endog = exog.sum(1) + g_errors + np.random.normal(size=300)
        mdf1 = MixedLM(endog, exog, groups).fit()
        mdf2 = MixedLM(endog, exog, groups, np.ones(300)).fit()
        assert_almost_equal(mdf1.params, mdf2.params, decimal=8)

    def test_formulas(self):

        exog = np.random.normal(size=(300,4))
        exog_re = np.random.normal(size=300)
        groups = np.kron(np.arange(100), [1,1,1])
        g_errors = exog_re * np.kron(np.random.normal(size=100),
                                     [1,1,1])
        endog = exog.sum(1) + g_errors + np.random.normal(size=300)

        mdf1 = MixedLM(endog, exog, groups, exog_re).fit()

        df = pd.DataFrame({"endog": endog})
        for k in range(exog.shape[1]):
            df["exog%d" % k] = exog[:,k]
        df["exog_re"] = exog_re
        md2 = MixedLM.from_formula(
            "endog ~ 0 + exog0 + exog1 + exog2 + exog3",
            groups=groups, data=df)
        md2.set_random("0 + exog_re", data=df)
        mdf2 = md2.fit()

        assert_almost_equal(mdf1.params, mdf2.params)


    def test_regularized(self):

        exog = np.random.normal(size=(400,5))
        groups = np.kron(np.arange(100), np.ones(4))
        expected_endog = exog[:,0] - exog[:,2]
        endog = expected_endog +\
                np.kron(np.random.normal(size=100), np.ones(4)) +\
                np.random.normal(size=400)

        # L1 regularization
        md = MixedLM(endog, exog, groups)
        mdf1 = md.fit_regularized(alpha=1.)
        mdf1.summary()

        # L1 regularization
        md = MixedLM(endog, exog, groups)
        mdf2 = md.fit_regularized(alpha=10*np.ones(5))
        mdf2.summary()

        # L2 regularization
        pen = penalties.L2()
        mdf3 = md.fit_regularized(method=pen, alpha=0.)
        mdf3.summary()

        # L2 regularization
        pen = penalties.L2()
        mdf4 = md.fit_regularized(method=pen, alpha=100.)
        mdf4.summary()

        # Pseudo-Huber regularization
        pen = penalties.PseudoHuber(0.3)
        mdf4 = md.fit_regularized(method=pen, alpha=1.)
        mdf4.summary()


    def do1(self, reml, irf, ds_ix):

        # No need to check independent random effects when there is
        # only one of them.
        if irf and ds_ix < 6:
            return

        irfs = "irf" if irf else "drf"

        meth = "reml" if reml else "ml"

        coef = globals()["coef_%s_%s_%d" % (meth, irfs, ds_ix)]
        vcov_r = globals()["vcov_%s_%s_%d" % (meth, irfs, ds_ix)]
        cov_re_r = globals()["cov_re_%s_%s_%d" % (meth, irfs, ds_ix)]
        sig2_r = globals()["sig2_%s_%s_%d" % (meth, irfs, ds_ix)]
        loglike = globals()["loglike_%s_%s_%d" % (meth, irfs, ds_ix)]

        if not irf:
            ranef_postmean = globals()["ranef_mean_%s_%s_%d" %
                                       (meth, irfs, ds_ix)]
            ranef_condvar = globals()["ranef_condvar_%s_%s_%d" %
                                      (meth, irfs, ds_ix)]
            ranef_condvar = np.atleast_2d(ranef_condvar)

        # Variance component MLE ~ 0 may require manual tweaking of
        # algorithm parameters, so exclude from tests for now.
        if np.min(np.diag(cov_re_r)) < 0.01:
            print "Skipping %d since solution is on boundary." % ds_ix
            return

        # Load the data file
        cur_dir = os.path.dirname(os.path.abspath(__file__))
        rdir = os.path.join(cur_dir, 'results')
        fname = os.path.join(rdir, "lme%02d.csv" % ds_ix)
        fid = open(fname)
        rdr = csv.reader(fid)
        header = rdr.next()
        data = [[float(x) for x in line] for line in rdr]
        data = np.asarray(data)

        # Split into exog, endog, etc.
        endog = data[:,header.index("endog")]
        groups = data[:,header.index("groups")]
        ii = [i for i,x in enumerate(header) if
              x.startswith("exog_fe")]
        exog_fe = data[:,ii]
        ii = [i for i,x in enumerate(header) if
              x.startswith("exog_re")]
        exog_re = data[:,ii]

        # Fit the model
        md = MixedLM(endog, exog_fe, groups, exog_re)
        if not irf: # Free random effects covariance
            mdf = md.fit(gtol=1e-8, reml=reml)
        else: # Independent random effects
            mdf = md.fit(reml=reml, gtol=1e-8,
                         free=(np.ones(exog_fe.shape[1]),
                               np.eye(exog_re.shape[1])))

        assert_almost_equal(mdf.fe_params, coef, decimal=4)
        assert_almost_equal(mdf.cov_re, cov_re_r, decimal=4)
        assert_almost_equal(mdf.sig2, sig2_r, decimal=4)

        pf = exog_fe.shape[1]
        assert_almost_equal(vcov_r, mdf.cov_params()[0:pf,0:pf],
                            decimal=3)

        assert_almost_equal(mdf.likeval, loglike[0], decimal=2)

        # Not supported in R
        if not irf:
            assert_almost_equal(mdf.ranef()[0], ranef_postmean,
                                decimal=3)
            assert_almost_equal(mdf.ranef_cov()[0], ranef_condvar,
                                decimal=3)

    # Run all the tests against R
    def test_r(self):

        cur_dir = os.path.dirname(os.path.abspath(__file__))
        rdir = os.path.join(cur_dir, 'results')
        fnames = os.listdir(rdir)
        fnames = [x for x in fnames if x.startswith("lme")
                  and x.endswith(".csv")]

        for fname in fnames:
            for reml in False,True:
                for irf in False,True:
                    ds_ix = int(fname[3:5])
                    print ds_ix, reml
                    yield self.do1, reml, irf, ds_ix



if  __name__=="__main__":

    import nose

    nose.runmodule(argv=[__file__,'-vvs','-x','--pdb', '--pdb-failure'],
                   exit=False)
