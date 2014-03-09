import numpy as np
from statsmodels.regression.lme import LME
from numpy.testing import assert_almost_equal
from lme_r_results import *
from scipy.misc import derivative
import os
import csv

class TestLME(object):


    # Test analytic scores using numeric differentiation
    def test_score(self):

        n = 200
        m = 5
        p = 3
        pr = 2

        for jl in 0,1:
            for reml in False,True:
                for pen in 0,10:

                    exog_fe = np.random.normal(size=(n*m, p))
                    exog_re = np.random.normal(size=(n*m, pr))
                    endog = exog_fe.sum(1) + np.random.normal(size=n*m)
                    groups = np.kron(range(n), np.ones(m))

                    md = LME(endog, exog_fe, exog_re, groups)
                    if jl == 0:
                        like = lambda x: -md.like_L(x, reml, pen)
                        score = lambda x: -md.score_L(x, reml, pen)
                    else:
                        like = lambda x: -md.like(x, reml, pen)
                        score = lambda x: -md.score(x, reml, pen)

                    for kr in range(5):
                        params_fe = np.random.normal(size=p)
                        revar = np.random.normal(size=(pr,pr))
                        revar = np.dot(revar.T, revar)
                        params_prof = md._pack(params_fe, revar)
                        gr = score(params_prof)

                        ngr = np.zeros_like(gr)
                        for k in range(len(ngr)):
                            def f(x):
                                pp = params_prof.copy()
                                pp[k] = x
                                return like(pp)
                            ngr[k] = derivative(f, params_prof[k], dx=1e-6)

                        assert_almost_equal(gr / ngr, np.ones(len(gr)),
                                            decimal=3)


    def do1(self, reml, ds_ix):

        meth = "reml" if reml else "ml"

        coef = globals()["coef_%s_%d" % (meth, ds_ix)]
        vcov_r = globals()["vcov_%s_%d" % (meth, ds_ix)]
        revar_r = globals()["revar_%s_%d" % (meth, ds_ix)]
        sig2_r = globals()["sig2_%s_%d" % (meth, ds_ix)]

        # Variance component MLE ~ 0 currently requires manual
        # tweaking of algorithm parameters, so exclude from tests.
        if np.min(np.diag(revar_r)) < 0.01:
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
        md = LME(endog, exog_fe, exog_re, groups)
        mdf = md.fit(reml=reml)

        assert_almost_equal(mdf.params_fe, coef, decimal=4)
        assert_almost_equal(mdf.revar, revar_r, decimal=4)
        assert_almost_equal(mdf.sig2, sig2_r, decimal=4)

        pf = exog_fe.shape[1]
        assert_almost_equal(vcov_r, mdf.cov_params()[0:pf,0:pf],
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
                ds_ix = int(fname[3:5])
                print ds_ix, reml
                yield self.do1, reml, ds_ix



if  __name__=="__main__":

    import nose

    nose.runmodule(argv=[__file__,'-vvs','-x','--pdb', '--pdb-failure'],
                   exit=False)
