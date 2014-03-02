import numpy as np
from statsmodels.regression.lme import LME
from numpy.testing import assert_almost_equal
from lme_r_results import *
import os
import csv

class TestLME(object):

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
        mdf = md.fit(reml=reml, num_em=50)
        params_fe, revar = md._unpack(mdf.params)
        sig2 = md.get_sig2(params_fe, revar, reml)
        revar *= sig2

        assert_almost_equal(params_fe, coef, decimal=4)
        assert_almost_equal(revar, revar_r, decimal=4)
        assert_almost_equal(sig2, sig2_r, decimal=4)

        # Our cov_params uses numerical differentiation, so
        # agreement won't be very high.
        pf = exog_fe.shape[1]
        assert_almost_equal(vcov_r, mdf.cov_params()[0:pf,0:pf],
                            decimal=3)


    # Run all the tests
    def test_r(self):

        cur_dir = os.path.dirname(os.path.abspath(__file__))
        rdir = os.path.join(cur_dir, 'results')
        fnames = os.listdir(rdir)
        fnames = [x for x in fnames if x.startswith("lme")
                  and x.endswith(".csv")]

        for fname in fnames:
            for reml in False,True:
                ds_ix = int(fname[3:5])
                yield self.do1, reml, ds_ix



if  __name__=="__main__":

    import nose

    nose.runmodule(argv=[__file__,'-vvs','-x','--pdb', '--pdb-failure'],
                   exit=False)
