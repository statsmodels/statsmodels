import os
import numpy as np
from statsmodels.sandbox.phreg import PHreg
from numpy.testing import assert_almost_equal

# All the R results
from survival_r_results import *

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

        # Needs to match the kronecker statement in survival.R
        strata = np.kron(range(5), np.ones(n/5))

        # No stratification or entry times
        phrb = PHreg(time, exog, status, ties=ties).fit(**args)
        coef = globals()["coef_%d_%d_%s" % (n, p, ties[0:3])]
        se = globals()["se_%d_%d_%s" % (n, p, ties[0:3])]
        assert_almost_equal(phrb.params, coef, decimal=4)
        assert_almost_equal(phrb.bse, se, decimal=4)

        # Entry times but no stratification
        phrb = PHreg(time, exog, status, entry=entry,
                     ties=ties).fit(**args)
        coef = globals()["coef_%d_%d_et_%s" % (n, p, ties[0:3])]
        se = globals()["se_%d_%d_et_%s" % (n, p, ties[0:3])]
        assert_almost_equal(phrb.params, coef, decimal=4)
        assert_almost_equal(phrb.bse, se, decimal=4)

        # Stratification but no entry times
        phrb = PHreg(time, exog, status, strata=strata,
                      ties=ties).fit(**args)
        coef = globals()["coef_%d_%d_st_%s" % (n, p, ties[0:3])]
        se = globals()["se_%d_%d_st_%s" % (n, p, ties[0:3])]
        assert_almost_equal(phrb.params, coef, decimal=4)
        assert_almost_equal(phrb.bse, se, decimal=4)

        # Stratification and entry times
        phrb = PHreg(time, exog, status, entry=entry,
                     strata=strata, ties=ties).fit(**args)
        coef = globals()["coef_%d_%d_et_st_%s" % (n, p, ties[0:3])]
        se = globals()["se_%d_%d_et_st_%s" % (n, p, ties[0:3])]
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



if  __name__=="__main__":

    import nose

    nose.runmodule(argv=[__file__,'-vvs','-x','--pdb', '--pdb-failure'],
                   exit=False)
