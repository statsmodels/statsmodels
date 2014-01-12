import os
import numpy as np
from statsmodels.sandbox.phreg import PHreg
from numpy.testing import assert_almost_equal


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


    # Read the R results from the results directory.  The first column
    # of the returned array is the params and the second column is
    # bse.
    def get_r_params(self, fname, entry, strata, ties):

        cur_dir = os.path.dirname(os.path.abspath(__file__))
        rdir = os.path.join(cur_dir, 'results')

        params_r = []

        u = []
        if entry:
            u.append("et")
        if strata:
            u.append("st")

        u.append(ties[0:3])

        fname1 = fname.replace(".csv", "_" + "_".join(u) + ".txt")

        fid = open(os.path.join(rdir, fname1))
        for line in fid:
            if line.startswith("exog"):
                lsp = line.split()
                params_r.append((float(lsp[1]), float(lsp[3])))
        params_r = np.asarray(params_r)
        params_r = params_r[0:params_r.shape[0]/2,:]
        return params_r



    # Run a single test against R output
    def do1(self, fname, ties, entry_f, strata_f):

        time, status, entry, exog = self.load_file(fname)
        n = len(time)

        if strata_f:
            # Needs to match the kronecker statement in survival.R
            strata = np.kron(range(5), np.ones(n/5))

        # No stratification or entry times
        if not entry_f and not strata_f:
            phrb = PHreg(time, status, exog,
                         ties=ties)

        # Entry times but no stratification
        elif entry_f and not strata_f:
            phrb = PHreg(time, status, exog,
                         entry=entry, ties=ties)

        # Stratification but no entry times
        elif strata_f and not entry_f:
            phrb = PHreg(time, status, exog,
                         strata=strata, ties=ties)

        # Stratification and entry times
        else:
            phrb = PHreg(time, status, exog,
                         entry=entry, strata=strata,
                         ties=ties)

        phr = phrb.fit(method='bfgs', disp=0)
        params = phr.params

        params_r = self.get_r_params(fname, entry_f,
                                     strata_f, ties)

        assert_almost_equal(params, params_r[:,0],
                            decimal=4)

        se = phr.bse
        assert_almost_equal(se, params_r[:,1], decimal=4)



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
