"""
Assesment of Generalized Estimating Equations using simulation.

Only Gaussian models are currently checked.
"""

##!!!! Delete before going to github
import sys
df = "/afs/umich.edu/user/k/s/kshedden/fork/statsmodels/"
sys.path.insert(0, df)


import numpy as np
from statsmodels.genmod.generalized_estimating_equations import GEE,\
    gee_setup_multicategorical,gee_ordinal_starting_values
from statsmodels.genmod.families import Gaussian,Binomial,Poisson
from statsmodels.genmod.dependence_structures import Exchangeable,\
    Independence,GlobalOddsRatio,Autoregressive,Nested
import statsmodels.formula.api as sm
from itertools import product


class GEE_simulator(object):

    #
    # Parameters that must be defined
    #

    # Number of groups
    ngroups = None

    # Standard deviation of the pure errors
    error_sd = None

    # The regression coefficients
    params = None

    # The parameters defining the dependence structure
    dparams = None


    #
    # Output parameters
    #

    # Matrix of exogeneous data (rows are cases, columns are
    # variables)
    exog = None

    # Matrix of endogeneous data (len(endog) = exog.shape[0])
    endog = None

    # Matrix of time information (time.shape[0] = len(endog))
    time = None

    # Group labels (len(groups) = len(endog))
    groups = None

    # Group sizes are random within this range
    group_size_range = [4, 11]

    # dparams_est is dparams with scale_inv appended
    def print_dparams(self, dparams_est):
        raise NotImplementedError


class AR_simulator(GEE_simulator):

    # The distance function for determining AR correlations.
    distfun = [lambda x, y: np.sqrt(np.sum((x-y)**2)),]


    def print_dparams(self, dparams_est):
        print "AR coefficient estimate:   %8.4f" % dparams_est[0]
        print "AR coefficient truth:      %8.4f" % self.dparams[0]
        print "Error variance estimate:   %8.4f" % dparams_est[1]
        print "Error variance truth:      %8.4f" % self.error_sd**2
        print

    def simulate(self):

        endog, exog, group, time = [], [], [], []

        for i in range(self.ngroups):

            gsize = np.random.randint(self.group_size_range[0],
                                      self.group_size_range[1])

            group.append([i,] * gsize)

            time1 = np.random.normal(size=(gsize,2))
            time.append(time1)

            exog1 = np.random.normal(size=(gsize, 5))
            exog1[:,0] = 1
            exog.append(exog1)

            # Pairwise distances within the cluster
            distances = np.zeros((gsize, gsize), dtype=np.float64)
            distfun = self.distfun[0]
            for j1 in range(gsize):
                for j2 in range(gsize):
                    distances[j1, j2] = \
                        distfun(time1[j1,:], time1[j2,:])

            # Pairwise correlations within the cluster
            correlations = self.dparams[0]**distances
            correlations_sr = np.linalg.cholesky(correlations)

            errors = np.dot(correlations_sr, np.random.normal(size=gsize))

            endog1 = np.dot(exog1, self.params) + errors * self.error_sd
            endog.append(endog1)

        self.exog = np.concatenate(exog, axis=0)
        self.endog = np.concatenate(endog)
        self.time = np.concatenate(time, axis=0)
        self.group = np.concatenate(group)



class Nested_simulator(GEE_simulator):

    # Vector containing list of nest sizes (used instead of
    # group_size_range).
    nest_sizes = None

    # Matrix of nest id's (an output parameter)
    id_matrix = None


    def print_dparams(self, dparams_est):
        for j in range(len(self.nest_sizes)):
            print "Nest %d variance estimate:  %8.4f" % \
                (j+1, dparams_est[j])
            print "Nest %d variance truth:     %8.4f" % \
                (j+1, self.dparams[j])

        print "Error variance estimate:   %8.4f" % \
            (dparams_est[-1] - sum(dparams_est[0:-1]))
        print "Error variance truth:      %8.4f" % self.error_sd**2
        print


    def simulate(self):

        group_effect_var = self.dparams[0]

        vcomp = self.dparams[1:]
        vcomp.append(0)

        endog, exog, group, id_matrix = [], [], [], []

        for i in range(self.ngroups):

            iterators = [xrange(n) for n in self.nest_sizes]

            # The random effects
            variances = [np.sqrt(v)*np.random.normal(size=n)
                         for v,n in zip(vcomp, self.nest_sizes)]

            gpe = np.random.normal() * np.sqrt(group_effect_var)

            nest_all = []
            for j in self.nest_sizes:
                nest_all.append(set())

            for nest in product(*iterators):

                group.append(i)

                # The sum of all random effects that apply to this
                # unit
                ref = gpe + sum([v[j] for v,j in zip(variances, nest)])

                exog1 = np.random.normal(size=5)
                exog1[0] = 1
                exog.append(exog1)

                error = ref + self.error_sd * np.random.normal()

                endog1 = np.dot(exog1, self.params) + error
                endog.append(endog1)

                for j in range(len(nest)):
                    nest_all[j].add(tuple(nest[0:j+1]))

                nest1 = [len(x)-1 for x in nest_all]
                id_matrix.append(nest1[0:-1])

        self.exog = np.array(exog)
        self.endog = np.array(endog)
        self.group = np.array(group)
        self.id_matrix = np.array(id_matrix)
        self.time = np.zeros_like(self.endog)



def check_dparams(gendat):
    """
    Check the estimation of the dependence parameters.
    """

    nrep = 10

    dparams = []
    for j in range(nrep):

        da,va = gendat()

        ga = Gaussian()

        md = GEE(da.endog, da.exog, da.group, da.time, ga, va)
        mdf = md.fit()

        scale_inv = 1 / md.estimate_scale()

        dparams.append(np.r_[va.dparams, scale_inv])

    dparams_mean = np.array(sum(dparams) / len(dparams))

    #v = list(da.dparams)
    #v.append(da.error_sd**2)
    #v = np.array(v)

    da.print_dparams(dparams_mean)


def check_regression(gendat):
    """
    Check the estimation of the regression coefficients.
    """

    nrep = 10

    params = []
    std_errors = []

    for j in range(nrep):

        da,va = gendat()

        ga = Gaussian()

        md = GEE(da.endog, da.exog, da.group, da.time, ga, va)
        mdf = md.fit()

        params.append(np.asarray(mdf.params))
        std_errors.append(np.asarray(mdf.standard_errors))

    params = np.array(params)
    eparams = params.mean(0)
    sdparams = params.std(0)
    std_errors = np.array(std_errors)
    std_errors = std_errors.mean(0)

    print "Checking parameter values"
    print "Observed:            ", eparams
    print "Expected:            ", da.params
    print "Absolute difference: ", eparams - da.params
    print "Relative difference: ", (eparams - da.params) / da.params
    print

    print "Checking standard errors"
    print "Observed:            ", sdparams
    print "Expected:            ", std_errors
    print "Absolute difference: ", sdparams - std_errors
    print "Relative difference: ", (sdparams - std_errors) / std_errors
    print


def check_constraint(gendat0):
    """
    Check the score testing of the parameter constraints.
    """

    nrep = 100
    pvalues = []

    for j in range(nrep):

        da,va = gendat()

        ga = Gaussian()

        lhs = np.array([[0., 1, 1, 0, 0],])
        rhs = np.r_[0.,]

        md = GEE(da.endog, da.exog, da.group, da.time, ga, va,
                 constraint=(lhs, rhs))
        mdf = md.fit()
        score = md.score_test_results
        pvalues.append(score["p-value"])

    pvalues.sort()

    print "Checking constrained estimation:"
    print "Observed   Expected"
    for q in np.arange(0.1, 0.91, 0.1):
        print "%10.3f %10.3f" % (pvalues[int(q*len(pvalues))], q)



def gen_gendat_ar0(ar):
    def gendat_ar0(msg = False):
        ars = AR_simulator()
        ars.ngroups = 200
        ars.params = np.r_[0, -1, 1, 0, 0.5]
        ars.error_sd = 2
        ars.dparams = [ar,]
        ars.simulate()
        return ars, Autoregressive()
    return gendat_ar0

def gen_gendat_ar1(ar):
    def gendat_ar1():
        ars = AR_simulator()
        ars.ngroups = 200
        ars.params = np.r_[0, -0.8, 1.2, 0, 0.5]
        ars.error_sd = 2
        ars.dparams = [ar,]
        ars.simulate()
        return ars, Autoregressive()
    return gendat_ar1

def gendat_nested0():
    ns = Nested_simulator()
    ns.error_sd = 1.
    ns.params = np.r_[0., 1, 1, -1, -1]
    ns.ngroups = 50
    ns.nest_sizes = [10, 5]
    ns.dparams = [2., 1.]
    ns.simulate()
    return ns, Nested(ns.id_matrix)

def gendat_nested1():
    ns = Nested_simulator()
    ns.error_sd = 2.
    ns.params = np.r_[0, 1, 1.3, -0.8, -1.2]
    ns.ngroups = 50
    ns.nest_sizes = [10, 5]
    ns.dparams = [1., 3.]
    ns.simulate()
    return ns, Nested(ns.id_matrix)

# Loop over data generating models
for j in 0,1:

    if j == 0:
        gendats = [gen_gendat_ar0(ar) for ar in 0, 0.3, 0.6]
        gendats.extend([gen_gendat_ar1(ar) for ar in 0, 0.3, 0.6])
    elif j == 1:
        gendats = [gendat_nested0, gendat_nested1]

    for gendat in gendats:

        check_dparams(gendat)

        check_regression(gendat)

        check_constraint(gendat)
