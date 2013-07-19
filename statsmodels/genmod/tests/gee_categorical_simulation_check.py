"""
Assesment of Generalized Estimating Equations using simulation.

This script checks the performance of ordinal and nominal models for
multinomial data.
"""

##!!!! Delete before going to github
import sys
df = "/afs/umich.edu/user/k/s/kshedden/fork/statsmodels/"
sys.path.insert(0, df)


import numpy as np
from statsmodels.genmod.generalized_estimating_equations import GEE,\
    gee_setup_multicategorical, gee_ordinal_starting_values, \
    Multinomial
from statsmodels.genmod.families import Gaussian,Binomial,Poisson
from statsmodels.genmod.dependence_structures import Exchangeable,\
    Independence,GlobalOddsRatio,Autoregressive,Nested
import statsmodels.formula.api as sm
from itertools import product
from scipy import stats

np.set_printoptions(formatter={'all': lambda x: "%8.3f" % x},
                    suppress=True)


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


    # The data after recoding as binary
    endog_ex = None
    exog_ex = None
    group_ex = None
    time_ex = None


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
    group = None

    # Group sizes are random within this range
    group_size_range = [4, 11]

    # dparams_est is dparams with scale_inv appended
    def print_dparams(self, dparams_est):
        raise NotImplementedError


class ordinal_simulator(GEE_simulator):

    # The thresholds where the latent continuous process is cut to
    # obtain the categorical values.
    threshold = None


    def true_params(self):
        return np.concatenate((self.thresholds, self.params))


    def starting_values(self):
        return gee_ordinal_starting_values(self.endog,
                                           len(self.params))


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

            exog1 = np.random.normal(size=(gsize, len(self.params)))
            exog.append(exog1)

            lp = np.dot(exog1, self.params)

            z = np.random.uniform(size=gsize)
            z = np.log(z / (1 - z)) + lp
            endog1 = np.array([np.sum(x > self.thresholds) for x in z])
            endog.append(endog1)

        self.exog = np.concatenate(exog, axis=0)
        self.endog = np.concatenate(endog)
        self.time = np.concatenate(time, axis=0)
        self.group = np.concatenate(group)
        self.offset = np.zeros(len(self.endog), dtype=np.float64)


class nominal_simulator(GEE_simulator):


    def starting_values(self):
        return None

    def true_params(self):
        return np.concatenate(self.params[:-1])

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

            exog1 = np.random.normal(size=(gsize, len(self.params[0])))
            exog.append(exog1)

            # Probabilities for each outcome
            prob = [np.exp(np.dot(exog1, p)) for p in self.params]
            prob = np.vstack(prob).T
            prob /= prob.sum(1)[:, None]

            m = len(self.params)
            endog1 = []
            for k in range(gsize):
                pdist = stats.rv_discrete(values=(range(m),
                                                  prob[k,:]))
                endog1.append(pdist.rvs())

            endog.append(np.asarray(endog1))

        self.exog = np.concatenate(exog, axis=0)
        self.endog = np.concatenate(endog).astype(np.int32)
        self.time = np.concatenate(time, axis=0)
        self.group = np.concatenate(group)
        self.offset = np.zeros(len(self.endog), dtype=np.float64)


def check_dparams(gendat):
    """
    Check the estimation of the dependence parameters.
    """

    nrep = 10

    dparams = []
    for j in range(nrep):

        da,va = gendat()

        ga = Gaussian()

        beta = gee_ordinal_starting_values(endog, exog.shape[1])

        md = GEE(da.endog, da.exog, da.group, da.time, ga, va)
        mdf = md.fit(starting_beta = beta)

        scale_inv = 1 / md.estimate_scale()

        dparams.append(np.r_[va.dparams, scale_inv])

    dparams_mean = np.array(sum(dparams) / len(dparams))

    da.print_dparams(dparams_mean)


def check_regression(gendat):
    """
    Check the estimation of the regression coefficients.
    """

    nrep = 20

    params = []
    std_errors = []

    for j in range(nrep):

        da, va, mt = gendat()

        beta = da.starting_values()

        md = GEE(da.endog_ex, da.exog_ex, da.group_ex, da.time_ex,
                 mt, va)
        mdf = md.fit(starting_beta = beta)

        params.append(np.asarray(mdf.params))
        std_errors.append(np.asarray(mdf.standard_errors))

    params = np.array(params)
    eparams = params.mean(0)
    sdparams = params.std(0)
    std_errors = np.array(std_errors)
    std_errors = std_errors.mean(0)
    true_params = da.true_params()

    print "Checking parameter values"
    print "Observed:            ", eparams
    print "Expected:            ", true_params
    print "Absolute difference: ", eparams - true_params
    print "Relative difference: ", \
        (eparams - true_params) / true_params
    print

    print "Checking standard errors"
    print "Observed:            ", sdparams
    print "Expected:            ", std_errors
    print "Absolute difference: ", sdparams - std_errors
    print "Relative difference: ", \
        (sdparams - std_errors) / std_errors
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



def gendat_ordinal():

    os = ordinal_simulator()
    os.params = np.r_[0., 1]
    os.ngroups = 200
    os.thresholds = [1, 0, -1]
    os.simulate()

    os.endog_ex, os.exog_ex, os.group_ex, os.time_ex, \
    os.offset_ex, os.nthresh = \
        gee_setup_multicategorical(os.endog, os.exog, os.group,
                                   os.time, os.offset, "ordinal")

    va = GlobalOddsRatio(4, "ordinal")

    return os, va, Binomial()


def gendat_nominal():

    ns = nominal_simulator()

    # The last component of params must be identically zero
    ns.params = [np.r_[0., 1], np.r_[-1., 0], np.r_[0., 0]]
    ns.ngroups = 200
    ns.simulate()

    ns.endog_ex, ns.exog_ex, ns.group_ex, ns.time_ex, \
    ns.offset_ex, ns.nthresh = \
        gee_setup_multicategorical(ns.endog, ns.exog, ns.group,
                                   ns.time, ns.offset, "nominal")

    va = GlobalOddsRatio(3, "nominal")

    return ns, va, Multinomial(2)


# Loop over data generating models
gendats = [gendat_nominal, gendat_ordinal]

for gendat in gendats:

    #check_dparams(gendat)

    check_regression(gendat)

    #check_constraint(gendat)
