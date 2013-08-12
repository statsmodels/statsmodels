"""
Assesment of Generalized Estimating Equations using simulation.

This script checks ordinal and nominal models for multinomial data.

See the generated file "gee_categorical_simulation_check.txt" for
results.
"""

##!!!! Delete before going to github
import sys
df = "/afs/umich.edu/user/k/s/kshedden/fork/statsmodels/"
sys.path.insert(0, df)


import numpy as np
from statsmodels.genmod.generalized_estimating_equations import GEE,\
    gee_setup_multicategorical, gee_ordinal_starting_values, \
    Multinomial, GEEMargins
from statsmodels.genmod.families import Gaussian,Binomial,Poisson
from statsmodels.genmod.dependence_structures import Exchangeable,\
    Independence,GlobalOddsRatio,Autoregressive,Nested
import statsmodels.formula.api as sm
from itertools import product
from scipy import stats

from gee_gaussian_simulation_check import GEE_simulator



class ordinal_simulator(GEE_simulator):

    # The thresholds where the latent continuous process is cut to
    # obtain the categorical values.
    thresholds = None


    def true_params(self):
        return np.concatenate((self.thresholds, self.params))


    def starting_values(self, nconstraints):
        beta = gee_ordinal_starting_values(self.endog,
                                           len(self.params))
        if nconstraints > 0:
            m = self.exog_ex.shape[1] - nconstraints
            beta = beta[0:m]

        return beta


    def print_dparams(self, dparams_est):
        OUT.write("Odds ratio estimate:   %8.4f\n" % dparams_est[0])
        OUT.write("Odds ratio truth:      %8.4f\n" %
                  self.dparams[0])
        OUT.write("\n")


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


    def starting_values(self, nconstraints):
        return None

    def true_params(self):
        return np.concatenate(self.params[:-1])

    def print_dparams(self, dparams_est):
        OUT.write("Odds ratio estimate:   %8.4f\n" % dparams_est[0])
        OUT.write("Odds ratio truth:      %8.4f\n" % self.dparams[0])
        OUT.write("\n")

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



def gendat_ordinal():

    os = ordinal_simulator()
    os.params = np.r_[0., 1]
    os.ngroups = 200
    os.thresholds = [1, 0, -1]
    os.dparams = [1.,]
    os.simulate()

    os.endog_ex, os.exog_ex, os.group_ex, os.time_ex, \
    os.offset_ex, os.nthresh = \
        gee_setup_multicategorical(os.endog, os.exog, os.group,
                                   os.time, os.offset, "ordinal")

    va = GlobalOddsRatio(4, "ordinal")

    lhs = np.array([[0., 0., 0, 1., 0.], [0., 0, 0, 0, 1]])
    rhs = np.r_[0., 1]

    return os, va, Binomial(), (lhs, rhs)


def gendat_nominal():

    ns = nominal_simulator()

    # The last component of params must be identically zero
    ns.params = [np.r_[0., 1], np.r_[-1., 0], np.r_[0., 0]]
    ns.ngroups = 200
    ns.dparams = [1., ]
    ns.simulate()

    ns.endog_ex, ns.exog_ex, ns.group_ex, ns.time_ex, \
    ns.offset_ex, ns.nthresh = \
        gee_setup_multicategorical(ns.endog, ns.exog, ns.group,
                                   ns.time, ns.offset, "nominal")

    va = GlobalOddsRatio(3, "nominal")

    lhs = np.array([[0., 1., 1, 0],])
    rhs = np.r_[0.,]

    return ns, va, Multinomial(2), (lhs, rhs)


if __name__ == '__main__':

    nrep = 100

    OUT = open("gee_categorical_simulation_check.txt", "w")

    np.set_printoptions(formatter={'all': lambda x: "%8.3f" % x},
                        suppress=True)

    # Loop over data generating models
    gendats = [gendat_nominal, gendat_ordinal]

    for gendat in gendats:

        dparams = []
        params = []
        std_errors = []
        pvalues = []

        for j in range(nrep):

            da, va, mt, constraint = gendat()

            beta = da.starting_values(0)

            md = GEE(da.endog_ex, da.exog_ex, da.group_ex, da.time_ex,
                     mt, va)
            mdf = md.fit(starting_params = beta)

            scale_inv = 1 / md.estimate_scale()

            dparams.append(np.r_[va.dparams, scale_inv])

            params.append(np.asarray(mdf.params))
            std_errors.append(np.asarray(mdf.standard_errors))

            da, va, mt, constraint = gendat()

            beta = da.starting_values(constraint[0].shape[0])

            md = GEE(da.endog_ex, da.exog_ex, da.group_ex, da.time_ex,
                     mt, va, constraint=constraint)
            mdf = md.fit(starting_params = beta)

            score = md.score_test_results
            pvalues.append(score["p-value"])


        dparams_mean = np.array(sum(dparams) / len(dparams))

        OUT.write("Checking dependence parameters:\n")
        da.print_dparams(dparams_mean)

        params = np.array(params)
        eparams = params.mean(0)
        sdparams = params.std(0)
        std_errors = np.array(std_errors)
        std_errors = std_errors.mean(0)
        true_params = da.true_params()

        OUT.write("Checking parameter values:\n")
        OUT.write("Observed:            ")
        OUT.write(np.array_str(eparams) + "\n")
        OUT.write("Expected:            ")
        OUT.write(np.array_str(true_params) + "\n")
        OUT.write("Absolute difference: ")
        OUT.write(np.array_str(eparams - true_params) + "\n")
        OUT.write("Relative difference: ")
        OUT.write(np.array_str((eparams - true_params) / true_params)
                  + "\n")
        OUT.write("\n")

        OUT.write("Checking standard errors:\n")
        OUT.write("Observed:            ")
        OUT.write(np.array_str(sdparams) + "\n")
        OUT.write("Expected:            ")
        OUT.write(np.array_str(std_errors) + "\n")
        OUT.write("Absolute difference: ")
        OUT.write(np.array_str(sdparams - std_errors) + "\n")
        OUT.write("Relative difference: ")
        OUT.write(np.array_str((sdparams - std_errors) / std_errors)
                  + "\n")
        OUT.write("\n")

        OUT.write("Checking constrained estimation:\n")
        OUT.write("Observed   Expected\n")

        pvalues.sort()
        for q in np.arange(0.1, 0.91, 0.1):
            OUT.write("%10.3f %10.3f\n" %
                      (pvalues[int(q*len(pvalues))], q))

        OUT.write("=" * 80 + "\n\n")

    OUT.close()
