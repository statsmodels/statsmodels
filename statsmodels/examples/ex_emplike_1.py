"""
This is a basic tutorial on how to conduct basic empirical likelihood
inference for descriptive statistics.  If matplotlib is installed
it also generates plots.

"""

from __future__ import print_function
import numpy as np
import statsmodels.api as sm
print('Welcome to El')
np.random.seed(634)  # No significance of the seed.
# Let's first generate some univariate data.
univariate = np.random.standard_normal(30)

# Now let's play with it
# Initiate an empirical likelihood descriptive statistics instance
eldescriptive = sm.emplike.DescStat(univariate)

# Empirical likelihood is (typically) a  method of inference,
# not estimation.  Therefore, there is no attribute eldescriptive.mean
# However, we can check the mean:
eldescriptive_mean = eldescriptive.endog.mean()  #.42

#Let's conduct a hypothesis test to see if the mean is 0
print('Hypothesis test results for the mean:')
print(eldescriptive.test_mean(0))


# The first value is is  -2 *log-likelihood ratio, which is distributed
#chi2.  The second value is the p-value.

# Let's see what the variance is:
eldescriptive_var = eldescriptive.endog.var()  # 1.01

#Let's test if the variance is 1:
print('Hypothesis test results for the variance:')
print(eldescriptive.test_var(1))

# Let's test if Skewness and Kurtosis are 0
print('Hypothesis test results for Skewness:')
print(eldescriptive.test_skew(0))
print('Hypothesis test results for the Kurtosis:')
print(eldescriptive.test_kurt(0))
# Note that the skewness and Kurtosis take longer.  This is because
# we have to optimize over the nuisance parameters (mean, variance).

# We can also test for the joint skewness and kurtoses
print(' Joint Skewness-Kurtosis test')
eldescriptive.test_joint_skew_kurt(0, 0)


# Let's try and get some confidence intervals
print('Confidence interval for the mean')
print(eldescriptive.ci_mean())
print('Confidence interval for the variance')
print(eldescriptive.ci_var())
print('Confidence interval for skewness')
print(eldescriptive.ci_skew())
print('Confidence interval for kurtosis')
print(eldescriptive.ci_kurt())


# if matplotlib is installed, we can get a contour plot for the mean
# and variance.
mean_variance_contour = eldescriptive.plot_contour(-.5, 1.2, .2, 2.5, .05, .05)
# This returns a figure instance.  Just type mean_var_contour.show()
# to see the plot.

# Once you close the plot, we can start some multivariate analysis.

x1 = np.random.exponential(2, (30, 1))
x2 = 2 * x1 + np.random.chisquare(4, (30, 1))
mv_data = np.concatenate((x1, x2), axis=1)
mv_elmodel = sm.emplike.DescStat(mv_data)
# For multivariate data, the only methods are mv_test_mean,
# mv mean contour and ci_corr and test_corr.

# Let's test the hypthesis that x1 has a mean of 2 and x2 has a mean of 7
print('Multivaraite mean hypothesis test')
print(mv_elmodel.mv_test_mean(np.array([2, 7])))

# Now let's get the confidence interval for correlation
print('Correlation Coefficient CI')
print(mv_elmodel.ci_corr())
# Note how this took much longer than previous functions.  That is
# because the function is optimizing over 4 nuisance parameters.
# We can also do a hypothesis test for correlation
print('Hypothesis test for correlation')
print(mv_elmodel.test_corr(.7))

# Finally, let's create a contour plot for the means of the data
means_contour = mv_elmodel.mv_mean_contour(1, 3, 6,9, .15,.15, plot_dta=1)
# This also returns a fig so we can type mean_contour.show() to see the figure
# Sometimes, the data is very dispersed and we would like to see the confidence
# intervals without the plotted data.  Let's see the difference when we set
# plot_dta=0

means_contour2 = mv_elmodel.mv_mean_contour(1, 3, 6,9, .05,.05, plot_dta=0)
