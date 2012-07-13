"""

Here is the midterm report and a test script for my empirical likelihood
GSOC project.  Please see my July 11 blog post at
http://landopystat.blogspot.com/ for a narrative of my GSOC experience.

For this script to work, install statsmodels from the emplike_reg branch.

This script can be ran feature by feature or all at once. However, plt.show()
is at the end of the file so to see the plots with the rest of the script
commented out, make sure to type plt.show().

"""
import numpy as np
from matplotlib import pyplot as plt
import statsmodels.api as sm
print 'Welcome to El'
np.random.seed(634) # No significance of the seed, just my favorite number
# Let's first generate some univariate data.
univariate = np.random.standard_normal(30)

# Now let's play with it
# Initiate an empirical likelihood descriptive statistics instance
eldescriptive=sm.emplike.DescStat(univariate)

# Empirical likelihood is (typically) a  method of inference,
# not estimation.  Therefore, there is no attribute eldescriptive.mean
# However, we can check the mean:
eldescriptive_mean= eldescriptive.endog.mean() #.42

#Let's conduct a hypothesis test to see if the mean is 0
print 'Hypothesis test results for the mean:'
print eldescriptive.hy_test_mean(0)


# The first value is the pvalue and the second is  -2 *log-likelihood ratio,
# which is distributed chi2.

# Let's see what the variance is:
eldescriptive_var = eldescriptive.endog.var() # 1.01

#Let's test if the variance is 1:
print 'Hypothesis test results for the variance:'
print eldescriptive.hy_test_var(1)

# Let's test if Skewness and Kurtosis are 0
print 'Hypothesis test results for Skewness:'
print eldescriptive.hy_test_skew(0)
print 'Hypothesis test results for the Kurtosis:'
print eldescriptive.hy_test_kurt(0)

# Note that the skewness and Kurtosis take longer.  This is because
# we have to optimizae over the nuisance parameters (mean, variance).

# Let's try and get some confidence intervals
print 'Confidence interval for the mean'
print eldescriptive.ci_mean()
print 'Confidence interval for the variance'
print eldescriptive.ci_var()
print 'Confidence interval for skewness'
print eldescriptive.ci_skew()
print 'Confidence interval for kurtosis'
print eldescriptive.ci_kurt()
# Note in 'Finding Ci for the variance' output.
# This is because we need to optimize over the variance and we need to
# set limits.  However, if we have a general idea, we cans et those and
# speed up the optimization.
print eldescriptive.ci_kurt(var_min=.7, var_max=1.5, mu_min=.2, mu_max=.7)
# In this case, since it was fast to find the CI for the variance, adding
# in the optional parameters didn't help much.  With more difficult data
# to analyze, this can be very helpful.

# if matplotlib is installed, we can get a contour plot for the mean
# and variance.
plt.figure(1)
eldescriptive.mean_var_contour(-.5, 1.2, .2, 2.5, .1, .1)

# The mean is on the X acis and Variance is on the Y.  I jsut realized
# that either those should be labeled or mentioned in the documentation

# Once you close the plot, we can start some multivariate analysis.

x1 = np.random.exponential(2, (30,1))
x2 = 2 * x1 +np.random.chisquare(4, (30,1))
mv_data = np.concatenate((x1, x2), axis=1)
mv_elmodel = sm.emplike.DescStat(mv_data)
# For multivariate data, the only attributes are mv_hy_test_mean,
# mv mean contour and ci_corr and hy_test_corr.  Maybe the multivariate
# functions should be their own class.

# Let test the hypthesis that x1 has a mean of 2 and x2 has a mean of 7
'Multivaraite mean hypothesis test'
print mv_elmodel.mv_hy_test_mean(np.array([2, 7]))

# Now let's get the confidence interval for correlation
print 'Correlation Coefficient CI'
print mv_elmodel.ci_corr()
# Note how this took much longer than previous functions.  That is
# because the function is optimizing over 4 nuisance parameters.
# We can also do a hypothesis test for correlation
print 'Hypothesis test for correlation'
print mv_elmodel.hy_test_corr(.7)

#Finally, let's create a contour plot for the means of the data
plt.figure(2)
mv_elmodel.mv_mean_contour(0,4, 4,12, .15,.15, plot_dta=1)



# Again, I suppose I need labels or something in the documentation to
# let the end user know which axis is which.

plt.show()
