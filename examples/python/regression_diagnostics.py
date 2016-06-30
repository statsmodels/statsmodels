
## Regression diagnostics

# This example file shows how to use a few of the ``statsmodels`` regression diagnostic tests in a real-life context. You can learn about more tests and find out more information abou the tests here on the [Regression Diagnostics page.](http://www.statsmodels.org/stable/diagnostic.html)
# 
# Note that most of the tests described here only return a tuple of numbers, without any annotation. A full description of outputs is always included in the docstring and in the online ``statsmodels`` documentation. For presentation purposes, we use the ``zip(name,test)`` construct to pretty-print(short descriptions in the examples below.

# ## Estimate a regression model

from __future__ import print_function
from statsmodels.compat import lzip
import statsmodels
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
import statsmodels.stats.api as sms

# Load data
url = 'http://vincentarelbundock.github.io/Rdatasets/csv/HistData/Guerry.csv'
dat = pd.read_csv(url)

# Fit regression model (using the natural log of one of the regressaors)
results = smf.ols('Lottery ~ Literacy + np.log(Pop1831)', data=dat).fit()

# Inspect the results
print(results.summary())


# ## Normality of the residuals

# Jarque-Bera test:

name = ['Jarque-Bera', 'Chi^2 two-tail prob.', 'Skew', 'Kurtosis']
test = sms.jarque_bera(results.resid)
lzip(name, test)


# Omni test:

name = ['Chi^2', 'Two-tail probability']
test = sms.omni_normtest(results.resid)
lzip(name, test)


# ## Influence tests
# 
# Once created, an object of class ``OLSInfluence`` holds attributes and methods that allow users to assess the influence of each observation. For example, we can compute and extract the first few rows of DFbetas by:

from statsmodels.stats.outliers_influence import OLSInfluence
test_class = OLSInfluence(results)
test_class.dfbetas[:5,:]


# Explore other options by typing ``dir(influence_test)``
# 
# Useful information on leverage can also be plotted:

from statsmodels.graphics.regressionplots import plot_leverage_resid2
print(plot_leverage_resid2(results))


# Other plotting options can be found on the [Graphics page.](http://www.statsmodels.org/stable/graphics.html)

# ## Multicollinearity
# 
# Condition number:

np.linalg.cond(results.model.exog)


# ## Heteroskedasticity tests
# 
# Breush-Pagan test:

name = ['Lagrange multiplier statistic', 'p-value', 
        'f-value', 'f p-value']
test = sms.het_breushpagan(results.resid, results.model.exog)
lzip(name, test)


# Goldfeld-Quandt test

name = ['F statistic', 'p-value']
test = sms.het_goldfeldquandt(results.resid, results.model.exog)
lzip(name, test)


# ## Linearity
# 
# Harvey-Collier multiplier test for Null hypothesis that the linear specification is correct:

name = ['t value', 'p value']
test = sms.linear_harvey_collier(results)
lzip(name, test)

