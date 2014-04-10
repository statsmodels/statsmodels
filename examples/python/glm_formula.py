
## Generalized Linear Models (Formula)

# This notebook illustrates how you can use R-style formulas to fit Generalized Linear Models.
# 
# To begin, we load the ``Star98`` dataset and we construct a formula and pre-process the data:

from __future__ import print_function
import statsmodels.api as sm
import statsmodels.formula.api as smf
star98 = sm.datasets.star98.load_pandas().data
formula = 'SUCCESS ~ LOWINC + PERASIAN + PERBLACK + PERHISP + PCTCHRT +            PCTYRRND + PERMINTE*AVYRSEXP*AVSALK + PERSPENK*PTRATIO*PCTAF'
dta = star98[['NABOVE', 'NBELOW', 'LOWINC', 'PERASIAN', 'PERBLACK', 'PERHISP',
              'PCTCHRT', 'PCTYRRND', 'PERMINTE', 'AVYRSEXP', 'AVSALK',
              'PERSPENK', 'PTRATIO', 'PCTAF']]
endog = dta['NABOVE'] / (dta['NABOVE'] + dta.pop('NBELOW'))
del dta['NABOVE']
dta['SUCCESS'] = endog


# Then, we fit the GLM model:

mod1 = smf.glm(formula=formula, data=dta, family=sm.families.Binomial()).fit()
mod1.summary()


# Finally, we define a function to operate customized data transformation using the formula framework:

def double_it(x):
    return 2 * x
formula = 'SUCCESS ~ double_it(LOWINC) + PERASIAN + PERBLACK + PERHISP + PCTCHRT +            PCTYRRND + PERMINTE*AVYRSEXP*AVSALK + PERSPENK*PTRATIO*PCTAF'
mod2 = smf.glm(formula=formula, data=dta, family=sm.families.Binomial()).fit()
mod2.summary()


# As expected, the coefficient for ``double_it(LOWINC)`` in the second model is half the size of the ``LOWINC`` coefficient from the first model:

print(mod1.params[1])
print(mod2.params[1] * 2)

