"""GLM Formula Example
"""

import statsmodels.api as sm
import numpy as np

star98 = sm.datasets.star98.load_pandas().data

formula = 'SUCCESS ~ LOWINC + PERASIAN + PERBLACK + PERHISP + PCTCHRT '
formula += '+ PCTYRRND + PERMINTE*AVYRSEXP*AVSALK + PERSPENK*PTRATIO*PCTAF'

dta = star98[["NABOVE", "NBELOW", "LOWINC", "PERASIAN", "PERBLACK", "PERHISP",
              "PCTCHRT", "PCTYRRND", "PERMINTE", "AVYRSEXP", "AVSALK",
              "PERSPENK", "PTRATIO", "PCTAF"]]

endog = dta["NABOVE"]/(dta["NABOVE"] + dta.pop("NBELOW"))
del dta["NABOVE"]
dta["SUCCESS"] = endog

mod = sm.GLM.from_formula(formula=formula, df=dta,
                          family=sm.families.Binomial()).fit()

# try passing a formula object, using arbitrary user-injected code

def double_it(x):
    return 2*x

formula = 'SUCCESS ~ double_it(LOWINC) + PERASIAN + PERBLACK + PERHISP + '
formula += 'PCTCHRT '
formula += '+ PCTYRRND + PERMINTE*AVYRSEXP*AVSALK + PERSPENK*PTRATIO*PCTAF'
mod2 = sm.GLM.from_formula(formula=formula, df=dta,
                           family=sm.families.Binomial()).fit()
