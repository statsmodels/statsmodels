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

endog = dta.pop("SUCCESS")
exog = dta
mod = sm.GLM(endog, exog, formula=formula, family=sm.families.Binomial()).fit()

# try passing a formula object, using user-injected code

def double_it(x):
    return 2*x

# What is the correct entry point for this? Should users be able to inject
# code into default_env or similar? I don't see a way to do this yet using
# the approach I have been using, it should be an argument to Desc
from charlton.builtins import builtins
builtins['double_it'] = double_it

formula = 'SUCCESS ~ double_it(LOWINC) + PERASIAN + PERBLACK + PERHISP + '
formula += 'PCTCHRT '
formula += '+ PCTYRRND + PERMINTE*AVYRSEXP*AVSALK + PERSPENK*PTRATIO*PCTAF'
mod2 = sm.GLM(endog, exog, formula=formula, family=sm.families.Binomial()).fit()
