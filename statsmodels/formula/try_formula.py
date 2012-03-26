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
