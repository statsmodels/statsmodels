# -*- coding: utf-8 -*-
"""
Copyright (c) 2013 Ana Martinez Pardo <anamartinezpardo@gmail.com>
License: BSD-3 [see LICENSE.txt]

Examples for Conditional logit - Statsmodels
See:
https://github.com/statsmodels/statsmodels/wiki/DCM:-Discrete-choice-models

"""
import pandas as pandas
import numpy as np
from patsy import dmatrices
import time

from dcm_clogit import CLogit
from dcm_clogit import CLogitResults

u"""
Examples
--------
See Greene, Econometric Analysis (5th Edition - 2003: Page 729)
21.7.8. APPLICATION: CONDITIONAL LOGIT MODEL FOR TRAVEL MODE CHOICE

    *four alternative-specific constants (αair, αtrain, αbus, αcar)
        αcar dropped for identification
    *two alternative specific variables (gc, ttme)
        with a generic coefficient (βG, βT)
    *one alternative specific variable (hinc_air)
        with an alternative specific coefficient (γH*di,air)

Ui j = αair*di,air + αtrain*di,train + αbus*di,bus + βG*GCij
        + βT*TTMEij + (γH*di,air)*HINCi + εij

Note: There's a typo on TABLE 21.11. βT isn't -0.19612 is -0.09612
    see TABLE 21.13 to check
"""

print u"""

Example 1. Replicate Greene (2003) results.
TABLE 21.11 Parameter Estimates. Unweighted Sample
    βG       βT      αair        γH         αtrain       αbus
[-0.015501  -0.09612   5.2074  0.01328757  3.86905293  3.16319074]

"""
# TODO: use datasets instead
url = "http://vincentarelbundock.github.io/Rdatasets/csv/Ecdat/ModeChoice.csv"
file_ = "ModeChoice.csv"
import os
if not os.path.exists(file_):
    import urllib
    urllib.urlretrieve(url, "ModeChoice.csv")
df = pandas.read_csv(file_)
pandas.set_printoptions(max_rows=1000, max_columns=20)
df.describe()

nchoices = 4
nobs = 210
choice_index = np.arange(nchoices*nobs) % nchoices
df['hinc_air'] = df['hinc']*(choice_index==0)

f = 'mode  ~ ttme+invc+invt+gc+hinc+psize+hinc_air'
y, X = dmatrices(f, df, return_type='dataframe')
y.head()
X.head()

endog = y.to_records()
endog = endog['mode'].reshape(-1, nchoices)

dta = X.to_records()
dta1 = np.array(dta)

xivar = [['gc', 'ttme', 'Intercept','hinc_air'],
         ['gc', 'ttme', 'Intercept'],
         ['gc', 'ttme', 'Intercept'],
         ['gc', 'ttme' ]]

xi = []

for ii in range(nchoices):
    xi.append(dta1[xivar[ii]][choice_index==ii])

# xifloat = [xx.view(float).reshape(nobs,-1) for xx in xi]
# xifloat = [X[xi_names][choice_index==ii].values for ii, xi_names in enumerate(xivar)]
xifloat = ( [X.ix[choice_index == ii, xi_names].values
            for ii, xi_names in enumerate(xivar)] )

start_time = time.time()
clogit_mod  = CLogit(endog, xifloat, 2)
# Iterations:  ¿ 957 ?
clogit_res =  clogit_mod.fit()
end_time = time.time()
print("the elapsed time to estimate the whole example 1 was %g seconds"
% (end_time - start_time))

exog_names = u'     βG         βT        αair          γH          αtrain   \
    αbus'.split()
print u'     βG         βT        αair          γH          αtrain       αbus'
print clogit_res.params


exog_names = u'G T const_air H const_train const_bus'.split()
print clogit_res.summary(yname='Travel Mode', xname=exog_names)

print CLogitResults(clogit_mod, clogit_res).summary()

hessian = clogit_mod.hessian(clogit_res.params)
s = 'The value of hessian hessian is '+ '\r' + str(hessian)
print s

"""
# R code for example 1

library("mlogit", "TravelMode")
names(TravelMode)<- c("individual", "mode", "choice", "ttme", "invc",
                         "invt", "gc", "hinc", "psize")
TravelMode$hinc_air <- with(TravelMode, hinc * (mode == "air"))
res <- mlogit(choice ~ gc + ttme + hinc_air, data = TravelMode,
            shape = "long", alt.var = "mode", reflevel = "car")
summary(res)
model$hessian       #the hessian of the log-likelihood at convergence

# R results for example 1

Call:
mlogit(formula = choice ~ gc + ttme + hinc_air, data = TravelMode,
    reflevel = "car", shape = "long", alt.var = "mode", method = "nr",
    print.level = 0)

Frequencies of alternatives:
    car     air   train     bus
0.28095 0.27619 0.30000 0.14286

nr method
5 iterations, 0h:0m:0s
g'(-H)^-1g = 0.000234
successive function values within tolerance limits

Coefficients :
                   Estimate Std. Error t-value  Pr(>|t|)
air:(intercept)    5.207433   0.779055  6.6843 2.320e-11 ***
train:(intercept)  3.869036   0.443127  8.7312 < 2.2e-16 ***
bus:(intercept)    3.163190   0.450266  7.0252 2.138e-12 ***
gc                -0.015501   0.004408 -3.5167  0.000437 ***
ttme              -0.096125   0.010440 -9.2075 < 2.2e-16 ***
hinc_air           0.013287   0.010262  1.2947  0.195414
---
Signif. codes:  0 ‘***’ 0.001 ‘**’ 0.01 ‘*’ 0.05 ‘.’ 0.1 ‘ ’ 1

Log-Likelihood: -199.13
McFadden R^2:  0.29825
Likelihood ratio test : chisq = 169.26 (p.value = < 2.22e-16)"""

print  u"""
Summary R results for example 1

air:(intercept) train:(intercept)   bus:(intercept)   gc
5.20743293        3.86903570        3.16319033       -0.01550151
ttme          hinc_air
-0.09612462    0.01328701

model$hessian       #the hessian of the log-likelihood at convergence,
                  air:(intercept) train:(intercept) bus:(intercept)           gc        ttme    hinc_air
air:(intercept)        -25.613627          7.715062        3.883696    192.37152  -1109.9784   -993.2641
train:(intercept)        7.715062        -28.707527        6.766574   -776.60445   -313.7511    284.4266
bus:(intercept)          3.883696          6.766574      -17.978427    -21.70683   -159.8403    144.5267
gc                     192.371522       -776.604449      -21.706830 -75474.20527 -16841.6889   7780.6315
ttme                 -1109.978447       -313.751079     -159.840260 -16841.68892 -91446.9712 -43448.0365
hinc_air              -993.264146        284.426623      144.526736   7780.63148 -43448.0365 -48054.1196
"""

print u"""
Example 2

    *four alternative-specific constants (αair, αtrain, αbus, αcar)
        αcar dropped for identification
    *one alternative specific variables (invc)
        with a generic coefficient (βinvc)

"""
xivar2 = [['invc', 'Intercept'],
          ['invc', 'Intercept'],
          ['invc', 'Intercept'],
          ['invc' ]]

xi2 = []

for ii in range(nchoices):
    xi2.append(dta1[xivar2[ii]][choice_index==ii])

xifloat2 = ( [X.ix[choice_index == ii, xi2_names].values
                for ii, xi2_names in enumerate(xivar2)] )

star_time = time.time()
clogit_mod2 = CLogit(endog, xifloat2, 1)
clogit_res2 = clogit_mod2.fit()
end_time = time.time()
print("the elapsed time to estimate the whole example 2 was %g seconds"
% (end_time - start_time))

print clogit_res2.params
exog_names = u'invc const_air const_train const_bus'.split()
print clogit_res2.summary(yname='Travel Mode', xname=exog_names)
print CLogitResults(clogit_mod2, clogit_res2).summary()

"""
    Call:
mlogit(formula = choice ~ invc, data = TravelMode, reflevel = "car",
    shape = "long", alt.var = "mode", print.level = 2, method = "nr")

Frequencies of alternatives:
    car     air   train     bus
0.28095 0.27619 0.30000 0.14286

nr method
4 iterations, 0h:0m:0s
g'(-H)^-1g = 0.000482
successive function values within tolerance limits

Coefficients :
                    Estimate Std. Error t-value Pr(>|t|)
air:(intercept)    0.8711172  0.3979705  2.1889  0.02860 *
train:(intercept)  0.4825992  0.2455787  1.9652  0.04940 *
bus:(intercept)   -0.5000892  0.2356369 -2.1223  0.03381 *
invc              -0.0138883  0.0055318 -2.5106  0.01205 *
---
Signif. codes:  0 ‘***’ 0.001 ‘**’ 0.01 ‘*’ 0.05 ‘.’ 0.1 ‘ ’ 1

Log-Likelihood: -280.54
McFadden R^2:  0.011351
Likelihood ratio test : chisq = 6.4418 (p.value = 0.011147)
"""

hessian2 = clogit_mod2.hessian(clogit_res2.params)

print hessian2

print u"""
R results

    model2$coefficient
air:(intercept) train:(intercept)   bus:(intercept)              invc
   0.87111722        0.48259924       -0.50008925       -0.01388828

    model2$hessian
                  air:(intercept) train:(intercept) bus:(intercept)          invc
air:(intercept)        -41.485888         17.171385        8.218602   -2022.67713
train:(intercept)       17.171385        -43.402569        8.885814     -81.87671
bus:(intercept)          8.218602          8.885814      -25.618418     455.92294
invc                 -2022.677132        -81.876710      455.922944 -157872.76175
"""
print u"""
Example 3
    *one alternative specific variables (gc)
        with a generic coefficient (βG)
"""

xivar3 = [['gc'],
         ['gc'],
         ['gc'],
         ['gc']]

xi = []

for ii in range(nchoices):
    xi.append(dta1[xivar3[ii]][choice_index==ii])

xifloat3 = ( [X.ix[choice_index == ii, xi_names].values
                for ii, xi_names in enumerate(xivar3)] )
start_time = time.time()
clogit_mod3  = CLogit(endog, xifloat3, 1)
# Iterations:  ¿ 957 ?
clogit_res3 =  clogit_mod3.fit()
end_time = time.time()
print("the elapsed time to estimate the whole example 3 was %g seconds"
% (end_time - start_time))

exog_names = u'βT        αai        αtrain       αbus'.split()
print u'βT        αai        αtrain       αbus'
print clogit_res3.params


exog_names = u'gc'.split()
print clogit_res3.summary(yname='Travel Mode', xname=exog_names)

print CLogitResults(clogit_mod3, clogit_res3).summary()

hessian3 = clogit_mod3.hessian(clogit_res3.params)
s = 'The value of hessian hessian is '+ '\r' + str(hessian3)
print s