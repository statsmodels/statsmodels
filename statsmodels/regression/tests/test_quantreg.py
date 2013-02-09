import pandas as pd
from patsy import dmatrices
data = pd.read_csv('http://vincentarelbundock.github.com/Rdatasets/csv/quantreg/engel.csv')
y, X = dmatrices('foodexp ~ income', data, return_type='dataframe')
mod = QuantReg(y, X)
res = mod.fit()
res.summary()

'''
R results

library(quantreg)
data(engel)
fit = rq(foodexp ~ income, tau=.5, engel)
summary(fit)

Call: rq(formula = foodexp ~ income, tau = 0.5, data = engel)
tau: [1] 0.5
Coefficients:
            coefficients lower bd  upper bd
(Intercept)  81.48225     53.25915 114.01156
income        0.56018      0.48702   0.60199
'''
