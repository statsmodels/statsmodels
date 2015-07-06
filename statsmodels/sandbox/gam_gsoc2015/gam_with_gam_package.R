# GAM using the gam library 

library('gam')

ris_mgcv = read.csv('Documents/statsmodels/statsmodels/sandbox/gam_gsoc2015/tests/results/prediction_from_mgcv.csv')

x = ris_mgcv$x
y = ris_mgcv$y

y_mgcv = ris_mgcv$y_est

data = data.frame('y'=y, 'x'=x)

g = gam(formula = y~s(x = x,  spar = .9), data = data)

# Tr value 
# > g$nl.chisq
# s(x = x, spar = 0.9) 
# 8.261499 


pred = predict(object = g, data = data, se.fit = T)




plot(x, y_mgcv, type = 'l')
points(x, pred$fit, type = 'l')





plot(x, pred$fit, type = 'l')
points(x, pred$fit + 1.96 * pred$se.fit, type='l')
points(x, pred$fit - 1.96 * pred$se.fit, type='l')

points(x, y)

