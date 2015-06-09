## This files contains the R code to generate some test for the GAM function ##
## Documentation for MGCV is available at: http://cran.r-project.org/web/packages/mgcv/mgcv.pdf

library('mgcv')

data = read.csv('Documents/statsmodels/statsmodels/sandbox/gam_gsoc2015/tests/spector_data.csv')
data$X = NULL

g = gam(formula = GRADE~s(TUCE, df = 4, spar = 0.2 ), data = data, family = 'binomial', )

y = predict(object = g, newdata = data, type = 'response')
plot(data$TUCE, y)



library('mgcv')
data = data.frame('x'=c(1,2,3,4,5), 'y'=c(1,0,0,0,1))

g = gam(y~poly(x, k = 4),family = 'binomial', data = data, scale = 0)
plot(g)
  