## This files contains the R code to generate some test for the GAM function ##
## Documentation for MGCV is available at: http://cran.r-project.org/web/packages/mgcv/mgcv.pdf

library('pracma')
library('mgcv')
#library('gam')
n = 100
x = seq(from = -1, to = 1, length.out = n)

funz = function(x){
  2 * x * x * x - x + rnorm(n = n, mean = 0, sd = .1)
}



y = funz(x)
data = data.frame('y'=y, 'x'=x)

# for mgcv
g = gam(y~s(x, k = 10, bs = "cr"), data = data, scale = 1)

y_gam = predict(g, newdata = data)
plot(x, y_gam, type='l', col='blue', ylim=c(min(y), max(y)))
points(x, y)


y_est = predict(g, newdata = data)
plot(data$x, y_est, ylim=c(min(y), max(y)), type='l')
points(data$x, data$y)

data$y_est = y_est

### Anova test for GLM ###
anova.gam(object = g, freq = T, p.type = -1)


### GAM BINOMIAL ###

mu = mean(y)
ybin = y
ybin[y>mu] = 1
ybin[y<=mu] = 0
data$ybin = ybin
gb = gam(ybin~s(x, k = 10, bs = "cr"), data = data, family = 'binomial', scale=10)

data$ybin_est = predict(gb, newdata = data)

plot(data$x, sigmoid(data$ybin_est), 'l', ylim=c(-1,2))
points(data$x, data$ybin)

#write.csv(data, '/home/donbeo/Documents/statsmodels/statsmodels/sandbox/gam_gsoc2015/tests/prediction_from_mgcv.csv')
