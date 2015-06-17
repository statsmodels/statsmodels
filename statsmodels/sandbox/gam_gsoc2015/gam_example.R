## This files contains the R code to generate some test for the GAM function ##
## Documentation for MGCV is available at: http://cran.r-project.org/web/packages/mgcv/mgcv.pdf

library('mgcv')
#library('gam')
n = 10000
x = seq(from = -1, to = 1, length.out = n)

funz = function(x){
  2 * x * x * x - x
}

y = funz(x)
data = data.frame('y'=y, 'x'=x)

# for mgcv
g = gam(y~s(x, k = 10, bs = "cr"), data = data, scale = 80)

# for gam
#g = gam(y~s(x, spar = 2), data = data)


y_gam = predict(g, newdata = data)
plot(x, y_gam, type='l', col='blue')
points(x, y)

new_data = data.frame('x'=seq(from = -1, to = 1, length.out = 100))
new_data$y = funz(new_data$x)

y_est = predict(g, newdata = new_data)
plot(new_data$x, y_est, ylim=c(-1, 1), type='l')
points(new_data$x, new_data$y)

new_data$y_est = y_est
write.csv(new_data, '/home/donbeo/Documents/statsmodels/statsmodels/sandbox/gam_gsoc2015/tests/prediction_from_mgcv.csv')
