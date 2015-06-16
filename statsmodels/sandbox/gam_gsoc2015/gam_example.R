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

s1 = s(x, k = 10, bs = "ps")
g = gam(y~s(x, k = 10, bs = "cr"), data = data, scale = 80)

y_gam = predict(g, newdata = data)
plot(x, y_gam, type='l', col='blue')
points(x, y)

new_data = data.frame('x'=seq(from = -1, to = 1, length.out = 100))
new_data$y = funz(new_data$x)

y_est = predict(g, newdata = new_data)
plot(new_data$x, y_est)
points(new_data$x, new_data$y, type='l')
