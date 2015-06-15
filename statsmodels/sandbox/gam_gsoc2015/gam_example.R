## This files contains the R code to generate some test for the GAM function ##
## Documentation for MGCV is available at: http://cran.r-project.org/web/packages/mgcv/mgcv.pdf

#library('mgcv')
library('gam')
n = 200
x = seq(from = 0, to = 1, length.out = n)
y = x * x * x - x + rnorm(n = n,mean = 0, sd = 0.01)
data = data.frame('y'=y, 'x'=x)

degree = 100
g = gam(y~poly(x = x, degree = degree), data = data)
pr = glm(y~poly(x = x, degree = degree), data = data)

y_gam = predict(g, newdata = data)
y_glm = predict(pr, newdata = data)
plot(y_glm, type = 'l', col='red')
plot(y_gam, type='l', col='blue')

points(y)
  