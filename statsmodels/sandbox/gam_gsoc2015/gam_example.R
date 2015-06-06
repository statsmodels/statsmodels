## This files contains the R code to generate some test for the GAM function ##
## Documentation for MGCV is available at: http://cran.r-project.org/web/packages/mgcv/mgcv.pdf

library('mgcv')

n = 200
x = seq(from = -10, to = 10, length.out = n)
poly = x*x

y = 1/(1+ exp(-poly)) #+ rnorm(n = n, mean = 0, sd = 0.01)
y01 = y
mu = mean(y)
y01[y>mu] = 1
y01[y<=mu] = 0
y01 = as.factor(y01)
table(y01)
df = data.frame(x,y01)

gam1 = gam(y01~s(x, k = 100, bs="ps", fx = F) , family = binomial(), data = df, scale=0)
plot(gam1, se=F)
points(x, poly, col='red')
gam1$coefficients


df_new = data.frame(x = seq(-10, 10, length.out = 10))
y_est = predict(gam1, newdata = df_new )
plot(y_est)
