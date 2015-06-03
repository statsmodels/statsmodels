## This files contains the R code to generate some test for the GAM function ##
## Documentation for MGCV is available at: http://cran.r-project.org/web/packages/mgcv/mgcv.pdf

library('mgcv')

n = 200
x = seq(from = -1, to = 1, length.out = n)
poly = x*x*x

y = 1/(1+ exp(-poly))
y01 = y
mu = mean(y)
y01[y>mu] = 1
y01[y<=mu] = 0
y01 = as.factor(y01)
table(y01)
df = data.frame(x,y01)

gam1 = gam(y01~s(x, k = 10, bs="ps" ) , family = binomial(), data = df, scale=0)
plot(gam1, se=F)

gam1$coefficients

