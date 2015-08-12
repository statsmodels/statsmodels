library('mgcv')

set.seed(2) ## simulate some data... 
dat <- gamSim(1,n=400,dist="normal",scale=2)
dat$x3 = NULL
dat$x1 = NULL
dat$f3 = NULL
dat$f1 = NULL
b <- gam(y~s(x0, bs = "cc")+s(x2, bs = "cc"), data=dat)
summary(b)

plot(b, pages=1)

y_est = predict(b, dat)
partial = predict(b, type = "terms")

res = cbind(dat, partial, y_est)

head(res)
#write.csv(x = res, file = 'Documenti/statsmodels/statsmodels/sandbox/gam_gsoc2015/tests/results/cubic_cyclic_splines_from_mgcv.csv')


### part 2 ###


n = 300
x = seq(-5, 5, length.out = n)
y = x*x*x - x*x + rnorm(n, 0, 10)


g_cc = gam(y~s(x, k = 30, bs = "cc" ))
g_cr = gam(y~s(x, k = 30, bs = "cr" ))
g_cs = gam(y~s(x, k=30, ))

y_est_cc = predict(g_cc)
y_est_cr = predict(g_cr)

plot(x, y)
lines(x, y_est_cc)
lines(x, y_est_cr)
