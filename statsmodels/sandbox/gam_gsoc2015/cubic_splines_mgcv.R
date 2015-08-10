library('mgcv')

n = 500
x1 = runif(n, -10, 10)
y = sin(x1)/x1 + rnorm(n, 0, .1)
y = y - mean(y)

plot(x1, y)

dat = data.frame(x1, y)


b <- gam(y~s(x1, bs = "cc", k = 10 )-1, data=dat)
summary(b)

y_est = predict(b, newdata = dat)

plot(x1, y_est, ylim = c(-1, 1.2))
points(x1, y, col='red')


x2 = runif(n, -1, 1)

y2 = y + x2**2
y2 = y2 - mean(y2)

plot(x1, y2)
dat2 = cbind(dat, x2, y2)

b2 = gam(y2~s(x1, bs="cc", k=10) + s(x2, bs="cc", k=10) - 1, data = dat2)

p = plot(b2)

partial1_x = p[[1]]$x
partial1_est = p[[1]]$fit
partial1_se = p[[1]]$se

partial2_x = p[[2]]$x
partial2_est = p[[2]]$fit
partial2_se = p[[2]]$se

res = data.frame(x1, x2, y, y_est, partial1_x, partial1_se, partial1_est, 
                 partial2_x, partial2_se, partial2_est)

write.csv(x = res, file = 'Documenti/statsmodels/statsmodels/sandbox/gam_gsoc2015/tests/results/cubic_splines_from_mgcv.csv')
