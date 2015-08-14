library('mgcv')
set.seed(0)
n = 200
x = seq(-10, 10, length.out = n)
y = 1/(1 + exp(-x*x)) + rnorm(n, 0, .06)
mu = mean(y)
y[y>mu] = 1
y[y<=mu] = 0

g = gam(y~s(x, k = 10), family = 'binomial', link='logit')

y_est = predict(g)

plot(x, 1/(1 + exp(-y_est)), 'l', ylim=c(-.1, 1.1))
points(x, y)

data = data.frame(x, y, y_est)
write.csv(x = data, file = 'Documenti/statsmodels/statsmodels/sandbox/tests/logit_gam_mgcv.csv')
