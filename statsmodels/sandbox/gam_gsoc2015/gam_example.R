## This files contains the R code to generate some test for the GAM function ##
## Documentation for MGCV is available at: http://cran.r-project.org/web/packages/mgcv/mgcv.pdf

library('mgcv')
#library('gam')

set.seed(0)
n = 100
x = seq(from = -1, to = 1, length.out = n)

funz = function(x){
  2 * x * x * x - x + rnorm(n = n, mean = 0, sd = .1)
}



y = funz(x)
data = data.frame('y'=y, 'x'=x)

# for mgcv
g = gam(y~s(x, k = 10, bs = "cr"), data = data, scale = 1)

pred = predict(g, newdata = data, se.fit = T)

y_gam = pred$fit
se = pred$se.fit

plot(x, y_gam, type='l', col='blue', ylim=c(min(y), max(y)))
points(x, y_gam + se, type = 'l')
points(x, y_gam - se, type = 'l')
points(x, y)


data$y_est = pred$fit
data$y_est_se = pred$se.fit

summary(g)

## DEGREE OF FREEDOM:
# the degrees of freedom are obtained as:
sum(g$edf1[2:10]) #[1] 3.996761

### Anova test for GLM ###
anova.gam(object = g, freq = T, p.type = 0)



### Manually compute prediction intervals ###
# code from http://stackoverflow.com/questions/18909234/coding-a-prediction-interval-from-a-generalized-additive-model-with-a-very-large
Designmat = predict(g,data=data,type="lpmatrix") #design mat is the equivalent of basis
predvar <- diag(Designmat %*% g$Vp %*% t(Designmat))
SE <- sqrt(predvar)

# get a prediction from the designmatrix
lin_pred = as.array(Designmat %*% g$coefficients)
norm(lin_pred - as.numeric(y_est))


#SE2 <- sqrt(predvar+g$sig2)
#tfrac <- qt(0.975, g$df.residual)
#interval = tfrac*SE2






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

#write.csv(data, '/home/donbeo/Documents/statsmodels/statsmodels/sandbox/gam_gsoc2015/tests/results/prediction_from_mgcv.csv')
