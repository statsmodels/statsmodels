library("forecast")
macrodata <- read.csv('../../../../datasets/macrodata/macrodata.csv')

# Here y is not a ts() object, so only is fit with a mean
y <- macrodata$cpi
lambda_d0s1 <- BoxCox.lambda(y, method = 'loglik', lower = -1, upper = 2)

# y is now a ts() object, so it now includes a time trend, but no seasonal
y <- ts(macrodata$cpi)
lambda_d1s1 <- BoxCox.lambda(y, method = 'loglik', lower = -1, upper = 2)

# y is now a ts() object w/ frequency, so it now includes a time trend
# and a seasonal
y <- ts(macrodata$cpi, frequency=12)
lambda_d1s12 <- BoxCox.lambda(y, method = 'loglik', lower = -1, upper = 2)
