### GLS Example with Longley Data 
### Done the long way...

d <- read.table('./longley.csv', sep=',', header=T)
attach(d)
m1 <- lm(TOTEMP ~ GNP + POP)
rho <- cor(m1$res[-1],m1$res[-16])
sigma <- diag(16)  # diagonal matrix of ones
sigma <- rho^abs(row(sigma)-col(sigma))
# row sigma is a matrix of the row index
# col sigma is a matrix of the column index
# this gives a upper-lower triangle with the
# covariance structure of an AR1 process...
sigma_inv <- solve(sigma)     # inverse of sigma
x <- model.matrix(m1)
xPrimexInv <- solve(t(x) %*% sigma_inv %*% x)
beta <- xPrimexInv %*% t(x) %*% sigma_inv %*% TOTEMP
beta
# residuals
res <- TOTEMP - x %*% beta
# whitened residuals, not sure if this is right
# xPrimexInv is different than cholsigmainv obviously...
wres = sigma_inv %*% TOTEMP - sigma_inv %*% x %*% beta

sig <- sqrt(sum(res^2)/m1$df)
wsig <- sqrt(sum(wres^2)/m1$df)
wvc <- sqrt(diag(xPrimexInv))*wsig
vc <- sqrt(diag(xPrimexInv))*sig
vc

### Attempt to use a varFunc for GLS
library(nlme)
m1 <- gls(TOTEMP ~ GNP + POP, correlation=corAR1(value=rho, fixed=TRUE))
results <- summary(m1)
bse <- sqrt(diag(vcov(m1)))


