### GLS Example with Longley Data 
### Done the long way...

d <- read.table('./longley.csv', sep=',', header=T)
attach(d)
m1 <- lm(TOTEMP ~ GNP + POP)
c <- cor(m1$res[-1],m1$res[-16])
sigma <- diag(16)  # diagonal matrix of ones
sigma <- c^abs(row(sigma)-col(sigma))
# row sigma is a matrix of the row index
# col sigma is a matrix of the column index
# this gives a upper-lower triangle with the
# covariance structure of an AR1 process...
sigma_inv <- solve(sigma)     # inverse of sigma
x <- model.matrix(m1)
xPrimexInv <- solve(t(x) %*% sigma_inv %*% x)
beta <- xPrimexInv %*% t(x) %*% sigma_inv %*% TOTEMP
beta
res <- TOTEMP - x %*% beta
sig <- sqrt(sum(res^2)/m1$df)
vc <- sqrt(diag(xPrimexInv))*sig
vc




