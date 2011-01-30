library(vars)

# German income/investment/consumption data used in Lutkepohl (2005)
## data <- read.table('data/e1.dat', skip=7,
##                   col.names=c('invest', 'income', 'cons'))

data <- read.csv('scikits/statsmodels/datasets/macrodata/macrodata.csv')
names <- colnames(data)
data <- data[c('realgdp', 'realcons', 'realinv')]

reorder.coefs <- function(coefs) {
 n <- dim(coefs)[1]
 # put constant first...
 coefs[c(n, seq(1:(n-1))),]
}

extract.mat <- function(lst, i) {
  sapply(lst, function(x) x[,i])
}

get.coefs <- function(est) {
  t(reorder.coefs(extract.mat(coef(est), 1)))
}

get.stderr <- function(est) {
  reorder.coefs(extract.mat(coef(est), 2))
}

get.results <- function(data, p=1) {
  sel <- VARselect(data)
  est <- VAR(data, p=p)

  list(coefs=get.coefs(est),
       stderr=get.stderr(est),
       obs=est$obs,
       totobs=est$totobs,
       type=est$type,
       crit=t(sel$criteria))
}

k <- dim(data)[2]
result <- get.results(data)

