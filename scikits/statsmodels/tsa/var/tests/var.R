library(vars)

data <- read.csv('/home/wesm/code/statsmodels/scikits/statsmodels/datasets/macrodata/macrodata.csv')
names <- colnames(data)
data <- log(data[c('realgdp', 'realcons', 'realinv')])
data <- sapply(data, diff)

reorder.coefs <- function(coefs) {
 n <- dim(coefs)[1]
 # put constant first...
 coefs[c(n, seq(1:(n-1))),]
}

extract.mat <- function(lst, i) {
  sapply(lst, function(x) x[,i])
}

get.coefs <- function(est) {
  reorder.coefs(extract.mat(coef(est), 1))
}

get.stderr <- function(est) {
  reorder.coefs(extract.mat(coef(est), 2))
}

get.results <- function(data, p=1) {
  sel <- VARselect(data, p) # do at most p

  est <- VAR(data, p=p)

  nirfs <- 5
  orth.irf <- irf(est, n.ahead=nirfs, boot=F)$irf
  irf <- irf(est, n.ahead=nirfs, boot=F, orth=F)$irf

  crit <- t(sel$criteria)
  colnames(crit) <- c('aic', 'hqic', 'sic', 'fpe')

  list(coefs=get.coefs(est),
       stderr=get.stderr(est),
       obs=est$obs,
       totobs=est$totobs,
       type=est$type,
       crit=as.list(crit[p,]),
       nirfs=nirfs,
       orthirf=orth.irf,
       irf=irf)
}

k <- dim(data)[2]
result <- get.results(data, p=2)

est = VAR(data, p=2)
