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

reorder.phi <- function(phis) {
  # Puts things in more proper C order for comparison purposes in Python

  k <- dim(phis)[1]
  n <- dim(phis)[3]

  arr <- array(dim=c(n, k, k))

  for (i in 1:n)
    arr[i,,] <- phis[,,i]

  arr
}

causality.matrix <- function(est) {
  names <- colnames(est$y)
  K <- est$K

  # p-values
  result <- matrix(0, nrow=K, ncol=)
  for (i in 1:K) {
    ## # causes
    ## result[i,1] <- causality(est, cause=names[i])$Granger$p.value

    # caused by others
    result[i,1] <- causality(est, cause=names[-i])$Granger$p.value
  }

  colnames(result) <- c("causedby")

  result
}

get.results <- function(data, p=1) {
  sel <- VARselect(data, p) # do at most p

  est <- VAR(data, p=p)

  K <- ncol(data)

  nirfs <- 5
  orth.irf <- irf(est, n.ahead=nirfs, boot=F)$irf
  irf <- irf(est, n.ahead=nirfs, boot=F, orth=F)$irf

  crit <- t(sel$criteria)
  colnames(crit) <- c('aic', 'hqic', 'sic', 'fpe')

  resid <- resid(est)
  detomega <- det(crossprod(resid) / (est$obs - K * p - 1))

  n.ahead <- 5

  list(coefs=get.coefs(est),
       stderr=get.stderr(est),
       obs=est$obs,
       totobs=est$totobs,
       type=est$type,
       crit=as.list(crit[p,]),
       nirfs=nirfs,
       orthirf=orth.irf,
       irf=irf,
       causality=causality.matrix(est),
       detomega=detomega,
       loglike=as.numeric(logLik(est)),
       nahead=n.ahead,
       phis=Phi(est, n.ahead))
}

k <- dim(data)[2]
result <- get.results(data, p=2)

est = VAR(data, p=2)
