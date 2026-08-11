library(plyr)
library(vars)
# library(MTS)
library(readstata13)
options(digits=15, scipen=999)

dta <- read.dta13('~/projects/statsmodels/statsmodels/tsa/tests/results/lutkepohl2.dta')

# TODO
# predict(res, dumvar=exog_fcast)
# vars::fevd(res)
# vars::irf(res)

extract_var_output <- function(res, k_trend=0, k_exog=0) {
  s <- summary(res)
  
  # Coefficient vectors
  coeffs <- vector("list", res$K)
  for(i in 1:res$K) {
    coeffs[[i]] <- t(res$varresult[[i]]$coefficients)
  }
  coeffs <- rbind.fill.matrix(coeffs)
  
  # Reorder
  k <- res$K * res$p
  k1 <- k + k_trend
  k2 <- ncol(coeffs)
  
  trend <- c()
  if (k_trend > 0) {
    trend <- t(coeffs[,(k + 1):k1])
  }
  
  var <- t(coeffs[,1:k])
  
  exog <- c()
  if (k_exog > 0) {
    exog <- t(coeffs[,(k1 + 1):k2])
  }
  
  # Error variance
  # Have to compute this by hand since the definition used
  # in the log-likelihood computation is not the same as that
  # given by summary
  resids <- resid(res)
  Sigma <- (t(resids) %*% resids) / res$obs
  U <- chol(Sigma)
  params <- c(trend, var, exog, U[upper.tri(U, diag=TRUE)])
  
  return(list(coeffs=coeffs, params=params, s=s))
}

# "res_basic"
endog <- dta[2:76,c('dln_inv', 'dln_inc', 'dln_consump')]
res <- vars::VAR(endog, p=2, type="none")
out <- extract_var_output(res, k_trend=0)
cat(out$s$logLik)
cat(out$params, sep=', ')
fcasts_basic <- as.data.frame(predict(res)$fcst)[c(1,4,5,8,9,12)]
names(fcasts_basic) <- paste0('basic.fcast.', names(fcasts_basic))
irfs_basic <- as.data.frame(vars::irf(res, ortho=FALSE, boot=FALSE)$irf)
names(irfs_basic) <- paste0('basic.irf.', names(irfs_basic))
irfs_basic_ortho <- as.data.frame(vars::irf(res, ortho=TRUE, boot=FALSE)$irf)
names(irfs_basic_ortho) <- paste0('basic.irf.ortho.', names(irfs_basic_ortho))
irfs_basic_cumu <- as.data.frame(vars::irf(res, cumulative=TRUE, boot=FALSE)$irf)
names(irfs_basic_cumu) <- paste0('basic.irf.cumu.', names(irfs_basic_cumu))
fevd_basic <- vars::fevd(res)
fevd_basic <- as.data.frame(cbind(fevd_basic[[1]], fevd_basic[[2]], fevd_basic[[3]]))
names(fevd_basic)  <- paste0('basic.fevd.', names(fevd_basic))

# "res_c"
endog <- dta[2:76,c('dln_inv', 'dln_inc', 'dln_consump')]
res <- vars::VAR(endog, p=2, type="const")
out <- extract_var_output(res, k_trend=1)
cat(out$s$logLik)
cat(out$params, sep=', ')
fcasts_c <- as.data.frame(predict(res)$fcst)[c(1,4,5,8,9,12)]
names(fcasts_c) <- paste0('c.fcast.', names(fcasts_c))
irfs_c <- as.data.frame(vars::irf(res, ortho=FALSE, boot=FALSE)$irf)
names(irfs_c) <- paste0('c.irf.', names(irfs_c))
irfs_c_ortho <- as.data.frame(vars::irf(res, ortho=TRUE, boot=FALSE)$irf)
names(irfs_c_ortho) <- paste0('c.irf.ortho.', names(irfs_c_ortho))
irfs_c_cumu <- as.data.frame(vars::irf(res, cumulative=TRUE, boot=FALSE)$irf)
names(irfs_c_cumu) <- paste0('c.irf.cumu.', names(irfs_c_cumu))
fevd_c <- vars::fevd(res)
fevd_c <- as.data.frame(cbind(fevd_c[[1]], fevd_c[[2]], fevd_c[[3]]))
names(fevd_c)  <- paste0('c.fevd.', names(fevd_c))

# "res_ct"
endog <- dta[2:76,c('dln_inv', 'dln_inc', 'dln_consump')]
res <- vars::VAR(endog, p=2, type="both")
out <- extract_var_output(res, k_trend=2)
cat(out$s$logLik)
cat(out$params, sep=', ')
fcasts_ct <- as.data.frame(predict(res)$fcst)[c(1,4,5,8,9,12)]
names(fcasts_ct) <- paste0('ct.fcast.', names(fcasts_ct))
irfs_ct <- as.data.frame(vars::irf(res, ortho=FALSE, boot=FALSE)$irf)
names(irfs_ct) <- paste0('ct.irf.', names(irfs_ct))
irfs_ct_ortho <- as.data.frame(vars::irf(res, ortho=TRUE, boot=FALSE)$irf)
names(irfs_ct_ortho) <- paste0('ct.irf.ortho.', names(irfs_ct_ortho))
irfs_ct_cumu <- as.data.frame(vars::irf(res, cumulative=TRUE, boot=FALSE)$irf)
names(irfs_ct_cumu) <- paste0('ct.irf.cumu.', names(irfs_ct_cumu))
fevd_ct <- vars::fevd(res)
fevd_ct <- as.data.frame(cbind(fevd_ct[[1]], fevd_ct[[2]], fevd_ct[[3]]))
names(fevd_ct)  <- paste0('ct.fevd.', names(fevd_ct))

# "res_ct_as_exog0"
# here we start the trend at zero
endog <- dta[2:76,c('dln_inv', 'dln_inc', 'dln_consump')]
exog <- cbind(rep(1, 75), 0:74)
exog_fcast <- cbind(rep(1, 10), 75:84)
# exog <- cbind(rep(1, 75), 1:75, (1:75)^2)
res <- vars::VAR(endog, p=2, type="none", exogen=exog)
out <- extract_var_output(res, k_trend=0, k_exog=2)
cat(out$s$logLik)
cat(out$params, sep=', ')
fcasts_ct_as_exog0 <- as.data.frame(predict(res, dumvar=exog_fcast)$fcst)[c(1,4,5,8,9,12)]
names(fcasts_ct_as_exog0) <- paste0('ct_as_exog0.fcast.', names(fcasts_ct_as_exog0))
irfs_ct_as_exog0 <- as.data.frame(vars::irf(res, ortho=FALSE, boot=FALSE)$irf)
names(irfs_ct_as_exog0) <- paste0('ct_as_exog0.irf.', names(irfs_ct_as_exog0))
irfs_ct_as_exog0_ortho <- as.data.frame(vars::irf(res, ortho=TRUE, boot=FALSE)$irf)
names(irfs_ct_as_exog0_ortho) <- paste0('ct_as_exog0.irf.ortho.', names(irfs_ct_as_exog0_ortho))
irfs_ct_as_exog0_cumu <- as.data.frame(vars::irf(res, cumulative=TRUE, boot=FALSE)$irf)
names(irfs_ct_as_exog0_cumu) <- paste0('ct_as_exog0.irf.cumu.', names(irfs_ct_as_exog0_cumu))
fevd_ct_as_exog0 <- vars::fevd(res)
fevd_ct_as_exog0 <- as.data.frame(cbind(fevd_ct_as_exog0[[1]], fevd_ct_as_exog0[[2]], fevd_ct_as_exog0[[3]]))
names(fevd_ct_as_exog0)  <- paste0('ct_as_exog0.fevd.', names(fevd_ct_as_exog0))

# "res_ctt_as_exog1"
endog <- dta[2:76,c('dln_inv', 'dln_inc', 'dln_consump')]
exog <- cbind(rep(1, 75), 1:75, (1:75)^2)
exog_fcast <- cbind(rep(1, 10), 76:85, (76:85)^2)
res <- vars::VAR(endog, p=2, type='none', exogen=exog)
out <- extract_var_output(res, k_trend=0, k_exog=3)
cat(out$s$logLik)
cat(out$params, sep=', ')
fcasts_ctt_as_exog1 <- as.data.frame(predict(res, dumvar=exog_fcast)$fcst)[c(1,4,5,8,9,12)]
names(fcasts_ctt_as_exog1) <- paste0('ctt_as_exog1.fcast.', names(fcasts_ctt_as_exog1))
irfs_ctt_as_exog1 <- as.data.frame(vars::irf(res, ortho=FALSE, boot=FALSE)$irf)
names(irfs_ctt_as_exog1) <- paste0('ctt_as_exog1.irf.', names(irfs_ctt_as_exog1))
irfs_ctt_as_exog1_ortho <- as.data.frame(vars::irf(res, ortho=TRUE, boot=FALSE)$irf)
names(irfs_ctt_as_exog1_ortho) <- paste0('ctt_as_exog1.irf.ortho.', names(irfs_ctt_as_exog1_ortho))
irfs_ctt_as_exog1_cumu <- as.data.frame(vars::irf(res, cumulative=TRUE, boot=FALSE)$irf)
names(irfs_ctt_as_exog1_cumu) <- paste0('ctt_as_exog1.irf.cumu.', names(irfs_ctt_as_exog1_cumu))
fevd_ctt_as_exog1 <- vars::fevd(res)
fevd_ctt_as_exog1 <- as.data.frame(cbind(fevd_ctt_as_exog1[[1]], fevd_ctt_as_exog1[[2]], fevd_ctt_as_exog1[[3]]))
names(fevd_ctt_as_exog1)  <- paste0('ctt_as_exog1.fevd.', names(fevd_ctt_as_exog1))

# "res_ct_exog"
endog <- dta[2:76,c('dln_inv', 'dln_inc', 'dln_consump')]
exog <- dta[2:76,c('inc')]
exog_fcast <- matrix(dta[77:86,'inc'], nrow=10, ncol=1)
names(exog_fcast) <- c('inc')
res <- vars::VAR(endog, p=2, type='both', exogen=exog)
out <- extract_var_output(res, k_trend=2, k_exog=1)
cat(out$s$logLik)
cat(out$params, sep=', ')
fcasts_ct_exog <- as.data.frame(predict(res, dumvar=exog_fcast)$fcst)[c(1,4,5,8,9,12)]
names(fcasts_ct_exog) <- paste0('ct_exog.fcast.', names(fcasts_ct_exog))
irfs_ct_exog <- as.data.frame(vars::irf(res, ortho=FALSE, boot=FALSE)$irf)
names(irfs_ct_exog) <- paste0('ct_exog.irf.', names(irfs_ct_exog))
irfs_ct_exog_ortho <- as.data.frame(vars::irf(res, ortho=TRUE, boot=FALSE)$irf)
names(irfs_ct_exog_ortho) <- paste0('ct_exog.irf.ortho.', names(irfs_ct_exog_ortho))
irfs_ct_exog_cumu <- as.data.frame(vars::irf(res, cumulative=TRUE, boot=FALSE)$irf)
names(irfs_ct_exog_cumu) <- paste0('ct_exog.irf.cumu.', names(irfs_ct_exog_cumu))
fevd_ct_exog <- vars::fevd(res)
fevd_ct_exog <- as.data.frame(cbind(fevd_ct_exog[[1]], fevd_ct_exog[[2]], fevd_ct_exog[[3]]))
names(fevd_ct_exog)  <- paste0('ct_exog.fevd.', names(fevd_ct_exog))


# "res_c_2exog"
endog <- dta[2:76,c('dln_inv', 'dln_inc', 'dln_consump')]
exog <- dta[2:76,c('inc', 'inv')]
exog_fcast <- dta[77:86,c('inc', 'inv')]
res <- vars::VAR(endog, p=2, type='const', exogen=exog)
out <- extract_var_output(res, k_trend=1, k_exog=2)
cat(out$s$logLik)
cat(out$params, sep=', ')
fcasts_c_2exog <- as.data.frame(predict(res, dumvar=exog_fcast)$fcst)[c(1,4,5,8,9,12)]
names(fcasts_c_2exog) <- paste0('c_2exog.fcast.', names(fcasts_c_2exog))
irfs_c_2exog <- as.data.frame(vars::irf(res, ortho=FALSE, boot=FALSE)$irf)
names(irfs_c_2exog) <- paste0('c_2exog.irf.', names(irfs_c_2exog))
irfs_c_2exog_ortho <- as.data.frame(vars::irf(res, ortho=TRUE, boot=FALSE)$irf)
names(irfs_c_2exog_ortho) <- paste0('c_2exog.irf.ortho.', names(irfs_c_2exog_ortho))
irfs_c_2exog_cumu <- as.data.frame(vars::irf(res, cumulative=TRUE, boot=FALSE)$irf)
names(irfs_c_2exog_cumu) <- paste0('c_2exog.irf.cumu.', names(irfs_c_2exog_cumu))
fevd_c_2exog <- vars::fevd(res)
fevd_c_2exog <- as.data.frame(cbind(fevd_c_2exog[[1]], fevd_c_2exog[[2]], fevd_c_2exog[[3]]))
names(fevd_c_2exog)  <- paste0('c_2exog.fevd.', names(fevd_c_2exog))

cbind.fill <- function(...){
  nm <- list(...) 
  nm <- lapply(nm, as.matrix)
  n <- max(sapply(nm, nrow)) 
  do.call(cbind, lapply(nm, function (x) 
    rbind(x, matrix(, n-nrow(x), ncol(x))))) 
}

# Combine forecasts, irfs, etc
output <- cbind.fill(
  fcasts_basic, fcasts_c, fcasts_ct, fcasts_ct_as_exog0,
  fcasts_ctt_as_exog1, fcasts_ct_exog, fcasts_c_2exog,
  irfs_basic, irfs_c, irfs_ct, irfs_ct_as_exog0,
  irfs_ctt_as_exog1, irfs_ct_exog, irfs_c_2exog,
  irfs_basic_ortho, irfs_c_ortho, irfs_ct_ortho, irfs_ct_as_exog0_ortho,
  irfs_ctt_as_exog1_ortho, irfs_ct_exog_ortho, irfs_c_2exog_ortho,
  irfs_basic_cumu, irfs_c_cumu, irfs_ct_cumu, irfs_ct_as_exog0_cumu,
  irfs_ctt_as_exog1_cumu, irfs_ct_exog_cumu, irfs_c_2exog_cumu,
  fevd_basic, fevd_c, fevd_ct, fevd_ct_as_exog0,
  fevd_ctt_as_exog1, fevd_ct_exog, fevd_c_2exog)
write.csv(output, '~/projects/statsmodels/statsmodels/tsa/statespace/tests/results/results_var_R_output.csv')
