dta <- read.csv("/home/skipper/statsmodels/statsmodels-skipper/statsmodels/datasets/macrodata/macrodata.csv")

cpi <- dta$cpi

/* this automatically suppresses the constant */
mod111 <- arima(cpi, order=c(1,1,1), method="CSS")
/*you can use xreg=1:length(cpi)*/

dcpi <- diff(cpi)

mod111 <- arima(dcpi, order=c(1,0,1), method="CSS")
bse <- sqrt(diag(mod111$var.coef))
tvalues <- mod111$coef / bse
pvalues <- (1 - pt(abs(tvalues), 198)) * 2

/* use starting values from X-ARIMA */
mod112 <- arima(dcpi, order=c(1,0,2), method="CSS", init=c(-0.692425, 1.07366, 0.172024, 0.905322))
bse <- sqrt(diag(mod112$var.coef))
tvalues <- mod112$coef / bse
pvalues <- (1 - pt(abs(tvalues), 198)) * 2

