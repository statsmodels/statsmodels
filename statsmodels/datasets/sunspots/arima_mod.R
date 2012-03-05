dta <- read.csv('./sunspots.csv')
attach(dta)
arma_mod <- arima(SUNACTIVITY, order=c(9,0,0), xreg=rep(1,309), include.mean=FALSE)
