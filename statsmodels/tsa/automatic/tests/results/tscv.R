library("forecast")
macrodata <- read.csv('../../../../datasets/macrodata/macrodata.csv')
x <- ts(macrodata$infl)
y <- ts(macrodata$cpi)
far2 <- function(x, h){forecast(Arima(x, order=c(1,0,0), init=c(0.5, 1.5),
                                      optim.control=list(maxit=0)), h=h)}
e <- tsCV(x, far2, h=1, window=30)
write.csv(e, file = "result_tscv.csv")

e <- tsCV(y, far2, h=1, window=30)
write.csv(e, file = "result_tscv_cpi.csv")

data_new <- read.csv('../CPIAPPNS.csv')
y<-ts(data_new$CPIAPPNS)
e <- tsCV(y, far2, h=1, window=30)
write.csv(e, file = "result_tscv_CPIAPPNS.csv")
