library("forecast")
macrodata <- read.csv('../../../../datasets/macrodata/macrodata.csv')
x <- ts(macrodata$infl)
far2 <- function(x, h){forecast(Arima(x, order=c(1,0,0), init=c(0.5, 1.5),
                                      optim.control=list(maxit=0)), h=h)}
e <- tsCV(x, far2, h=1, window=30)
write.csv(e, file = "result_tscv.csv")
