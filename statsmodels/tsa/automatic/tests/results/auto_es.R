library("forecast")
macrodata <- read.csv('../../../../datasets/macrodata/macrodata.csv')
y <- ts(macrodata$cpi)
fit <- ets(y)
