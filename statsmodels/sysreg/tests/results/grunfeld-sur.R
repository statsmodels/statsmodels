data <- read.csv('/Users/alexandre/Documents/soc2012/code/statsmodels/datasets/grunfeld/grunfeld.csv')
data <- data[data$firm %in% c('General Motors','Chrysler','General Electric','Westinghouse','US Steel'),]
attach(data)
library('plm')
library('systemfit')
panel <- plm.data(data,c('firm','year'))
formula <- invest ~ value + capital
SUR <- systemfit(formula,method='SUR',data=panel)

