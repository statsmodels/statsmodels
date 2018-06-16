library("forecast")
macrodata <- read.csv('/Users/abhijeet/gsoc/dev/statsmodels/statsmodels/datasets/macrodata/macrodata.csv')
lambda <- BoxCox.lambda(macrodata['cpi'], method = 'loglik', lower = -1, upper = 2)