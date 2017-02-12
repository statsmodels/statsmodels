library(quantreg)
library(strucchange)

macrodata <- read.csv('/Users/fulton/projects/statsmodels/statsmodels/datasets/macrodata/macrodata.csv')

y <- macrodata$cpi
X <- cbind(rep(1, nrow(macrodata)), macrodata$m1)
rec_beta <- lm.fit.recursive(X, y, int=FALSE)
rec_resid <- recresid(cpi ~ 1 + m1, data=macrodata)
cusum <- efp(cpi ~ 1 + m1, data=macrodata, type=c("Rec-CUSUM"))

output <- data.frame(cbind(t(rec_beta[,3:203]), rec_resid, cusum$process[2:202]))
names(output) <- c('beta1', 'beta2', 'rec_resid', 'cusum')
write.csv(output, '/Users/fulton/projects/statsmodels/statsmodels/regression/tests/results/results_rls_R.csv', row.names=FALSE)
print(cusum$sigma)  # 9.163565
