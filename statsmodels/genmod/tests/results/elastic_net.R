library(glmnet)
library(R2nparray)

rslt = list()

for (dtype in c("binomial", "poisson")) {

    ik = 0

    data = read.csv(sprintf("enet_%s.csv", dtype))

    endog = data[, 1]
    exog = data[, 2:dim(data)[2]]
    exog = as.matrix(exog)

    for (k in 1:dim(exog)[2]) {
        exog[,k] = exog[,k] - mean(exog[,k])
        exog[,k] = exog[,k] / sd(exog[,k])
    }

    for (alpha in c(0, 0.5, 1)) {

        fit = glmnet(exog, endog, family=dtype, intercept=FALSE,
            standardize=FALSE, alpha=alpha)

        for (q in c(0.3, 0.5, 0.7)) {
            ii = round(q * length(fit$lambda))
            coefs = coef(fit, s=fit$lambda[ii])
            coefs = coefs[2:length(coefs)]
            rname = sprintf("rslt_%s_%d", dtype, ik)
            ik = ik + 1
            rslt[[rname]] = c(alpha, fit$lambda[ii], coefs)
        }
    }
}

R2nparray(rslt, fname="glmnet_r_results.py")
