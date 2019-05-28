library(glmnet)
library(R2nparray)

# Run the glmnet lasso (elastic net) on all the test data sets

data = read.csv("lasso_data.csv", header=FALSE)

ik = 0
rslt = list()

for (n in c(100, 200, 300)) {
    for (p in c(2, 3, 5)) {

        endog = data[1:n, 1]
        exog = data[1:n, 2:(p+1)]
        exog = as.matrix(exog)

        endog = (endog - mean(endog)) / sd(endog)
        for (k in 1:p) {
            exog[,k] = exog[,k] - mean(exog[,k])
            exog[,k] = exog[,k] / sd(exog[,k])
        }

        for (alpha in c(0, 0.5, 1)) {

            fit = glmnet(exog, endog, intercept=FALSE, standardize=FALSE, alpha=alpha)

            for (q in c(0.3, 0.5, 0.7)) {
                ii = round(q * length(fit$lambda))
                coefs = coef(fit, s=fit$lambda[ii])
                coefs = coefs[2:length(coefs)]
                rname = sprintf("rslt_%d", ik)
                ik = ik + 1
                rslt[[rname]] = c(n, p, alpha, fit$lambda[ii], coefs)
            }
        }
    }
}

R2nparray(rslt, fname="glmnet_r_results.py")
