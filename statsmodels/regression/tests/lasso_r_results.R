library(glmnet)
library(R2nparray)

# Run the glmnet lasso (elastic net) on all the test data sets

files = list.files(path="results", pattern="lasso_data_...csv")

rslt = list()
ik = 0

for (file in files) {

    data = read.csv(paste("results", file, sep="/"), header=FALSE)

    endog = data[,1]
    exog = data[,2:dim(data)[2]]
    exog = as.matrix(exog)

    for (alpha in c(0, 0.5, 1)) {

        fit = glmnet(exog, endog, intercept=FALSE, standardize=FALSE, alpha=alpha)

        ii = length(fit$lambda) * c(0.3, 0.5, 0.7)
        ii = round(ii)

        for (q in c(0.3, 0.5, 0.7)) {
            ii = round(q * length(fit$lambda))
            coefs = coef(fit, s=fit$lambda[ii])
            coefs = coefs[2:length(coefs)]
            rname = sprintf("rslt_%d", ik)
            ik = ik + 1
            fix = substr(file, 12, 13)
            fix = as.integer(fix)
            rslt[[rname]] = c(fix, alpha, fit$lambda[ii], coefs)
        }
    }
}

R2nparray(rslt, fname="glmnet_r_results.py")
