library(glmnet)
library(survival)
library(R2nparray)

# List containing (sample size, # covariates) for each data set
ixd = list(c(50,2), c(100,5))

res = list()

for (ix in ixd) {

    fname = sprintf("results/survival_data_%d_%d.csv", ix[1], ix[2])
    data = read.table(fname)

    time = data[,1]
    status = data[,2]
    entry = data[,3]
    exog = data[,4:dim(data)[2]]
    exog = as.matrix(exog)
    n = dim(exog)[1]
    p = dim(exog)[2]

    for (k in 1:p) {
        exog[,k] = (exog[,k] - mean(exog[,k])) / sd(exog[,k])
    }

    surv = Surv(time, status)
    md = glmnet(exog, surv, family="cox", lambda=c(0, 0.1))
    #md1 = coxph(surv ~ exog, ties="breslow")
    tag = sprintf("%d_%d", n, p)
    res[[sprintf("coef_%s_0", tag)]] = as.vector(coef(md, s=0))
    res[[sprintf("coef_%s_1", tag)]] = as.vector(coef(md, s=0.1))
}

R2nparray(res, fname="survival_enet_r_results.py")
