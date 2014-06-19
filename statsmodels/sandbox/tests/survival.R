library(survival)
library(R2nparray)

ixd = list(c(20,1), c(50,1), c(50,2), c(100,5), c(1000,10))

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

    # Needs to match the kronecker statement in test_phreg.py
    strata = kronecker(seq(5), array(1, n/5))

    for (ties in c("breslow", "efron")) {

        ti = substr(ties, 1, 3)

        # Base model
        surv = Surv(time, status)
        md = coxph(surv ~ exog, ties=ties)
        tag = sprintf("%d_%d_%s", n, p, ti)
        res[[sprintf("coef_%s", tag)]] = md$coef
        res[[sprintf("se_%s", tag)]] = sqrt(diag(md$var))
        #bhaz = basehaz(md)
        bhaz = survfit(md, type="aalen")
        #bhaz = survfit(md, type="efron")
        res[[sprintf("time_%s", tag)]] = bhaz$time
        res[[sprintf("hazard_%s", tag)]] = -log(bhaz$surv)

        # With entry time
        surv = Surv(entry, time, status)
        md = coxph(surv ~ exog, ties=ties)
        tag = sprintf("%d_%d_et_%s", n, p, ti)
        res[[sprintf("coef_%s", tag)]] = md$coef
        res[[sprintf("se_%s", tag)]] = sqrt(diag(md$var))
        res[[sprintf("time_%s", tag)]] = c(0)
        res[[sprintf("hazard_%s", tag)]] = c(0)

        # With strata
        surv = Surv(time, status)
        md = coxph(surv ~ exog + strata(strata), ties=ties)
        tag = sprintf("%d_%d_st_%s", n, p, ti)
        res[[sprintf("coef_%s", tag)]] = md$coef
        res[[sprintf("se_%s", tag)]] = sqrt(diag(md$var))
        res[[sprintf("time_%s", tag)]] = c(0)
        res[[sprintf("hazard_%s", tag)]] = c(0)

        # With entry time and strata
        surv = Surv(entry, time, status)
        md = coxph(surv ~ exog + strata(strata), ties=ties)
        tag = sprintf("%d_%d_et_st_%s", n, p, ti)
        res[[sprintf("coef_%s", tag)]] = md$coef
        res[[sprintf("se_%s", tag)]] = sqrt(diag(md$var))
        res[[sprintf("time_%s", tag)]] = c(0)
        res[[sprintf("hazard_%s", tag)]] = c(0)
    }
}

R2nparray(res, fname="survival_r_results.py")
