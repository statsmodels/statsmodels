library(survival)

ixd = list(c(20,1), c(50,1), c(50,2), c(100,5), c(1000,10))

for (ix in ixd) {
    fname = sprintf("survival_%d_%d.csv", ix[1], ix[2])
    data = read.table(fname)

    time = data[,1]
    status = data[,2]
    entry = data[,3]
    exog = data[,4:dim(data)[2]]
    exog = as.matrix(exog)
    n = dim(exog)[1]

    # Needs to match the kronecker statement in test_phreg.py
    strata = kronecker(seq(5), array(1, n/5))

    for (ties in c("breslow", "efron")) {

        ti = substr(ties, 1, 3)

        fname1 = sub(".csv", "", fname)

        # Base model
        surv = Surv(time, status)
        md = coxph(surv ~ exog, ties=ties)
        fname2 = sprintf("%s_%s.txt", fname1, ti)
        output = capture.output(summary(md))
        cat(output, sep="\n", file=fname2)

        # With entry time
        surv = Surv(entry, time, status)
        md = coxph(surv ~ exog, ties=ties)
        fname2 = sprintf("%s_et_%s.txt", fname1, ti)
        output = capture.output(summary(md))
        cat(output, sep="\n", file=fname2)

        # With strata
        surv = Surv(time, status)
        md = coxph(surv ~ exog + strata(strata), ties=ties)
        fname2 = sprintf("%s_st_%s.txt", fname1, ti)
        output = capture.output(summary(md))
        cat(output, sep="\n", file=fname2)

        # With entry time and strata
        surv = Surv(entry, time, status)
        md = coxph(surv ~ exog + strata(strata), ties=ties)
        fname2 = sprintf("%s_et_st_%s.txt", fname1, ti)
        output = capture.output(summary(md))
        cat(output, sep="\n", file=fname2)
    }
}
