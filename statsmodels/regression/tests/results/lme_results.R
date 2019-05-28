library(lme4)
library(R2nparray)

files = list.files(path="results", pattern="lme...csv")

rslt = list()

# Check the code with a sequence of data files with different
# dimensions and random effects structures.
for (file in files) {

    # Fit a model with independent random effects (irf=TRUE) or
    # dependent random effects (irf=FALSE).
    for (irf in c(FALSE, TRUE)) {

        if (irf) {
            rf = "irf"
        } else {
            rf = "drf"
        }

        # Use REML or ML estimation
        for (reml in c(FALSE, TRUE)) {

            if (reml) {
                meth = "reml"
            } else {
                meth = "ml"
            }

            data = read.csv(paste("results", file, sep="/"))

            exog_fe_ix = grep("exog_fe", names(data))
            exog_re_ix = grep("exog_re", names(data))

            # No need to check independent random effects when there is
            # only one of them.
            if (irf & (length(exog_re_ix) == 1)) {
                next
            }

            pr = length(exog_re_ix)
            fml_fe = paste(names(data)[exog_fe_ix], collapse="+")

            if (irf) {
                st = NULL
                for (ik in exog_re_ix) {
                    st[length(st) + 1] = sprintf("(0 + %s | groups)", names(data)[ik])
                }
                fml_re = paste(st, collapse="+")

                fml = sprintf("endog ~ 0 + %s + %s", fml_fe, fml_re)
            } else {
                fml_re = paste(names(data)[exog_re_ix], collapse="+")
                fml = sprintf("endog ~ 0 + %s + (0 + %s | groups)", fml_fe, fml_re)
            }

            md = lmer(as.formula(fml), data=data, REML=reml)

            ds_ix = as.integer(substr(file, 4, 6))
            rslt[[sprintf("coef_%s_%s_%d", meth, rf, ds_ix)]] = as.vector(fixef(md))
            rslt[[sprintf("vcov_%s_%s_%d", meth, rf, ds_ix)]] = as.matrix(vcov(md))
            if (irf) {
                rev = NULL
                for (k in 1:length(exog_re_ix)) {
                    rev[k] = as.numeric(VarCorr(md)[[k]])
                }
                rev = diag(rev)
                rslt[[sprintf("cov_re_%s_%s_%d", meth, rf, ds_ix)]] = rev
            } else {
                rslt[[sprintf("cov_re_%s_%s_%d", meth, rf, ds_ix)]] = array(as.numeric(VarCorr(md)$groups),
                        c(pr, pr))
            }
            rslt[[sprintf("scale_%s_%s_%d", meth, rf, ds_ix)]] = attr(VarCorr(md), "sc")^2
            rslt[[sprintf("loglike_%s_%s_%d", meth, rf, ds_ix)]] = as.numeric(logLik(md))

            # Apparently lmer does not support these things when the random effects
            # are independent.
            if (!irf) {
                reo = ranef(md, condVar=TRUE)
                re = as.matrix(reo$groups)
                condvar = attr(reo$groups, "postVar")
                rslt[[sprintf("ranef_mean_%s_%s_%d", meth, rf, ds_ix)]] = re[1,]
                rslt[[sprintf("ranef_condvar_%s_%s_%d", meth, rf, ds_ix)]] = condvar[,,1]
            }
        }
    }
}

R2nparray(rslt, fname="lme_r_results.py")
