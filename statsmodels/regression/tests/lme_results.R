library(lme4)
library(R2nparray)

files = list.files(path="results", pattern="lme...csv")

rslt = list()

for (file in files) {

    data = read.csv(paste("results", file, sep="/"))

    exog_fe_ix = grep("exog_fe", names(data))
    exog_re_ix = grep("exog_re", names(data))

    pr = length(exog_re_ix)
    fml_re = paste(names(data)[exog_re_ix], collapse="+")
    fml_fe = paste(names(data)[exog_fe_ix], collapse="+")
    fml = sprintf("endog ~ 0 + %s + (0 + %s | groups)", fml_fe, fml_re)

    md = lmer(as.formula(fml), data=data, REML=FALSE)

    ds_ix = as.integer(substr(file, 4, 6))
    rslt[[sprintf("coef_%d", ds_ix)]] = as.vector(fixef(md))
    rslt[[sprintf("vcov_%d", ds_ix)]] = as.matrix(vcov(md))
    rslt[[sprintf("revar_%d", ds_ix)]] = array(as.numeric(VarCorr(md)$groups),
            c(pr, pr))
    rslt[[sprintf("sig2_%d", ds_ix)]] = attr(VarCorr(md), "sc")^2
}

R2nparray(rslt, fname="lme_r_results.py")
