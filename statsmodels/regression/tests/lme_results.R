library(lme4)

files = list.files(path="results", pattern="lme...csv")

for (file in files) {

    data = read.csv(paste("results", file, sep="/"))

    exog_fe_ix = grep("exog_fe", names(data))
    exog_re_ix = grep("exog_re", names(data))

    fml_re = paste(names(data)[exog_re_ix], sep=" + ", collapse="")
    fml_fe = paste(names(data)[exog_fe_ix], sep=" + ", collapse="")
    fml = sprintf("endog ~ %s + (%s | groups)", fml_fe, fml_re)

    md = lmer(as.formula(fml), data=data)
    stop()
}
