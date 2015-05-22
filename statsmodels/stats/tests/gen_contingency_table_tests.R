library(coin)
library(exact2x2)

tables = list()

tables[[1]] = matrix(c(23, 15, 19, 31), ncol=2, byrow=TRUE)

tables[[2]] = matrix(c(144, 33, 84, 126,
          2, 4, 14, 29,
          0, 2, 6, 25,
          0, 0, 1, 5), ncol=4, byrow=TRUE)

tables[[3]] = matrix(c(20, 10, 5,
          3, 30, 15,
          0, 5, 40),
          ncol=3, byrow=TRUE)

results = array(0, c(length(tables), 17))

for (k in 1:3) {

    table = as.table(tables[[k]])
    names(attributes(table)$dimnames) = c("x", "y")

    # Nominal homogeneity test with chi^2 reference
    rslt = mh_test(table)
    results[k, 1] = rslt@statistic@teststatistic
    results[k, 2] = rslt@statistic@df

    # Nominal homogeneity test with binomial reference
    if (prod(dim(table)) == 4) {
        rslt = mcnemar.exact(table)
        results[k, 3] = rslt$p.value
    }

    # Nominal homogeneity test with continuity correction
    if (prod(dim(table)) == 4) {
        rslt = mcnemar.test(table)
        results[k, 4] = rslt$p.value
    }

    # Linear-by-linear homogeneity test with linear weights
    scores = list(x=seq(dim(table)[1]), y=seq(dim(table)[2]))
    rslt = lbl_test(table, scores=scores)
    results[k, 5] = rslt@statistic@linearstatistic
    results[k, 6] = rslt@statistic@expectation
    results[k, 7] = rslt@statistic@covariance@variance
    results[k, 8] = rslt@statistic@teststatistic
    results[k, 9] = rslt@distribution@pvalue(rslt@statistic@teststatistic)

    # Linear-by-linear homogeneity test with quadratic column weights
    scores = list(x=seq(dim(table)[1]), y=seq(dim(table)[2])^2)
    rslt = lbl_test(table, scores=scores)
    results[k, 10] = rslt@statistic@linearstatistic
    results[k, 11] = rslt@statistic@expectation
    results[k, 12] = rslt@statistic@covariance@variance
    results[k, 13] = rslt@statistic@teststatistic
    results[k, 14] = rslt@distribution@pvalue(rslt@statistic@teststatistic)

    # Bowker symmetry test (apparently mcnemar.test performs a
    # symmetry test when dim>2 although this is not documented).
    rslt = mcnemar.test(table, correct=FALSE)
    results[k, 15] = rslt$statistic
    results[k, 16] = rslt$parameter[1]
    results[k, 17] = rslt$p.value
}

colnames(results) = c("homog_stat", "homog_df", "homog_binom_p",
            "homog_cont_p", "lbl_stat", "lbl_expval", "lbl_var",
            "lbl_chi2", "lbl_pvalue", "lbl2_stat", "lbl2_expval",
            "lbl2_var", "lbl2_chi2", "lbl2_pvalue", "bowker_stat",
            "bowker_df", "bowker_pvalue")

write.csv(results, file="contingency_table_r_results.csv", row.names=FALSE)
