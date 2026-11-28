library(metafor)

# TestStratified4
data = array(c(16, 11, 5, 20,
               12, 16, 7, 19),
             dim=c(2, 2, 2))

# TestStratified5
data = array(c(185, 33, 189, 26,
               169, 49, 165, 57,
               156, 48, 104, 58,
               130, 80, 123, 118),
             dim=c(2, 2, 4))

# Map the 2x2 tables from statsmodels' convention
# [[a, b],
#  [c, d]]
# to the corresponding ai, bi, ci, di values used by metafor::rma.mh.
# In the R array representation below, these values appear as:
#      [,1] [,2]
# [1,]   a    c
# [2,]   b    d

a <- data[1, 1, ]
b <- data[2, 1, ]
c <- data[1, 2, ]
d <- data[2, 2, ]

or <- rma.mh( ai = a, bi = b, ci = c, di = d, measure = "OR", correct = FALSE )
rr <- rma.mh( ai = a, bi = b, ci = c, di = d, measure = "RR" )
rd <- rma.mh( ai = a, bi = b, ci = c, di = d, measure = "RD" )

invisible(
  cat(
    "cls.oddsratio_pooled = ", round( exp( or$b ), 6 ), "\n",
    "cls.logodds_pooled = ", round( or$b, 6 ), "\n",
    "cls.or_lcb = ", round( exp( or$ci.lb ), 6 ), "\n",
    "cls.or_ucb = ", round( exp( or$ci.ub ), 6 ), "\n",
    "cls.mh_stat = ", round( or$MH, 6 ), "\n",
    "cls.mh_pvalue = ",round( or$MHp, 6 ), "\n",
    "cls.or_homog_adj = ", round( or$TA, 6 ), "\n",
    "cls.or_homog_adj_p = ", round( or$TAp, 6 ), "\n",
    "\n",
    "cls.riskratio_pooled = ", round( exp( rr$b ), 6 ), "\n",
    "cls.logriskratio_pooled = ", round( rr$b, 6 ), "\n",
    "cls.rr_lcb = ", round( exp( rr$ci.lb ), 6 ), "\n",
    "cls.rr_ucb = ", round( exp( rr$ci.ub ), 6 ), "\n",
    "\n",
    "cls.risk_diff = ", round( rd$b, 6 ), "\n",
    "cls.risk_diff_se = ", round( rd$se, 6 ), "\n",
    "cls.rd_lcb = ", round( rd$ci.lb, 6 ), "\n",
    "cls.rd_ucb = ", round( rd$ci.ub, 6 ), "\n",
    "cls.rd_stat = ", round( rd$zval, 6 ), "\n",
    "cls.rd_pvalue = ", round( rd$pval, 6 ), "\n",
    sep=''
  )
)
