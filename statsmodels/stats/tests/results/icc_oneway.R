# Reference values for test_oneway.py::TestIccOneway
# R 4.5.3, ICC 2.4.0, irr 0.85
library(ICC)
library(irr)

options(digits = 15)

show <- function(name, y, g) {
  x <- factor(g)
  est <- ICCest(x, y, alpha = 0.05, CI.type = "THD")
  tab <- anova(aov(y ~ x))
  cat(name, "\n")
  cat("icc", est$ICC, "\n")
  cat("conf_int", est$LowerCI, est$UpperCI, "\n")
  cat("var_between", est$vara, "\n")
  cat("var_within", est$varw, "\n")
  cat("nobs_mean", est$k, "\n")
  cat("statistic", tab$"F value"[1], "\n")
  cat("pvalue", tab$"Pr(>F)"[1], "\n")
  cat("df", tab$Df[1], tab$Df[2], "\n\n")
}

# unbalanced, groups of size 2 to 5
y1 <- c(5.1, 4.8, 6.0, 7.2, 6.9, 7.5, 7.0, 3.9, 4.4, 4.1, 5.5, 5.9, 6.2,
        5.0, 5.3, 8.1, 7.7)
g1 <- c(1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 4, 4, 4, 5, 5, 6, 6)
show("unbalanced", y1, g1)

# unbalanced with a group of size one
y2 <- c(2.3, 2.9, 2.6, 4.1, 3.8, 1.7, 2.2, 2.0, 1.5, 3.3)
g2 <- c("a", "a", "a", "b", "b", "c", "c", "c", "c", "d")
show("single", y2, g2)

# negative estimate, between group variance smaller than expected
y3 <- c(1.0, 3.0, 2.1, 1.1, 2.9, 2.0, 0.9, 3.1, 1.9, 1.0, 3.0, 2.0)
g3 <- c(1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4)
show("negative", y3, g3)

# balanced, 6 subjects rated 4 times, also irr::icc with the oneway model
ratings <- matrix(c(9, 2, 5, 8,
                    6, 1, 3, 2,
                    8, 4, 6, 8,
                    7, 1, 2, 6,
                    10, 5, 6, 9,
                    6, 2, 4, 7), ncol = 4, byrow = TRUE)
y4 <- as.vector(t(ratings))
g4 <- rep(1:6, each = 4)
show("balanced", y4, g4)
r <- icc(ratings, model = "oneway", unit = "single", conf.level = 0.95)
cat("irr icc", r$value, "F", r$Fvalue, "p", r$p.value, "ci", r$lbound,
    r$ubound, "\n")
