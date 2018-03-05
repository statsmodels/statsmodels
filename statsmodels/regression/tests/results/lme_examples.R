library(lme4)
library(MASS)

md = lmer(size ~ Time + (1 | tree), Sitka)

md = lmer(size ~ Time + (1 + Time | tree), Sitka)
