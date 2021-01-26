library(survival)
library(survMisc)

df = read.csv("bmt.csv")

sf = survfit(Surv(df$T, df$Status) ~ df$Group)
cb = ci(sf, how="hall", trans="log", tL=1, tU=100)


#df = df[df$Group != "ALL",]
#df$Strata = seq(dim(df)[1]) %% 5

#f = survfit(Surv(df$T, df$Status) ~ df$Group)
#mc = comp(f, FHp=1, FHq=1)

#ff = survdiff(Surv(df$T, df$Status) ~ df$Group + strata(df$Strata), rho=1)


