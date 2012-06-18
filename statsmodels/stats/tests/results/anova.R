dta <- read.table('data.dat', header=TRUE)
dta$Duration <- factor(dta$Duration)
dta$Weight <- factor(dta$Weight)
dta$logDays <- log(dta$Days + 1) # Use log days to "stabilize" variance

attach(dta)
library(car)
source('/home/skipper/statsmodels/statsmodels/tools/topy.R')

sum.lm = lm(logDays ~ Duration * Weight, contrasts=list(Duration=contr.sum, 
                                                        Weight=contr.sum))

anova.lm.sum <- anova(sum.lm)

for(name in names(anova.lm.sum)) {
    mkarray2(anova.lm.sum[[name]], name, TRUE)
        }; cat("\n")

anova.lm.interaction <- anova(lm(logDays ~ Duration + Weight), sum.lm)

for(name in names(anova.lm.interaction)) {
    mkarray2(anova.lm.interaction[[name]], name, TRUE)
        }; cat("\n")

anova.lm.variable <- anova(lm(logDays ~ Duration), lm(logDays ~ Duration + Weight))

anova.lm.variable2 <- anova(lm(logDays ~ Weight), lm(logDays ~ Duration + Weight))

anova.i <- anova(sum.lm)
anova.ii <- Anova(sum.lm, type='II')
anova.iii <- Anova(sum.lm, type='III')

nosum.lm = lm(logDays ~ Duration * Weight, contrasts=list(Duration=contr.treatment, Weight=contr.treatment))

anova.i.nosum <- anova(nosum.lm)
anova.ii.nosum <- Anova(nosum.lm, type='II')
anova.iii.nosum <- Anova(nosum.lm, type='III')

dta.dropped <- dta[4:60, ]
sum.lm.dropped <- lm(logDays ~ Duration * Weight,  dta.dropped,
                    contrasts=list(Duration=contr.sum, Weight=contr.sum))

anova.i.dropped <- anova(sum.lm.dropped)
anova.ii.dropped <- Anova(sum.lm.dropped, type='II')
anova.iii.dropped <- Anova(sum.lm.dropped, type='III')

for(name in names(anova.ii.dropped)) {
    mkarray2(anova.ii.dropped[[name]], name, TRUE)
        }; cat("\n")

for(name in names(anova.iii.dropped)) {
    mkarray2(anova.iii.dropped[[name]], name, TRUE)
        }; cat("\n")

anova.iii.dropped <- Anova(sum.lm.dropped, white="hc0", type='III')
for(name in names(anova.iii.dropped)) {
    mkarray2(anova.iii.dropped[[name]], name, TRUE)
        }; cat("\n")
anova.iii.dropped <- Anova(sum.lm.dropped, white="hc1", type='III')
for(name in names(anova.iii.dropped)) {
    mkarray2(anova.iii.dropped[[name]], name, TRUE)
        }; cat("\n")
anova.iii.dropped <- Anova(sum.lm.dropped, white="hc2", type='III')
for(name in names(anova.iii.dropped)) {
    mkarray2(anova.iii.dropped[[name]], name, TRUE)
        }; cat("\n")
anova.iii.dropped <- Anova(sum.lm.dropped, white="hc3", type='III')
for(name in names(anova.iii.dropped)) {
    mkarray2(anova.iii.dropped[[name]], name, TRUE)
        }; cat("\n")


anova.ii.dropped <- Anova(sum.lm.dropped, type='II', white="hc0")
for(name in names(anova.ii.dropped)) {
    mkarray2(anova.ii.dropped[[name]], name, TRUE)
        }; cat("\n")

anova.ii.dropped <- Anova(sum.lm.dropped, type='II', white="hc1")
for(name in names(anova.ii.dropped)) {
    mkarray2(anova.ii.dropped[[name]], name, TRUE)
        }; cat("\n")

anova.ii.dropped <- Anova(sum.lm.dropped, type='II', white="hc2")
for(name in names(anova.ii.dropped)) {
    mkarray2(anova.ii.dropped[[name]], name, TRUE)
        }; cat("\n")

anova.ii.dropped <- Anova(sum.lm.dropped, type='II', white="hc3")
for(name in names(anova.ii.dropped)) {
    mkarray2(anova.ii.dropped[[name]], name, TRUE)
        }; cat("\n")

