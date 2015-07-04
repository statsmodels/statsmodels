require(lme4)

hdp <- read.csv("http://www.ats.ucla.edu/stat/data/hdp.csv")
hdp <- within(hdp, {
  Married <- factor(Married, levels = 0:1, labels = c("no", "yes"))
  DID <- factor(DID)
  HID <- factor(HID)
})

m <- glmer(remission ~ IL6 + (1 | DID), data = hdp, family = binomial, nAGQ = 1)
print(logLik(m))