# install.packages("tsDyn")  # comment in, if package is not installed yet
library(tsDyn)

dta = read.table("E6_jmulti.csv", header = FALSE, sep = " ")

det.terms <- c("co", "cc", "colt", "cclt", "colc", "cclc")

for(dt in det.terms){
  det.outside.coint <- "none"
  det.inside.coint <- "none"

  if(grepl("co", dt))  # python: if "co" in dt:
  {
    det.outside.coint <- "const"
  }
  if(grepl("lt", dt))  # python: if "lt" in dt:
  {
    det.outside.coint <- ifelse(det.outside.coint=="const", "both", "trend")
  }
  if(grepl("cc", dt))  # python: if "cc" in dt:
  {
    det.inside.coint <- "const"
  }
  if(grepl("lc", dt))  # python: if "lc" in dt:
  {
    det.outside.coint <- ifelse(det.inside.coint=="const", "both", "trend")
  }
  
  vecm.jo <- VECM(dta, lag=3, estim="ML", include = det.outside.coint, LRinclude = det.inside.coint)
  sum.jo <- summary(vecm.jo)
  
  file.name.start <- paste("e6_r_tsDyn_", dt, sep="")  # paste() concatenates strings
  sink(paste(file.name.start, "_alpha_gamma.txt", sep=""))  # put output of the code that follows into the given file
  print(sum.jo$coefMat)
  sink()  # stop writing to this file

  sink(paste(file.name.start, "_beta.txt", sep=""))  # put output of the code that follows into the given file
  print(vecm.jo$model.specific$beta)
  sink()  # stop writing to this file
  
  sink(paste(file.name.start, "_df_resid.txt", sep=""))  # put output of the code that follows into the given file
  print(sum.jo$df.residual)
  sink()  # stop writing to this file
}

