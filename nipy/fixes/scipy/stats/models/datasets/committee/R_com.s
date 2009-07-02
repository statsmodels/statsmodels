### Setup ###
com.data <- as.matrix(read.csv("./committee.csv",header=T))
com.factors <- data.frame(104th =com.data[,2], SIZE=com.data[,3], 
    SUBS=com.data[,4], LNSTAFF=com.data[,5], PRESTIGE=com.data[,6], 
    103rd=com.data[,7] )
attach(com.factors)

### Model ###
com.nbinomal <- nb()

### FINISH WHEN NEGATIVE BINOMIAL IS ADDED TO GLM...
