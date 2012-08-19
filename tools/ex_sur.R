data <- read.csv('E:\\path_to_repo\\statsmodels\\datasets\\grunfeld\\grunfeld.csv')

data <- data[data$firm %in% c('General Motors','Chrysler','General Electric','Westinghouse','US Steel'),]
attach(data)
library('plm')
library('systemfit')
panel <- plm.data(data,c('firm','year'))
formula <- invest ~ value + capital
SUR <- systemfit(formula,method='SUR',data=panel)
f <- fitted(SUR)
ff <- c(f[,'Chrysler'],f[,'General.Electric'],f[,'General.Motors'],f[,'US.Steel'],f[,'Westinghouse'])

# save results to python module
#load functions, (windows path separators)
source("E:\\path_to_repo\\tools\\R2nparray\\R\\R2nparray.R")
source("E:\\path_to_repo\\tools\\topy.R")

#translation table for names  (could be dict in python)
translate = list(coefficients="params",
                 coefCov="cov_params",
                 residCovEst="resid_cov_est",
                 residCov="resid_cov",
                 df_residual="df_resid",
                 df_residual_sys="df_resid_sys",
                 #nCoef="k_vars",    #not sure about this
                 fitted_values="fittedvalues"
                 )

fname = "tmp_sur_0.py"
append = FALSE #TRUE

#redirect output to file
sink(file=fname, append=append)
write_header()
cat("\nsur = Bunch()\n")

cat_items(SUR, prefix="sur.", blacklist=c("eq", "control"), trans=translate)

equations = SUR[["eq"]]
for (ii in c(1:length(equations))) {
equ_name = paste("sur.equ", ii, sep="")
cat("\n\n", equ_name, sep=""); cat(" = Bunch()\n")
cat_items(equations[[ii]], prefix=paste(equ_name, ".", sep=""), trans=translate)
}

sink()
