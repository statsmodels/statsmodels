
source("M:\\josef_new\\eclipse_ws\\statsmodels\\statsmodels_py34_pr\\tools\\R2nparray\\R\\R2nparray.R")
library('mgcv')
library('gamair')
d = data(mpg)

#gam_a = gam(city.mpg ~ fuel + drive + s(weight,bs="cc",k=7) + s(hp,bs="cc",k=6), data = mpg,
#            sp = c(6.46225497484073, 0.81532465890585))


#gam_a = gam(city.mpg ~ fuel + drive + s(weight,bs="bs",k=7) + s(hp,bs="bs",k=6), data = mpg)

sm_knots_w = c(1488., 1488., 1488., 1488., 1953.22222222, 2118.77777778, 2275., 2383.88888889, 2515.55555556, 2757.33333333, 3016.44444444, 3208.11111111, 4066., 4066., 4066., 4066.)
sm_knots_hp = c(48.0, 48.0, 48.0, 48.0, 68.0, 73.0, 88.0, 101.0, 116.0, 152.28571428571428, 288.0, 288.0, 288.0, 288.0)

knots_w <- data.frame(weight=sm_knots_w)
knots_h <- data.frame(hp=sm_knots_hp)
gam_a = gam(city.mpg ~ fuel + drive + s(weight,bs="bs",k=12) + s(hp,bs="bs",k=10), data = mpg, knots=c(knots_w, knots_h), family=poisson)


pls = gam_a

fname = "results_mpg_bs_poisson.py"
append = FALSE #TRUE

#redirect output to file
sink(file=fname, append=append)
write_header()
mod_name = "mpg_bs_poisson."
cat("\nmpg_bs_poisson = Bunch()\n")
cat(paste("\n", mod_name, "smooth0 = Bunch()\n", sep=""))

sm1 <- gam_a$smooth[[1]]
cat_items(sm1, prefix=paste(mod_name, "smooth0.", sep=""))
#, blacklist=c())
cat("\n")
cat(convert.numeric(pls$smooth[[1]]$S[[1]], name=paste(mod_name, "smooth0.S", sep="")))
cat("\n")
cat_items(pls, prefix=mod_name, blacklist=c("eq", "control"))
cat("\n")
pls_summ = summary(pls)
cat_items(pls_summ, prefix=mod_name, blacklist=c("eq", "control"))

params = coef(pls)
cat("\n")
cat(convert.numeric(params, name=paste(mod_name, "params", sep="")))

# stop redirecting output
sink()
