dta <- read.csv('/home/skipper/statsmodels/statsmodels/statsmodels/datasets/co2/co2.csv')

# simple non-ts stuff

co2.filt <- filter(dta$co2, filter=c(.75, .25))
co2.filt.one <- filter(dta$co2, filter=c(.75, .25), sides=1)
co2.filt.r <- filter(dta$co2, filter=c(.75, .25), method='recursive')
co2.filt.r.init <- filter(dta$co2, filter=c(.75, .25), method='recursive', init=c(300,300))




# ts-stuff

# use frequency to tell it that it is monthly sample but correct unit is 12
#co2.ts <- ts(dta$co2, start=c(1952, 3, 29), end=c(2001, 12, 29), frequency=12)
co2.ts <- ts(dta$co2, start=c(1958, 3, 29), frequency=52)
library(zoo)
dataset <- data.frame(date = as.Date("1958-03-29")+seq(1, nrow(dta)*7, 7))

# interpolate missing values
co2.ts <- na.approx(co2.ts)
dataset$co2 <- as.numeric(co2.ts)
co2.monthly <- aggregate(dataset$co2, by=list(Mon=as.yearmon(dataset$date)), mean)


co2.ts = ts(co2.monthly$x, start=c(1958, 3), frequency=12)

co2.decompose <- decompose(co2.ts)
co2.decompose.m <- decompose(co2.ts, "multiplicative")


co2.ts <- ts(co2.ts, start=c(1958, 3, 29), end=c(2001, 12, 29), frequency=12)
co2.decompose.monthly <- decompose(co2.ts)
co2.decompose.monthly.m <- decompose(co2.ts, "multiplicative")

## just load in monthly data aggregated in pandas. R is a pita.
#dta = sm.datasets.co2.load_pandas().data
#dta.co2.interpolate(inplace=True)
#dta = dta.co2.resample('M')
#dta.to_csv("/home/skipper/statsmodels/statsmodels-skipper/statsmodels/tsa/tests/monthly_co2.csv")

co2.ts <- read.csv("/home/skipper/statsmodels/statsmodels-skipper/statsmodels/tsa/tests/monthly_co2.csv")
co2.ts <- ts(co2.ts$co2, start=c(1958, 3), end=c(2001, 12), frequency=12)
co2.decompose.monthly <- decompose(co2.ts)



# exponential smoothing
library(forecast)
library(R2nparray)

# SES

type <- 'ANN'
alpha <- .9
beta <- 0
gamma <- 0
phi <- 0
x <- na.omit(co2.decompose.monthly$random)
co2.ses <- ses(na.omit(co2.decompose.monthly$random), h=48, alpha=.9, 
               initial="simple")
options(digits = 12)
R2nparray(list(fitted=co2.ses$fitted, resid=-co2.ses$residuals, 
               level=co2.ses$model$states, forecasts=co2.ses$mean), 
          fname='/home/skipper/statsmodels/statsmodels-skipper/statsmodels/tsa/tests/results/co2_ses_results.py')

co2.ses.optimal <- ses(co2.ts, h=48, alpha=.9, initial="optimal")

R2nparray(list(fitted=co2.ses$fitted, resid=-co2.ses$residuals, 
          forecasts=co2.ses$mean), fname='/home/skipper/statsmodels/statsmodels-skipper/statsmodels/tsa/tests/results/co2_ses_results_optimal.py')

# double-exponential, non-seasonal holt-winters
typ <- "AAN"

x <- as.ts(na.omit(co2.decompose.monthly$trend + co2.decompose.monthly$random))

#co2.hw2 <- ets(x, typ, damped=FALSE, alpha=.86, beta=.523, gamma=0,
#               additive.only=TRUE)
co2.hw2 <- holt(x, alpha=.86, beta=.523, initial="simple", h=48)
co2.hw <- HoltWinters(x, alpha=.86, beta=.523, gamma=FALSE)
# they uses a different notion of starting values plus they burn the
# starting values, they don't treat them as pre-values like holt and we do
co2.hw.forecast <- forecast.HoltWinters(co2.hw, h=48)
R2nparray(list(xhat=co2.hw2$fitted, 
               level=co2.hw2$model$states[,1],
               trend=co2.hw2$model$states[,2],
               resid=co2.hw2$residuals,
               forecasts=co2.hw2$mean), 
               fname='/home/skipper/statsmodels/statsmodels-skipper/statsmodels/tsa/tests/results/co2_holt_des_results.py')

# multiplicative trend
co2.hw2.m <- holt(x, alpha=.86, beta=.523, initial="simple", exponential=TRUE,
                  h=48)
R2nparray(list(xhat=co2.hw2.m$fitted, 
               level=co2.hw2.m$model$states[,1],
               trend=co2.hw2.m$model$states[,2],
               resid=co2.hw2.m$residuals,
               forecasts=co2.hw2.m$mean), 
               fname='/home/skipper/statsmodels/statsmodels-skipper/statsmodels/tsa/tests/results/co2_holt_des_mult_results.py')

# holt winters
x <- as.ts(na.omit(co2.decompose.monthly$seasonal + co2.decompose.monthly$random))
co2.hw2.seas <- hw(x, alpha=.0043, beta=FALSE, gamma=.2586,
                   initial="simple", exponential=FALSE, h=48)
R2nparray(list(xhat=co2.hw2.seas$fitted, 
               level=co2.hw2.seas$model$states[,1],
               resid=co2.hw2.seas$residuals,
               forecasts=co2.hw2.seas$mean), 
               fname='/home/skipper/statsmodels/statsmodels-skipper/statsmodels/tsa/tests/results/co2_holt_seas_results.py')

x <- na.omit(co2.decompose.monthly$seasonal + co2.decompose.monthly$random) + 5

co2.hw2.seas.m <- hw(x, alpha=.0043, beta=FALSE, gamma=.2586, 
                     initial="simple", seasonal='multiplicative',
                     damp=FALSE, h=48)
R2nparray(list(xhat=co2.hw2.seas.m$fitted, 
               level=co2.hw2.seas.m$model$states[,1],
               resid=co2.hw2.seas.m$residuals,
               forecasts=co2.hw2.seas.m$mean), 
               fname='/home/skipper/statsmodels/statsmodels-skipper/statsmodels/tsa/tests/results/co2_holt_seas_mult_results.py')

# damped only makes sense with a trend
#co2.hw2.seas.damped <- hw(x, alpha=NULL, beta=NULL, gamma=NULL, 
#                     initial="simple", damped=TRUE)
#R2nparray(list(xhat=co2.hw2.seas.m$fitted, 
#               level=co2.hw2.seas.m$model$states[,1],
#               resid=co2.hw2.seas.m$residuals,
#               forecasts=co2.hw2.seas.m$mean), 
#               fname='/home/skipper/statsmodels/statsmodels-skipper/statsmodels/tsa/tests/results/co2_holt_seas_damped_results.py')
#

# reverse the trend
tt <- co2.decompose.monthly$trend
x <- na.omit(replace(tt, TRUE, rev(tt)) + co2.decompose.monthly$random)

co2.damped = ets(x, model="AAN", damped=TRUE, alpha=.7826, beta=.0299, phi=.98)
#co2.damped.h = holt(x, damped=TRUE, alpha=.7826, beta=.0299, phi=.98)
R2nparray(list(xhat=co2.damped$fitted, 
               level=co2.damped$states[,1],
               trend=co2.damped$states[,2],
               resid=co2.damped$residuals,
               forecasts=forecast(co2.damped, h=48)$mean),
               fname='/home/skipper/statsmodels/statsmodels-skipper/statsmodels/tsa/tests/results/co2_damped_results.py')


co2.damped.m = ets(x, model="MMN", damped=TRUE, alpha=.7826, beta=.0299, 
                   phi=.98)
R2nparray(list(xhat=co2.damped.m$fitted, 
               level=co2.damped.m$states[,1],
               trend=co2.damped.m$states[,2],
               resid=co2.damped.m$residuals,
               forecasts=forecast(co2.damped.m, 48)$mean),
               fname='/home/skipper/statsmodels/statsmodels-skipper/statsmodels/tsa/tests/results/co2_damped_mult_results.py')


# multiplicative damped trend not allowed by ets
# co2.damped.m = ets(x, model="AMN", damped=TRUE) 

co2.multmult = ets(co2.ts, model="MMM", alpha=.7198, beta=.0387, gamma=.01,
                   damped=FALSE)
R2nparray(list(xhat=co2.multmult$fitted, 
               level=co2.multmult$states[,1],
               trend=co2.multmult$states[,2],
               resid=co2.multmult$residuals,
               forecasts=forecast(co2.multmult, 48)$mean),
               fname='/home/skipper/statsmodels/statsmodels-skipper/statsmodels/tsa/tests/results/co2_mult_mult_results.py')


