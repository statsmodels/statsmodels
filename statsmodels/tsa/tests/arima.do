insheet using "/home/skipper/statsmodels/statsmodels-skipper/statsmodels/datasets/macrodata/macrodata.csv"

gen qtrdate=yq(year,quarter)
format qtrdate %tq
tsset qtrdate

/* here for safe keeping 
arima cpi, arima(4,1,1) from(0.92687596 -0.5558627   0.32086541  0.25225289  0.11362493  0.93914412 0.78437817334640536, copy)
predict xb, xb dynamic(.)
predict y, y dynamic(.)
*/

arima cpi, arima(1,1,1)

mat llf=e(ll)
mat nobs=e(N)
// number of parameters
mat k=e(k)
// number of dependent variables
mat k_exog=e(k_dv)
mat sigma=e(sigma)
mat chi2=e(chi2)
mat df_model=e(df_m)
mat k_ar=e(ar_max)
mat k_ma=e(ma_max)
mat params=e(b)
mat cov_params=e(V)

// don't append because you'll rewrite the bunch class
// mat2nparray llf nobs k k_exog sigma chi2 df_model k_ar k_ma params cov_params, saving("/home/skipper/statsmodels/statsmodels-skipper/statsmodels/tsa/tests/results/arima_results.py") format("%16.0g") replace

predict xb
predict y, y
predict resid, resid
predict yr, yr
predict mse, mse
predict stdp, stdp
estat ic

mat icstats=r(S)
mkmat xb xb
mkmat y y
mkmat yr yr
mkmat mse mse
mkmat stdp stdp
mkmat resid resid

mat2nparray llf nobs k k_exog sigma chi2 df_model k_ar k_ma params cov_params xb y resid yr mse stdp icstats, saving("/home/skipper/statsmodels/statsmodels-skipper/statsmodels/tsa/tests/results/arima111_results.py") format("%16.0g") replace

clear

/* do it with no constant */

insheet using "/home/skipper/statsmodels/statsmodels-skipper/statsmodels/datasets/macrodata/macrodata.csv"

gen qtrdate=yq(year,quarter)
format qtrdate %tq
tsset qtrdate

arima cpi, arima(1,1,1) noconstant

mat llf=e(ll)
mat nobs=e(N)
// number of parameters
mat k=e(k)
// number of dependent variables
mat k_exog=e(k_dv)
mat sigma=e(sigma)
mat chi2=e(chi2)
mat df_model=e(df_m)
mat k_ar=e(ar_max)
mat k_ma=e(ma_max)
mat params=e(b)
mat cov_params=e(V)

// don't append because you'll rewrite the bunch class
// mat2nparray llf nobs k k_exog sigma chi2 df_model k_ar k_ma params cov_params, saving("/home/skipper/statsmodels/statsmodels-skipper/statsmodels/tsa/tests/results/arima_results.py") format("%16.0g") replace

predict xb
predict y, y
predict resid, resid
predict yr, yr
predict mse, mse
/*predict stdp, stdp*/
estat ic

mat icstats=r(S)
mkmat xb xb
mkmat y y
mkmat yr yr
mkmat mse mse
/*mkmat stdp stdp*/
mkmat resid resid

mat2nparray llf nobs k k_exog sigma chi2 df_model k_ar k_ma params cov_params xb y resid yr mse stdp icstats, saving("/home/skipper/statsmodels/statsmodels-skipper/statsmodels/tsa/tests/results/arima111nc_results.py") format("%16.0g") replace

clear

/* Now do conditional */

insheet using "/home/skipper/statsmodels/statsmodels-skipper/statsmodels/datasets/macrodata/macrodata.csv"

gen qtrdate=yq(year,quarter)
format qtrdate %tq
tsset qtrdate

arima cpi, arima(1,1,1) condition

mat llf=e(ll)
mat nobs=e(N)
// number of parameters
mat k=e(k)
// number of dependent variables
mat k_exog=e(k_dv)
mat sigma=e(sigma)
mat chi2=e(chi2)
mat df_model=e(df_m)
mat k_ar=e(ar_max)
mat k_ma=e(ma_max)
mat params=e(b)
mat cov_params=e(V)

// don't append because you'll rewrite the bunch class
// mat2nparray llf nobs k k_exog sigma chi2 df_model k_ar k_ma params cov_params, saving("/home/skipper/statsmodels/statsmodels-skipper/statsmodels/tsa/tests/results/arima_results.py") format("%16.0g") replace

predict xb
predict y, y
predict resid, resid
predict yr, yr
predict mse, mse
predict stdp, stdp
estat ic

mat icstats=r(S)
mkmat xb xb
mkmat y y
mkmat yr yr
mkmat mse mse
mkmat stdp stdp
mkmat resid resid

mat2nparray llf nobs k k_exog sigma chi2 df_model k_ar k_ma params cov_params xb y resid yr mse stdp icstats, saving("/home/skipper/statsmodels/statsmodels-skipper/statsmodels/tsa/tests/results/arima111_css_results.py") format("%16.0g") replace

clear

/* do it with no constant */

insheet using "/home/skipper/statsmodels/statsmodels-skipper/statsmodels/datasets/macrodata/macrodata.csv"

gen qtrdate=yq(year,quarter)
format qtrdate %tq
tsset qtrdate

arima cpi, arima(1,1,1) noconstant condition

mat llf=e(ll)
mat nobs=e(N)
// number of parameters
mat k=e(k)
// number of dependent variables
mat k_exog=e(k_dv)
mat sigma=e(sigma)
mat chi2=e(chi2)
mat df_model=e(df_m)
mat k_ar=e(ar_max)
mat k_ma=e(ma_max)
mat params=e(b)
mat cov_params=e(V)

// don't append because you'll rewrite the bunch class
// mat2nparray llf nobs k k_exog sigma chi2 df_model k_ar k_ma params cov_params, saving("/home/skipper/statsmodels/statsmodels-skipper/statsmodels/tsa/tests/results/arima_results.py") format("%16.0g") replace

predict xb
predict y, y
predict resid, resid
predict yr, yr
predict mse, mse
predict stdp, stdp
estat ic

mat icstats=r(S)
mkmat xb xb
mkmat y y
mkmat yr yr
mkmat mse mse
mkmat stdp stdp
mkmat resid resid

mat2nparray llf nobs k k_exog sigma chi2 df_model k_ar k_ma params cov_params xb y resid yr mse stdp icstats, saving("/home/skipper/statsmodels/statsmodels-skipper/statsmodels/tsa/tests/results/arima111nc_css_results.py") format("%16.0g") replace

