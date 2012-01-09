insheet using "/home/skipper/statsmodels/statsmodels-git/scikits/statsmodels/datasets/fair/fair.csv", double clear

tobit affairs rate_marriage age yrs_married children religious educ occupation occupation_husb, ll(0)

mat cov_params = e(V)
mat llf = e(ll)
mat llnull = e(ll_0)
mat n_lcens = e(N_lc)
mat n_rcens = e(N_rc)
mat n_ucens = e(N_unc)
mat chi2 = e(chi2)
mat df_model = e(df_m)
mat df_resid = e(df_r)
mat params = e(b)
mata: bse = sqrt(diagonal(st_matrix("cov_params"))); st_matrix("bse", bse)

estat ic
mat r = r(S)
mat rr = r[1,5..6]

* linear prediction
predict xb
* standard error of the prediction
* which can be
* thought of as the standard error of the predicted expected value or
* mean for the observation's covariate pattern.  The standard error of
* the prediction is also referred to as the standard error of the
* fitted value.
predict stdp, stdp
* calculates Pr(a < xb + u < b), the probability that y|x would be
* observed in the interval (a,b). Takes inf and neg. inf
predict pr2535, pr(25,35)
* calculates E(xb + u | a < xb + u < b), the expected value of y|x
* conditional on y|x being in the interval (a,b), meaning that y|x is
* censored. 
predict e2535, e(25, 35)
* calculates E(y*), where y* = a if xb + u < a, y* = b if
* xb + u > b, and y* = xb+u otherwise, meaning that y* is truncated.
predict ystar2535, ystar(25, 35)
* calculates equation-level score variables.
* The first new variable will contain the derivative of the log
* likelihood with respect to the regression equation.
* The second new variable will contain the derivative of the log
* likelihood with respect to the scale equation (sigma).
predict score_params score_sigma, scores

mat2nparray cov_params llf llnull n_lcens n_rcens n_ucens chi2 df_model df_resid params bse, saving("/home/skipper/statsmodels/statsmodels-git/scikits/statsmodels/regression/tests/results/tobit_left") replace

** FAKE MODEL TO TEST RIGHT CENSORING
insheet using "/home/skipper/statsmodels/statsmodels-git/scikits/statsmodels/datasets/fair/fair.csv", double clear


tobit affairs rate_marriage age yrs_married children religious educ occupation occupation_husb, ul(2)

mat cov_params = e(V)
mat llf = e(ll)
mat llnull = e(ll_0)
mat n_lcens = e(N_lc)
mat n_rcens = e(N_rc)
mat n_ucens = e(N_unc)
mat chi2 = e(chi2)
mat df_model = e(df_m)
mat df_resid = e(df_r)
mat params = e(b)
mata: bse = sqrt(diagonal(st_matrix("cov_params"))); st_matrix("bse", bse)

estat ic
mat r = r(S)
mat rr = r[1,5..6]

* linear prediction
predict xb
* standard error of the prediction
* which can be
* thought of as the standard error of the predicted expected value or
* mean for the observation's covariate pattern.  The standard error of
* the prediction is also referred to as the standard error of the
* fitted value.
predict stdp, stdp
* calculates Pr(a < xb + u < b), the probability that y|x would be
* observed in the interval (a,b). Takes inf and neg. inf
predict pr2535, pr(25,35)
* calculates E(xb + u | a < xb + u < b), the expected value of y|x
* conditional on y|x being in the interval (a,b), meaning that y|x is
* censored. 
predict e2535, e(25, 35)
* calculates E(y*), where y* = a if xb + u < a, y* = b if
* xb + u > b, and y* = xb+u otherwise, meaning that y* is truncated.
predict ystar2535, ystar(25, 35)
* calculates equation-level score variables.
* The first new variable will contain the derivative of the log
* likelihood with respect to the regression equation.
* The second new variable will contain the derivative of the log
* likelihood with respect to the scale equation (sigma).
predict score_params score_sigma, scores

mat2nparray cov_params llf llnull n_lcens n_rcens n_ucens chi2 df_model df_resid params bse, saving("/home/skipper/statsmodels/statsmodels-git/scikits/statsmodels/regression/tests/results/tobit_right") replace


** FAKE MODEL TO TEST lower and upper censoring
insheet using "/home/skipper/statsmodels/statsmodels-git/scikits/statsmodels/datasets/fair/fair.csv", double clear


tobit affairs rate_marriage age yrs_married children religious educ occupation occupation_husb, ll(0) ul(2)

mat cov_params = e(V)
mat llf = e(ll)
mat llnull = e(ll_0)
mat n_lcens = e(N_lc)
mat n_rcens = e(N_rc)
mat n_ucens = e(N_unc)
mat chi2 = e(chi2)
mat df_model = e(df_m)
mat df_resid = e(df_r)
mat params = e(b)
mata: stderr = sqrt(diagonal(st_matrix("cov_params"))); st_matrix("stderr", stderr)

estat ic
mat r = r(S)
mat rr = r[1,5..6]

* linear prediction
predict xb
* standard error of the prediction
* which can be
* thought of as the standard error of the predicted expected value or
* mean for the observation's covariate pattern.  The standard error of
* the prediction is also referred to as the standard error of the
* fitted value.
predict stdp, stdp
* calculates Pr(a < xb + u < b), the probability that y|x would be
* observed in the interval (a,b). Takes inf and neg. inf
predict pr2535, pr(25,35)
* calculates E(xb + u | a < xb + u < b), the expected value of y|x
* conditional on y|x being in the interval (a,b), meaning that y|x is
* censored. 
predict e2535, e(25, 35)
* calculates E(y*), where y* = a if xb + u < a, y* = b if
* xb + u > b, and y* = xb+u otherwise, meaning that y* is truncated.
predict ystar2535, ystar(25, 35)
* calculates equation-level score variables.
* The first new variable will contain the derivative of the log
* likelihood with respect to the regression equation.
* The second new variable will contain the derivative of the log
* likelihood with respect to the scale equation (sigma).
predict score_params score_sigma, scores

mat2nparray cov_params llf llnull n_lcens n_rcens n_ucens chi2 df_model df_resid params bse, saving("/home/skipper/statsmodels/statsmodels-git/scikits/statsmodels/regression/tests/results/tobit_both") replace
