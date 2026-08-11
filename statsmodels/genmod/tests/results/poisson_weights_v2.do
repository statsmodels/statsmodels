clear
tempname filename
local filename = "M:\josef_new\stata_work\results_glm_poisson_weights.py"

insheet using M:\josef_new\eclipse_ws\statsmodels\statsmodels_py34_pr\statsmodels\datasets\cpunish\cpunish.csv
generate LN_VC100k96 = log(vc100k96)
generate var10 = 1 in 1
replace var10 = 1 in 2
replace var10 = 1 in 3
replace var10 = 2 in 4
replace var10 = 2 in 5
replace var10 = 2 in 6
replace var10 = 3 in 7
replace var10 = 3 in 8
replace var10 = 3 in 9
replace var10 = 1 in 10
replace var10 = 1 in 11
replace var10 = 1 in 12
replace var10 = 2 in 13
replace var10 = 2 in 14
replace var10 = 2 in 15
replace var10 = 3 in 16
replace var10 = 3 in 17
label variable var10 "fweight"
rename var10 fweight
label variable LN_VC100k96 "LN_VC100k96"

/* for checkin Poisson produces the same, poisson does not allow aweights */
/*poisson executions income perpoverty perblack LN_VC100k96 south degree */


glm executions income perpoverty perblack LN_VC100k96 south degree, family(poisson)

/* copied from glm_logit_constrained.do */
/*glm grade  gpa tuce psi, family(binomial) constraints(2) */

/* boiler plate, add matrices if needed */
tempname cov
tempname params_table
matrix cov = e(V)
matrix params_table = r(table)'
estat ic
matrix infocrit = r(S)
predict predict_mu, mu
predict predict_linpred_std, stdp
predict predict_hat, hat

predict score_factor, score
predict resid_anscombe, anscombe
predict resid_deviance, deviance
predict resid_response, response
predict resid_pearson, pearson
predict resid_working, working

local pred predict_mu predict_linpred_std predict_hat
local res score_factor resid_response resid_anscombe resid_deviance resid_pearson resid_working
mkmat `res', matrix(resids)
mkmat `pred', matrix(predicted)
estmat2nparray params_table cov infocrit predicted resids, saving(`filename') format("%16.0g") resname("poisson_none_nonrobust") replace
/*------------------*/

drop `pred' `res'


glm executions income perpoverty perblack LN_VC100k96 south degree [fweight=fweight], family(poisson)

/* boiler plate, add matrices if needed */
tempname cov
tempname params_table
matrix cov = e(V)
matrix params_table = r(table)'
estat ic
matrix infocrit = r(S)
predict predict_mu, mu
predict predict_linpred_std, stdp
predict predict_hat, hat

predict score_factor, score
predict resid_anscombe, anscombe
predict resid_deviance, deviance
predict resid_response, response
predict resid_pearson, pearson
predict resid_working, working

local pred predict_mu predict_linpred_std predict_hat
local res score_factor resid_response resid_anscombe resid_deviance resid_pearson resid_working
mkmat `res', matrix(resids)
mkmat `pred', matrix(predicted)
estmat2nparray params_table cov infocrit predicted resids, saving(`filename') format("%16.0g") resname("poisson_fweight_nonrobust") append

/*------------------*/

drop `pred' `res'

glm executions income perpoverty perblack LN_VC100k96 south degree [aweight=fweight], family(poisson)

/* boiler plate, add matrices if needed */
tempname cov
tempname params_table
matrix cov = e(V)
matrix params_table = r(table)'
estat ic
matrix infocrit = r(S)
predict predict_mu, mu
predict predict_linpred_std, stdp
predict predict_hat, hat

predict score_factor, score
predict resid_anscombe, anscombe
predict resid_deviance, deviance
predict resid_response, response
predict resid_pearson, pearson
predict resid_working, working

local pred predict_mu predict_linpred_std predict_hat
local res score_factor resid_response resid_anscombe resid_deviance resid_pearson resid_working
mkmat `res', matrix(resids)
mkmat `pred', matrix(predicted)
estmat2nparray params_table cov infocrit predicted resids, saving(`filename') format("%16.0g") resname("poisson_aweight_nonrobust") append
/*------------------*/


drop `pred' `res'

glm executions income perpoverty perblack LN_VC100k96 south degree [pweight=fweight], family(poisson)

/* boiler plate, add matrices if needed */
tempname cov
tempname params_table
matrix cov = e(V)
matrix params_table = r(table)'
estat ic
matrix infocrit = r(S)

predict predict_mu, mu
predict predict_linpred_std, stdp
//predict predict_hat, hat      not allowed after robust, pweights implies HC1

predict score_factor, score
predict resid_anscombe, anscombe
predict resid_deviance, deviance
predict resid_response, response
predict resid_pearson, pearson
predict resid_working, working

local pred predict_mu predict_linpred_std //predict_hat
local res score_factor resid_response resid_anscombe resid_deviance resid_pearson resid_working
mkmat `res', matrix(resids)
mkmat `pred', matrix(predicted)
estmat2nparray params_table cov infocrit predicted resids, saving(`filename') format("%16.0g") resname("poisson_pweight_nonrobust") append
/*------------------*/

/*******************************************************************/
/***********  next with robust = HC1, do not save resid and similar */

drop `pred' `res'
glm executions income perpoverty perblack LN_VC100k96 south degree, family(poisson) vce(robust)

/* boiler plate, add matrices if needed */
tempname cov
tempname params_table
matrix cov = e(V)
matrix params_table = r(table)'
estat ic
matrix infocrit = r(S)
predict predict_mu, mu
predict predict_linpred_std, stdp
//predict predict_hat, hat

local pred predict_mu predict_linpred_std //predict_hat
mkmat `pred', matrix(predicted)
estmat2nparray params_table cov infocrit predicted, saving(`filename') format("%16.0g") resname("poisson_none_hc1") append
/*------------------*/

drop `pred'

glm executions income perpoverty perblack LN_VC100k96 south degree [fweight=fweight], family(poisson) vce(robust)

/* boiler plate, add matrices if needed */
tempname cov
tempname params_table
matrix cov = e(V)
matrix params_table = r(table)'
estat ic
matrix infocrit = r(S)
predict predict_mu, mu
predict predict_linpred_std, stdp
//predict predict_hat, hat

local pred predict_mu predict_linpred_std //predict_hat
mkmat `pred', matrix(predicted)
estmat2nparray params_table cov infocrit predicted, saving(`filename') format("%16.0g") resname("poisson_fweight_hc1") append
/*------------------*/

drop `pred'

glm executions income perpoverty perblack LN_VC100k96 south degree [aweight=fweight], family(poisson) vce(robust)

/* boiler plate, add matrices if needed */
tempname cov
tempname params_table
matrix cov = e(V)
matrix params_table = r(table)'
estat ic
matrix infocrit = r(S)
predict predict_mu, mu
predict predict_linpred_std, stdp
//predict predict_hat, hat

local pred predict_mu predict_linpred_std //predict_hat
mkmat `pred', matrix(predicted)
estmat2nparray params_table cov infocrit predicted, saving(`filename') format("%16.0g") resname("poisson_aweight_hc1") append
/*------------------*/


drop `pred'

glm executions income perpoverty perblack LN_VC100k96 south degree [pweight=fweight], family(poisson) vce(robust)

/* boiler plate, add matrices if needed */
tempname cov
tempname params_table
matrix cov = e(V)
matrix params_table = r(table)'
estat ic
matrix infocrit = r(S)
predict predict_mu, mu
predict predict_linpred_std, stdp
//predict predict_hat, hat

local pred predict_mu predict_linpred_std //predict_hat
mkmat `pred', matrix(predicted)
estmat2nparray params_table cov infocrit predicted, saving(`filename') format("%16.0g") resname("poisson_pweight_hc1") append
/*------------------*/




/*****************************************************************/
/*************************** with cluster robust standard errors */
gen id = (_n - mod(_n, 2)) / 2

drop `pred'
glm executions income perpoverty perblack LN_VC100k96 south degree, family(poisson) vce(cluster id)

/* boiler plate, add matrices if needed */
tempname cov
tempname params_table
matrix cov = e(V)
matrix params_table = r(table)'
estat ic
matrix infocrit = r(S)
predict predict_mu, mu
predict predict_linpred_std, stdp
//predict predict_hat, hat

local pred predict_mu predict_linpred_std //predict_hat
mkmat `pred', matrix(predicted)
estmat2nparray params_table cov infocrit predicted, saving(`filename') format("%16.0g") resname("poisson_none_clu1") append
/*------------------*/

drop `pred'

glm executions income perpoverty perblack LN_VC100k96 south degree [fweight=fweight], family(poisson) vce(cluster id)

/* boiler plate, add matrices if needed */
tempname cov
tempname params_table
matrix cov = e(V)
matrix params_table = r(table)'
estat ic
matrix infocrit = r(S)
predict predict_mu, mu
predict predict_linpred_std, stdp
//predict predict_hat, hat

local pred predict_mu predict_linpred_std //predict_hat
mkmat `pred', matrix(predicted)
estmat2nparray params_table cov infocrit predicted, saving(`filename') format("%16.0g") resname("poisson_fweight_clu1") append
/*------------------*/

drop `pred'

glm executions income perpoverty perblack LN_VC100k96 south degree [aweight=fweight], family(poisson) vce(cluster id)

/* boiler plate, add matrices if needed */
tempname cov
tempname params_table
matrix cov = e(V)
matrix params_table = r(table)'
estat ic
matrix infocrit = r(S)
predict predict_mu, mu
predict predict_linpred_std, stdp
//predict predict_hat, hat

local pred predict_mu predict_linpred_std //predict_hat
mkmat `pred', matrix(predicted)
estmat2nparray params_table cov infocrit predicted, saving(`filename') format("%16.0g") resname("poisson_aweight_clu1") append
/*------------------*/


drop `pred'

glm executions income perpoverty perblack LN_VC100k96 south degree [pweight=fweight], family(poisson) vce(cluster id)

/* boiler plate, add matrices if needed */
tempname cov
tempname params_table
matrix cov = e(V)
matrix params_table = r(table)'
estat ic
matrix infocrit = r(S)
predict predict_mu, mu
predict predict_linpred_std, stdp
//predict predict_hat, hat

local pred predict_mu predict_linpred_std //predict_hat
mkmat `pred', matrix(predicted)
estmat2nparray params_table cov infocrit predicted, saving(`filename') format("%16.0g") resname("poisson_pweight_clu1") append
/*------------------*/



/*****************************************************************/
/*
gen idn = _n
glm executions income perpoverty perblack LN_VC100k96 south degree, family(poisson) vce(cluster idn)
*/

/*
glm grade  gpa tuce psi, family(binomial) constraints(2) vce(robust)
/* boiler plate, add matrices if needed */
tempname cov
tempname params_table
matrix cov = e(V)
matrix params_table = r(table)'
estat ic
matrix infocrit = r(S)
estmat2nparray params_table cov infocrit, saving(`filename') format("%16.0g") resname("constraint2_robust") append
/*------------------*/


capture drop fittedvalues fittedvalues_se
predict fittedvalues, xb
predict fittedvalues_se, stdp
outsheet fittedvalues fittedvalues_se using theil_predict.csv, comma replace
/*------------------*/

glm executions income perpoverty perblack LN_VC100k96 south degree [aweight=fweight], family(poisson)
predict resid_deviance_aw, deviance


*/

/*------------------*/

drop `pred'

regress executions income perpoverty perblack LN_VC100k96 south degree [aweight=fweight], vce(robust)

/* boiler plate, add matrices if needed */
tempname cov
tempname params_table
matrix cov = e(V)
matrix params_table = r(table)'
estat ic
matrix infocrit = r(S)
predict predict_mu, xb
predict predict_std, stdp
//predict predict_stf, stdf
//predict predict_str, stdr
//predict predict_hat, hat

local pred predict_mu predict_std //predict_stf predict_str predict_hat
mkmat `pred', matrix(predicted)
estmat2nparray params_table cov infocrit predicted, saving(`filename') format("%16.0g") resname("wls_aweight_robust") append
/*------------------*/

drop `pred'

regress executions income perpoverty perblack LN_VC100k96 south degree [aweight=fweight], vce(cluster id)

/* boiler plate, add matrices if needed */
tempname cov
tempname params_table
matrix cov = e(V)
matrix params_table = r(table)'
estat ic
matrix infocrit = r(S)
predict predict_mu, xb
predict predict_std, stdp
//predict predict_stf, stdf
//predict predict_str, stdr
//predict predict_hat, hat

local pred predict_mu predict_std //predict_stf predict_str predict_hat
mkmat `pred', matrix(predicted)
estmat2nparray params_table cov infocrit predicted, saving(`filename') format("%16.0g") resname("wls_aweight_clu1") append
/*------------------*/


drop `pred'

regress executions income perpoverty perblack LN_VC100k96 south degree [fweight=fweight], vce(cluster id)

/* boiler plate, add matrices if needed */
tempname cov
tempname params_table
matrix cov = e(V)
matrix params_table = r(table)'
estat ic
matrix infocrit = r(S)
predict predict_mu, xb
predict predict_std, stdp
//predict predict_stf, stdf
//predict predict_str, stdr
//predict predict_hat, hat

local pred predict_mu predict_std //predict_stf predict_str predict_hat
mkmat `pred', matrix(predicted)
estmat2nparray params_table cov infocrit predicted, saving(`filename') format("%16.0g") resname("wls_fweight_clu1") append
/*------------------*/

drop `pred'

regress executions income perpoverty perblack LN_VC100k96 south degree [pweight=fweight], vce(cluster id)

/* boiler plate, add matrices if needed */
tempname cov
tempname params_table
matrix cov = e(V)
matrix params_table = r(table)'
estat ic
matrix infocrit = r(S)
predict predict_mu, xb
predict predict_std, stdp
//predict predict_stf, stdf
//predict predict_str, stdr
//predict predict_hat, hat

local pred predict_mu predict_std //predict_stf predict_str predict_hat
mkmat `pred', matrix(predicted)
estmat2nparray params_table cov infocrit predicted, saving(`filename') format("%16.0g") resname("wls_pweight_clu1") append
/*------------------*/

drop `pred'
