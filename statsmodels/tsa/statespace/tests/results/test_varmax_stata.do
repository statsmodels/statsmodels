webuse lutkepohl2, clear
tsset

// VAR(1)
dfactor (dln_inv dln_inc dln_consump = , ar(1) arstructure(general) noconstant covstructure(unstructured)) if qtr<=tq(1978q4)
estat ic

// These are predict in-sample + forecast out-of-sample (1979q1 is first out-of sample obs)
predict predict_1, dynamic(tq(1979q1)) equation(dln_inv)
predict predict_2, dynamic(tq(1979q1)) equation(dln_inc)
predict predict_3, dynamic(tq(1979q1)) equation(dln_consump)

// These are predict in-sample for first 3-observations (1960q2-1960q4), then
// dynamic predict for the rest of in-sample observations (1961q1-1978q4), then
// forecast for the remaining periods 1979q1 - 1982q4
predict dyn_predict_1, dynamic(tq(1961q1)) equation(dln_inv)
predict dyn_predict_2, dynamic(tq(1961q1)) equation(dln_inc)
predict dyn_predict_3, dynamic(tq(1961q1)) equation(dln_consump)

// VAR(1), diagonal covariance
dfactor (dln_inv dln_inc dln_consump = , ar(1) arstructure(general) noconstant covstructure(diagonal)) if qtr<=tq(1978q4)
estat ic

// predict + forecast, see above
predict predict_diag1, dynamic(tq(1979q1)) equation(dln_inv)
predict predict_diag2, dynamic(tq(1979q1)) equation(dln_inc)
predict predict_diag3, dynamic(tq(1979q1)) equation(dln_consump)

// predict + dynamic predict + forecast, see above
predict dyn_predict_diag1, dynamic(tq(1961q1)) equation(dln_inv)
predict dyn_predict_diag2, dynamic(tq(1961q1)) equation(dln_inc)
predict dyn_predict_diag3, dynamic(tq(1961q1)) equation(dln_consump)

// VAR(1), diagonal covariance + observation intercept
dfactor (dln_inv dln_inc dln_consump = , ar(1) arstructure(general) covstructure(diagonal)) if qtr<=tq(1978q4)
estat ic

// predict + forecast, see above
predict predict_int1, dynamic(tq(1979q1)) equation(dln_inv)
predict predict_int2, dynamic(tq(1979q1)) equation(dln_inc)
predict predict_int3, dynamic(tq(1979q1)) equation(dln_consump)

// predict + dynamic predict + forecast, see above
predict dyn_predict_int1, dynamic(tq(1961q1)) equation(dln_inv)
predict dyn_predict_int2, dynamic(tq(1961q1)) equation(dln_inc)
predict dyn_predict_int3, dynamic(tq(1961q1)) equation(dln_consump)

// VAR(1), diagonal covariance + 1 exog
gen t = _n
//dfactor (dln_inv dln_inc dln_consump = t, ar(1) arstructure(general) noconstant covstructure(diagonal)) if qtr<=tq(1978q4)
var dln_inv dln_inc dln_consump if qtr<=tq(1978q4), lags(1) noconstant exog(t)

// predict, see above (Note: uses actual data for forecasting, so we will want
// to ignore the predictions after 1978q4, see below
predict predict_exog1_1, equation(dln_inv)
predict predict_exog1_2, equation(dln_inc)
predict predict_exog1_3, equation(dln_consump)

// We will want to use these values to compare for forecasting, but note that
// this also includes in the columns the value for 1978q4 (i.e. a VAR(1) needs
// 1 sample from which to compute forecasts.
fcast compute fcast_exog1_ , dynamic(tq(1979q1)) step(16) replace

// VAR(1), diagonal covariance + 2 exog
gen c = 1
//dfactor (dln_inv dln_inc dln_consump = t, ar(1) arstructure(general) noconstant covstructure(diagonal)) if qtr<=tq(1978q4)
var dln_inv dln_inc dln_consump if qtr<=tq(1978q4), lags(1) noconstant exog(c t)

// predict, see above (Note: uses actual data for forecasting, so we will want
// to ignore the predictions after 1978q4, see below
predict predict_exog2_1, equation(dln_inv)
predict predict_exog2_2, equation(dln_inc)
predict predict_exog2_3, equation(dln_consump)

// We will want to use these values to compare for forecasting, but note that
// this also includes in the columns the value for 1978q4 (i.e. a VAR(1) needs
// 1 sample from which to compute forecasts.
fcast compute fcast_exog2_ , dynamic(tq(1979q1)) step(16) replace

// VAR(2)
dfactor (dln_inv dln_inc = , ar(1/2) arstructure(general) noconstant covstructure(unstructured)) if qtr<=tq(1978q4)
estat ic

// predict + forecast, see above
predict predict_var2_1, dynamic(tq(1979q1)) equation(dln_inv)
predict predict_var2_2, dynamic(tq(1979q1)) equation(dln_inc)

// predict + dynamic predict + forecast, see above
predict dyn_predict_var2_1, dynamic(tq(1961q1)) equation(dln_inv)
predict dyn_predict_var2_2, dynamic(tq(1961q1)) equation(dln_inc)

outsheet pred* dyn* fcas* using results_var_stata.csv, comma replace

// VARMA(1,1)
// Note: Stata does not have this built-in, so we need to create the state space form ourselves
// This replicates example 4 from the Stata documentation
use https://www.stata-press.com/data/r12/manufac, clear

gen dlncaputil = D.lncaputil
gen dlnhours = D.lnhours

constraint 1 [u1]L.u2 = 1
constraint 2 [u1]e.u1 = 1
constraint 3 [u3]e.u3 = 1
constraint 4 [dlncaputil]u1 = 1
constraint 5 [dlnhours]u3 = 1

sspace (u1 L.u1 L.u2 e.u1, state noconstant) ///
       (u2 e.u1, state noconstant) ///
       (u3 L.u1 L.u3 e.u3, state noconstant) ///
       (dlncaputil u1, noconstant) ///
       (dlnhours u3, noconstant), ///
       constraints(1/5) technique(nr) covstate(diagonal)

// in-sample predict + forecast 5 new observations (sample ends at 1999m12)
tsappend, add(5)
predict predict_varma11_1, dynamic(tm(2009m1)) equation(dlncaputil)
predict predict_varma11_2, dynamic(tm(2009m1)) equation(dlnhours)

// predict + dynamic predict + forecast
predict dyn_predict_varma11_1, dynamic(tm(2000m1)) equation(dlncaputil)
predict dyn_predict_varma11_2, dynamic(tm(2000m1)) equation(dlnhours)

// VMA(1)
// Note: Stata does not have this built-in, so we need to create the state space form ourselves
constraint 1 [u1]L.u2 = 1
constraint 2 [u1]e.u1 = 1
constraint 3 [u3]e.u3 = 1
constraint 4 [dlncaputil]u1 = 1
constraint 5 [dlnhours]u3 = 1

sspace (u1 L.u2 e.u1, state noconstant) ///
       (u2 e.u1, state noconstant) ///
       (u3 e.u3, state noconstant) ///
       (dlncaputil u1, noconstant) ///
       (dlnhours u3, noconstant), ///
       constraints(1/5) technique(nr) covstate(diagonal)

// in-sample predict + forecast 5 new observations (sample ends at 1999m12)
predict predict_vma1_1, dynamic(tm(2009m1)) equation(dlncaputil)
predict predict_vma1_2, dynamic(tm(2009m1)) equation(dlnhours)

// predict + dynamic predict + forecast
predict dyn_predict_vma1_1, dynamic(tm(2000m1)) equation(dlncaputil)
predict dyn_predict_vma1_2, dynamic(tm(2000m1)) equation(dlnhours)


outsheet pred* dyn* using results_varmax_stata.csv, comma replace
