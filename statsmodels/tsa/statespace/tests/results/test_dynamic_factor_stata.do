// Helpful commands:
// matrix list e(b)
// matrix d = vecdiag(e(V))
// matrix list d

webuse lutkepohl2, clear
tsset
// use lutkepohl2_s12, clear
// tsset qtr

// Dynamic factors
dfactor (dln_inv dln_inc dln_consump = , noconstant )  (f = , ar(1/2)) if qtr<=tq(1978q4)

// These are predict in-sample + forecast out-of-sample (1979q1 is first out-of sample obs)
predict predict_dfm_1, dynamic(tq(1979q1)) equation(dln_inv)
predict predict_dfm_2, dynamic(tq(1979q1)) equation(dln_inc)
predict predict_dfm_3, dynamic(tq(1979q1)) equation(dln_consump)

// These are predict in-sample for first 3-observations (1960q2-1960q4), then
// dynamic predict for the rest of in-sample observations (1961q1-1978q4), then
// forecast for the remaining periods 1979q1 - 1982q4
predict dyn_predict_dfm_1, dynamic(tq(1961q1)) equation(dln_inv)
predict dyn_predict_dfm_2, dynamic(tq(1961q1)) equation(dln_inc)
predict dyn_predict_dfm_3, dynamic(tq(1961q1)) equation(dln_consump)

// Dynamic factors, with 2 factors
// Note: this does not converge even if we do not enforce iter(#), but this is
// good enough for testing
dfactor (dln_inv dln_inc dln_consump = , noconstant )  (f1 f2 = , ar(1) arstructure(general)) if qtr<=tq(1978q4), iter(1)

// predict + forecast, see above
predict predict_dfm2_1, dynamic(tq(1979q1)) equation(dln_inv)
predict predict_dfm2_2, dynamic(tq(1979q1)) equation(dln_inc)
predict predict_dfm2_3, dynamic(tq(1979q1)) equation(dln_consump)

// predict + dynamic predict + forecast, see above
predict dyn_predict_dfm2_1, dynamic(tq(1961q1)) equation(dln_inv)
predict dyn_predict_dfm2_2, dynamic(tq(1961q1)) equation(dln_inc)
predict dyn_predict_dfm2_3, dynamic(tq(1961q1)) equation(dln_consump)

// Dynamic factors, with 1 exog
gen c = 1
dfactor (dln_inv dln_inc dln_consump = c, noconstant )  (f = , ar(1)) if qtr<=tq(1978q4)

// predict + forecast, see above
predict predict_dfm_exog1_1, dynamic(tq(1979q1)) equation(dln_inv)
predict predict_dfm_exog1_2, dynamic(tq(1979q1)) equation(dln_inc)
predict predict_dfm_exog1_3, dynamic(tq(1979q1)) equation(dln_consump)

// predict + dynamic predict + forecast, see above
predict dyn_predict_dfm_exog1_1, dynamic(tq(1961q1)) equation(dln_inv)
predict dyn_predict_dfm_exog1_2, dynamic(tq(1961q1)) equation(dln_inc)
predict dyn_predict_dfm_exog1_3, dynamic(tq(1961q1)) equation(dln_consump)

// Dynamic factors, with 2 exog
gen t = _n
dfactor (dln_inv dln_inc dln_consump = c t, noconstant )  (f = , ar(1)) if qtr<=tq(1978q4)

// predict + forecast, see above
predict predict_dfm_exog2_1, dynamic(tq(1979q1)) equation(dln_inv)
predict predict_dfm_exog2_2, dynamic(tq(1979q1)) equation(dln_inc)
predict predict_dfm_exog2_3, dynamic(tq(1979q1)) equation(dln_consump)

// predict + dynamic predict + forecast, see above
predict dyn_predict_dfm_exog2_1, dynamic(tq(1961q1)) equation(dln_inv)
predict dyn_predict_dfm_exog2_2, dynamic(tq(1961q1)) equation(dln_inc)
predict dyn_predict_dfm_exog2_3, dynamic(tq(1961q1)) equation(dln_consump)

// Dynamic factors, with general errors (VAR + unstructured), on demeaned data
// (have to demean, otherwise get non-stationarity start params for the error
// VAR)
dfactor (dln_inv dln_inc dln_consump = , noconstant ar(1) arstructure(general) covstructure(unstructured))  (f = , ar(1)) if qtr<=tq(1978q4)

// predict + forecast, see above
predict predict_dfm_gen_1, dynamic(tq(1979q1)) equation(dln_inv)
predict predict_dfm_gen_2, dynamic(tq(1979q1)) equation(dln_inc)
predict predict_dfm_gen_3, dynamic(tq(1979q1)) equation(dln_consump)

// predict + dynamic predict + forecast, see above
predict dyn_predict_dfm_gen_1, dynamic(tq(1961q1)) equation(dln_inv)
predict dyn_predict_dfm_gen_2, dynamic(tq(1961q1)) equation(dln_inc)
predict dyn_predict_dfm_gen_3, dynamic(tq(1961q1)) equation(dln_consump)

// Dynamic factors, with AR(2) errors
dfactor (dln_inv dln_inc dln_consump = , noconstant ar(1/2))  (f = , ar(1)) if qtr<=tq(1978q4)

// predict + forecast, see above
predict predict_dfm_ar2_1, dynamic(tq(1979q1)) equation(dln_inv)
predict predict_dfm_ar2_2, dynamic(tq(1979q1)) equation(dln_inc)
predict predict_dfm_ar2_3, dynamic(tq(1979q1)) equation(dln_consump)

// predict + dynamic predict + forecast, see above
predict dyn_predict_dfm_ar2_1, dynamic(tq(1961q1)) equation(dln_inv)
predict dyn_predict_dfm_ar2_2, dynamic(tq(1961q1)) equation(dln_inc)
predict dyn_predict_dfm_ar2_3, dynamic(tq(1961q1)) equation(dln_consump)

// Dynamic factors, with scalar error covariance + constant (1 exog)
// Note: estimation does not converge w/o constant term
dfactor (dln_inv dln_inc dln_consump = c, noconstant covstructure(dscalar))  (f = , ar(1)) if qtr<=tq(1978q4)

// predict + forecast, see above
predict predict_dfm_scalar_1, dynamic(tq(1979q1)) equation(dln_inv)
predict predict_dfm_scalar_2, dynamic(tq(1979q1)) equation(dln_inc)
predict predict_dfm_scalar_3, dynamic(tq(1979q1)) equation(dln_consump)

// predict + dynamic predict + forecast, see above
predict dyn_predict_dfm_scalar_1, dynamic(tq(1961q1)) equation(dln_inv)
predict dyn_predict_dfm_scalar_2, dynamic(tq(1961q1)) equation(dln_inc)
predict dyn_predict_dfm_scalar_3, dynamic(tq(1961q1)) equation(dln_consump)

// Static factor (factor with no autocorrelation)
// Note: estimation does not converge w/o constant term
dfactor (dln_inv dln_inc dln_consump = , noconstant)  (f = , ) if qtr<=tq(1978q4)

// predict + forecast, see above
predict predict_sfm_1, dynamic(tq(1979q1)) equation(dln_inv)
predict predict_sfm_2, dynamic(tq(1979q1)) equation(dln_inc)
predict predict_sfm_3, dynamic(tq(1979q1)) equation(dln_consump)

// predict + dynamic predict + forecast, see above
predict dyn_predict_sfm_1, dynamic(tq(1961q1)) equation(dln_inv)
predict dyn_predict_sfm_2, dynamic(tq(1961q1)) equation(dln_inc)
predict dyn_predict_sfm_3, dynamic(tq(1961q1)) equation(dln_consump)

// SUR (exog, no autocorrelation, correlated innovations)
dfactor (dln_inv dln_inc dln_consump = c t, noconstant covstructure(unstructured)) if qtr<=tq(1978q4)

// predict + forecast, see above
predict predict_sur_1, dynamic(tq(1979q1)) equation(dln_inv)
predict predict_sur_2, dynamic(tq(1979q1)) equation(dln_inc)
predict predict_sur_3, dynamic(tq(1979q1)) equation(dln_consump)

// predict + dynamic predict + forecast, see above
predict dyn_predict_sur_1, dynamic(tq(1961q1)) equation(dln_inv)
predict dyn_predict_sur_2, dynamic(tq(1961q1)) equation(dln_inc)
predict dyn_predict_sur_3, dynamic(tq(1961q1)) equation(dln_consump)

// SUR (exog, vector error autocorrelation, uncorrelated innovations)
dfactor (dln_inv dln_inc = c t, noconstant ar(1)) if qtr<=tq(1978q4)

// predict + forecast, see above
predict predict_sur_auto_1, dynamic(tq(1979q1)) equation(dln_inv)
predict predict_sur_auto_2, dynamic(tq(1979q1)) equation(dln_inc)

// predict + dynamic predict + forecast, see above
predict dyn_predict_sur_auto_1, dynamic(tq(1961q1)) equation(dln_inv)
predict dyn_predict_sur_auto_2, dynamic(tq(1961q1)) equation(dln_inc)

outsheet pred* dyn* using results_dynamic_factor_stata.csv, comma replace
