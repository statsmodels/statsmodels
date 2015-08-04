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

outsheet pred* dyn* using results_varmax_stata.csv, comma replace
