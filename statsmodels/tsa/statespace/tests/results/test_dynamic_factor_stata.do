webuse lutkepohl2, clear
tsset

// Dynamic Factors: Static Factor model
dfactor (dln_inv dln_inc dln_consump = , noconstant )  (f = , ar(1/2)) if qtr<=tq(1978q4)

// predict + forecast, see above
predict predict_dfm_1, dynamic(tq(1979q1)) equation(dln_inv)
predict predict_dfm_2, dynamic(tq(1979q1)) equation(dln_inc)
predict predict_dfm_3, dynamic(tq(1979q1)) equation(dln_consump)

// predict + dynamic predict + forecast, see above
predict dyn_predict_dfm_1, dynamic(tq(1961q1)) equation(dln_inv)
predict dyn_predict_dfm_2, dynamic(tq(1961q1)) equation(dln_inc)
predict dyn_predict_dfm_3, dynamic(tq(1961q1)) equation(dln_consump)

// Dynamic Factors: Static Factor model with 2 factors
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

outsheet pred* dyn* using results_dynamic_factor_stata.csv, comma replace

