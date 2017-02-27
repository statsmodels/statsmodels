webuse lutkepohl2, clear
tsset

// AR(1)
// For results output, do:
// matrix list e(b)
// matrix list e(Sigma)
// for loglike, aic, sbic, hqic, fpe:
// ereturn list
// for non-Lutkepohl IC, do:
// estat ic
// matrix list r(S)
var dln_inv if qtr<=tq(1978q4), lags(1)
estat ic
var dln_inv if qtr<=tq(1978q4), lags(1) lutstats
estat ic
var dln_inv if qtr<=tq(1978q4), lags(1) lutstats dfk
estat ic

// VAR(1)
// For results output, do:
// matrix list e(b)
// matrix list e(Sigma), format(%13.12f)
// for loglike, aic, sbic, hqic, fpe:
// ereturn list
// for non-Lutkepohl IC, do:
// estat ic
// matrix list r(S)
var dln_inv dln_inc dln_consump if qtr<=tq(1978q4), noconstant lags(1)
var dln_inv dln_inc dln_consump if qtr<=tq(1978q4), noconstant lags(1) lutstats
var dln_inv dln_inc dln_consump if qtr<=tq(1978q4), noconstant lags(1) lutstats dfk
