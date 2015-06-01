clear
insheet using results_realgdpar_stata.csv
keep value

gen lgdp = log(value)
gen dlgdp = lgdp - lgdp[_n-1]
gen quarter = _n
tsset quarter

// Estimate an ARMA(3,0)
arima dlgdp if ~missing(dlgdp), arima(3,0,0) noconstant
matrix b = e(b)
matrix b = (b[1,1..3],1,1,1,b[1,4]^2)

// Estimate via a state-space model
constraint 1 [dlgdp]u1 = 1
constraint 2 [u2]L.u1 = 1
constraint 3 [u3]L.u2 = 1

sspace (u1 L.u1 L.u2 L.u3, state noconstant) ///
       (u2 L.u1, state noconstant noerror) ///
       (u3 L.u2, state noconstant noerror) ///
       (dlgdp u1, noconstant noerror) if ~missing(dlgdp), ///
       constraints(1/3) covstate(diagonal) from(b)

// Estimate an ARMA(12,0)
arima dlgdp if ~missing(dlgdp), arima(12,0,0) noconstant
matrix b = e(b)
matrix b = (b[1,1..12],1,1,1, 1,1,1, 1,1,1, 1,1,1,b[1,13]^2)

// Estimate via a state-space model
constraint 1 [dlgdp]u1 = 1
constraint 2 [u2]L.u1 = 1
constraint 3 [u3]L.u2 = 1
constraint 4 [u4]L.u3 = 1
constraint 5 [u5]L.u4 = 1
constraint 6 [u6]L.u5 = 1
constraint 7 [u7]L.u6 = 1
constraint 8 [u8]L.u7 = 1
constraint 9 [u9]L.u8 = 1
constraint 10 [u10]L.u9 = 1
constraint 11 [u11]L.u10 = 1
constraint 12 [u12]L.u11 = 1

sspace (u1 L.u1 L.u2 L.u3 L.u4 L.u5 L.u6 L.u7 L.u8 L.u9 L.u10 L.u11 L.u12, state noconstant) ///
       (u2 L.u1, state noconstant noerror) ///
       (u3 L.u2, state noconstant noerror) ///
       (u4 L.u3, state noconstant noerror) ///
       (u5 L.u4, state noconstant noerror) ///
       (u6 L.u5, state noconstant noerror) ///
       (u7 L.u6, state noconstant noerror) ///
       (u8 L.u7, state noconstant noerror) ///
       (u9 L.u8, state noconstant noerror) ///
       (u10 L.u9, state noconstant noerror) ///
       (u11 L.u10, state noconstant noerror) ///
       (u12 L.u11, state noconstant noerror) ///
       (dlgdp u1, noconstant noerror) if ~missing(dlgdp), ///
       constraints(1/12) covstate(diagonal) from (b)

// Save the estimated states
predict est_u1-est_u12, states equation(u1 u2 u3 u4 u5 u6 u7 u8 u9 u10 u11 u12)

// Save the filtered states
predict u1-u12, states equation(u1 u2 u3 u4 u5 u6 u7 u8 u9 u10 u11 u12) smethod(filter)

// Save the standardized residuals
predict rstd, rstandard

// Output
outsheet value u1-u12 est_u1-est_u12 rstd using results_realgdpar_stata.csv, comma replace
