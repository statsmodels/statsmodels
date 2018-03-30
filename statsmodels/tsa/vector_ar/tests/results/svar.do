insheet using "/home/skipper/statsmodels/statsmodels-skipper/scikits/statsmodels/datasets/macrodata/macrodata.csv", double clear

gen qtrdate=yq(year,quarter)
format qtrdate %tq
tsset qtrdate

gen lgdp = log(realgdp)
gen lcons = log(realcons)
gen linv = log(realinv)

gen gdp = D.lgdp
gen cons = D.lcons
gen inv = D.linv

matrix A = (1,0,0\.,1,0\.,.,1)
matrix B = (.,0,0\0,.,0\0,0,.)

svar gdp cons inv, aeq(A) beq(B) lags(1/3) var

