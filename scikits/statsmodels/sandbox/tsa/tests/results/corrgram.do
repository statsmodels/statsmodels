* Stata do file for getting test results
insheet using "/home/skipper/statsmodels/statsmodels-skipper/scikits/statsmodels/datasets/macrodata/macrodata.csv", double clear
gen qtrdate=yq(year,quarter)
format qtrdate %tq
tsset qtrdate
ac realgdp, gen(acvar)
ac realgdp, gen(acvarfft) fft
corrgram realgdp
matrix Q = r(Q)'
svmat Q, names(Q)
matrix PAC = r(PAC)'
svmat PAC, names(PAC)
rename PAC1 PACOLS
pac realgdp, yw gen(PACYW)
outsheet acvar acvarfft Q1 PACOLS PACYW using "/home/skipper/statsmodels/statsmodels-skipper/scikits/statsmodels/sandbox/tsa/tests/results/results_corrgram.csv", comma replace

