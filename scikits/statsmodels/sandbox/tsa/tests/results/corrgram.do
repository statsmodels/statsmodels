* Stata do file for getting test results
insheet using "/home/skipper/statsmodels/statsmodels-skipper/scikits/statsmodels/datasets/macrodata/macrodata.csv", double
ac realgdp, gen(acvar)
ac realgdp, gen(acvarfft) fft
corrgram realgdp
matrix Q = r(Q)'
svmat Q, names(Q)
matrix PAC = r(PAC)'
svmat PAC, names(PAC)
