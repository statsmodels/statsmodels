insheet using /Users/fulton/projects/statsmodels/statsmodels/datasets/macrodata/macrodata.csv, clear

gen tq = yq(year, quarter)
format tq %tq
tsset tq

cusum6 cpi m1, rr(rr) cs(cusum) lw(lw) uw(uw) cs2(cusum2) lww(lww) uww(uww) noplot

outsheet rr-lww using results_rls_stata.csv, comma replace

// Section for restricted least squares

constraint 1 m1 + unemp = 1
cnsreg infl m1 unemp, c(1)

disp %18.13f e(ll) // -534.4292052931121
matrix list e(b), format(%15.13f) //  -0.0018477514060   1.0018477514060  -0.7001083844336

mata: se=sqrt(diagonal(st_matrix("e(V)")))
mata: se // .0005369357, .0005369357, .4699552366
