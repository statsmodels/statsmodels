insheet using /Users/fulton/projects/statsmodels/statsmodels/datasets/macrodata/macrodata.csv, clear

gen tq = yq(year, quarter)
format tq %tq
tsset tq

cusum6 cpi m1, rr(rr) cs(cusum) lw(lw) uw(uw) cs2(cusum2) lww(lww) uww(uww) noplot

outsheet rr-lww using results_rls_stata.csv, comma replace
