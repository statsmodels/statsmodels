set more off
insheet using /home/skipper/statsmodels/statsmodels/statsmodels/datasets/grunfeld/grunfeld.csv, clear double
encode firm, generate(firmn)
xtset firmn year
xtreg invest value capital, fe

log using /home/skipper/scratch/panel_log.txt, text replace
set trace on
xtreg invest value capital, fe
log close

/* Code to estimate two-way effects
xtreg invest value capital i.year, fe

/*Generate the time fixed effects. Express them in terms of deviations from
the conditional mean of the sample rather than in terms of exclusion from the
base case (1935)*/


qui tabulate year, generate(yr)
local j 0
forvalues i = 1936/1954 {
    local ++j
    rename yr`j' yr`i'
    qui replace yr`i' = yr`i' - yr1
}
drop yr1

xtreg invest value capital yr*, fe
*/
