set more off
insheet using /home/skipper/statsmodels/statsmodels/statsmodels/datasets/grunfeld/grunfeld.csv, clear double
encode firm, generate(firmn)
xtset firmn year
xtreg invest value capital, fe

log using /home/skipper/scratch/panel_log.txt, text replace
set trace on
xtreg invest value capital, fe
log close
