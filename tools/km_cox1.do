/* Run Survival models and save results

Author: Josef Perktold
based on example from Stata help

*/

clear
*basic Kaplan-Meier
capture use "E:\Josef\statawork\stan3.dta", clear
if _rc != 0 webuse stan3
capture save "E:\Josef\statawork\stan3.dta"
capture erase surf.dta
sts list,  saving("surf")
use "E:\Josef\statawork\surf.dta", clear
outsheet using "surv_km.csv", comma replace

* Kaplan-Meier with by
use "E:\Josef\statawork\stan3.dta", clear
capture erase surf2.dta
sts list, by(posttran)  saving("surf2")
use "E:\Josef\statawork\surf2.dta", clear
outsheet using "surv_km2.csv", comma replace


* Cox Proportional Hazard
use "E:\Josef\statawork\stan3.dta", clear
stcox age posttran ,  estimate norobust
ereturn list,
matlist e(V)
matlist e(p)

* the next doesn't work
* predict predictall, hr xb stdp basesurv basechazard basehc mgale csnell deviance ldisplace lmax effects
/* generate in python:
>>> for i in 'hr xb stdp basesurv basechazard basehc mgale csnell deviance ldisplace lmax effects'.split(): print 'predict %s, %s' % (i,i)
*/
predict hr, hr
predict xb, xb
predict stdp, stdp
predict basesurv, basesurv
predict basechazard, basechazard
predict basehc, basehc
predict mgale, mgale
predict csnell, csnell
predict deviance, deviance
predict ldisplace, ldisplace
predict lmax, lmax
*capture predict effects, effects

outsheet hr xb stdp basesurv basechazard basehc mgale csnell deviance ldisplace lmax using "surv_coxph.csv", comma replace

* replay
stcox

matrix cov = e(V)
svmat cov, names(cov)

* get the colnames and rownames
capture drop nacol narow
matrix params_table = r(table)'
gen nacol = "`: colnames params_table'"
gen narow = "`: rownames params_table'"
di nacol
di narow

*2nd version

capture drop rown2
local colnms: coln params_table
gen str rown2 = "`colnms'"
di rown2

svmat params_table, names(params_table)
estmat2nparray params_table cov, saving("results_coxphrobust.py") format("%16.0g") append
* other options, no matrices or no est results
*estmat2nparray params_table cov, saving("results_coxphrobust_2.py") format("%16.0g") append noest
*estmat2nparray , saving("results_coxphrobust_2.py") format("%16.0g") append
