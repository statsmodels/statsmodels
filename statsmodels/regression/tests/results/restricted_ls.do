
set type double

tempname filename
local filename = "results_restrictedls.py"

insheet using "E:\Josef\eclipsegworkspace\statsmodels-git\statsmodels-all-new2_py27\statsmodels\statsmodels\regression\tests\rlsdata.txt"
gen y2 = y*y
egen ystd = std(y)
gen ystd2 = ystd * ystd


constraint 1 ne + nc + s + w=0
cnsreg g y y2 ne nc s w, constraints(1) collinear

/* boiler plate, add matrices if needed */
tempname cov
tempname cov_prior
tempname params_table
matrix cov = e(V)
matrix cov_prior = e(Vprior)
matrix params_table = r(table)'
estmat2nparray params_table cov cov_prior, saving(`filename') format("%16.0g") resname("rls1_nonrobust") replace
/*------------------*/

cnsreg g y y2 ne nc s w, constraints(1) collinear vce(robust)

* boiler plate, add matrices if needed */
tempname cov
tempname cov_prior
tempname params_table
matrix cov = e(V)
matrix cov_prior = e(Vprior)
matrix params_table = r(table)'
estmat2nparray params_table cov cov_prior, saving(`filename') format("%16.0g") resname("rls1_robust") append
/*------------------*/



cnsreg g ystd ystd2 ne nc s w, constraints(1) collinear

* boiler plate, add matrices if needed */
tempname cov
tempname cov_prior
tempname params_table
matrix cov = e(V)
matrix cov_prior = e(Vprior)
matrix params_table = r(table)'
estmat2nparray params_table cov cov_prior, saving(`filename') format("%16.0g") resname("rls2_nonrobust") append
/*------------------*/

test (ne + nc + s + w = 0)
test (ystd=0) (ne + nc = 0)
test (ystd=0) (ne + nc + s + w = 0)
test (ystd=0)


cnsreg g ystd ystd2 ne nc s w, constraints(1) collinear vce(robust)

* boiler plate, add matrices if needed */
tempname cov
tempname cov_prior
tempname params_table
matrix cov = e(V)
matrix cov_prior = e(Vprior)
matrix params_table = r(table)'
estmat2nparray params_table cov cov_prior, saving(`filename') format("%16.0g") resname("rls2_robust") append
/*------------------*/


gen groups = 1*ne + 2*nc + 3*s + 4*w
cnsreg g ystd ystd2 ne nc s w, constraints(1) collinear vce(cluster groups)

* boiler plate, add matrices if needed */
tempname cov
tempname cov_prior
tempname params_table
matrix cov = e(V)
matrix cov_prior = e(Vprior)
matrix params_table = r(table)'
estmat2nparray params_table cov cov_prior, saving(`filename') format("%16.0g") resname("rls2_cluster") append
/*------------------*/


cnsreg g ystd ne nc s w [pweight=ystd2], constraints(1) collinear

* boiler plate, add matrices if needed */
tempname cov
tempname cov_prior
tempname params_table
matrix cov = e(V)
matrix cov_prior = e(Vprior)
matrix params_table = r(table)'
estmat2nparray params_table cov cov_prior, saving(`filename') format("%16.0g") resname("rls3_weights") append
/*------------------*/

