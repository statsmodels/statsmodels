// Dataset
use https://www.stata-press.com/data/r12/wpi1, clear
rename t time

set more off
set maxiter 100

// 25, 43, 48, 52

// Create data for exog test
gen x = (wpi - floor(wpi))^2

// Create data for deterministic trend tests
gen c = 1
gen t = _n
gen t2 = t^2
gen t3 = t^3

// Dummy column for saving LLFs
gen mod = ""
gen llf = .
gen parameters = ""

// Program to save the results
program drop save_results
program save_results
    args i
    replace mod = e(cmdline) in `i'
    replace llf = e(ll) in `i'
    
    matrix b = e(b)
    local params = ""
    local params = string(b[1,1])
    local nparams = e(k)
    if `nparams' > 1 {
        foreach j of numlist 2/`nparams' {
            local params = "`params'," + string(b[1,`j'])
        }
    }
    replace parameters = "`params'" in `i'
end

// AR: (p,0,0) x (0,0,0,0)
capture arima wpi, arima(3,0,0) noconstant vce(oim)
save_results 1

// AR and deterministic trends ('nc', 'c', 'ct', polynomial)

// 'c'
capture arima wpi c, arima(3,0,0) noconstant vce(oim)
save_results 2

// 'ct'
capture arima wpi c t, arima(3,0,0) noconstant vce(oim)
save_results 3

// polynomial [1,0,0,1]
capture arima wpi c t3, arima(3,0,0) noconstant vce(oim)
save_results 4

// AR and I(d): (p,d,0) x (0,0,0,0)
capture arima wpi, arima(3,2,0) noconstant vce(oim)
save_results 5

// AR and I(D): (p,0,0) x (0,D,0,s)
capture arima wpi, arima(3,0,0) sarima(0,2,0,4) noconstant vce(oim)
save_results 6

// AR and diffuse initialization
capture arima wpi, arima(3,0,0) noconstant vce(oim) diffuse
save_results 7

// ARX
capture arima wpi x, arima(3,0,0) noconstant vce(oim)
save_results 8

// MA: (0,0,q) x (0,0,0,0)
capture arima wpi, arima(0,0,3) noconstant vce(oim)
save_results 9

// MA and deterministic trends ('nc', 'c', 'ct', polynomial)

// 'c'
capture arima wpi c, arima(0,0,3) noconstant vce(oim)
save_results 10

// 'ct'
capture arima wpi c t, arima(0,0,3) noconstant vce(oim)
save_results 11

// polynomial [1,0,0,1]
capture arima wpi c t3, arima(0,0,3) noconstant vce(oim)
save_results 12

// MA and I(d): (0,d,q) x (0,0,0,0)
capture arima wpi, arima(0,2,3) noconstant vce(oim)
save_results 13

// MA and I(D): (p,0,0) x (0,D,0,s)
capture arima wpi, arima(0,0,3) sarima(0,2,0,4) noconstant vce(oim)
save_results 14

// MA and diffuse initialization
capture arima wpi, arima(0,0,3) noconstant vce(oim) diffuse
save_results 15

// MAX
capture arima wpi x, arima(0,0,3) noconstant vce(oim)
save_results 16


// ARMA: (p,0,q) x (0,0,0,0)
capture arima wpi, arima(3,0,3) noconstant vce(oim)
save_results 17

// ARMA and deterministic trends ('nc', 'c', 'ct', polynomial)

// 'c'
capture arima wpi c, arima(3,0,2) noconstant vce(oim)
save_results 18

// 'ct'
capture arima wpi c t, arima(3,0,2) noconstant vce(oim)
save_results 19

// polynomial [1,0,0,1]
capture arima wpi c t3, arima(3,0,2) noconstant vce(oim)
save_results 20

// ARMA and I(d): (p,d,q) x (0,0,0,0)
capture arima wpi, arima(3,2,2) noconstant vce(oim)
save_results 21

// ARMA and I(D): (p,0,q) x (0,D,0,s)
capture arima wpi, arima(3,0,2) sarima(0,2,0,4) noconstant vce(oim)
save_results 22

// ARMA and I(d) and I(D): (p,d,q) x (0,D,0,s)
capture arima wpi, arima(3,2,2) sarima(0,2,0,4) noconstant vce(oim)
save_results 23

// ARMA and diffuse initialization
capture arima wpi, arima(3,0,2) noconstant vce(oim) diffuse
save_results 24

// ARMAX
capture arima wpi x, arima(3,0,2) noconstant vce(oim)
save_results 25

// SAR: (0,0,0) x (P,0,0,s)
capture arima wpi, sarima(3,0,0,4) noconstant vce(oim)
save_results 26

// SAR and deterministic trends ('nc', 'c', 'ct', polynomial)

// 'c'
capture arima wpi c, sarima(3,0,0,4) noconstant vce(oim)
save_results 27

// 'ct'
capture arima wpi c t, sarima(3,0,0,4) noconstant vce(oim)
save_results 28

// polynomial [1,0,0,1]
capture arima wpi c t3, sarima(3,0,0,4) noconstant vce(oim)
save_results 29

// SAR and I(d): (0,d,0) x (P,0,0,s)
capture arima wpi, arima(0,2,0) sarima(3,0,0,4) noconstant vce(oim)
save_results 30

// SAR and I(D): (0,0,0) x (P,D,0,s)
capture arima wpi, sarima(3,2,0,4) noconstant vce(oim)
save_results 31

// SAR and diffuse initialization
capture arima wpi, sarima(3,0,0,4) noconstant vce(oim) diffuse
save_results 32

// SARX
capture arima wpi x, sarima(3,0,0,4) noconstant vce(oim)
save_results 33

// SMA
capture arima wpi, sarima(0,0,3,4) noconstant vce(oim)
save_results 34

// SMA and deterministic trends ('nc', 'c', 'ct', polynomial)

// 'c'
capture arima wpi c, sarima(0,0,3,4) noconstant vce(oim)
save_results 35

// 'ct'
capture arima wpi c t, sarima(0,0,3,4) noconstant vce(oim)
save_results 36

// polynomial [1,0,0,1]
capture arima wpi c t3, sarima(0,0,3,4) noconstant vce(oim)
save_results 37

// SMA and I(d): (0,d,0) x (0,0,Q,s)
capture arima wpi, arima(0,2,0) sarima(0,0,3,4) noconstant vce(oim)
save_results 38

// SAR and I(D): (0,0,0) x (0,D,Q,s)

capture arima wpi, sarima(0,2,3,4) noconstant vce(oim)
save_results 39

// SMA and diffuse initialization
capture arima wpi, sarima(0,0,3,4) noconstant vce(oim) diffuse
save_results 40

// SMAX
capture arima wpi x, sarima(0,0,3,4) noconstant vce(oim)
save_results 41

// SARMA: (0,0,0) x (P,0,Q,s)
capture arima wpi, sarima(3,0,2,4) noconstant vce(oim)
save_results 42

// SARMA and deterministic trends ('nc', 'c', 'ct', polynomial)

// 'c'
capture arima wpi c, sarima(3,0,2,4) noconstant vce(oim)
save_results 43

// 'ct'
capture arima wpi c t, sarima(3,0,2,4) noconstant vce(oim)
save_results 44

// polynomial [1,0,0,1]
capture arima wpi c t3, sarima(3,0,2,4) noconstant vce(oim)
save_results 45

// SARMA and I(d): (0,d,0) x (P,0,Q,s)
capture arima wpi, arima(0,2,0) sarima(3,0,2,4) noconstant vce(oim)
save_results 46

// SARMA and I(D): (0,0,0) x (P,D,Q,s)
capture arima wpi, sarima(3,2,2,4) noconstant vce(oim)
save_results 47

// SARMA and I(d) and I(D): (0,d,0) x (P,D,Q,s)
capture arima wpi, arima(0,2,0) sarima(3,2,2,4) noconstant vce(oim)
save_results 48

// SARMA and diffuse initialization
capture arima wpi, sarima(3,0,2,4) noconstant vce(oim) diffuse
save_results 49

// SARMAX
capture arima wpi x, sarima(3,0,2,4) noconstant vce(oim)
save_results 50

// SARIMAX and exogenous
capture arima wpi x, arima(3,2,2) sarima(3,2,2,4) noconstant vce(oim)
save_results 51

// SARIMAX and exogenous and diffuse
capture arima wpi x, arima(3,2,2) sarima(3,2,2,4) noconstant vce(oim) diffuse
save_results 52

// ARMA and exogenous and trend polynomial and missing
gen wpi2 = wpi
gen t32 = (t-1)^3
replace wpi2 = . in 10/19
capture arima D.wpi2 t32 x, arima(3,0,2) noconstant vce(oim)
save_results 53

// Write results
outsheet mod llf parameters using "results_sarimax_coverage.csv" in 1/53, comma replace
