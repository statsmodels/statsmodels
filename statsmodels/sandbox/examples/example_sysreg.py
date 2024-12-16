"""Example: statsmodels.sandbox.sysreg
"""
#TODO: this is going to change significantly once we have a panel data structure
from statsmodels.compat.python import asbytes, lmap

import numpy as np

import statsmodels.api as sm
from statsmodels.sandbox.regression.gmm import IV2SLS
from statsmodels.sandbox.sysreg import SUR, Sem2SLS

#for Python 3 compatibility

# Seemingly Unrelated Regressions (SUR) Model

# This example uses the subset of the Grunfeld data in Greene's Econometric
# Analysis Chapter 14 (5th Edition)

grun_data = sm.datasets.grunfeld.load()

firms = ['General Motors', 'Chrysler', 'General Electric', 'Westinghouse',
        'US Steel']
#for Python 3 compatibility
firms = lmap(asbytes, firms)

grun_exog = grun_data.exog
grun_endog = grun_data.endog

# Right now takes SUR takes a list of arrays
# The array alternates between the LHS of an equation and RHS side of an
# equation
# This is very likely to change
grun_sys = []
for i in firms:
    index = grun_exog['firm'] == i
    grun_sys.append(grun_endog[index])
    exog = grun_exog[index][['value','capital']].view(float).reshape(-1,2)
    exog = sm.add_constant(exog, prepend=True)
    grun_sys.append(exog)

# Note that the results in Greene (5th edition) uses a slightly different
# version of the Grunfeld data. To reproduce Table 14.1 the following changes
# are necessary.
grun_sys[-2][5] = 261.6
grun_sys[-2][-3] = 645.2
grun_sys[-1][11,2] = 232.6

grun_mod = SUR(grun_sys)
grun_res = grun_mod.fit()
print("Results for the 2-step GLS")
print("Compare to Greene Table 14.1, 5th edition")
print(grun_res.params)
# or you can do an iterative fit
# you have to define a new model though this will be fixed
# TODO: note the above
print("Results for iterative GLS (equivalent to MLE)")
print("Compare to Greene Table 14.3")
#TODO: these are slightly off, could be a convergence issue
# or might use a different default DOF correction?
grun_imod = SUR(grun_sys)
grun_ires = grun_imod.fit(igls=True)
print(grun_ires.params)

# Two-Stage Least Squares for Simultaneous Equations
#TODO: we are going to need *some kind* of formula framework

# This follows the simple macroeconomic model given in
# Greene Example 15.1 (5th Edition)
# The data however is from statsmodels and is not the same as
# Greene's

# The model is
# consumption: c_{t} = \alpha_{0} + \alpha_{1}y_{t} + \alpha_{2}c_{t-1} + \epsilon_{t1}
# investment: i_{t} = \beta_{0} + \beta_{1}r_{t} + \beta_{2}\left(y_{t}-y_{t-1}\right) + \epsilon_{t2}
# demand: y_{t} = c_{t} + I_{t} + g_{t}

# See Greene's Econometric Analysis for more information

# Load the data
macrodata = sm.datasets.macrodata.load().data

# Not needed, but make sure the data is sorted
macrodata = np.sort(macrodata, order=['year','quarter'])

# Impose the demand restriction
y = macrodata['realcons'] + macrodata['realinv'] + macrodata['realgovt']

# Build the system
macro_sys = []
# First equation LHS
macro_sys.append(macrodata['realcons'][1:]) # leave off first date
# First equation RHS
exog1 = np.column_stack((y[1:],macrodata['realcons'][:-1]))
#TODO: it might be nice to have "lag" and "lead" functions
exog1 = sm.add_constant(exog1, prepend=True)
macro_sys.append(exog1)
# Second equation LHS
macro_sys.append(macrodata['realinv'][1:])
# Second equation RHS
exog2 = np.column_stack((macrodata['tbilrate'][1:], np.diff(y)))
exog2 = sm.add_constant(exog2, prepend=True)
macro_sys.append(exog2)

# We need to say that y_{t} in the RHS of equation 1 is an endogenous regressor
# We will call these independent endogenous variables
# Right now, we use a dictionary to declare these
indep_endog = {0 : [1]}

# We also need to create a design of our instruments
# This will be done automatically in the future
instruments = np.column_stack((macrodata[['realgovt',
    'tbilrate']][1:].view(float).reshape(-1,2),macrodata['realcons'][:-1],
    y[:-1]))
instruments = sm.add_constant(instruments, prepend=True)
macro_mod = Sem2SLS(macro_sys, indep_endog=indep_endog, instruments=instruments)
# Right now this only returns parameters
macro_params = macro_mod.fit()
print("The parameters for the first equation are correct.")
print("The parameters for the second equation are not.")
print(macro_params)

#TODO: Note that the above is incorrect, because we have no way of telling the
# model that *part* of the y_{t} - y_{t-1} is an independent endogenous variable
# To correct for this we would have to do the following
y_instrumented = macro_mod.wexog[0][:,1]
whitened_ydiff = y_instrumented - y[:-1]
wexog = np.column_stack((macrodata['tbilrate'][1:],whitened_ydiff))
wexog = sm.add_constant(wexog, prepend=True)
correct_params = sm.GLS(macrodata['realinv'][1:], wexog).fit().params

print("If we correctly instrument everything, then these are the parameters")
print("for the second equation")
print(correct_params)
print("Compare to output of R script statsmodels/sandbox/tests/macrodata.s")

print('\nUsing IV2SLS')
miv = IV2SLS(macro_sys[0], macro_sys[1], instruments)
resiv = miv.fit()
print("equation 1")
print(resiv.params)
miv2 = IV2SLS(macro_sys[2], macro_sys[3], instruments)
resiv2 = miv2.fit()
print("equation 2")
print(resiv2.params)

### Below is the same example using Greene's data ###

run_greene = 0
if run_greene:
    try:
        data3 = np.genfromtxt('/home/skipper/school/MetricsII/Greene \
TableF5-1.txt', names=True)
    except Exception:
        raise ValueError("Based on Greene TableF5-1.  You should download it "
                         "from his web site and edit this script accordingly.")

    # Example 15.1 in Greene 5th Edition
# c_t = constant + y_t + c_t-1
# i_t = constant + r_t + (y_t - y_t-1)
# y_t = c_t + i_t + g_t
    sys3 = []
    sys3.append(data3['realcons'][1:])  # have to leave off a beg. date
# impose 3rd equation on y
    y = data3['realcons'] + data3['realinvs'] + data3['realgovt']

    exog1 = np.column_stack((y[1:],data3['realcons'][:-1]))
    exog1 = sm.add_constant(exog1, prepend=False)
    sys3.append(exog1)
    sys3.append(data3['realinvs'][1:])
    exog2 = np.column_stack((data3['tbilrate'][1:],
        np.diff(y)))
    # realint is missing 1st observation
    exog2 = sm.add_constant(exog2, prepend=False)
    sys3.append(exog2)
    indep_endog = {0 : [0]} # need to be able to say that y_1 is an instrument..
    instruments = np.column_stack((data3[['realgovt',
        'tbilrate']][1:].view(float).reshape(-1,2),data3['realcons'][:-1],
        y[:-1]))
    instruments = sm.add_constant(instruments, prepend=False)
    sem_mod = Sem2SLS(sys3, indep_endog = indep_endog, instruments=instruments)
    sem_params = sem_mod.fit()  # first equation is right, but not second?
                                # should y_t in the diff be instrumented?
                                # how would R know this in the script?
    # well, let's check...
    y_instr = sem_mod.wexog[0][:,0]
    wyd = y_instr - y[:-1]
    wexog = np.column_stack((data3['tbilrate'][1:],wyd))
    wexog = sm.add_constant(wexog, prepend=False)
    params = sm.GLS(data3['realinvs'][1:], wexog).fit().params

    print("These are the simultaneous equation estimates for Greene's \
example 13-1 (Also application 13-1 in 6th edition.")
    print(sem_params)
    print("The first set of parameters is correct.  The second set is not.")
    print("Compare to the solution manual at \
http://pages.stern.nyu.edu/~wgreene/Text/econometricanalysis.htm")
    print("The reason is the restriction on (y_t - y_1)")
    print("Compare to R script GreeneEx15_1.s")
    print("Somehow R carries y.1 in yd to know that it needs to be \
instrumented")
    print("If we replace our estimate with the instrumented one")
    print(params)
    print("We get the right estimate")
    print("Without a formula framework we have to be able to do restrictions.")
# yep!, but how in the world does R know this when we just fed it yd??
# must be implicit in the formula framework...
# we are going to need to keep the two equations separate and use
# a restrictions matrix.  Ugh, is a formula framework really, necessary to get
# around this?
