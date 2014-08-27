"""
Test module for the Heckman module.

All tests pass if file runs without error.
"""


import numpy as np
import pandas as pd
import statsmodels.api as sm
import imp
import csv
import urllib
from numpy.testing import assert_
import pdb


from statsmodels.regression import heckman

imp.reload(heckman)

def _prep_female_labor_supply_data():
	############################################################################
	# Gets Greene's female labor supply data, and prepares it for use.
	############################################################################

	VAR_NAMES_ROW = 0
	FIRST_DATA_ROW = 1
	N = 753

	data_url = 'http://people.stern.nyu.edu/wgreene/Text/Edition7/TableF5-1.csv'
	data_prepped = pd.io.parsers.read_csv(data_url)
	data_prepped.columns = [v.strip(' ') for v in data_prepped.columns]

	return data_prepped


def _prep_censored_wage_heckman_exampledata():
	## Get female labor supply data, and construct additional variables ##
	data_prepped = _prep_female_labor_supply_data()

	data = data_prepped[['WW','LFP','AX','WE','CIT','WA','FAMINC','KL6','K618']].copy()
	del data_prepped

	data['AX2'] = data['AX']**2
	data['WA2'] = data['WA']**2
	data['K'] = 0

	data.loc[(data['KL6'] + data['K618'])>0, 'K'] = 1

	# create wage variable that is missing if not working
	data['wage'] = pd.Series(
		[ data['WW'][i] if data['LFP'][i]==1 else np.nan for i in data.index ],
		index=data.index)

	## Split data into dependent variable, independent variables for regression equation,
	## and independent variables for selection equation
	Y = data['wage']


	X = data[['AX','AX2','WE','CIT']]
	X = sm.add_constant(X, prepend=True)

	Z = data[['WA','WA2','FAMINC','WE','K']]
	Z = sm.add_constant(Z, prepend=True)

	## Return as three vars
	return Y, X, Z


def test_heckman_2step():
	############################################################################
	# Tests to make sure that the Heckman 2 step estimates produced by the
	# Heckman module are the same as/very close to the estimates produced by
	# Stata for the female labor supply data.
	############################################################################

	### Fit data with Heckman model using 2 step ###
	## With pandas input with named variables ##
	Y, X, Z = _prep_censored_wage_heckman_exampledata()

	heckman_model = heckman.Heckman(Y,X,Z)
	heckman_res = heckman_model.fit(method='twostep')
	heckman_smry = heckman_res.summary(disp=True)

	## With list input (no names) ##
	heckman_basic_model = heckman.Heckman(Y.tolist(), X.as_matrix().tolist(), Z.as_matrix().tolist())
	heckman_basic_res = heckman_basic_model.fit(method='twostep')
	heckman_basic_smry = heckman_basic_res.summary(disp=True)

	### Check against Stata's estimates ###
	'''
	Stata estimates can be produced with this Stata code:

	insheet using "http://people.stern.nyu.edu/wgreene/Text/Edition7/TableF5-1.csv", comma clear
	assert _N==753
	rename *, upper

	gen AX2 = AX^2
	gen WA2 = WA^2
	gen K = (KL6+K618)>0

	replace WW = . if LFP==0
	heckman WW AX AX2 WE CIT, select(WA WA2 FAMINC WE K) twostep


	which produces the following output

	Heckman selection model -- two-step estimates   Number of obs      =       753
	(regression model with sample selection)        Censored obs       =       325
	                                                Uncensored obs     =       428

	                                                Wald chi2(4)       =     23.33
	                                                Prob > chi2        =    0.0001

	------------------------------------------------------------------------------
	          WW |      Coef.   Std. Err.      z    P>|z|     [95% Conf. Interval]
	-------------+----------------------------------------------------------------
	WW           |
	          AX |    .021061   .0624646     0.34   0.736    -.1013674    .1434893
	         AX2 |   .0001371   .0018782     0.07   0.942    -.0035441    .0038183
	          WE |   .4170174   .1002497     4.16   0.000     .2205316    .6135032
	         CIT |   .4438379   .3158984     1.41   0.160    -.1753116    1.062987
	       _cons |  -.9712003   2.059351    -0.47   0.637    -5.007453    3.065053
	-------------+----------------------------------------------------------------
	select       |
	          WA |   .1853951   .0659667     2.81   0.005     .0561028    .3146874
	         WA2 |  -.0024259   .0007735    -3.14   0.002     -.003942   -.0009098
	      FAMINC |   4.58e-06   4.21e-06     1.09   0.276    -3.66e-06    .0000128
	          WE |   .0981823   .0229841     4.27   0.000     .0531342    .1432303
	           K |  -.4489867   .1309115    -3.43   0.001    -.7055686   -.1924049
	       _cons |  -4.156807   1.402086    -2.96   0.003    -6.904845   -1.408769
	-------------+----------------------------------------------------------------
	mills        |
	      lambda |  -1.097619   1.265986    -0.87   0.386    -3.578906    1.383667
	-------------+----------------------------------------------------------------
	         rho |   -0.34300
	       sigma |  3.2000643
	------------------------------------------------------------------------------

	'''

	## Stata's estimates ##

	stata_reg_coef = {
	          'AX' :    .021061,
	         'AX2' :   .0001371,
	          'WE' :   .4170174,
	         'CIT' :   .4438379,
	       'const' :  -.9712003
	       }

	stata_reg_stderr = {
	          'AX' : .0624646,
	         'AX2' : .0018782,
	          'WE' : .1002497,
	         'CIT' : .3158984,
	       'const' : 2.059351
	       }

	stata_select_coef = {
	          'WA' :   .1853951,
	         'WA2' :  -.0024259,
	      'FAMINC' :   4.58e-06,
	          'WE' :   .0981823,
	           'K' :  -.4489867,
	       'const' :  -4.156807
	       }

	stata_select_stderr = {
	          'WA' :  .0659667,
	         'WA2' :  .0007735,
	      'FAMINC' :  4.21e-06,
	          'WE' :  .0229841,
	           'K' :  .1309115,
	       'const' :  1.402086
	       }

	stata_lambda_coef = -1.097619
	stata_lambda_stderr = 1.265986

	stata_rho = -0.34300
	stata_sigma = 3.2000643


	## check against those estimates
	stata_regvar_ordered = ['const','AX','AX2','WE','CIT']
	stata_selectvar_ordered = ['const','WA','WA2','FAMINC','WE','K']

	stata_reg_coef_arr = np.array([stata_reg_coef[k] for k in stata_regvar_ordered])
	stata_reg_stderr_arr = np.array([stata_reg_stderr[k] for k in stata_regvar_ordered])

	stata_select_coef_arr = np.array([stata_select_coef[k] for k in stata_selectvar_ordered])
	stata_select_stderr_arr = np.array([stata_select_stderr[k] for k in stata_selectvar_ordered])

	## for pandas input with var names ##
	_check_heckman_to_stata(
		stata_reg_coef_arr, stata_reg_stderr_arr,
		stata_select_coef_arr, stata_select_stderr_arr,
		stata_lambda_coef, stata_lambda_stderr,
		stata_rho, stata_sigma,
		heckman_res,
		TOL=1e-3)

	## for basic list input ##
	_check_heckman_to_stata(
		stata_reg_coef_arr, stata_reg_stderr_arr,
		stata_select_coef_arr, stata_select_stderr_arr,
		stata_lambda_coef, stata_lambda_stderr,
		stata_rho, stata_sigma,
		heckman_basic_res,
		TOL=1e-3)


def _check_heckman_to_stata(
		stata_reg_coef_arr, stata_reg_stderr_arr,
		stata_select_coef_arr, stata_select_stderr_arr,
		stata_lambda_coef, stata_lambda_stderr,
		stata_rho, stata_sigma,
		heckman_res,
		TOL=1e-3):
	# this function checks output from the Heckman module against known Stata output;
	# an error occurs if they are different, otherwise function will simply return



	test_list = [
		{'trueval' : stata_reg_coef_arr, 'estval': heckman_res.params},
		{'trueval' : stata_reg_stderr_arr, 'estval': np.sqrt(np.diag(heckman_res.scale*heckman_res.normalized_cov_params))},
		{'trueval' : stata_select_coef_arr, 'estval': heckman_res.select_res.params},
		{'trueval' : stata_select_stderr_arr, 'estval': np.sqrt(np.diag(heckman_res.select_res.scale*heckman_res.select_res.normalized_cov_params))},
		{'trueval' : stata_lambda_coef, 'estval': heckman_res.param_inverse_mills},
		{'trueval' : stata_lambda_stderr, 'estval': heckman_res.stderr_inverse_mills},
		{'trueval' : stata_rho, 'estval': heckman_res.corr_eqnerrors},
		{'trueval' : stata_sigma, 'estval': heckman_res.var_reg_error}
		]

	for test_set in test_list:
		t = np.array(test_set['trueval'])
		e = np.array(test_set['estval'])
		assert_( all(np.atleast_1d( (t-e)/t < TOL )) )


#TODO: add missing non-Pandas data test

	'''
	Stata estimates can be produced with this Stata code:

	insheet using "http://people.stern.nyu.edu/wgreene/Text/Edition7/TableF5-1.csv", comma clear
	assert _N==753
	rename *, upper

	gen AX2 = AX^2
	gen WA2 = WA^2
	gen K = (KL6+K618)>0

	replace WW = . if _n==1
	replace AX = . if _n==2
	replace WA = . if _n==3

	replace WW = . if LFP==0
	heckman WW AX AX2 WE CIT, select(WA WA2 FAMINC WE K) twostep


	which produces the following output

	Heckman selection model -- two-step estimates   Number of obs      =       751
	(regression model with sample selection)        Censored obs       =       326
	                                                Uncensored obs     =       425

	                                                Wald chi2(4)       =     22.85
	                                                Prob > chi2        =    0.0001

	------------------------------------------------------------------------------
	          WW |      Coef.   Std. Err.      z    P>|z|     [95% Conf. Interval]
	-------------+----------------------------------------------------------------
	WW           |
	          AX |   .0178018   .0628961     0.28   0.777    -.1054722    .1410759
	         AX2 |   .0002058   .0018881     0.11   0.913    -.0034948    .0039064
	          WE |    .415142   .1003871     4.14   0.000      .218387     .611897
	         CIT |   .4571676    .318401     1.44   0.151     -.166887    1.081222
	       _cons |  -.9155062   2.065008    -0.44   0.658    -4.962847    3.131835
	-------------+----------------------------------------------------------------
	select       |
	          WA |   .1951469   .0661367     2.95   0.003     .0655214    .3247724
	         WA2 |  -.0025278   .0007753    -3.26   0.001    -.0040474   -.0010083
	      FAMINC |   4.65e-06   4.21e-06     1.11   0.269    -3.59e-06    .0000129
	          WE |   .0988935   .0229896     4.30   0.000     .0538348    .1439522
	           K |  -.4527965   .1308841    -3.46   0.001    -.7093247   -.1962683
	       _cons |  -4.394952   1.406405    -3.12   0.002    -7.151456   -1.638448
	-------------+----------------------------------------------------------------
	mills        |
	      lambda |  -1.107076    1.25778    -0.88   0.379     -3.57228    1.358128
	-------------+----------------------------------------------------------------
	         rho |   -0.34488
	       sigma |  3.2100548
	------------------------------------------------------------------------------


	'''


#TODO: add constant option test


## Run tests if file is run ##
if __name__ == '__main__':
	test_heckman_2step()
