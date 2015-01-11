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
    X = X.astype(float)

    Z = data[['WA','WA2','FAMINC','WE','K']]
    Z = sm.add_constant(Z, prepend=True)
    Z = Z.astype(float)

    ## Return as three vars
    return Y, X, Z


def _load_stata_femalewage_2step_estimates():
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

    then get predictions with

    predict y_pred

    '''

    ## Stata's estimates ##
    # parameter estimates
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

    # predictions on first and last rows of data
    y_new_firstobs = 4.354729
    y_new_lastobs = 3.498265

    retdict = {
        'stata_reg_coef': stata_reg_coef,
        'stata_reg_stderr': stata_reg_stderr,
        'stata_select_coef': stata_select_coef,
        'stata_select_stderr': stata_select_stderr,
        'stata_lambda_coef': stata_lambda_coef,
        'stata_lambda_stderr': stata_lambda_stderr,
        'stata_rho': stata_rho,
        'stata_sigma': stata_sigma,
        'y_new_firstobs': y_new_firstobs,
        'y_new_lastobs': y_new_lastobs
        }

    return retdict


def _load_stata_femalewage_missing_2step_estimates():
    '''
    Stata estimates can be produced with this Stata code:

    insheet using "http://people.stern.nyu.edu/wgreene/Text/Edition7/TableF5-1.csv", comma clear
    assert _N==753
    rename *, upper

    gen AX2 = AX^2
    gen WA2 = WA^2
    gen K = (KL6+K618)>0

    /* start introducing some missings */
    replace WW = . if _n==1
    replace AX = . if _n==2
    replace WA = . if _n==3

    replace WW = . if _n==101
    replace CIT = . if _n==102
    replace FAMINC = . if _n==103

    replace WW = . if _n==201
    replace WE = . if _n==202
    replace K = . if _n==203

    /* last of introduced missings */

    replace WW = . if LFP==0
    heckman WW AX AX2 WE CIT, select(WA WA2 FAMINC WE K) twostep


    which produces the following output

    Heckman selection model -- two-step estimates   Number of obs      =       747
    (regression model with sample selection)        Censored obs       =       328
                                                    Uncensored obs     =       419

                                                    Wald chi2(4)       =     21.90
                                                    Prob > chi2        =    0.0002

    ------------------------------------------------------------------------------
              WW |      Coef.   Std. Err.      z    P>|z|     [95% Conf. Interval]
    -------------+----------------------------------------------------------------
    WW           |
              AX |   .0197836   .0634079     0.31   0.755    -.1044936    .1440607
             AX2 |    .000154   .0018978     0.08   0.935    -.0035657    .0038737
              WE |   .4091202   .1008091     4.06   0.000      .211538    .6067023
             CIT |   .4468051   .3207769     1.39   0.164    -.1819061    1.075516
           _cons |   -.946634   2.087786    -0.45   0.650     -5.03862    3.145352
    -------------+----------------------------------------------------------------
    select       |
              WA |   .1903152   .0661414     2.88   0.004     .0606804      .31995
             WA2 |  -.0024649   .0007751    -3.18   0.001     -.003984   -.0009458
          FAMINC |   4.22e-06   4.21e-06     1.00   0.316    -4.02e-06    .0000125
              WE |   .0989903   .0230147     4.30   0.000     .0538823    .1440982
               K |  -.4578685    .131184    -3.49   0.000    -.7149844   -.2007527
           _cons |  -4.306956   1.406577    -3.06   0.002    -7.063796   -1.550116
    -------------+----------------------------------------------------------------
    mills        |
          lambda |  -.9595565   1.269555    -0.76   0.450    -3.447839    1.528726
    -------------+----------------------------------------------------------------
             rho |   -0.30079
           sigma |  3.1901153
    ------------------------------------------------------------------------------

    then get predictions with

    predict y_pred

    '''

    ## Stata's estimates ##
    # parameter estimates
    stata_reg_coef = {
              'AX' :   .0197836,
             'AX2' :    .000154,
              'WE' :   .4091202,
             'CIT' :   .4468051,
           'const' :   -.946634
           }

    stata_reg_stderr = {
              'AX' :  .0634079,
             'AX2' :  .0018978,
              'WE' :  .1008091,
             'CIT' :  .3207769,
           'const' :  2.087786
           }

    stata_select_coef = {
              'WA' :   .1903152,
             'WA2' :  -.0024649,
          'FAMINC' :   4.22e-06,
              'WE' :   .0989903,
               'K' :  -.4578685,
           'const' :  -4.306956
           }

    stata_select_stderr = {
              'WA' :  .0661414,
             'WA2' :  .0007751,
          'FAMINC' :  4.21e-06,
              'WE' :  .0230147,
               'K' :   .131184,
           'const' :  1.406577
           }

    stata_lambda_coef = -.9595565
    stata_lambda_stderr = 1.269555

    stata_rho = -0.30079
    stata_sigma = 3.1901153

    # predictions on first and last rows of data
    y_new_firstobs = 4.269959

    y_new_lastobs = 3.441829


    retdict = {
        'stata_reg_coef': stata_reg_coef,
        'stata_reg_stderr': stata_reg_stderr,
        'stata_select_coef': stata_select_coef,
        'stata_select_stderr': stata_select_stderr,
        'stata_lambda_coef': stata_lambda_coef,
        'stata_lambda_stderr': stata_lambda_stderr,
        'stata_rho': stata_rho,
        'stata_sigma': stata_sigma,
        'y_new_firstobs': y_new_firstobs,
        'y_new_lastobs': y_new_lastobs
        }

    return retdict


def _check_heckman_to_stata(
        stata_reg_coef_arr, stata_reg_stderr_arr,
        stata_select_coef_arr, stata_select_stderr_arr,
        stata_lambda_coef, stata_lambda_stderr,
        stata_rho, stata_sigma,
        heckman_res,
        TOL=1e-3):
    # This function checks output from the Heckman module against known Stata output;
    # an error occurs if they are different, otherwise function will simply return.
    # This function will not perform assertion test on a particular estimate if None
    # is inputted.

    test_list = [
        {'trueval' : stata_reg_coef_arr, 'estval': heckman_res.params},
        {'trueval' : stata_reg_stderr_arr,
            'estval': np.sqrt(np.diag(heckman_res.scale*heckman_res.normalized_cov_params))},
        {'trueval' : stata_select_coef_arr, 'estval': heckman_res.select_res.params},
        {'trueval' : stata_select_stderr_arr,
            'estval': np.sqrt(np.diag(heckman_res.select_res.scale*heckman_res.select_res.normalized_cov_params))},
        {'trueval' : stata_lambda_coef, 'estval': heckman_res.param_inverse_mills},
        {'trueval' : stata_lambda_stderr, 'estval': heckman_res.stderr_inverse_mills},
        {'trueval' : stata_rho, 'estval': heckman_res.corr_eqnerrors},
        {'trueval' : stata_sigma, 'estval': heckman_res.var_reg_error}
        ]

    for test_set in test_list:
        t = np.array(test_set['trueval'])
        e = np.array(test_set['estval'])

        if t is not None and t.ndim>0:
            assert_( all(np.atleast_1d( (t-e)/t < TOL )) )


def test_heckman_2step(verbose=True):
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
    heckman_smry = heckman_res.summary(disp=verbose)

    ## With list input (no names) ##
    heckman_basic_model = heckman.Heckman(Y.tolist(), X.as_matrix().tolist(), Z.as_matrix().tolist())
    heckman_basic_res = heckman_basic_model.fit(method='twostep')
    heckman_basic_smry = heckman_basic_res.summary(disp=verbose)

    ### Check against Stata's estimates ###
    ## Load Stata's estimates
    retdict = _load_stata_femalewage_2step_estimates()

    stata_reg_coef = retdict['stata_reg_coef']
    stata_reg_stderr = retdict['stata_reg_stderr']
    stata_select_coef = retdict['stata_select_coef']
    stata_select_stderr = retdict['stata_select_stderr']
    stata_lambda_coef = retdict['stata_lambda_coef']
    stata_lambda_stderr = retdict['stata_lambda_stderr']
    stata_rho = retdict['stata_rho']
    stata_sigma = retdict['stata_sigma']
    y_new_firstobs = retdict['y_new_firstobs']
    y_new_lastobs = retdict['y_new_lastobs']


    ## check against those estimates
    stata_regvar_ordered = ['const','AX','AX2','WE','CIT']
    stata_selectvar_ordered = ['const','WA','WA2','FAMINC','WE','K']

    stata_reg_coef_arr = np.array([stata_reg_coef[k] for k in stata_regvar_ordered])
    stata_reg_stderr_arr = np.array([stata_reg_stderr[k] for k in stata_regvar_ordered])

    stata_select_coef_arr = np.array([stata_select_coef[k] for k in stata_selectvar_ordered])
    stata_select_stderr_arr = np.array([stata_select_stderr[k] for k in stata_selectvar_ordered])

    # for pandas input with var names #
    TOL=1e-3

    _check_heckman_to_stata(
        stata_reg_coef_arr, stata_reg_stderr_arr,
        stata_select_coef_arr, stata_select_stderr_arr,
        stata_lambda_coef, stata_lambda_stderr,
        stata_rho, stata_sigma,
        heckman_res,
        TOL=TOL)

    # for basic list input #
    _check_heckman_to_stata(
        stata_reg_coef_arr, stata_reg_stderr_arr,
        stata_select_coef_arr, stata_select_stderr_arr,
        stata_lambda_coef, stata_lambda_stderr,
        stata_rho, stata_sigma,
        heckman_basic_res,
        TOL=TOL)

    ## check that predict method works
    y_pred = heckman_basic_res.predict()
    assert_( (y_new_firstobs-y_pred[0])/y_new_firstobs < TOL )
    assert_( (y_new_lastobs-y_pred[-1])/y_new_lastobs < TOL )


def test_heckman_mle(verbose=True):
    ############################################################################
    # Tests to make sure that the Heckman MLE estimates produced by the
    # Heckman module are the same as/very close to the estimates produced by
    # Stata for the female labor supply data.
    ############################################################################

    ### Fit data with Heckman model using MLE ###
    ## With pandas input with named variables ##
    # get the testing data
    Y, X, Z = _prep_censored_wage_heckman_exampledata()

    # specify the model
    heckman_model = heckman.Heckman(Y,X,Z)

    # fit the model
    heckman_res = heckman_model.fit(method='mle', method_mle='ncg', disp=False)

    # produce the fitted model summary object
    heckman_smry = heckman_res.summary(disp=verbose)


    ### Stata's two-step estimates ###
    retdict = _load_stata_femalewage_2step_estimates()

    stata_reg_coef = retdict['stata_reg_coef']
    stata_reg_stderr = retdict['stata_reg_stderr']
    stata_select_coef = retdict['stata_select_coef']
    stata_select_stderr = retdict['stata_select_stderr']
    stata_lambda_coef = retdict['stata_lambda_coef']
    stata_lambda_stderr = retdict['stata_lambda_stderr']
    stata_rho = retdict['stata_rho']
    stata_sigma = retdict['stata_sigma']
    y_new_firstobs = retdict['y_new_firstobs']
    y_new_lastobs = retdict['y_new_lastobs']


    ### Use Stata's two-step estimates to check against MLE estimates ###
    ## coef estimates should be similar
    stata_regvar_ordered = ['const','AX','AX2','WE','CIT']
    stata_selectvar_ordered = ['const','WA','WA2','FAMINC','WE','K']

    stata_reg_coef_arr = np.array([stata_reg_coef[k] for k in stata_regvar_ordered])
    stata_reg_stderr_arr = np.array([stata_reg_stderr[k] for k in stata_regvar_ordered])

    stata_select_coef_arr = np.array([stata_select_coef[k] for k in stata_selectvar_ordered])
    stata_select_stderr_arr = np.array([stata_select_stderr[k] for k in stata_selectvar_ordered])


    TOL=1e-1

    _check_heckman_to_stata(
        stata_reg_coef_arr, None,
        stata_select_coef_arr, None,
        stata_lambda_coef, None,
        stata_rho, stata_sigma,
        heckman_res,
        TOL=TOL)


    ### check that predict method works
    y_pred = heckman_res.predict()
    assert_( (y_new_firstobs-y_pred[0])/y_new_firstobs < TOL )
    assert_( (y_new_lastobs-y_pred[-1])/y_new_lastobs < TOL )


    ## MLE is more efficient, so standard errors should be lower than two-step,
    ## so here check that they aren't much smaller
    stderr_stata_list = np.hstack([
        stata_reg_stderr_arr,
        stata_select_stderr_arr,
        stata_lambda_stderr
        ])

    stderr_est_list = np.hstack([
        np.sqrt(np.diag(heckman_res.scale*heckman_res.normalized_cov_params)),
        np.sqrt(np.diag(heckman_res.select_res.scale*heckman_res.select_res.normalized_cov_params)),
        heckman_res.stderr_inverse_mills
        ])

    assert len(stderr_stata_list) == len(stderr_est_list)

    # check that all std err estimates are not nan
    assert_(not any(np.isnan(stderr_est_list)))

    # check that MLE estimates are less than two-step estimates within a tolerance
    for i in range(len(stderr_stata_list)):
        t = stderr_stata_list[i]
        e = stderr_est_list[i]

        assert_(e<=t+TOL)



def test_heckman_2step_missingdata(verbose=True):
    ############################################################################
    # Tests to make sure that the Heckman 2 step estimates produced by the
    # Heckman module WITH MISSING DATA and DROP OPTION are the same as/very
    # close to the estimates produced by Stata for the female labor supply data.
    ############################################################################

    ### Fit data with Heckman model using 2 step ###
    ## With pandas input with named variables ##
    Y, X, Z = _prep_censored_wage_heckman_exampledata()

    ## delete some data ##
    '''
    /* start introducing some missings */
    replace WW = . if _n==1
    replace AX = . if _n==2
    replace WA = . if _n==3

    replace WW = . if _n==101
    replace CIT = . if _n==102
    replace FAMINC = . if _n==103

    replace WW = . if _n==201
    replace WE = . if _n==202
    replace K = . if _n==203

    /* last of introduced missings */
    '''

    Y.ix[1-1] = np.nan
    X.ix[2-1,'AX'] = np.nan
    Z.ix[3-1,'WA'] = np.nan

    Y.ix[101-1] = np.nan
    X.ix[102-1,'CIT'] = np.nan
    Z.ix[103-1,'FAMINC'] = np.nan

    Y.ix[201-1] = np.nan
    X.ix[202-1,'WE'] = np.nan
    Z.ix[203-1,'K'] = np.nan

    ## fit it

    heckman_model = heckman.Heckman(Y,X,Z,
        missing='drop')
    heckman_res = heckman_model.fit(method='twostep')
    heckman_smry = heckman_res.summary(disp=verbose)

    ## With list input (no names) ##
    heckman_basic_model = heckman.Heckman(Y.tolist(), X.as_matrix().tolist(), Z.as_matrix().tolist(),
        missing='drop')
    heckman_basic_res = heckman_basic_model.fit(method='twostep')
    heckman_basic_smry = heckman_basic_res.summary(disp=verbose)

    ### Check against Stata's estimates ###
    ## Load Stata's estimates
    retdict = _load_stata_femalewage_missing_2step_estimates()

    stata_reg_coef = retdict['stata_reg_coef']
    stata_reg_stderr = retdict['stata_reg_stderr']
    stata_select_coef = retdict['stata_select_coef']
    stata_select_stderr = retdict['stata_select_stderr']
    stata_lambda_coef = retdict['stata_lambda_coef']
    stata_lambda_stderr = retdict['stata_lambda_stderr']
    stata_rho = retdict['stata_rho']
    stata_sigma = retdict['stata_sigma']
    y_new_firstobs = retdict['y_new_firstobs']
    y_new_lastobs = retdict['y_new_lastobs']


    ## check against those estimates
    stata_regvar_ordered = ['const','AX','AX2','WE','CIT']
    stata_selectvar_ordered = ['const','WA','WA2','FAMINC','WE','K']

    stata_reg_coef_arr = np.array([stata_reg_coef[k] for k in stata_regvar_ordered])
    stata_reg_stderr_arr = np.array([stata_reg_stderr[k] for k in stata_regvar_ordered])

    stata_select_coef_arr = np.array([stata_select_coef[k] for k in stata_selectvar_ordered])
    stata_select_stderr_arr = np.array([stata_select_stderr[k] for k in stata_selectvar_ordered])

    # for pandas input with var names #
    TOL=1e-3

    _check_heckman_to_stata(
        stata_reg_coef_arr, stata_reg_stderr_arr,
        stata_select_coef_arr, stata_select_stderr_arr,
        stata_lambda_coef, stata_lambda_stderr,
        stata_rho, stata_sigma,
        heckman_res,
        TOL=TOL)

    # for basic list input #
    _check_heckman_to_stata(
        stata_reg_coef_arr, stata_reg_stderr_arr,
        stata_select_coef_arr, stata_select_stderr_arr,
        stata_lambda_coef, stata_lambda_stderr,
        stata_rho, stata_sigma,
        heckman_basic_res,
        TOL=TOL)

    ## check that predict method works
    y_pred = heckman_basic_res.predict()
    assert_( (y_new_firstobs-y_pred[0])/y_new_firstobs < TOL )
    assert_( (y_new_lastobs-y_pred[-1])/y_new_lastobs < TOL )




## Run tests if file is ran ##
if __name__ == '__main__':
    test_heckman_2step(verbose=False)
    test_heckman_mle(verbose=False)
    test_heckman_2step_missingdata(verbose=False)
