
from __future__ import print_function
import numpy as np

import statsmodels.stats.outliers_influence as oi


if __name__ == '__main__':

    import statsmodels.api as sm

    data = np.array('''\
    64 57  8
    71 59 10
    53 49  6
    67 62 11
    55 51  8
    58 50  7
    77 55 10
    57 48  9
    56 42 10
    51 42  6
    76 61 12
    68 57  9'''.split(), float).reshape(-1,3)
    varnames = 'weight height age'.split()

    endog = data[:,0]
    exog = sm.add_constant(data[:,2])


    res_ols = sm.OLS(endog, exog).fit()

    hh = (res_ols.model.exog * res_ols.model.pinv_wexog.T).sum(1)
    x = res_ols.model.exog
    hh_check = np.diag(np.dot(x, np.dot(res_ols.model.normalized_cov_params, x.T)))

    from numpy.testing import assert_almost_equal
    assert_almost_equal(hh, hh_check, decimal=13)

    res = res_ols #alias

    #http://en.wikipedia.org/wiki/PRESS_statistic
    #predicted residuals, leave one out predicted residuals
    resid_press = res.resid / (1-hh)
    ess_press = np.dot(resid_press, resid_press)

    sigma2_est = np.sqrt(res.mse_resid) #can be replace by different estimators of sigma
    sigma_est = np.sqrt(sigma2_est)
    resid_studentized = res.resid / sigma_est / np.sqrt(1 - hh)
    #http://en.wikipedia.org/wiki/DFFITS:
    dffits = resid_studentized * np.sqrt(hh / (1 - hh))

    nobs, k_vars = res.model.exog.shape
    #Belsley, Kuh and Welsch (1980) suggest a threshold for abs(DFFITS)
    dffits_threshold = 2 * np.sqrt(k_vars/nobs)

    res_ols.df_modelwc = res_ols.df_model + 1
    n_params = res.model.exog.shape[1]
    #http://en.wikipedia.org/wiki/Cook%27s_distance
    cooks_d = res.resid**2 / sigma2_est / res_ols.df_modelwc * hh / (1 - hh)**2
    #or
    #Eubank p.93, 94
    cooks_d2 = resid_studentized**2 / res_ols.df_modelwc * hh / (1 - hh)
    #threshold if normal, also Wikipedia
    from scipy import stats
    alpha = 0.1
    #df looks wrong
    print(stats.f.isf(1-alpha, n_params, res.df_resid))
    print(stats.f.sf(cooks_d, n_params, res.df_resid))


    print('Cooks Distance')
    print(cooks_d)
    print(cooks_d2)

    doplot = 0
    if doplot:
        import matplotlib.pyplot as plt
        fig = plt.figure()
#        ax = fig.add_subplot(3,1,1)
#        plt.plot(andrew_results.weights, 'o', label='rlm weights')
#        plt.legend(loc='lower left')
        ax = fig.add_subplot(3,1,2)
        plt.plot(cooks_d, 'o', label="Cook's distance")
        plt.legend(loc='upper left')
        ax2 = fig.add_subplot(3,1,3)
        plt.plot(resid_studentized, 'o', label='studentized_resid')
        plt.plot(dffits, 'o', label='DFFITS')
        leg = plt.legend(loc='lower left', fancybox=True)
        leg.get_frame().set_alpha(0.5) #, fontsize='small')
        ltext = leg.get_texts() # all the text.Text instance in the legend
        plt.setp(ltext, fontsize='small') # the legend text fontsize


    print(oi.reset_ramsey(res, degree=3))

    #note, constant in last column
    for i in range(1):
        print(oi.variance_inflation_factor(res.model.exog, i))

    infl = oi.OLSInfluence(res_ols)
    print(infl.resid_studentized_external)
    print(infl.resid_studentized_internal)
    print(infl.summary_table())
    print(oi.summary_table(res, alpha=0.05)[0])

'''
>>> res.resid
array([  4.28571429,   4.        ,   0.57142857,  -3.64285714,
        -4.71428571,   1.92857143,  10.        ,  -6.35714286,
       -11.        ,  -1.42857143,   1.71428571,   4.64285714])
>>> infl.hat_matrix_diag
array([ 0.10084034,  0.11764706,  0.28571429,  0.20168067,  0.10084034,
        0.16806723,  0.11764706,  0.08403361,  0.11764706,  0.28571429,
        0.33613445,  0.08403361])
>>> infl.resid_press
array([  4.76635514,   4.53333333,   0.8       ,  -4.56315789,
        -5.24299065,   2.31818182,  11.33333333,  -6.94036697,
       -12.46666667,  -2.        ,   2.58227848,   5.06880734])
>>> infl.ess_press
465.98646628086374
'''
