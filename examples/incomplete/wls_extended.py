"""
Weighted Least Squares

example is extended to look at the meaning of rsquared in WLS,
at outliers, compares with RLM and a short bootstrap

"""
from __future__ import print_function
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt

data = sm.datasets.ccard.load()
data.exog = sm.add_constant(data.exog, prepend=False)
ols_fit = sm.OLS(data.endog, data.exog).fit()

# perhaps the residuals from this fit depend on the square of income
incomesq = data.exog[:,2]
plt.scatter(incomesq, ols_fit.resid)
#@savefig wls_resid_check.png
plt.grid()


# If we think that the variance is proportional to income**2
# we would want to weight the regression by income
# the weights argument in WLS weights the regression by its square root
# and since income enters the equation, if we have income/income
# it becomes the constant, so we would want to perform
# this type of regression without an explicit constant in the design

#..data.exog = data.exog[:,:-1]
wls_fit = sm.WLS(data.endog, data.exog[:,:-1], weights=1/incomesq).fit()

# This however, leads to difficulties in interpreting the post-estimation
# statistics.  Statsmodels does not yet handle this elegantly, but
# the following may be more appropriate

# explained sum of squares
ess = wls_fit.uncentered_tss - wls_fit.ssr
# rsquared
rsquared = ess/wls_fit.uncentered_tss
# mean squared error of the model
mse_model = ess/(wls_fit.df_model + 1) # add back the dof of the constant
# f statistic
fvalue = mse_model/wls_fit.mse_resid
# adjusted r-squared
rsquared_adj = 1 -(wls_fit.nobs)/(wls_fit.df_resid)*(1-rsquared)



#Trying to figure out what's going on in this example
#----------------------------------------------------

#JP: I need to look at this again. Even if I exclude the weight variable
# from the regressors and keep the constant in then the reported rsquared
# stays small. Below also compared using squared or sqrt of weight variable.
# TODO: need to add 45 degree line to graphs
wls_fit3 = sm.WLS(data.endog, data.exog[:,(0,1,3,4)], weights=1/incomesq).fit()
print(wls_fit3.summary())
print('corrected rsquared')
print((wls_fit3.uncentered_tss - wls_fit3.ssr)/wls_fit3.uncentered_tss)
plt.figure();
plt.title('WLS dropping heteroscedasticity variable from regressors');
plt.plot(data.endog, wls_fit3.fittedvalues, 'o');
plt.xlim([0,2000]);
#@savefig wls_drop_het.png
plt.ylim([0,2000]);
print('raw correlation of endog and fittedvalues')
print(np.corrcoef(data.endog, wls_fit.fittedvalues))
print('raw correlation coefficient of endog and fittedvalues squared')
print(np.corrcoef(data.endog, wls_fit.fittedvalues)[0,1]**2)

# compare with robust regression,
# heteroscedasticity correction downweights the outliers
rlm_fit = sm.RLM(data.endog, data.exog).fit()
plt.figure();
plt.title('using robust for comparison');
plt.plot(data.endog, rlm_fit.fittedvalues, 'o');
plt.xlim([0,2000]);
#@savefig wls_robust_compare.png
plt.ylim([0,2000]);

#What is going on? A more systematic look at the data
#----------------------------------------------------

# two helper functions

def getrsq(fitresult):
    '''calculates rsquared residual, total and explained sums of squares

    Parameters
    ----------
    fitresult : instance of Regression Result class, or tuple of (resid, endog) arrays
        regression residuals and endogenous variable

    Returns
    -------
    rsquared
    residual sum of squares
    (centered) total sum of squares
    explained sum of squares (for centered)
    '''
    if hasattr(fitresult, 'resid') and hasattr(fitresult, 'model'):
        resid = fitresult.resid
        endog = fitresult.model.endog
        nobs = fitresult.nobs
    else:
        resid = fitresult[0]
        endog = fitresult[1]
        nobs = resid.shape[0]


    rss = np.dot(resid, resid)
    tss = np.var(endog)*nobs
    return 1-rss/tss, rss, tss, tss-rss


def index_trim_outlier(resid, k):
    '''returns indices to residual array with k outliers removed

    Parameters
    ----------
    resid : array_like, 1d
        data vector, usually residuals of a regression
    k : int
        number of outliers to remove

    Returns
    -------
    trimmed_index : array, 1d
        index array with k outliers removed
    outlier_index : array, 1d
        index array of k outliers

    Notes
    -----

    Outliers are defined as the k observations with the largest
    absolute values.

    '''
    sort_index = np.argsort(np.abs(resid))
    # index of non-outlier
    trimmed_index = np.sort(sort_index[:-k])
    outlier_index = np.sort(sort_index[-k:])
    return trimmed_index, outlier_index


#Comparing estimation results for ols, rlm and wls with and without outliers
#---------------------------------------------------------------------------

#ols_test_fit = sm.OLS(data.endog, data.exog).fit()
olskeep, olsoutl = index_trim_outlier(ols_fit.resid, 2)
print('ols outliers', olsoutl, ols_fit.resid[olsoutl])
ols_fit_rm2 = sm.OLS(data.endog[olskeep], data.exog[olskeep,:]).fit()
rlm_fit_rm2 = sm.RLM(data.endog[olskeep], data.exog[olskeep,:]).fit()
#weights = 1/incomesq

results = [ols_fit, ols_fit_rm2, rlm_fit, rlm_fit_rm2]
#Note: I think incomesq is already square
for weights in [1/incomesq, 1/incomesq**2, np.sqrt(incomesq)]:
    print('\nComparison OLS and WLS with and without outliers')
    wls_fit0 = sm.WLS(data.endog, data.exog, weights=weights).fit()
    wls_fit_rm2 = sm.WLS(data.endog[olskeep], data.exog[olskeep,:],
                         weights=weights[olskeep]).fit()
    wlskeep, wlsoutl = index_trim_outlier(ols_fit.resid, 2)
    print('2 outliers candidates and residuals')
    print(wlsoutl, wls_fit.resid[olsoutl])
    # redundant because ols and wls outliers are the same:
    ##wls_fit_rm2_ = sm.WLS(data.endog[wlskeep], data.exog[wlskeep,:],
    ##                     weights=1/incomesq[wlskeep]).fit()

    print('outliers ols, wls:', olsoutl, wlsoutl)

    print('rsquared')
    print('ols vs ols rm2', ols_fit.rsquared, ols_fit_rm2.rsquared)
    print('wls vs wls rm2', wls_fit0.rsquared, wls_fit_rm2.rsquared) #, wls_fit_rm2_.rsquared
    print('compare R2_resid  versus  R2_wresid')
    print('ols minus 2', getrsq(ols_fit_rm2)[0],)
    print(getrsq((ols_fit_rm2.wresid, ols_fit_rm2.model.wendog))[0])
    print('wls        ', getrsq(wls_fit)[0],)
    print(getrsq((wls_fit.wresid, wls_fit.model.wendog))[0])

    print('wls minus 2', getrsq(wls_fit_rm2)[0])
    # next is same as wls_fit_rm2.rsquared for cross checking
    print(getrsq((wls_fit_rm2.wresid, wls_fit_rm2.model.wendog))[0])
    #print(getrsq(wls_fit_rm2_)[0],
    #print(getrsq((wls_fit_rm2_.wresid, wls_fit_rm2_.model.wendog))[0]
    results.extend([wls_fit0, wls_fit_rm2])

print('     ols             ols_rm2       rlm           rlm_rm2     wls (lin)    wls_rm2 (lin)   wls (squ)   wls_rm2 (squ)  wls (sqrt)   wls_rm2 (sqrt)')
print('Parameter estimates')
print(np.column_stack([r.params for r in results]))
print('R2 original data, next line R2 weighted data')
print(np.column_stack([getattr(r, 'rsquared', None) for r in results]))

print('Standard errors')
print(np.column_stack([getattr(r, 'bse', None) for r in results]))
print('Heteroscedasticity robust standard errors (with ols)')
print('with outliers')
print(np.column_stack([getattr(ols_fit, se, None) for se in ['HC0_se', 'HC1_se', 'HC2_se', 'HC3_se']]))

#..'''
#..
#..     ols             ols_rm2       rlm           rlm_rm2     wls (lin)    wls_rm2 (lin)   wls (squ)   wls_rm2 (squ)  wls (sqrt)   wls_rm2 (sqrt)
#..Parameter estimates
#..[[  -3.08181404   -5.06103843   -4.98510966   -5.34410309   -2.69418516    -3.1305703    -1.43815462   -1.58893054   -3.57074829   -6.80053364]
#.. [ 234.34702702  115.08753715  129.85391456  109.01433492  158.42697752   128.38182357   60.95113284  100.25000841  254.82166855  103.75834726]
#.. [ -14.99684418   -5.77558429   -6.46204829   -4.77409191   -7.24928987    -7.41228893    6.84943071   -3.34972494  -16.40524256   -4.5924465 ]
#.. [  27.94090839   85.46566835   89.91389709   95.85086459   60.44877369    79.7759146    55.9884469    60.97199734   -3.8085159    84.69170048]
#.. [-237.1465136    39.51639838  -15.50014814   31.39771833 -114.10886935   -40.04207242   -6.41976501  -38.83583228 -260.72084271  117.20540179]]
#..
#..R2 original data, next line R2 weighted data
#..[[   0.24357792    0.31745994    0.19220308    0.30527648    0.22861236     0.3112333     0.06573949    0.29366904    0.24114325    0.31218669]]
#..[[   0.24357791    0.31745994    None          None          0.05936888     0.0679071     0.06661848    0.12769654    0.35326686    0.54681225]]
#..
#..-> R2 with weighted data is jumping all over
#..
#..standard errors
#..[[   5.51471653    3.31028758    2.61580069    2.39537089    3.80730631     2.90027255    2.71141739    2.46959477    6.37593755    3.39477842]
#.. [  80.36595035   49.35949263   38.12005692   35.71722666   76.39115431    58.35231328   87.18452039   80.30086861   86.99568216   47.58202096]
#.. [   7.46933695    4.55366113    3.54293763    3.29509357    9.72433732     7.41259156   15.15205888   14.10674821    7.18302629    3.91640711]
#.. [  82.92232357   50.54681754   39.33262384   36.57639175   58.55088753    44.82218676   43.11017757   39.31097542   96.4077482    52.57314209]
#.. [ 199.35166485  122.1287718    94.55866295   88.3741058   139.68749646   106.89445525  115.79258539  105.99258363  239.38105863  130.32619908]]
#..
#..robust standard errors (with ols)
#..with outliers
#..      HC0_se         HC1_se       HC2_se        HC3_se'
#..[[   3.30166123    3.42264107    3.4477148     3.60462409]
#.. [  88.86635165   92.12260235   92.08368378   95.48159869]
#.. [   6.94456348    7.19902694    7.19953754    7.47634779]
#.. [  92.18777672   95.56573144   95.67211143   99.31427277]
#.. [ 212.9905298   220.79495237  221.08892661  229.57434782]]
#..
#..removing 2 outliers
#..[[   2.57840843    2.67574088    2.68958007    2.80968452]
#.. [  36.21720995   37.58437497   37.69555106   39.51362437]
#.. [   3.1156149     3.23322638    3.27353882    3.49104794]
#.. [  50.09789409   51.98904166   51.89530067   53.79478834]
#.. [  94.27094886   97.82958699   98.25588281  102.60375381]]
#..
#..
#..'''

# a quick bootstrap analysis
# --------------------------
#
#(I didn't check whether this is fully correct statistically)

#**With OLS on full sample**

nobs, nvar = data.exog.shape
niter = 2000
bootres = np.zeros((niter, nvar*2))

for it in range(niter):
    rind = np.random.randint(nobs, size=nobs)
    endog = data.endog[rind]
    exog = data.exog[rind,:]
    res = sm.OLS(endog, exog).fit()
    bootres[it, :nvar] = res.params
    bootres[it, nvar:] = res.bse

np.set_print(options(linewidth=200))
print('Bootstrap Results of parameters and parameter standard deviation  OLS')
print('Parameter estimates')
print('median', np.median(bootres[:,:5], 0))
print('mean  ', np.mean(bootres[:,:5], 0))
print('std   ', np.std(bootres[:,:5], 0))

print('Standard deviation of parameter estimates')
print('median', np.median(bootres[:,5:], 0))
print('mean  ', np.mean(bootres[:,5:], 0))
print('std   ', np.std(bootres[:,5:], 0))

plt.figure()
for i in range(4):
    plt.subplot(2,2,i+1)
    plt.hist(bootres[:,i],50)
    plt.title('var%d'%i)
#@savefig wls_bootstrap.png
plt.figtext(0.5, 0.935,  'OLS Bootstrap',
               ha='center', color='black', weight='bold', size='large')

#**With WLS on sample with outliers removed**

data_endog = data.endog[olskeep]
data_exog = data.exog[olskeep,:]
incomesq_rm2 = incomesq[olskeep]

nobs, nvar = data_exog.shape
niter = 500  # a bit slow
bootreswls = np.zeros((niter, nvar*2))

for it in range(niter):
    rind = np.random.randint(nobs, size=nobs)
    endog = data_endog[rind]
    exog = data_exog[rind,:]
    res = sm.WLS(endog, exog, weights=1/incomesq[rind,:]).fit()
    bootreswls[it, :nvar] = res.params
    bootreswls[it, nvar:] = res.bse

print('Bootstrap Results of parameters and parameter standard deviation',)
print('WLS removed 2 outliers from sample')
print('Parameter estimates')
print('median', np.median(bootreswls[:,:5], 0))
print('mean  ', np.mean(bootreswls[:,:5], 0))
print('std   ', np.std(bootreswls[:,:5], 0))

print('Standard deviation of parameter estimates')
print('median', np.median(bootreswls[:,5:], 0))
print('mean  ', np.mean(bootreswls[:,5:], 0))
print('std   ', np.std(bootreswls[:,5:], 0))

plt.figure()
for i in range(4):
    plt.subplot(2,2,i+1)
    plt.hist(bootreswls[:,i],50)
    plt.title('var%d'%i)
#@savefig wls_bootstrap_rm2.png
plt.figtext(0.5, 0.935,  'WLS rm2 Bootstrap',
               ha='center', color='black', weight='bold', size='large')


#..plt.show()
#..plt.close('all')

#::
#
#    The following a random variables not fixed by a seed
#
#    Bootstrap Results of parameters and parameter standard deviation
#    OLS
#
#    Parameter estimates
#    median [  -3.26216383  228.52546429  -14.57239967   34.27155426 -227.02816597]
#    mean   [  -2.89855173  234.37139359  -14.98726881   27.96375666 -243.18361746]
#    std    [   3.78704907   97.35797802    9.16316538   94.65031973  221.79444244]
#
#    Standard deviation of parameter estimates
#    median [   5.44701033   81.96921398    7.58642431   80.64906783  200.19167735]
#    mean   [   5.44840542   86.02554883    8.56750041   80.41864084  201.81196849]
#    std    [   1.43425083   29.74806562    4.22063268   19.14973277   55.34848348]
#
#    Bootstrap Results of parameters and parameter standard deviation
#    WLS removed 2 outliers from sample
#
#    Parameter estimates
#    median [  -3.95876112  137.10419042   -9.29131131   88.40265447  -44.21091869]
#    mean   [  -3.67485724  135.42681207   -8.7499235    89.74703443  -46.38622848]
#    std    [   2.96908679   56.36648967    7.03870751   48.51201918  106.92466097]
#
#    Standard deviation of parameter estimates
#    median [   2.89349748   59.19454402    6.70583332   45.40987953  119.05241283]
#    mean   [   2.97600894   60.14540249    6.92102065   45.66077486  121.35519673]
#    std    [   0.55378808   11.77831934    1.69289179    7.4911526    23.72821085]
#
#
#
#Conclusion: problem with outliers and possibly heteroscedasticity
#-----------------------------------------------------------------
#
#in bootstrap results
#
#* bse in OLS underestimates the standard deviation of the parameters
#  compared to standard deviation in bootstrap
#* OLS heteroscedasticity corrected standard errors for the original
#  data (above) are close to bootstrap std
#* using WLS with 2 outliers removed has a relatively good match between
#  the mean or median bse and the std of the parameter estimates in the
#  bootstrap
#
#We could also include rsquared in bootstrap, and do it also for RLM.
#The problems could also mean that the linearity assumption is violated,
#e.g. try non-linear transformation of exog variables, but linear
#in parameters.
#
#
#for statsmodels
#
# * In this case rsquared for original data looks less random/arbitrary.
# * Don't change definition of rsquared from centered tss to uncentered
#   tss when calculating rsquared in WLS if the original exog contains
#   a constant. The increase in rsquared because of a change in definition
#   will be very misleading.
# * Whether there is a constant in the transformed exog, wexog, or not,
#   might affect also the degrees of freedom calculation, but I haven't
#   checked this. I would guess that the df_model should stay the same,
#   but needs to be verified with a textbook.
# * df_model has to be adjusted if the original data does not have a
#   constant, e.g. when regressing an endog on a single exog variable
#   without constant. This case might require also a redefinition of
#   the rsquare and f statistic for the regression anova to use the
#   uncentered tss.
#   This can be done through keyword parameter to model.__init__ or
#   through autodedection with hasconst = (exog.var(0)<1e-10).any()
#   I'm not sure about fixed effects with a full dummy set but
#   without a constant. In this case autodedection wouldn't work this
#   way. Also, I'm not sure whether a ddof keyword parameter can also
#   handle the hasconst case.

