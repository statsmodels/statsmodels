

import numpy as np
from scipy import optimize
import scikits.statsmodels as sm
from scikits.statsmodels.sandbox.regression.numdiff import approx_fprime1, approx_hess
from scikits.statsmodels.model import LikelihoodModel, LikelihoodModelResults
from scikits.statsmodels.regression import RegressionResults, OLS


def maxabs(x):
    return np.abs(x).max()


class IV2SLS(LikelihoodModel):

    def __init__(self, endog, exog, instrument=None):
        self.instrument = instrument
        super(IV2SLS, self).__init__(endog, exog)
        # where is this supposed to be handled
        #Note: Greene p.77/78 dof correction is not necessary (because only
        #       asy results), but most packages to it anyway
        self.df_resid = exog.shape[0] - exog.shape[1] + 1

    def initialize(self):
        self.wendog = self.endog
        self.wexog = self.exog

    def whiten(self, X):
        pass

    def fit(self):
        #Greene 5th edt., p.78 section 5.4
        #move this maybe
        y,x,z = self.endog, self.exog, self.instrument
        ztz = np.dot(z.T, z)
        ztx = np.dot(z.T, x)
        xhatparams = np.linalg.solve(ztz, ztx)
        print 'x.T.shape, xhatparams.shape', x.shape, xhatparams.shape
        F = xhat = np.dot(x, xhatparams)
        FtF = np.dot(F.T, F)
        Ftx = np.dot(F.T, x)
        Fty = np.dot(F.T, y)
        params = np.linalg.solve(FtF, Fty)
        Ftxinv = np.linalg.inv(Ftx)
        self.normalized_cov_params = np.dot(Ftxinv.T, np.dot(FtF, Ftxinv))

        lfit = RegressionResults(self, params,
                       normalized_cov_params=self.normalized_cov_params)
        self._results = lfit
        return lfit

    #copied from GLS
    def predict(self, exog, params=None):
        """
        Return linear predicted values from a design matrix.

        Parameters
        ----------
        exog : array-like
            Design / exogenous data
        params : array-like, optional after fit has been called
            Parameters of a linear model

        Returns
        -------
        An array of fitted values

        Notes
        -----
        If the model as not yet been fit, params is not optional.
        """
        #JP: this doesn't look correct for GLMAR
        #SS: it needs its own predict method
        if self._results is None and params is None:
            raise ValueError, "If the model has not been fit, then you must specify the params argument."
        if self._results is not None:
            return np.dot(exog, self._results.params)
        else:
            return np.dot(exog, params)



###############  GMM with standalone functions, only for development

#copied from distributions estimation and not fully adjusted
def fitgmm(momcond, args, start, weights=None, fixed=None, weightsoptimal=True):
    '''estimate parameters of distribution function for binned data using GMM

    Parameters
    ----------
    momcond : function
        needs to return (nobs, nmoms) array


    Returns
    -------
    paramest : array
        estimated parameters

    Notes
    -----
    todo: add fixed parameter option

    added factorial

    '''
    if not fixed is None:
        raise NotImplementedError

    tmp = momcond(start, *args)
    nmoms = tmp.shape[-1]
    if weights is None:
        if weightsoptimal:
            raise NotImplementedError
            weights = freq/float(nobs)
        else:
            weights = np.eye(nmoms)
    # skip turning weights into matrix diag(freq/float(nobs))
    def gmmobjective(params):
        '''negative loglikelihood function of binned data

        corresponds to multinomial
        '''

        moms = momcond(params, *args)
        #return np.dot(moms*weights, moms)
        return np.dot(np.dot(moms.sum(0),weights), moms.sum(0))
    return optimize.fmin(gmmobjective, start)

def gmmiter(momcond, start, maxiter=2, args=None, centered_weights=True):
    '''iteration over gmm estimation with updating of optimal weighting matrix

    '''
    w = np.eye(len(start))
    for i in range(maxiter):
        resgmm = fitgmm(momcond, args, start, weights=w, fixed=None, weightsoptimal=False)
        moms = momcond(resgmm, *args)
        if centered_weights:
            w = np.cov(moms, rowvar=0) # note: I need this also for cov_params
        else:
            w = np.dot(moms.T, moms)
        start = resgmm
    return resgmm

def gmmcov_params(moms, gradmoms, weights=None, has_optimal_weights=True, centered_weights=True):
    '''calculate covariance of parameter estimates

    not all options tried out yet
    '''

    nobs = moms.shape[0]
    if centered_weights:
        # note: code duplication from gmmiter
        omegahat = np.cov(moms, rowvar=0)  # estimate of moment covariance
    else:
        omegahat = np.dot(moms.T, moms)/nobs
    #add other options, Barzen, ...  longrun var estimators
    if weights is None: #has_optimal_weights:
        cov = np.linalg.inv(np.dot(gradmoms.T,
                                   np.dot(np.linalg.inv(omegahat), gradmoms)))
    else:
        gw = np.dot(gradmoms.T, weights)
        gwginv = np.linalg.inv(np.dot(gw, gradmoms))
        cov = np.dot(np.dot(gwginv, np.dot(np.dot(gw, omegahat), gw.T)), gwginv)

    return cov/nobs


########## end standalone GMM functions


class GMM(object):
    '''
    Class for estimation by Generalized Method of Moments

    needs to be subclassed, where the subclass defined the moment conditions
    `momcond`

    Parameters
    ----------
    endog : array
        endogenous variable, see notes
    exog : array
        array of exogenous variables, see notes
    instrument : array
        array of instruments, see notes
    nmoms : None or int
        number of moment conditions, if None then it is set equal to the
        number of columns of instruments. Mainly needed to determin the shape
        or size of start parameters and starting weighting matrix.
    kwds : anything
        this is mainly if additional variables need to be stored for the
        calculations of the moment conditions

    Attributes
    ----------
    results : instance of GMMResults
        currently just a storage class for params and cov_params without it's
        own methods
    bse : property
        return bse


    Methods
    -------
    fit
    cov_params

    other methods

    fititer
    fitgmm
    calc_weightmatrix


    Notes
    -----
    The GMM class only uses the moment conditions and does not use any data
    directly. endog, exog, instrument and kwds in the creation of the class
    instance are only used to store them for access in the moment conditions.
    Which of this are required and how they are used depends on the moment
    conditions of the subclass.

    Warning:

    Options for various methods have not been fully implemented and
    are still missing in several methods.



    '''

    def __init__(self, endog, exog, instrument, nmoms=None, **kwds):
        '''
        maybe drop and use mixin instead

        GMM doesn't really care about the data, just the moment conditions
        '''
        self.endog = endog
        self.exog = exog
        self.instrument = instrument
        self.nmoms = nmoms or instrument.shape[1]
        self.results = GMMResults()
        self.__dict__.update(kwds)
        self.epsilon_iter = 1e-6

    def fit(self):
        '''
        Estimate the parameters using default settings.

        For estimation with more options use fititer method.

        Returns
        -------
        results : instance of GMMResults
            this is also attached as attribute results

        Notes
        -----
        this function attaches the estimated parameters, params, the
        weighting matrix of the final iteration, weights, and the value
        of the GMM objective function, jval to results


        '''
        params, weights = self.fititer(start, maxiter=2, start_weights=None,
                                        weights_method='cov', wargs=())
        self.results.params = params
        self.results.weights = weights
        self.results.jval = self.gmmobjective(params, weights)

        return self.results


    def fitgmm(self, momcond, args, start, weights=None):
        '''estimate parameters using GMM

        Parameters
        ----------
        momcond : function
            needs to return (nobs, nmoms) array


        Returns
        -------
        paramest : array
            estimated parameters

        Notes
        -----
        todo: add fixed parameter option, not here ???

        added factorial

        '''
        if not fixed is None:
            raise NotImplementedError

        tmp = momcond(start, *args)
        nmoms = tmp.shape[-1]
        if weights is None:
            weights = np.eye(nmoms)


##        def gmmobjective(self, params):
##            '''
##            objective function for GMM minimization
##
##            Parameters
##            ----------
##            params : array
##               parameter values at which objective is evaluated
##
##            uses weights from outer scope
##
##            '''
##            moms = momcond(params, *args)
##            return np.dot(np.dot(moms.sum(0),weights), moms.sum(0))
        return optimize.fmin(self.gmmobjective, start, (weights,), disp=0)

    def gmmobjective(self, params, weights):
        '''
        objective function for GMM minimization

        Parameters
        ----------
        params : array
            parameter values at which objective is evaluated
        weights : array
            weighting matrix

        Returns
        -------
        jval : float
            value of objective function

        '''
        moms = self.momcond(params)
        return np.dot(np.dot(moms.sum(0),weights), moms.sum(0))


    def fititer(self, start, maxiter=2, start_weights=None,
                    weights_method='cov', wargs=()):
        '''iterative estimation with updating of optimal weighting matrix

        stopping criteria are maxiter or change in parameter estimate less
        than self.epsilon_iter, with default 1e-6.

        Parameters
        ----------
        start : array
            starting value for parameters
        maxiter : int
            maximum number of iterations
        start_weights : array (nmoms, nmoms)
            initial weighting matrix; if None, then the identity matrix
            is used
        weights_method : {'cov', ...}
            method to use to estimate the optimal weighting matrix,
            see calc_weightmatrix for details

        Returns
        -------
        params : array
            estimated parameters
        weights : array
            optimal weighting matrix calculated with final parameter
            estimates

        Notes
        -----




        '''
        momcond = self.momcond

        if start_weights is None:
            w = np.eye(len(start))
        else:
            w = start_weights

        #call fitgmm function
        #args = (self.endog, self.exog, self.instrument)
        #args is not used in the method version
        for it in range(maxiter):
            resgmm = fitgmm(momcond, (), start, weights=w, fixed=None,
                            weightsoptimal=False)

            moms = momcond(resgmm)
            w = self.calc_weightmatrix(moms, method='cov', wargs=())

            if it > 2 and maxabs(resgmm - start) < self.epsilon_iter:
                #check rule for early stopping
                break
            start = resgmm
        return resgmm, w

    #todo: check if there is a matrix inverse missing somewhere, after
    #   converting to method
    def calc_weightmatrix(self, moms, method='momcov', wargs=()):
        '''calculate omega or the weighting matrix

        Parameters
        ----------

        moms : array, (nobs, nmoms)
            moment conditions for all observations evaluated at a parameter
            value
        method : 'momcov', anything else
            If method='momcov' is cov then the matrix is calculated as simple
            covariance of the moment conditions. For anything else, the
            uncentered moments are used. (The latter is not recommended.)
        wargs : tuple
            parameters that are required by some kernel methods to
            estimate the long-run covariance. Not used yet.

        Returns
        -------
        w : array (nmoms, nmoms)
            estimate for the weighting matrix or covariance of the moment
            condition



        TODO: implement long-run cov estimators, kernel-based

        Newey-West
        Andrews
        Andrews-Moy????

        References
        ----------
        Greene
        Hansen, Bruce

        '''
        nobs = moms.shape[0]
        if method == 'momcov':
            w = np.cov(moms, rowvar=0)
        elif method == 'fakekernel':
            #uniform cut-off window
            moms_centered = moms - moms.mean()
            maxlag = 5
            h = np.ones(maxlag)
            w = np.dot(moms.T, moms)/nobs
            for i in range(1,maxlag+1):
                w += (h * np.dot(moms_centered[i:].T, moms_centered[:-i]) /
                                                                  (nobs-i))
        else:
            w = np.dot(moms.T, moms)/nobs

        return w


    def momcond_mean(self, params):
        '''
        mean of moment conditions,

        '''

        #endog, exog = args
        return self.momcond(params).mean(0)

    def gradient_momcond(self, params, epsilon=1e-4, method='centered'):

        momcond = self.momcond_mean
        if method == 'centered':
            gradmoms = (approx_fprime1(params, momcond, epsilon=epsilon) +
                    approx_fprime1(params, momcond, epsilon=-epsilon))/2
        else:
            gradmoms = approx_fprime1(params, momcond, epsilon=epsilon)

        return gradmoms


    def cov_params(self):  #TODO add options ???
        if not hasattr(self.results, 'params'):
            raise ValueError('the model has to be fit first')

        if hasattr(self.results, '_cov_params'):
            #replace with decorator later
            return self.results._cov_params

        gradmoms = self.gradient_momcond(self.results.params)
        moms = self.momcond(self.results.params)
        covparams = self.calc_cov_params(moms, gradmoms)
        self.results._cov_params = covparams
        return self.results._cov_params



    #still needs to be fully converted to method
    def calc_cov_params(self, moms, gradmoms, weights=None, has_optimal_weights=True, centered_weights=True):
        '''calculate covariance of parameter estimates

        not all options tried out yet
        '''

        nobs = moms.shape[0]
        if centered_weights:
            # note: code duplication from gmmiter
            omegahat = np.cov(moms, rowvar=0)  # estimate of moment covariance
        else:
            omegahat = np.dot(moms.T, moms)/nobs
        #add other options, Barzen, ...  longrun var estimators
        if weights is None: #has_optimal_weights:
            cov = np.linalg.inv(np.dot(gradmoms.T,
                                       np.dot(np.linalg.inv(omegahat), gradmoms)))
        else:
            gw = np.dot(gradmoms.T, weights)
            gwginv = np.linalg.inv(np.dot(gw, gradmoms))
            cov = np.dot(np.dot(gwginv, np.dot(np.dot(gw, omegahat), gw.T)), gwginv)

        return cov/nobs

    @property
    def bse(self):
        return self.get_bse()

    def get_bse(self, method=None):
        '''

        method option not defined yet
        '''
        return np.sqrt(np.diag(self.cov_params()))

class GMMResults(object):
    '''just a storage class right now'''
    pass

class IVGMM(GMM):
    '''
    Class for linear instrumental variables estimation with homoscedastic
    errors

    currently mainly a test case, doesn't exploit linear structure

    '''

    def momcond(self, params):
        endog, exog, instrum = self.endog, self.exog, self.instrument
        return instrum * (endog - np.dot(exog, params))[:,None]

class NonlinearIVGMM(GMM):
    '''
    Class for linear instrumental variables estimation with homoscedastic
    errors

    currently mainly a test case, doesn't exploit linear structure

    '''

    def __init__(self, endog, exog, instrument, **kwds):
        self.func = func

    def momcond(self, params):
        endog, exog, instrum = self.endog, self.exog, self.instrument
        return instrum * (endog - self.func(params, exog))[:,None]

######## original examples of moment conditions:

def momcondOLS(params, endog, exog):
    #print exog.T.shape, endog.shape, params.shape
    #print np.dot(exog, params).shape
    #return np.dot(exog.T, endog - np.dot(exog, params))
    return exog * (endog - np.dot(exog, params))[:,None]

def momcondIVLS(params, endog, exog, instrum):
    return instrum * (endog - np.dot(exog, params))[:,None]


if __name__ == '__main__':

    exampledata = 'iv' #'ols'
    nobs = nsample = 5000
    sige = 10


    x = np.linspace(0,10, nobs)
    X = sm.add_constant(np.column_stack((x, x**2)))
    beta = np.array([1, 0.1, 10])

    def sample_ols(exog):
        endog = np.dot(exog, beta) + sige*np.random.normal(size=nobs)
        return endog, exog, None

    def sample_iv(exog):
        X = exog
        e = sige * np.random.normal(size=nobs)
        endog = np.dot(X, beta) + e
        X[:,0] += 0.01 * e
        z1 = X.sum(1) + np.random.normal(size=nobs)
        z2 = X[:,1]
        z3 = (np.dot(X, np.array([2,1, 0])) +
                        sige/2. * np.random.normal(size=nobs))
        z4 = X[:,1] + np.random.normal(size=nobs)
        instrument = np.column_stack([z1, z2, z3, z4])
        return endog, exog, instrument

    if exampledata == 'ols':
        endog, exog, _ = sample_ols(X)
        instrument = exog
    else:
        endog, exog, instrument = sample_iv(exog)





    results = sm.OLS(y, X).fit()
    start = beta * 0.9
    resgmm = fitgmm(momcondOLS, (endog, exog), start, fixed=None, weightsoptimal=False)
    print resgmm
    print results.params

    print gmmiter(momcondOLS, start, maxiter=3, args=(endog, exog))
    ##w = np.eye(len(start))
    ##for i in range(10):
    ##    resgmm = fitgmm(momcondOLS, (y,X), start, weights=w, fixed=None, weightsoptimal=False)
    ##    moms = momcondOLS(resgmm, y, X)
    ##    w = np.dot(moms.T,moms)
    ##    start = resgmm
    ##
    ##    print resgmm


    # OLS using IV
    resgmmiv = fitgmm(momcondIVLS, (endog, exog, exog), start, fixed=None, weightsoptimal=False)
    print gmmiter(momcondIVLS, start, maxiter=3, args=(endog, exog, exog))

    endog, exog = y, X
    def momcond(params):
        #endog, exog = args
        return momcondOLS(params, endog, exog).mean(0)

    #maybe grad first and then sum, need 3d return from approx_fprime1
    #   I don't think so, sum/mean and differentiation can be interchanged (?)
    gradgmm = approx_fprime1(resgmm, momcond)
    moms = momcondOLS(resgmm, y, X)
    w = np.dot(moms.T,moms)

    gradmoms2s = (approx_fprime1(resgmm, momcond, epsilon=1e-4)+approx_fprime1(resgmm, momcond, epsilon=-1e-4))/2

    covgmm = gmmcov_params(moms, gradmoms2s)
    print results.cov_params()
    print covgmm
    print 'percent difference'
    print (results.cov_params() - covgmm) / results.cov_params() *100
    print results.bse
    print np.sqrt(np.diag(covgmm))
    covgmmw = gmmcov_params(moms, gradmoms2s,
                            weights=np.linalg.inv(np.dot(moms.T, moms)))
    covgmmw2 = gmmcov_params(moms, gradmoms2s,
                             weights=np.linalg.inv(np.cov(moms, rowvar=0)))


    #using GMM and IV2SLS classes
    #----------------------------

    mod = IVGMM(endog, exog, instrument, nmoms=instrument.shape[1])
    res = mod.fit()
    modls = IV2SLS(endog, exog, instrument)
    resls = modls.fit()
    modols = OLS(endog, exog)
    resols = modols.fit()

    print '\nIV case'
    print 'params'
    print 'IV2SLS', resls.params
    print 'OLS   ', resols.params
    print 'GMM   ', res.params
    print 'diff  ', res.params - resls.params
    print '\nbse'
    print 'IV2SLS', resls.bse
    print 'OLS   ', resols.bse
    print 'GMM   ', mod.bse   #bse currently only attached to model not results
    print 'diff  ', mod.bse - resls.bse
    print '%-diff', resls.bse / mod.bse * 100 - 100
