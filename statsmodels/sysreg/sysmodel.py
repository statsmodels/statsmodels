import numpy as np
from statsmodels.base.model import LikelihoodModel, LikelihoodModelResults
from scipy.linalg import block_diag

class SysModel(object):
    '''
    A multiple regressions model. The class SysModel itself is not to be used.

    SysModel lays out the methods and attributes expected of any subclass.

    Notations
    ---------
    G : number of equations
    N : number of observations (same for each equation)
    K_i : number of regressors in equation i including the intercept if one is
    included in the data.

    Parameters
    ----------
    sys : list of dict
        [eq_1,...,eq_G]
    
    eq_i['endog'] : array-like (N x 1)
    eq_i['exog'] : array-like (N x K_i)
    #eq_i['instruments'] : list of array-like

    Attributes
    ----------
    neqs : int
        Number of equations
    nobs : int
        Number of observations
    endog : array (G x N)
        LHS variables for each equation stacked next to each other in row.
    exog : array (N x sum(K_i))
        RHS variables for each equation stacked next to each other in column.
    sp_exog : CSR sparse matrix
        Contains a block diagonal sparse matrix of the design so that
        eq_i['exog']  are on the diagonal.
    '''

    def __init__(self, sys):
        # TODO : check sys is correctly specified

        neqs = len(sys)
        nobs = len(sys[0]['endog']) # TODO : check nobs is the same for each eq
        endog = np.column_stack((np.asarray(eq['endog']) for eq in sys)).T
        exog = np.column_stack((np.asarray(eq['exog']) for eq in sys))
        # TODO : convert to a sparse matrix
        sp_exog = block_diag(*(np.asarray(eq['exog']) for eq in sys))

        self.neqs = neqs
        self.nobs = nobs
        self.endog = endog
        self.exog = exog
        self.sp_exog = sp_exog

    def fit(self):
        raise NotImplementedError

    def predict(self, params, exog=None, *args, **kwargs):
        raise NotImplementedError

class SysGLS(SysModel):
    '''
    Parameters
    ----------
    sigma : scalar or array
           `sigma` is the weighting matrix of the contemporaneous covariance.
           The default is None for no scaling.  If `sigma` is a scalar, it is
           assumed that `sigma` is an G x G diagonal matrix with the given
           scalar, `sigma` as the value of each diagonal element.  If `sigma`
           is an G-length vector, then `sigma` is assumed to be a diagonal
           matrix with the given `sigma` on the diagonal.  This should be the
           same as WLS.

    Attributes
    ----------
    cholsigmainv : array
        The transpose of the Cholesky decomposition of the pseudoinverse of
        the contemporaneous covariance sigma.
    wendog : array ((G*N) x 1)
        endogenous variables whitened by cholsigmainv and stacked into a single column.
    wexog : sparse matrix
        whitened exogenous variables sp_exog.
    pinv_wexog : array
        `pinv_wexog` is the Moore-Penrose pseudoinverse of `wexog`.
    normalized_cov_params : array
    '''

    def __init__(self, sys, sigma=None):
        super(SysGLS, self).__init__(sys)
        
        if sigma is None:
            self.sigma = np.diag(np.ones(self.neqs))
        # sigma = scalar
        elif sigma.shape == ():
            self.sigma = np.diag(np.ones(self.neqs)*sigma)
        # sigma = 1d vector
        elif (sigma.ndim == 1) and sigma.size == self.neqs:
            self.sigma = np.diag(sigma)
        # sigma = GxG matrix
        elif sigma.shape == (self.neqs,self.neqs):
            self.sigma = sigma
        else:
            raise ValueError("sigma is not correctly specified")
        self.cholsigmainv = np.linalg.cholesky(np.linalg.pinv(self.sigma)).T
        self.initialize()

    def initialize(self):
        self.wexog = self.whiten(self.sp_exog)
        self.wendog = self.whiten(self.endog.reshape(-1,1))
        self.pinv_wexog = np.linalg.pinv(self.wexog)
        self.normalized_cov_params = np.dot(self.pinv_wexog,
                np.transpose(self.pinv_wexog))
     
    def whiten(self, X):
        '''
        SysGLS whiten method

        Paramaters
        ----------
        X : array
            Data to be whitened
        '''
        return np.dot(np.kron(self.cholsigmainv,np.eye(self.nobs)), X)

    def fit(self):
        beta = np.dot(self.pinv_wexog, self.wendog)
        return SysResults(self, beta, self.normalized_cov_params)

class SysWLS(SysGLS):
    '''
    Parameters
    ----------
    weights : array-like, optional
        1d array of weights or scalar
    '''
    def __init__(self, sys, weights=1.0):
        weights = np.asarray(weights)
        neqs = len(sys)

        # weights = scalar
        if weights.shape == ():
            sigma = np.diag(np.ones(neqs)*weights)
        # weights = 1d vector
        elif weights.ndim == 1 and weights.size == neqs:
            sigma = np.diag(weights)
        else:
            raise ValueError("weights is not correctly specified")

        super(SysWLS, self).__init__(sys, sigma=sigma)

class SysOLS(SysWLS):
    def __init__(self, sys):
        super(SysWLS, self).__init__(sys)

class SysResults(LikelihoodModelResults):
    """
    Not implemented yet.
    """
    def __init__(self, model, params, normalized_cov_params=None, scale=1.):
        super(SysResults, self).__init__(model, params,
                normalized_cov_params, scale)
        self._get_results()

    def _get_results(self):
        pass

# Testing/Debugging
if __name__ == '__main__':
    from statsmodels.tools import add_constant
    
    nobs = 10
    (y1,y2) = (np.random.rand(nobs), np.random.rand(nobs))
    (x1,x2) = (np.random.rand(nobs,3), np.random.rand(nobs,4))
    (x1,x2) = (add_constant(x1,prepend=True),add_constant(x2,prepend=True))

    (eq1, eq2) = ({}, {})
    eq1['endog'] = y1
    eq1['exog'] = x1
    eq2['endog'] = y2
    eq2['exog'] = x2
    
    sys = [eq1, eq2]
    s1 = SysGLS(sys)

    from statsmodels.sysreg.sysreg import SUR
    s2 = SUR([y1,x1,y2,x2])

    s3 = SysGLS(sys, sigma=s2.sigma)
    s4 = SysWLS(sys, weights=3.0)
    s5 = SysOLS(sys)

