import numpy as np
import scipy as sp
from statsmodels.sysreg.sysmodel import SysModel, SysResults
from statsmodels.compatnp.sparse import block_diag as sp_block_diag
import statsmodels.tools.tools as tools
import scipy.sparse as sparse

def unique_rows(a):
    unique_a = np.unique(a.view([('', a.dtype)]*a.shape[1]))
    return unique_a.view(a.dtype).reshape((unique_a.shape[0], a.shape[1]))

def unique_cols(a):
    return unique_rows(a.T.copy()).T

class SysSEM(SysModel):
    """
    Least Squares for Simultaneous equations

    Parameters
    ----------
    sys : list of dict
        cf. SysModel. Each equation has now an 'indep_endog' key which is a list
        of the column numbers of the independent endogenous regressors.
    instruments : array, optional
        Array of additional instruments.
    dkf :
    sigma : 

    Attributes
    ----------
    fullexog : ndarray
        The larger set of instruments is used for estimation, including
        all of the exogenous variables in the system and the others instruments
        provided in 'instruments' parameters. 
    """

    def __init__(self, sys, instruments=None, sigma=None, dfk=None):
        super(SysSEM, self).__init__(sys, dfk=dfk)

        ## Handle sigma
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
        
        ## Handle restrictions: TODO

        ## Handle instruments design
        if (instruments is not None) and (instruments.shape[0] != self.nobs):
            raise ValueError("instruments is not correctly specified")
        self.instruments = instruments
        exogs = []
        for eq in self.sys:
            id_exog = list(set(range(eq['exog'].shape[1])).difference(
                    eq['indep_endog']))
            exogs.append(eq['exog'][:, id_exog])
        fullexog = np.column_stack(exogs)
        if not(self.instruments is None):
            fullexog = np.hstack((self.instruments, fullexog))
            # Note : the constant is not in the first column. 
        # Delete reoccuring cols
        self.fullexog = z = unique_cols(fullexog)

        ## Handle first-step
        ztzinv = np.linalg.inv(np.dot(z.T, z))
        #TODO: Josef: "some streamlining in the linear algebra is necessary. 
        #For example the projection matrix Pz, as you use it in the 1st stage 
        #regression is (nobs, nobs), which is large and inefficient for 
        #larger samples."
        Pz = np.dot(np.dot(z, ztzinv), z.T) 
        xhats = [np.dot(Pz, eq['exog']) for eq in self.sys]
        self.sp_xhat = sp_block_diag(xhats)
        # Identification conditions
        xhats_ranks = [tools.rank(cur_xhat) for cur_xhat in xhats]
        nexogs = [eq['exog'].shape[1] for eq in self.sys]
        if not(xhats_ranks == nexogs):
            raise ValueError('identification conditions are not statisfied')

        self.initialize()

    def initialize(self):
        self.cholsigmainv = np.linalg.cholesky(np.linalg.pinv(self.sigma)).T
        self.wxhat = self.whiten(self.sp_xhat)
        self.wendog = self.whiten(self.endog.T.reshape(-1,1))
        self.pinv_wxhat = np.linalg.pinv(self.wxhat)
    
    def whiten(self, X):
        '''
        SysSEM whiten method

        Parameters
        ----------
        X : ndarray
            Data to be whitened
        '''
        if sparse.issparse(X):
            return np.asarray((sparse.kron(self.cholsigmainv, 
                sparse.eye(self.nobs, self.nobs))*X).todense())
        else:
            return np.dot(np.kron(self.cholsigmainv,np.eye(self.nobs)), X)

    def _compute_res(self):
        params = np.squeeze(np.dot(self.pinv_wxhat, self.wendog))
        normalized_cov_params = np.dot(self.pinv_wxhat, self.pinv_wxhat.T)
        return (params, normalized_cov_params)

    def fit(self, igls=False, tol=1e-5, maxiter=100):
        res = self._compute_res()
        if not(igls):
            return SysResults(self, res[0], res[1])

        betas = [res[0], np.inf]
        iterations = 1

        while np.any(np.abs(betas[0] - betas[1]) > tol) \
                and iterations < maxiter:
            # Update sigma
            fittedvalues = (self.sp_exog*betas[0]).reshape(self.neqs,-1).T
            resids = self.endog - fittedvalues
            self.sigma = self._compute_sigma(resids)
            # Update attributes
            self.initialize()
            # Next iteration
            res = self._compute_res()
            betas = [res[0], betas[0]]
            iterations += 1
       
        self.iterations = iterations
        beta = betas[0]
        normalized_cov_params = self._compute_res()[1]
        return SysResults(self, beta, normalized_cov_params)

    def predict(self, params, exog=None):
        '''
        Parameters
        ----------
        exog : None or list of ndarray
            List of individual design (one for each equation)
        '''
        if exog is None:
            sp_exog = self.sp_exog
        else:
            sp_exog = sp_block_diag(exog)

        return sp_exog * params

class Sys2SLS(SysSEM):
    def __init__(self, sys, instruments=None, dfk=None):
        super(Sys2SLS, self).__init__(sys, instruments=instruments,
                sigma=None, dfk=dfk)
    
    def fit(self):
        res_fit = super(Sys2SLS, self).fit()
        params = res_fit.params
        # Covariance matrix of the parameters computed using new residuals as in systemfit
        self.sigma = np.diag(np.diag(res_fit.cov_resids))
        self.initialize()
        normalized_cov_params = np.dot(self.pinv_wxhat, self.pinv_wxhat.T)
        return SysResults(self, params, normalized_cov_params)

    def _compute_sigma(self, resids):
        '''
        Parameters
        ----------
        resids : ndarray (N x G)
            Residuals for each equation stacked in column.
        '''
        s = np.diag(np.diag(np.dot(resids.T, resids)))
        if self.dfk is None:
            return s / self.nobs
        elif self.dfk == 'dfk1':
            return s / self._div_dfk1
        else:
            return s / self._div_dfk2

class Sys3SLS(SysSEM):
    def __init__(self, sys, instruments=None, dfk=None):
        super(Sys3SLS, self).__init__(sys, instruments=instruments,
                sigma=None, dfk=dfk)

        # Estimate sigma with a first-step 2SLS
        sigma = Sys2SLS(self.sys, self.instruments, self.dfk).fit().cov_resids
        self.sigma = sigma

        self.initialize()

