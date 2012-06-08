import numpy as np
from scipy.linalg import block_diag
import statsmodels.tools.tools as tools 
from statsmodels.base.model import LikelihoodModel, LikelihoodModelResults
from statsmodels.regression.linear_model import OLS

class SysModel(LikelihoodModel):
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
    
    eq_i['endog'] : ndarray (N x 1)
    eq_i['exog'] : ndarray (N x K_i)
    # For SEM classes
    eq_i['instruments'] : ndarray (N x L_i)

    Attributes
    ----------
    neqs : int
        Number of equations
    nobs : int
        Number of observations
    endog : ndarray (G x N)
        LHS variables for each equation stacked next to each other in row.
    exog : ndarray (N x sum(K_i))
        RHS variables for each equation stacked next to each other in column.
    sp_exog : sparse matrix
        Contains a block diagonal sparse matrix of the design so that
        eq_i['exog'] are on the diagonal.
    df_model : ndarray (G x 1)
        Model degrees of freedom of each equation. K_i - 1 where K_i is
        the number of regressors for equation i and one is subtracted
        for the constant.
    df_resid : ndarray (G x 1)
        Residual degrees of freedom of each equation. Number of observations
        less the number of parameters.
    '''

    def __init__(self, sys):
        # TODO : check sys is correctly specified
        self.sys = sys
        self.neqs = len(sys)
        self.nobs = len(sys[0]['endog']) # TODO : check nobs is the same for each eq
        self.endog = np.column_stack((np.asarray(eq['endog']) for eq in sys)).T
        self.exog = np.column_stack((np.asarray(eq['exog']) for eq in sys))
        # TODO : convert to a sparse matrix (need scipy >= 0.11dev for sp.block_diag)
        self.sp_exog = block_diag(*(np.asarray(eq['exog']) for eq in sys))

        # Degrees of Freedom
        (df_model, df_resid) = ([], [])
        for eq in sys:
            rank = tools.rank(eq['exog'])
            df_model.append(rank - 1)
            df_resid.append(self.nobs - rank)
        (self.df_model, self.df_resid) = (np.asarray(df_model), np.asarray(df_resid))

        self.initialize()

    def initialize(self):
        pass

class SysGLS(SysModel):
    '''
    Parameters
    ----------
    sys : list of dict
        cf. SysModel
    sigma : scalar or array
        `sigma` the contemporaneous matrix covariance.
        The default is None for no scaling (<=> OLS).  If `sigma` is a scalar, it is
        assumed that `sigma` is an G x G diagonal matrix with the given
        scalar, `sigma` as the value of each diagonal element.  If `sigma`
        is an G-length vector, then `sigma` is assumed to be a diagonal
        matrix with the given `sigma` on the diagonal (<=> WLS).
    dfk : None, 'dfk1', or 'dfk2'
        Default is None.  Correction for the degrees of freedom
        should be specified for small samples.  See the notes for more
        information.

    Attributes
    ----------
    cholsigmainv : array
        The transpose of the Cholesky decomposition of the pseudoinverse of
        the contemporaneous covariance matrix.
    wendog : ndarray (G*N) x 1
        endogenous variables whitened by cholsigmainv and stacked into a single column.
    wexog : matrix (is sparse?)
        whitened exogenous variables sp_exog.
    pinv_wexog : array
        `pinv_wexog` is the Moore-Penrose pseudoinverse of `wexog`.
    normalized_cov_params : array
    '''

    def __init__(self, sys, sigma=None):
        neqs = len(sys)
        if sigma is None:
            self.sigma = np.diag(np.ones(neqs))
        # sigma = scalar
        elif sigma.shape == ():
            self.sigma = np.diag(np.ones(neqs)*sigma)
        # sigma = 1d vector
        elif (sigma.ndim == 1) and sigma.size == neqs:
            self.sigma = np.diag(sigma)
        # sigma = GxG matrix
        elif sigma.shape == (neqs,neqs):
            self.sigma = sigma
        else:
            raise ValueError("sigma is not correctly specified")
        super(SysGLS, self).__init__(sys)

    def initialize(self):
        self.cholsigmainv = np.linalg.cholesky(np.linalg.pinv(self.sigma)).T
        self.wexog = self.whiten(self.sp_exog)
        self.wendog = self.whiten(self.endog.reshape(-1,1))
        self.pinv_wexog = np.linalg.pinv(self.wexog)

    def whiten(self, X):
        '''
        SysGLS whiten method

        Parameters
        ----------
        X : ndarray
            Data to be whitened
        '''
        return np.dot(np.kron(self.cholsigmainv,np.eye(self.nobs)), X)

    def fit(self):
        beta = np.dot(self.pinv_wexog, self.wendog)
        normalized_cov_params = np.dot(self.pinv_wexog, self.pinv_wexog.T)
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
            designs = []
            cur_col = 0
            for eq in range(self.neqs):
                designs.append(exog[:,cur_col:cur_col+self.df_model[eq]+1])
                cur_col += self.df_model[eq]+1
            sp_exog = block_diag(*designs)

        return np.dot(sp_exog, params)


class SysWLS(SysGLS):
    '''
    Parameters
    ----------
    weights : 1d array or scalar, optional
        Variances of each equation. If weights is a scalar then homoscedasticity
        is assumed. Default is no scaling.
    '''
    def __init__(self, sys, weights=1.0):
        neqs = len(sys)
        weights = np.asarray(weights)
        # weights = scalar
        if weights.shape == ():
            sigma = np.diag(np.ones(neqs)*weights)
        # weights = 1d vector
        elif weights.ndim == 1 and weights.size == neqs:
            sigma = np.diag(weights)
        else:
            raise ValueError("weights is not correctly specified")
        super(SysWLS, self).__init__(sys, sigma)

class SysOLS(SysWLS):
    def __init__(self, sys):
        super(SysOLS, self).__init__(sys)

class SysSUR(SysGLS):
    def __init__(self, sys, dfk=None):
        super(SysSUR, self).__init__(sys, None)
        # TODO : check dfk in {None, dfk1, dfk2}
        self.dfk = dfk
        # Compute sigma
        ## OLS equation by equation
        resids = []
        for eq in sys:
            res = OLS(eq['endog'], eq['exog']).fit()
            resids.append(res.resid)
        resids = np.column_stack(resids)
        
        self.sigma = self._compute_sigma(resids)
        self.initialize()

    def _compute_sigma(self, resids):
        '''
        Parameters
        ----------
        resids : ndarray (N x G)
            OLS residuals for each equation stacked in column.
        '''
        nobs = resids.shape[0] # nobs should already be accessible by self.nobs
        if self.dfk is None:
            div = nobs
        elif self.dfk.lower() == 'dfk1':
            div = np.zeros((self.neqs, self.neqs))
            for i in range(self.neqs):
                for j in range(self.neqs):
                    div[i,j] = (self.df_model[i]+1)*(self.df_model[j]+1)**(1/2)
        else:
            div = np.zeros((self.neqs, self.neqs))
            for i in range(self.neqs):
                for j in range(self.neqs):
                    div[i,j] = nobs - np.max((self.df_model[i]+1, 
                                             self.df_model[j]+1))
        return (np.dot(resids.T, resids) / div)

class SysSURI(SysOLS):
    '''
    SUR estimation with identical regressors in each equation.
    It's equivalent to an OLS equation-by-equation.

    Parameters
    ----------
    endogs : list of array
        List endog variable for each equation.
    exog : ndarray
        Common exog variables in each equation.
    '''
    def __init__(self, endogs, exog):
        # Build the corresponding system
        sys = []
        for endog in endogs:
            eq = {}
            eq['endog'] = endog
            eq['exog'] = exog
            sys.append(eq)
        super(SysSURI, self).__init__(sys)

class SysResults(LikelihoodModelResults):
    """
    Not implemented yet.
    """
    def __init__(self, model, params, normalized_cov_params=None, scale=1.):
        super(SysResults, self).__init__(model, params,
                normalized_cov_params, scale)

# Testing/Debugging
if __name__ == '__main__':
    from statsmodels.tools import add_constant
    
    nobs = 100
    (y1,y2) = (np.random.rand(nobs), np.random.rand(nobs))
    (x1,x2) = (np.random.rand(nobs,3), np.random.rand(nobs,4))
    (x1,x2) = (add_constant(x1,prepend=True),add_constant(x2,prepend=True))

    (eq1, eq2) = ({}, {})
    eq1['endog'] = y1
    eq1['exog'] = x1
    eq2['endog'] = y2
    eq2['exog'] = x2
    
    sys = [eq1, eq2]
    old_sys = [y1,x1,y2,x2]

    from statsmodels.sysreg.sysreg import SUR
    s,s1,s2 = SysSUR(sys), SysSUR(sys, dfk='dfk1'), SysSUR(sys, dfk='dfk2')
    #p,p1,p2 = SUR(old_sys), SUR(old_sys, dfk='dfk1'), SUR(old_sys, dfk='dfk2') # Bug

