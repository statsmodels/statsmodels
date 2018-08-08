from statsmodels.compat.python import iterkeys
from statsmodels.regression.linear_model import GLS
import numpy as np
from statsmodels.base.model import LikelihoodModelResults
from scipy import sparse
from statsmodels.compat.numpy import np_matrix_rank

#http://www.irisa.fr/aladin/wg-statlin/WORKSHOPS/RENNES02/SLIDES/Foschi.pdf

__all__ = ['SUR']

#probably should have a SystemModel superclass
# TODO: does it make sense of SUR equations to have
# independent endogenous regressors?  If so, then
# change docs to LHS = RHS
#TODO: make a dictionary that holds equation specific information
#rather than these cryptic lists?  Slower to get a dict value?
#TODO: refine sigma definition
class SUR(object):
    """
    Seemingly Unrelated Regression

    Parameters
    ----------
    sys : list
        [endog1, exog1, endog2, exog2,...] It will be of length 2 x M,
        where M is the number of equations endog = exog.
    sigma : array-like
        M x M array where sigma[i,j] is the covariance between equation i and j
    dfk : None, 'dfk1', or 'dfk2'
        Default is None.  Correction for the degrees of freedom
        should be specified for small samples.  See the notes for more
        information.

    Attributes
    ----------
    cholsigmainv : array
        The transpose of the Cholesky decomposition of `pinv_wexog`
    df_model : array
        Model degrees of freedom of each equation. p_{m} - 1 where p is
        the number of regressors for each equation m and one is subtracted
        for the constant.
    df_resid : array
        Residual degrees of freedom of each equation. Number of observations
        less the number of parameters.
    endog : array
        The LHS variables for each equation in the system.
        It is a M x nobs array where M is the number of equations.
    exog : array
        The RHS variable for each equation in the system.
        It is a nobs x sum(p_{m}) array.  Which is just each
        RHS array stacked next to each other in columns.
    history : dict
        Contains the history of fitting the model. Probably not of interest
        if the model is fit with `igls` = False.
    iterations : int
        The number of iterations until convergence if the model is fit
        iteratively.
    nobs : float
        The number of observations of the equations.
    normalized_cov_params : array
        sum(p_{m}) x sum(p_{m}) array
        :math:`\\left[X^{T}\\left(\\Sigma^{-1}\\otimes\\boldsymbol{I}\\right)X\\right]^{-1}`
    pinv_wexog : array
        The pseudo-inverse of the `wexog`
    sigma : array
        M x M covariance matrix of the cross-equation disturbances. See notes.
    sp_exog : CSR sparse matrix
        Contains a block diagonal sparse matrix of the design so that
        exog1 ... exogM are on the diagonal.
    wendog : array
        M * nobs x 1 array of the endogenous variables whitened by
        `cholsigmainv` and stacked into a single column.
    wexog : array
        M*nobs x sum(p_{m}) array of the whitened exogenous variables.

    Notes
    -----
    All individual equations are assumed to be well-behaved, homoeskedastic
    iid errors.  This is basically an extension of GLS, using sparse matrices.

    .. math:: \\Sigma=\\left[\\begin{array}{cccc}
              \\sigma_{11} & \\sigma_{12} & \\cdots & \\sigma_{1M}\\\\
              \\sigma_{21} & \\sigma_{22} & \\cdots & \\sigma_{2M}\\\\
              \\vdots & \\vdots & \\ddots & \\vdots\\\\
              \\sigma_{M1} & \\sigma_{M2} & \\cdots & \\sigma_{MM}\\end{array}\\right]

    References
    ----------
    Zellner (1962), Greene (2003)
    """
#TODO: Does each equation need nobs to be the same?
    def __init__(self, sys, sigma=None, dfk=None):
        if len(sys) % 2 != 0:
            raise ValueError("sys must be a list of pairs of endogenous and \
exogenous variables.  Got length %s" % len(sys))
        if dfk:
            if not dfk.lower() in ['dfk1','dfk2']:
                raise ValueError("dfk option %s not understood" % (dfk))
        self._dfk = dfk
        M = len(sys[1::2])
        self._M = M
#        exog = np.zeros((M,M), dtype=object)
#        for i,eq in enumerate(sys[1::2]):
#            exog[i,i] = np.asarray(eq)  # not sure this exog is needed
                                        # used to compute resids for now
        exog = np.column_stack(np.asarray(sys[1::2][i]) for i in range(M))
#       exog = np.vstack(np.asarray(sys[1::2][i]) for i in range(M))
        self.exog = exog # 2d ndarray exog is better
# Endog, might just go ahead and reshape this?
        endog = np.asarray(sys[::2])
        self.endog = endog
        self.nobs = float(self.endog[0].shape[0]) # assumes all the same length

# Degrees of Freedom
        df_resid = []
        df_model = []
        [df_resid.append(self.nobs - np_matrix_rank(_)) for _ in sys[1::2]]
        [df_model.append(np_matrix_rank(_) - 1) for _ in sys[1::2]]
        self.df_resid = np.asarray(df_resid)
        self.df_model = np.asarray(df_model)

# "Block-diagonal" sparse matrix of exog
        sp_exog = sparse.lil_matrix((int(self.nobs*M),
            int(np.sum(self.df_model+1)))) # linked lists to build
        self._cols = np.cumsum(np.hstack((0, self.df_model+1)))
        for i in range(M):
            sp_exog[i*self.nobs:(i+1)*self.nobs,
                    self._cols[i]:self._cols[i+1]] = sys[1::2][i]
        self.sp_exog = sp_exog.tocsr() # cast to compressed for efficiency
# Deal with sigma, check shape earlier if given
        if np.any(sigma):
            sigma = np.asarray(sigma) # check shape
        elif sigma == None:
            resids = []
            for i in range(M):
                resids.append(GLS(endog[i],exog[:,
                    self._cols[i]:self._cols[i+1]]).fit().resid)
            resids = np.asarray(resids).reshape(M,-1)
            sigma = self._compute_sigma(resids)
        self.sigma = sigma
        self.cholsigmainv = np.linalg.cholesky(np.linalg.pinv(\
                    self.sigma)).T
        self.initialize()

    def initialize(self):
        self.wendog = self.whiten(self.endog)
        self.wexog = self.whiten(self.sp_exog)
        self.pinv_wexog = np.linalg.pinv(self.wexog)
        self.normalized_cov_params = np.dot(self.pinv_wexog,
                np.transpose(self.pinv_wexog))
        self.history = {'params' : [np.inf]}
        self.iterations = 0

    def _update_history(self, params):
        self.history['params'].append(params)

    def _compute_sigma(self, resids):
        """
        Computes the sigma matrix and update the cholesky decomposition.
        """
        M = self._M
        nobs = self.nobs
        sig = np.dot(resids, resids.T)  # faster way to do this?
        if not self._dfk:
            div = nobs
        elif self._dfk.lower() == 'dfk1':
            div = np.zeros(M**2)
            for i in range(M):
                for j in range(M):
                    div[i+j] = ((self.df_model[i]+1) *\
                            (self.df_model[j]+1))**(1/2)
            div.reshape(M,M)
        else: # 'dfk2' error checking is done earlier
            div = np.zeros(M**2)
            for i in range(M):
                for j in range(M):
                    div[i+j] = nobs - np.max(self.df_model[i]+1,
                        self.df_model[j]+1)
            div.reshape(M,M)
# doesn't handle (#,)
        self.cholsigmainv = np.linalg.cholesky(np.linalg.pinv(sig/div)).T
        return sig/div

    def whiten(self, X):
        """
        SUR whiten method.

        Parameters
        -----------
        X : list of arrays
            Data to be whitened.

        Returns
        -------
        If X is the exogenous RHS of the system.
        ``np.dot(np.kron(cholsigmainv,np.eye(M)),np.diag(X))``

        If X is the endogenous LHS of the system.

        """
        nobs = self.nobs
        if X is self.endog: # definitely not a robust check
            return np.dot(np.kron(self.cholsigmainv,np.eye(nobs)),
                X.reshape(-1,1))
        elif X is self.sp_exog:
            return (sparse.kron(self.cholsigmainv,
                sparse.eye(nobs,nobs))*X).toarray()#*=dot until cast to array

    def fit(self, igls=False, tol=1e-5, maxiter=100):
        """
        igls : bool
            Iterate until estimates converge if sigma is None instead of
            two-step GLS, which is the default is sigma is None.

        tol : float

        maxiter : int

        Notes
        -----
        This ia naive implementation that does not exploit the block
        diagonal structure. It should work for ill-conditioned `sigma`
        but this is untested.
        """

        if not np.any(self.sigma):
            self.sigma = self._compute_sigma(self.endog, self.exog)
        M = self._M
        beta = np.dot(self.pinv_wexog, self.wendog)
        self._update_history(beta)
        self.iterations += 1
        if not igls:
            sur_fit = SysResults(self, beta, self.normalized_cov_params)
            return sur_fit

        conv = self.history['params']
        while igls and (np.any(np.abs(conv[-2] - conv[-1]) > tol)) and \
                (self.iterations < maxiter):
            fittedvalues = (self.sp_exog*beta).reshape(M,-1)
            resids = self.endog - fittedvalues # don't attach results yet
            self.sigma = self._compute_sigma(resids) # need to attach for compute?
            self.wendog = self.whiten(self.endog)
            self.wexog = self.whiten(self.sp_exog)
            self.pinv_wexog = np.linalg.pinv(self.wexog)
            self.normalized_cov_params = np.dot(self.pinv_wexog,
                    np.transpose(self.pinv_wexog))
            beta = np.dot(self.pinv_wexog, self.wendog)
            self._update_history(beta)
            self.iterations += 1
        sur_fit = SysResults(self, beta, self.normalized_cov_params)
        return sur_fit

    def predict(self, design):
        pass


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
