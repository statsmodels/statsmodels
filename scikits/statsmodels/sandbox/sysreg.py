from scikits.statsmodels.regression import GLS
import numpy as np
from scikits.statsmodels import tools
from scikits.statsmodels.model import LikelihoodModelResults
from scipy import sparse

#http://www.irisa.fr/aladin/wg-statlin/WORKSHOPS/RENNES02/SLIDES/Foschi.pdf

#probably should have a SystemModel superclass

class SUR(object):
    """
    Seemingly Unrelated Regression

    Parameters
    ----------
    sys : list
        [endog, exog, endog, exog, ...], length 2*M, where M is the number
        of equations endog = exog.

    sigma : array-like
       M x M array where sigma[i,j] is the covariance between equation i and j
#TODO: refine this definition

    dfk : None, 'dfk1', or 'dfk2'.
        Default is None.  Correction for the degrees of freedom
        should be specified for small samples.  See the notes for more
        information.

    Notes
    ------
    All individual equations are assumed to be well-behaved, homoeskedastic
    iid errors.  This is currently under development and not tested.

    References
    ---------
    Zellner (1962), Greene

    """

    def __init__(self, sys, sigma=None, dfk=None):
        if len(sys) % 2 != 0:
            raise ValueError, "sys must be a list of pairs of endogenous and \
exogenous variables.  Got length %s" % len(sys)
        if dfk:
            if not dfk.lower() in ['dfk1','dfk2']:
                raise ValueError, "dfk option %s not understood" % (dfk)
        self._dfk = dfk
        M = len(sys[1::2])
        self._M = M
#        exog = np.zeros((M,M), dtype=object)
#        for i,eq in enumerate(sys[1::2]):
#            exog[i,i] = np.asarray(eq)  # not sure this exog is needed
                                        # used to compute resids for now
        exog = np.column_stack(np.asarray(sys[1::2][i]) for i in range(M))
        self.exog = exog # 2d ndarray exog is better
# Endog, might just go ahead and reshape this?
        endog = np.asarray(sys[::2])
        self.endog = endog
        self.nobs = float(self.endog[0].shape[0]) # assumes all the same length

# Degrees of Freedom
        df_resid = []
        df_model = []
        [df_resid.append(self.nobs - tools.rank(_)) \
                for _ in sys[1::2]]
        [df_model.append(tools.rank(_) - 1) for _ in sys[1::2]]
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
                resids.append(GLS(endog[i],exog[:,self._cols[i]:self._cols[i+1]]).fit().resid)
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
        self.normalized_cov_params = np.dot(self.pinv_wexog, np.transpose(self.pinv_wexog))
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
        np.dot(np.kron(cholsigmainv,np.eye(M)),np.diag(X))

        If X is the endogenous LHS of the system.

        """
        nobs = self.nobs
        if X is self.endog: # definitely not a robust check
            return np.dot(np.kron(self.cholsigmainv,np.eye(nobs)),
                X.reshape(-1,1))
        elif X is self.sp_exog:
#            return np.dot(sparse.kron(self.cholsigmainv,
#                sparse.eye(nobs,nobs)),X).todense()
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
        while igls and (np.any(np.abs(conv[-2] - conv[-1]) > tol)) and (self.iterations\
                < maxiter):
#            resids = np.dot(self.exog,beta).reshape(M,-1) #change this to use fittedvalues and be
                                                         # cleaner
            fittedvalues = (self.sp_exog*beta).reshape(M,-1)
            resids = self.endog - fittedvalues # don't attach results yet
            self.sigma = self._compute_sigma(resids) # need to attach for compute?
            self.wendog = self.whiten(self.endog)
            self.wexog = self.whiten(self.sp_exog)
            self.pinv_wexog = np.linalg.pinv(self.wexog)
            self.normalized_cov_params = np.dot(self.pinv_wexog, np.transpose(self.pinv_wexog))
            beta = np.dot(self.pinv_wexog, self.wendog)
            self._update_history(beta)
            self.iterations += 1
        sur_fit = SysResults(self, beta, self.normalized_cov_params)
        return sur_fit

    def predict(self, design):
        pass

class SysResults(LikelihoodModelResults):
    """
    """
    def __init__(self, model, params, normalized_cov_params=None, scale=1.):
        super(SysResults, self).__init__(model, params, normalized_cov_params, scale)
        self._get_results()

    def _get_results(self):
        pass

if __name__=='__main__':
    try:
        data = np.genfromtxt('./hsb2.csv', delimiter=",",
                dtype=[",".join(["f8"]*11)][0], names=True)
    except:
        raise ValueError, "You don't have the file, because I'm not sure if \
it's public domain.  You can download it here \
http://www.ats.ucla.edu/stat/R/faq/hsb2.csv"

    # eq 1: science = math female
    # eq 2: write = read female

    import scikits.statsmodels as sm
    import time

    endog1 = data['science'].view(float)
    exog1 = sm.add_constant(data[['math','female']].view(float).reshape(-1,2))
    endog2 = data['write'].view(float)
    exog2 = sm.add_constant(data[['read','female']].view(float).reshape(-1,2))
    sys = [endog1,exog1,endog2,exog2]
# just for a test
    endog3 = data['write'].view(float)
    exog3 = sm.add_constant(data[['write','female','read']].view(float).reshape(-1,3))
    sys2 = [endog1,exog1,endog2,exog2,endog3,exog3]
    t = time.time()
    sur_model = SUR(sys)
    sur_results_fgls = sur_model.fit()  # this is correct vs.
    #http://www.ats.ucla.edu/stat/sas/webbooks/reg/chapter4/sasreg4.htm
    print "This ran in %s seconds" % str(time.time() - t)
    sur_model2 = SUR(sys)
    sur_results_ifgls = sur_model2.fit(igls=True) # this doesn't look right and can't run an iterated
                                                  # fit an fgls fit on a model, because it updates...
#TODO: finish the results class, ie., R-squared, LR test, verify F tests, covariance matrix, standard
# errors, confidence intervals, etc.
#TODO: need to add tests, even though the parameter estimation is correct
#    print "Results from sysreg.SUR"
#    print sur_results_fgls.params
#    print "Results from UCLA SAS page"
#    print np.array([-2.18934, .625141, 20.13265, 5.453748, .535484, 21.83439])

# timings for the old version run without csr
#This ran in 0.228526115417 seconds
#This ran in 0.228340148926 seconds
#This ran in 0.228056907654 seconds
#This ran in 0.229265928268 seconds
#This ran in 0.229331970215 seconds
#This ran in 0.23272895813 seconds
#This ran in 0.22826218605 seconds
#This ran in 0.228145122528 seconds
#This ran in 0.229871034622 seconds
#This ran in 0.22944188118 seconds

# with casting to csr
#This ran in 0.232534885406 seconds
#This ran in 0.238698959351 seconds
#This ran in 0.233359098434 seconds
#This ran in 0.232124090195 seconds
#This ran in 0.232531070709 seconds
#This ran in 0.231685161591 seconds
#This ran in 0.232370138168 seconds
#This ran in 0.232092142105 seconds
#This ran in 0.230885028839 seconds
#This ran in 0.231520175934 seconds

# Looks marginally slower, though it may be a scalability issue or the dot?

