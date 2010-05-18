"""
This is the VAR class adapted from pymaclab.
"""
from __future__ import division
import copy as COP
import scipy as S
import numpy as np
from numpy import matlib as MAT
from scipy import linalg as LIN #TODO: get rid of this once cleaned up
from scipy import linalg, sparse
import scikits.statsmodels as sm    # maybe can be replaced later
from scikits.statsmodels import GLS
from scikits.statsmodels.sandbox.tools.tools_tsa import lagmat

#chain_dot was taken from pandas and is awesome!
def chain_dot(*matrices):
    """
    Returns the dot product of the given matrices.

    Parameters
    ----------
    matrices: argument list of ndarray
    """
    return reduce(lambda x, y: np.dot(y, x), matrices[::-1])


# Refactor of VAR to be like statsmodels
#inherit GLS?
class VAR2(object):
    """
    Parameters
    ----------
    data
    laglen
    useconst
    """
    def __init__(self, data=None, laglen=1, useconst=True):
        self.data = data
        self.laglen = float(laglen)
        self._useconst = useconst
        nobs = float(data.shape[0])
        self.nobs = nobs
        self.nvars = data.shape[1]
        self.avobs = nobs - laglen # what's a better name for this? autonobs? lagnobs?

#TODO: rename to fit or something?
    def ols(self):
        """
        The OLS Function does....

        It just calls GLS
        """
        #recast indices to integers
        avobs = int(self.avobs)
        laglen = int(self.laglen)
        nobs = int(self.nobs)
        nvars = int(self.nvars)
        data = self.data
        # trim from the front, unravel in F-contiguous way
        Y = data[laglen:,:].ravel('F')
        X = np.zeros((avobs,nvars*laglen))
        for x1 in range(0,laglen):
            X[:,x1*nvars:(x1+1)*nvars] = data[(laglen-1)-x1:(nobs-1)-x1,:]
        assert np.all(X == lagmat(data, laglen-1, trim="backward")[:-laglen])
        #which I don't understand yet...
        if self._useconst: # let user handle this?
            X = sm.add_constant(X,prepend=True)
#TODO:change to sparse matrices?
        diag_X = linalg.block_diag(*[X]*nvars)
#        spdiag_X = sparse.lil_matrix(diag_X.shape)
#        for i in range(nvars):
#            spdiag_X[i*shape0:shape0*(i+1),i*shape1:(i+1)*shape1] = X
#TODO: the below will be ok (get feedback on other ones from ML)
#could also use SUR for this.
#        spX = sparse.kron(sparse.eye(20,20),X).todia()
        results = GLS(Y,diag_X).fit()

#TODO: return a results class just like everything else
        self.results = results
        #params <-- betas
        params = results.params.reshape(nvars,-1)
        ncoefs = params.shape[1] # used in omega_beta_ols
        self.params = params # does it make sense to partition like old VAR?
        #yfit
        self.fittedvalues = results.fittedvalues.reshape(-1,nvars,order='F')
        #resid
        resid = results.resid.reshape(-1,nvars,order='F')
        self.resid = resid
        #omega, better name? is this somewhere in the results already?
        #I believe this is the between equation covariance
        omega = np.dot(resid.T,resid)/avobs
        self.omega = omega
        #omega_beta_ols, the covariance of each equation
        XTXinv = results.normalized_cov_params[:ncoefs,:ncoefs]
        obols = map(np.multiply, [XTXinv]*nvars, np.diag(omega))
        self.omega_beta_ols = np.asarray(obols)
#TODO: this is deja vu... this is all in SUR is it not?
        self.omega_beta_va = map(np.diag, obols)
        #Get GLS Covariance
        self.X = X # remove after testing
        self.XTXinv =XTXinv # iXX in old VAR
        # this is just a list of length nvars, with each
        # XeeX where e is the residuals for that equation
        # really just a scaling argument
        XeeX = [chain_dot(X.T, resid[:,i][:,None], resid[:,i][:,None].T, X) for i in range(nvars)]
#        XeeX = map(np.dot, map(np.dot, [X.T]*nvars,resid.T),
#            map(np.dot, resid.T, [X]*nvars))
        self.XeeX = XeeX #TODO: REMOVE
        obgls = np.array(map(chain_dot, [XTXinv]*nvars, XeeX,
                [XTXinv]*nvars))
        self.omega_beta_gls = obgls
        self.omega_beta_gls_va = map(np.diag, obgls)
# the last three properties have rounding error stemming from fittedvalues
# dot vs matrix multiplication, test with real data and not random
        self.ssr = results.ssr # rss in old VAR
        self.aic = linalg.det(omega)+2.*laglen*nvars**2/nobs
        self.bic = linalg.det(omega)+np.log(nobs)/nobs*laglen*nvars**2

#could this just be a standalone function?
    def irf(self, spos=1, nperiods=20):
        self.spos = spos
        self.nperiods = nperiods
        nvars = self.nvars
        laglen = self.laglen
        nobs = self.nobs
        avobs = self.avobs
        useconst = self._useconst

        #demean y
        data = self.data
        self.dmdata = data - data.mean(0) # need to attach everything?
        if useconst:
            # takes demeaned data
            # does OLS_comp
            pass

#TODO: None of these really make sense as methods of a VAR class
    def olscomp(self):
        """
        Does OLS in Companion Matrix Form.

        Uses demeaned data.
        """
        nvars = self.nvars
        laglen = int(self.laglen)
        nobs = self.nobs
        avobs = self.avobs
        useconst = self._useconst
        data = self.data

        # Stack Y,X
        # not sure about the last index being general
        y_stack = lagmat(data, laglen-1, trim="both")[laglen-1:]
        X_stack = lagmat(data, laglen-1, trim="backward")[:-laglen]




# Setting standard VAR options
VAR_opts = {}
VAR_opts['IRF_periods'] = 20




###############THE VECTOR AUTOREGRESSION CLASS (WORKS)###############
class VAR:
    """
    This is the vector autoregression class. It supports estimation,
    doing IRFs and other stuff.
    """
    def __init__(self,laglen=1,data='none',set_useconst='const'):
        self.VAR_attr = {}
        self.getdata(data)
        if set_useconst == 'noconst':
            self.setuseconst(0)
        elif set_useconst == 'const':
            self.setuseconst(1)
        self.setlaglen(laglen)

    def getdata(self,data):
        self.data = np.mat(data)
        self.VAR_attr['nuofobs'] = self.data.shape[0]
        self.VAR_attr['veclen'] = self.data.shape[1]

    def setuseconst(self,useconst):
        self.VAR_attr['useconst'] = useconst

    def setlaglen(self,laglen):
        self.VAR_attr['laglen'] = laglen
        self.VAR_attr['avobs'] = self.VAR_attr['nuofobs'] - self.VAR_attr['laglen']

    #This is the OLS function and does just that, no input
    def ols(self):
        self.ols_results = {}
        data = self.data
        VAR_attr = self.VAR_attr
        veclen = VAR_attr['veclen']
        laglen = VAR_attr['laglen']
        nuofobs = VAR_attr['nuofobs']
        avobs = VAR_attr['avobs']
        useconst = VAR_attr['useconst']
        y = data[laglen:,:]
        X = MAT.zeros((avobs,veclen*laglen))
        for x1 in range(0,laglen,1):
            X[:,x1*veclen:(x1+1)*veclen] = data[(laglen-1)-x1:(nuofobs-1)-x1,:]
        if self.VAR_attr['useconst'] == 1:
            X = np.hstack((MAT.ones((avobs,1)),X[:,:]))
        self.ols_results['y'] = y
        self.ols_results['X'] = X

        if useconst == 1:
            beta = MAT.zeros((veclen,1+veclen*laglen))
        else:
            beta = MAT.zeros((veclen,veclen*laglen))
        XX = X.T*X
        try:
            iXX = XX.I
        except:
            err = TS_err('Singular Matrix')
            return
        self.iXX = iXX
        for i in range(0,veclen,1):
            Xy = X.T*y[:,i]
            beta[i,:] = (iXX*Xy).T
        self.ols_results['beta'] = beta
        yfit = X*beta.T
        self.ols_results['yfit'] = yfit
        resid = y-yfit
        self.ols_results['resid'] = resid
        #Separate out beta's into B matrices
        self.ols_results['BBs'] = {}
        if self.VAR_attr['useconst'] == 0:
            i1 = 1
            for x1 in range(0,veclen*laglen,laglen):
                self.ols_results['BBs']['BB'+str(i1)] = beta[:,x1:x1+veclen]
                i1 = i1 + 1
        elif self.VAR_attr['useconst'] == 1:
            self.ols_results['BBs']['BB0'] = beta[:,0]
            i1 = 1
            for x in range(0,veclen*laglen,laglen):
                self.ols_results['BBs']['BB'+str(i1)] = beta[:,x+1:x+veclen+1]
                i1 = i1 + 1

        #Make variance-covariance matrix of residuals
        omega = (resid.T*resid)/avobs
        self.ols_results['omega'] = omega
        #Make variance-covariance matrix for est. coefficients under OLS
        omega_beta_ols = (np.multiply(iXX,np.diag(omega)[0]),)
        for i1 in range(1,veclen,1):
            omega_beta_ols = omega_beta_ols+(np.multiply(iXX,np.diag(omega)[i1]),)
        self.ols_results['omega_beta'] = omega_beta_ols
        #Extract just the diagonal variances of omega_beta_ols
        omega_beta_ols_va = (np.diag(omega_beta_ols[0]),)
        for i1 in range(1,veclen,1):
            omega_beta_ols_va = omega_beta_ols_va+(np.diag(omega_beta_ols[i1]),)
        self.ols_results['omega_beta_va'] = omega_beta_ols_va
        #Make variance-covariance matrix of est. coefficients under GLS
        XeeX = X.T*resid[:,0]*resid[:,0].T*X
        self.X = X  #TODO: REMOVE
        self.resid = resid #TODO: REMOVE
        self.XeeX = XeeX #TODO: REMOVE
        omega_beta_gls = (iXX*XeeX*iXX,)
        for i1 in range(1,veclen,1):
            XeeX = X.T*resid[:,i1]*resid[:,i1].T*X
            omega_beta_gls = omega_beta_gls+(iXX*XeeX*iXX,)
        self.ols_results['omega_beta_gls'] = omega_beta_gls
        #Extract just the diagonal variances of omega_beta_gls
        omega_beta_gls_va = np.diag(omega_beta_gls[0])
        for i1 in range(1,veclen,1):
            omega_beta_gls_va = (omega_beta_gls_va,)+(np.diag(omega_beta_gls[i1]),)
        self.ols_results['omega_beta_gls_va'] = omega_beta_gls_va
        sqresid = np.power(resid,2)
        rss = np.sum(sqresid)
        self.ols_results['rss'] = rss
        AIC = S.linalg.det(omega)+(2.0*laglen*veclen**2)/nuofobs
        self.ols_results['AIC'] = AIC
        BIC = S.linalg.det(omega)+(np.log(nuofobs)/nuofobs)*laglen*veclen**2
        self.ols_results['BIC'] = BIC

    def do_irf(self,spos=1,plen=VAR_opts['IRF_periods']):
        self.IRF_attr = {}
        self.IRF_attr['spos'] = spos
        self.IRF_attr['plen'] = plen
        VAR_attr = self.VAR_attr
        veclen = VAR_attr['veclen']
        laglen = VAR_attr['laglen']
        nuofobs = VAR_attr['nuofobs']
        avobs = VAR_attr['avobs']
        useconst = VAR_attr['useconst']
        self.irf_results = {}
        # Strip out the means of vector y
        data = self.data
        dmeans = np.mat(np.average(data,0))
        dmeans = np.mat(dmeans.tolist()*nuofobs)
        self.dmeans = dmeans
        dmdata = data - dmeans
        self.IRF_attr['dmdata'] = dmdata
        # Do OLS on de-meaned series and collect BBs
        if self.VAR_attr['useconst'] == 1:
            self.setuseconst(0)
            self.data = dmdata
            self.ols_comp()
            self.IRF_attr['beta'] = COP.deepcopy(self.ols_comp_results['beta'])
            self.ols()
            omega = COP.deepcopy(self.ols_results['omega'])
            self.IRF_attr['omega'] = COP.deepcopy(self.ols_results['omega'])
            self.setuseconst(1)
            self.data = data
            self.ols_comp()
        elif self.VAR_attr['useconst'] == 0:
            self.data = dmdata
            self.ols_comp()
            self.IRF_attr['beta'] = COP.deepcopy(self.ols_comp_results['beta'])
            self.ols()
            omega = self.ols_results['omega']
            self.IRF_attr['omega'] = COP.deepcopy(self.ols_results['omega'])
            self.data = data
            self.ols_comp()

        # Calculate Cholesky decomposition of omega
        A0 = np.mat(S.linalg.cholesky(omega))
        A0 = A0.T
        A0 = A0.I
        self.IRF_attr['A0'] = A0
        beta = self.IRF_attr['beta']

        #Calculate IRFs using ols_comp_results
        ee = MAT.zeros((veclen*laglen,1))
        ee[spos-1,:] = 1
        shock = np.vstack((A0.I*ee[0:veclen,:],ee[veclen:,:]))
        Gam = shock.T
        Gam_2 = Gam[0,0:veclen]
        for x1 in range(0,plen,1):
            Gam_X = beta*Gam[x1,:].T
            Gam_2 = np.vstack((Gam_2,Gam_X.T[:,0:veclen]))
            Gam = np.vstack((Gam,Gam_X.T))

        self.irf_results['Gammas'] = Gam_2

    #This does OLS in first-order companion form (stacked), no input
    def ols_comp(self):
        self.ols_comp_results = {}
        self.make_y_X_stack()
        VAR_attr = self.VAR_attr
        veclen = VAR_attr['veclen']
        laglen = VAR_attr['laglen']
        nuofobs = VAR_attr['nuofobs']
        avobs = VAR_attr['avobs']
        useconst = VAR_attr['useconst']
        veclen = veclen*laglen
        laglen = 1
        X = self.ols_comp_results['X']
        y = self.ols_comp_results['y']

        if useconst == 1:
            beta = MAT.zeros((veclen,1+veclen*laglen))
        else:
            beta = MAT.zeros((veclen,veclen*laglen))
        XX = X.T*X
        iXX = XX.I
        for i in range(0,veclen,1):
            Xy = X.T*y[:,i]
            beta[i,:] = (iXX*Xy).T

        veclen2 = VAR_attr['veclen']
        for x1 in range(0,beta.shape[0],1):
            for x2 in range(0,beta.shape[1],1):
                if beta[x1,x2] < 1e-7 and x1 > veclen2-1:
                    beta[x1,x2] = 0.0

        for x1 in range(0,beta.shape[0],1):
            for x2 in range(0,beta.shape[1],1):
                if  0.9999999 < beta[x1,x2] < 1.0000001 and x1 > veclen2-1:
                    beta[x1,x2] = 1.0

        self.ols_comp_results['beta'] = beta
        yfit = X*beta.T
        self.ols_comp_results['yfit'] = yfit
        resid = y-yfit
        self.ols_comp_results['resid'] = resid
        #Make variance-covariance matrix of residuals
        omega = (resid.T*resid)/avobs
        self.ols_comp_results['omega'] = omega
        #Make variance-covariance matrix for est. coefficients under OLS
        omega_beta_ols = (np.diag(omega)[0]*iXX,)
        for i1 in range(1,veclen,1):
            omega_beta_ols = omega_beta_ols + (np.diag(omega)[i1]*iXX,)
        self.ols_comp_results['omega_beta'] = omega_beta_ols
        #Extract just the diagonal variances of omega_beta_ols
        omega_beta_ols_va = (np.diag(omega_beta_ols[0]),)
        for i1 in range(1,veclen,1):
            omega_beta_ols_va = omega_beta_ols_va+(np.diag(omega_beta_ols[i1]),)
        self.ols_comp_results['omega_beta_va'] = omega_beta_ols_va
        #Make variance-covariance matrix of est. coefficients under GLS
        XeeX = X.T*resid[:,0]*resid[:,0].T*X
        omega_beta_gls = (iXX*XeeX*iXX,)
        for i1 in range(1,veclen,1):
            XeeX = X.T*resid[:,i1]*resid[:,i1].T*X
            omega_beta_gls = omega_beta_gls+(iXX*XeeX*iXX,)
        self.ols_comp_results['omega_beta_gls'] = omega_beta_gls
        #Extract just the diagonal variances of omega_beta_gls
        omega_beta_gls_va = (np.diag(omega_beta_gls[0]),)
        for i1 in range(1,veclen,1):
            omega_beta_gls_va = omega_beta_gls_va+(np.diag(omega_beta_gls[i1]),)
        self.ols_comp_results['omega_beta_gls_va'] = omega_beta_gls_va
        sqresid = np.power(resid,2)
        rss = np.sum(sqresid)
        self.ols_comp_results['rss'] = rss
        AIC = S.linalg.det(omega[0:veclen,0:veclen])+(2.0*laglen*veclen**2)/nuofobs
        self.ols_comp_results['AIC'] = AIC
        BIC = S.linalg.det(omega[0:veclen,0:veclen])+(np.log(nuofobs)/nuofobs)*laglen*veclen**2
        self.ols_comp_results['BIC'] = BIC

    #This is a function which sets lag by AIC, input is maximum lags over which to search
    def lag_by_AIC(self,lags):
        laglen = self.VAR_attr['laglen']
        lagrange = np.arange(1,lags+1,1)
        i1 = 0
        for i in lagrange:
            self.laglen = i
            self.ols()
            if i1 == 0:
                AIC = self.AIC
                i1 = i1 + 1
            else:
                AIC = append(AIC,self.AIC)
                i1 = i1 + 1
        index = argsort(AIC)
        i1 = 1
        for element in index:
            if element == 0:
                lag_AIC = i1
                break
            else:
                i1 = i1 + 1
        self.VAR_attr['lag_AIC'] = lag_AIC
        self.VAR_attr['laglen'] = lag_AIC

    #This is a function which sets lag by BIC, input is maximum lags over which to search
    def lag_by_BIC(self,lags):
        laglen = self.VAR_attr['laglen']
        lagrange = np.arange(1,lags+1,1)
        i1 = 0
        for i in lagrange:
            self.laglen = i
            self.ols()
            if i1 == 0:
                BIC = self.BIC
                i1 = i1 + 1
            else:
                BIC = append(BIC,self.BIC)
                i1 = i1 + 1
        index = argsort(BIC)
        i1 = 1
        for element in index:
            if element == 0:
                lag_BIC = i1
                break
            else:
                i1 = i1 + 1
        self.VAR_attr['lag_BIC'] = lag_BIC
        self.VAR_attr['laglen'] = lag_BIC

    #Auxiliary function, creates the y and X matrices for OLS_comp
    def make_y_X_stack(self):
        VAR_attr = self.VAR_attr
        veclen = VAR_attr['veclen']
        laglen = VAR_attr['laglen']
        nuofobs = VAR_attr['nuofobs']
        avobs = VAR_attr['avobs']
        useconst = VAR_attr['useconst']
        data = self.data

        y = data[laglen:,:]
        X = MAT.zeros((avobs,veclen*laglen))
        for x1 in range(0,laglen,1):
            X[:,x1*veclen:(x1+1)*veclen] = data[(laglen-1)-x1:(nuofobs-1)-x1,:]
        if self.VAR_attr['useconst'] == 1:
            X = np.hstack((MAT.ones((avobs,1)),X[:,:]))
        try:
            self.ols_results['y'] = y
            self.ols_results['X'] = X
        except:
            self.ols()
            self.ols_results['y'] = y
            self.ols_results['X'] = X

        if useconst == 0:
            y_stack = MAT.zeros((avobs,veclen*laglen))
            y_stack_1 = MAT.zeros((avobs,veclen*laglen))
            y_stack[:,0:veclen] = y
            y_stack_1 = X

            y_stack = np.hstack((y_stack[:,0:veclen],y_stack_1[:,0:veclen*(laglen-1)]))

            self.ols_comp_results['X'] = y_stack_1
            self.ols_comp_results['y'] = y_stack
        else:
            y_stack = MAT.zeros((avobs,veclen*laglen))
            y_stack_1 = MAT.zeros((avobs,1+veclen*laglen))
            y_stack_1[:,0] = MAT.ones((avobs,1))[:,0]
            y_stack[:,0:veclen] = y
            y_stack_1 = X

            y_stack = np.hstack((y_stack[:,0:veclen],y_stack_1[:,1:veclen*(laglen-1)+1]))

            self.ols_comp_results['X'] = y_stack_1
            self.ols_comp_results['y'] = y_stack
"""***********************************************************"""

if __name__ == "__main__":
    import numpy as np
#    vr = VAR(data = np.random.randn(50,3))
#    vr.ols()
#    vr.ols_comp()
#    vr.do_irf()
#    print vr.irf_results.keys()
#    print vr.ols_comp_results.keys()
#    print vr.ols_results.keys()
#    print dir(vr)
    np.random.seed(12345)
    data = np.random.rand(50,3)
    vr = VAR(data = data, laglen=2)
    vr2 = VAR2(data = data, laglen=2)
    dataset = sm.datasets.macrodata.Load()
    data = dataset.data
    XX = data[['realinv','realgdp','realcons']].view(float).reshape(-1,3)
    XX = np.diff(np.log(XX), axis=0)
    vrx = VAR(data=XX,laglen=2)
    vrx2 = VAR2(data=XX, laglen=2)




