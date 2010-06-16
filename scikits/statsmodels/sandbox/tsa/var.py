"""
This is the VAR class refactored from pymaclab.
"""
from __future__ import division
import copy as COP
import scipy as S
import numpy as np
from numpy import matlib as MAT
from scipy import linalg as LIN #TODO: get rid of this once cleaned up
from scipy import linalg, sparse
import scikits.statsmodels as sm    # maybe can be replaced later
from scikits.statsmodels import GLS, chain_dot
from scikits.statsmodels.sandbox.tools.tools_tsa import lagmat
from scikits.statsmodels.model import LikelihoodModelResults
from scikits.statsmodels.decorators import *

# Refactor of VAR to be like statsmodels
#inherit GLS, SUR?
class VAR2(object):
    def __init__(self, endog=None, exog=None, laglen=1, useconst=True):
        """
        Parameters
        ----------
        endog
        exog
        laglen
        useconst

        Notes
        -----
        Exogenous variables are not supported yet
        """
        self.endog = endog    #TOD):rename endog
        self.laglen = float(laglen)
        self._useconst = useconst
        nobs = float(endog.shape[0])
        self.nobs = nobs
        self.nvars = endog.shape[1] # should this be neqs since we might have
                                   # exogenous data?
        self.neqs = endog.shape[1]
        self.avobs = nobs - laglen
        # what's a better name for this? autonobs? lagnobs?

#TODO: make ols comp default
    def fit(self, method="ols", structural=None, dfk=None):
        """
        Fit the VAR model

        Parameters
        ----------
        method : str
            "ols_comp" fit with OLS in companion form, defaul
            "ols" fit equation by equation with OLS
            "yw" fit with yule walker
            "mle" fit with unconditional maximum likelihood
        structural : str, optional
            If 'BQ' - Blanchard - Quah identification scheme is used.
            This imposes Long
        dfk : int, optional
            Small-sample bias correction.  If None, dfk = neqs * nlags +
            number of exogenous variables. Run restrictions.  Details in Lyx
            notes.

        Notes
        -----
        Not sure what to do with structural. Restrictions would be on
        coefficients or on omega.  So should it be short run (array),
        long run (array), or sign (str)?  Recursive?
        """
        if dfk is None:
            self.dfk = self.laglen * self.neqs
        else:
            self.dfk = dfk
        # What's cleaner? Logic handled here and private functions or all here?
        if method == "ols_comp":
            return self._ols_comp()
        if method == "ols":
            return self._ols()
#TODO: should 'BQ' just have it's own method?


    def _ols(self, structural=None):
        """
        The OLS Function does....

        It just calls GLS with no arguments.
        """
        #recast indices to integers
        avobs = int(self.avobs)
        laglen = int(self.laglen)
        nobs = int(self.nobs)
        nvars = int(self.nvars)
        neqs = int(self.neqs)
        endog = self.endog
        # trim from the front, unravel in F-contiguous way
        Y = endog[laglen:,:].ravel('F')
        X = np.zeros((avobs,nvars*laglen))
        self.X = X #TODO: rename or refactor? (exog?) lagged_exog?
        for x1 in range(0,laglen):
            X[:,x1*nvars:(x1+1)*nvars] = endog[(laglen-1)-x1:(nobs-1)-x1,:]
        assert np.all(X == lagmat(endog, laglen-1, trim="backward")[:-laglen])
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
        params = results.params.reshape(neqs,-1)
#TODO: make a separate SVAR class or this is going to get really messy
        if structural and structural.lower() == 'bq':
            phi = np.swapaxes(params.reshape(neqs,laglen,neqs), 1,0)
            I_phi_inv = np.linalg.inv(np.eye(n) - phi.sum(0))
            omega = np.dot(results.resid.T,resid)/(avobs - self.dfk)
            shock_var = chain_dot(I_phi_inv, omega, I_phi_inv.T)
            R = np.linalg.cholesky(shock_var)
            phi_normalize = np.dot(I_phi_inv,R)
            params = np.zeros_like(phi)
            #TODO: apply a dot product along an axis?
            for i in range(laglen):
                params[i] = np.dot(phi_normalize, phi[i])
                params = np.swapaxes(params, 1,0).reshape(neqs,laglen*neqs)
        return VARMAResults(self, results, params)






#TODO: None of these really make sense as methods
#make a fit method with the choice of fit -- OLS, OLS Companion, Unconditional
#MLE, etc.
    def _olscomp(self, demean=False):
        """
        Does OLS in Companion Matrix Form.
        """
        nvars = self.nvars
        laglen = int(self.laglen)
        nobs = self.nobs
        avobs = self.avobs
        useconst = self._useconst
        endog = self.endog

        # Stack Y,X
        # not sure about the last index being general
        y_stack = lagmat(endog, laglen-1, trim="both")[laglen-1:]
        X_stack = lagmat(endog, laglen-1, trim="backward")[:-laglen]
#TODO: finish this




# Setting standard VAR options
VAR_opts = {}
VAR_opts['IRF_periods'] = 20

#from scikits.statsmodels.sandbox.output import SimpleTable

#TODO: correct results if fit by 'BQ'
class VARMAResults(object):
    """
    """
    def __init__(self, model, results, params):
        self.results = results
        self.model = model
        self.avobs = model.avobs
        self.dfk = model.dfk
        self.neqs = model.neqs
        self.laglen = model.laglen
        self.nobs = model.nobs
        self.params = params
        self.ncoefs = self.params.shape[1]
        self.df_resid = model.avobs - self.ncoefs # normalize sigma by this

#    @cache_readonly
#    def params(self):
# note the order of this
# it's lag1 of y1, lag1 of y2, lag1 of y3 ... lag2 of y1, lag2 of y2 ...
# and each row is a separate equation
#        return self.results.params.reshape(self.neqs,-1)

    @cache_readonly
    def fittedvalues(self):
        return self.results.fittedvalues.reshape(-1, self.neqs, order='F')

    @cache_readonly
    def resid(self):
        return self.results.resid.reshape(-1,self.neqs,order='F')

#TODO: pass in from fit like regression models?
    @cache_readonly
    def omega(self): # variance of 'shocks' across equations
        resid = self.resid
        return np.dot(resid.T,resid)/(self.avobs - self.dfk)

    @cache_readonly
    def omega_beta_ols(self): # the covariance of each equation (check)
        ncoefs = self.params.shape[1]
        XTXinv = self.results.normalized_cov_params[:ncoefs,:ncoefs]
        # above is iXX in old VAR
        obols = map(np.multiply, [XTXinv]*self.neqs, np.diag(self.omega))
        return np.asarray(obols)

    @cache_readonly
    def omega_beta_va(self):
        return map(np.diag, self.omega_beta_ols)

    @cache_readonly
    def omega_beta_gls(self):
        X = self.model.X
        resid = self.resid
        neqs = self.neqs
        XTXinv = self.results.normalized_cov_params[:ncoefs,:ncoefs]
        # Get GLS Covariance
        # this is just a list of length nvars, with each
        # XeeX where e is the residuals for that equation
        # really just a scaling argument
        XeeX = [chain_dot(X.T, resid[:,i][:,None], resid[:,i][:,None].T,
            X) for i in range(neqs)]
        obgls = np.array(map(chain_dot, [XTXinv]*neqs, XeeX,
                [XTXinv]*neqs))
        return obgls

    @cache_readonly
    def omega_beta_gls_va(self):
        return map(np.diag, self.omega_beta_gls_va)

# the next three properties have rounding error stemming from fittedvalues
# dot vs matrix multiplication vs. old VAR, test with another package

    @cache_readonly
    def ssr(self):
        return self.results.ssr # rss in old VAR

    @cache_readonly
    def aic(self):
        return linalg.det(self.omega)+2.*self.laglen*self.neqs**2/self.nobs

    @cache_readonly
    def bic(self):
        nobs = self.nobs
        linalg.det(self.omega)+np.log(nobs)/nobs*self.laglen*self.nvars**2

    @wrap
    def wrap(self, attr, *args):
        return self.__getattribute__(attr, *args)

    #could this just be a standalone function?
    def irf(self, shock, params=None, nperiods=100):
        """
        Make the impulse response function.

        Parameters
        -----------
        shock : array-like
            An array of shocks must be provided that is shape (neqs,)

        If params is None, uses the model params. Note that no normalizing is
        done to the parameters.  Ie., this assumes the the coefficients are
        the identified structural coefficients.

        Notes
        -----
        TODO: Allow for common recursive structures.
        """
        neqs = self.neqs
        shock = np.asarray(shock)
        if shock.shape[0] != neqs:
            raise ValueError("Each shock must be specified even if it's zero")
        if shock.ndim > 1:
            shock = np.squeeze(shock)   # more robust check vs neqs
        if params == None:
            params = self.params
        laglen = int(self.laglen)
        nobs = self.nobs
        avobs = self.avobs
#Needed?
#        useconst = self._useconst
        responses = np.zeros((neqs,laglen+nperiods))
        responses[:,laglen] = shock # shock in the first period with
                                    # all lags set to zero
        for i in range(laglen,laglen+nperiods):
            # flatten lagged responses to broadcast with
            # current layout of params
            # each equation is in a row
            # row i needs to be y1_t-1 y2_t-1 y3_t-1 ... y1_t-2 y1_t-2...
            laggedres = responses[:,i-laglen:i][:,::-1].ravel('F')
            responses[:,i] = responses[:,i] + np.sum(laggedres * params,
                    axis = 1)
        return responses


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




