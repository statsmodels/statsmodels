from numpy import matlib as MAT
from scipy.stats import norm
import scipy as S
import numpy as np

from scikits.statsmodels.regression.linear_model import GLS
from scikits.statsmodels.tools.tools import chain_dot
from scikits.statsmodels.tsa.tsatools import add_trend
from scikits.statsmodels.base.model import LikelihoodModel
from scikits.statsmodels.tools.decorators import cache_readonly
from scikits.statsmodels.tools.compatibility import np_slogdet

__all__ = ['VAR2']

# Refactor of VAR to be like statsmodels
#inherit GLS, SUR?
#class VAR2(object):
class VAR2(LikelihoodModel):
    """
    Vector Autoregression (VAR, VARX) models.

    Parameters
    ----------
    endog
    exog
    laglen

    Notes
    -----
    Exogenous variables are not supported yet

    """
    def __init__(self, endog, exog=None):
        """
        Vector Autoregression (VAR, VARX) models.

        Parameters
        ----------
        endog
        exog
        laglen

        Notes
        -----
        Exogenous variables are not supported yet

        """
        nobs = float(endog.shape[0])
        self.nobs = nobs
        self.nvars = endog.shape[1] # should this be neqs since we might have
                                   # exogenous data?
                                   #NOTE: Yes
        self.neqs = endog.shape[1]
        super(VAR2, self).__init__(endog, exog)

    def loglike(self, params, omega):
        """
        Returns the value of the VAR(p) log-likelihood.

        Parameters
        ----------
        params : array-like
            The parameter estimates
        omega : ndarray
            Sigma hat matrix.  Each element i,j is the average product of the
            OLS residual for variable i and the OLS residual for variable j or
            np.dot(resid.T,resid)/avobs.  There should be no correction for the
            degrees of freedom.


        Returns
        -------
        loglike : float
            The value of the loglikelihood function for a VAR(p) model

        Notes
        -----
        The loglikelihood function for the VAR(p) is

        .. math:: -\\left(\\frac{T}{2}\\right)\\left(\\ln\\left|\\Omega\\right|-K\\ln\\left(2\\pi\\right)-K\\right)

        """
        params = np.asarray(params)
        omega = np.asarray(omega)
        logdet = np_slogdet(omega)
        if logdet[0] == -1:
            raise ValueError("Omega matrix is not positive definite")
        elif logdet[0] == 0:
            raise ValueError("Omega matrix is singluar")
        else:
            logdet = logdet[1]
        avobs = self.avobs
        neqs = self.neqs
        return -(avobs/2.)*(neqs*np.log(2*np.pi)+logdet+neqs)

#TODO: IRF, lag length selection
    def fit(self, method="ols", structural=None, dfk=None, maxlag=None,
            ic=None, trend="c"):
        """
        Fit the VAR model

        Parameters
        ----------
        method : str
            "ols" fit equation by equation with OLS
            "yw" fit with yule walker
            "mle" fit with unconditional maximum likelihood
            Only OLS is currently implemented.
        structural : str, optional
            If 'BQ' - Blanchard - Quah identification scheme is used.
            This imposes long run restrictions. Not yet implemented.
        dfk : int or Bool optional
            Small-sample bias correction.  If None, dfk = 0.
            If True, dfk = neqs * nlags + number of exogenous variables.  The
            user can also provide a number for dfk. Omega is divided by (avobs -
            dfk).
        maxlag : int, optional
            The highest lag order for lag length selection according to `ic`.
            The default is 12 * (nobs/100.)**(1./4).  If ic=None, maxlag
            is the number of lags that are fit for each equation.
        ic : str {"aic","bic","hqic", "fpe"} or None, optional
            Information criteria to maximize for lag length selection.
            Not yet implemented for VAR.
        trend, str {"c", "ct", "ctt", "nc"}
            "c" - add constant
            "ct" - constant and trend
            "ctt" - constant, linear and quadratic trend
            "nc" - co constant, no trend
            Note that these are prepended to the columns of the dataset.


        Notes
        -----
        Not sure what to do with structural. Restrictions would be on
        coefficients or on omega.  So should it be short run (array),
        long run (array), or sign (str)?  Recursive?
        """
        if dfk is None:
            self.dfk = 0
        elif dkf is True:
            self.dfk = self.X.shape[1] #TODO: change when we accept
                                          # equations for endog and exog
        else:
            self.dfk = dfk

        nobs = int(self.nobs)

        self.avobs = nobs - maxlag # available obs (sample - pre-sample)


#        #recast indices to integers #TODO: really?  Is it easier to just use
                                     # floats in other places or import
                                     # division?

        # need to recompute after lag length selection
        avobs = int(self.avobs)
        if maxlag is None:
            maxlag = round(12*(nobs/100.)**(1/4.))
        self.laglen = maxlag #TODO: change when IC selection is sorted
#        laglen = se
        nvars = int(self.nvars)
        neqs = int(self.neqs)
        endog = self.endog
        laglen = maxlag
        Y = endog[laglen:,:]

        # Make lagged endogenous RHS
        X = np.zeros((avobs,nvars*laglen))
        for x1 in xrange(laglen):
            X[:,x1*nvars:(x1+1)*nvars] = endog[(laglen-1)-x1:(nobs-1)-x1,:]
#NOTE: the above loop is faster than lagmat
#        assert np.all(X == lagmat(endog, laglen-1, trim="backward")[:-laglen])

        # Prepend Exogenous variables
        if self.exog is not None:
            X = np.column_stack((self.exog[laglen:,:], X))

        # Handle constant, etc.
        if trend == 'c':
            trendorder = 1
        elif trend == 'nc':
            trendorder = 0
        elif trend == 'ct':
            trendorder = 2
        elif trend == 'ctt':
            trendorder = 3
        if trend != 'nc':
            X = add_trend(X,prepend=True, trend=trend)
        self.trendorder = trendorder

        self.Y = Y
        self.X = X

# Two ways to do block diagonal, but they are slow
# diag
#        diag_X = linalg.block_diag(*[X]*nvars)
#Sparse: Similar to SUR
#        spdiag_X = sparse.lil_matrix(diag_X.shape)
#        for i in range(nvars):
#            spdiag_X[i*shape0:shape0*(i+1),i*shape1:(i+1)*shape1] = X
#        spX = sparse.kron(sparse.eye(20,20),X).todia()
#        results = GLS(Y,diag_X).fit()

        lagstart = trendorder
        if self.exog is not None:
            lagstart += self.exog.shape[1] #TODO: is there a variable that
                                           #      holds exog.shapep[1]?


#NOTE: just use GLS directly
        results = []
        for y in Y.T:
            results.append(GLS(y,X).fit())
        params = np.vstack((_.params for _ in results))

#TODO: For coefficient restrictions, will have to use SUR


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


# Setting standard VAR options
VAR_opts = {}
VAR_opts['IRF_periods'] = 20

#from scikits.statsmodels.sandbox.output import SimpleTable

#TODO: correct results if fit by 'BQ'
class VARMAResults(object):
    """
    Holds the results for VAR models.

    Parameters
    ----------
    model
    results
    params

    Attributes
    ----------
    aic : float
        (Lutkepohl 2004)
    avobs : float
        Available observations for estimation.  The size of the whole sample
        less the pre-sample observations needed for lags.
    bic : float
        Bayesian information criterium
    df_resid : float
        Residual degrees of freedom.
    dfk : float
        Degrees of freedom correction.  Not currently used. MLE estimator of
        omega is used everywhere.
    fittedvalues
    fpe : float
        (Lutkepohl 2005, p 146-7). See notes.
    laglen :
    model :
    ncoefs :
    neqs :
    nobs : int
        Total number of observations in the sample.
    omega : ndarray
        Sigma hat matrix.  Each element i,j is the average product of the OLS
        residual for variable i and the OLS residual for variable j or
        np.dot(resid.T,resid)/avobs.  There is no correction for the degrees
        of freedom.  This is the maximum likelihood estimator of Omega.
    omega_beta_gls :
    omega_beta_gls_va :
    omega_beta_ols :
    omega_beta_va :
    params : array
        The fitted parameters for each equation.  Note that the rows are the
        equations and that each row holds lags first then variables, so
        it is the first lag for `neqs` variables, the second lag for `neqs`
        variables, etc. exogenous variables and then the trend variables are
        prepended as columns.
    results : list
        Each entry is the equation by equation OLS results if VAR was fit by
        OLS.

    Methods
    -------
    summary
       a summary table of the estimation results


    Notes
    ------
    FPE formula

    .. math:: \\left[\\frac{T+Kp+t}{T-Kp-t}\\right]^{K}\\left|\\Omega\\right|

    Where T = `avobs`
          K = `neqs`
          p = `laglength`
          t = `trendorder`

    """
    def __init__(self, model, results, params):
        self.results = results # most of this won't work, keep?
        self.model = model
        self.avobs = model.avobs
        self.dfk = model.dfk
        self.neqs = model.neqs
        self.laglen = model.laglen
        self.nobs = model.nobs
# it's lag1 of y1, lag1 of y2, lag1 of y3 ... lag2 of y1, lag2 of y2 ...
        self.params = params
        self.ncoefs = self.params.shape[1]
        self.df_resid = model.avobs - self.ncoefs # normalize sigma by this
        self.trendorder = model.trendorder

    @cache_readonly
    def fittedvalues(self):
        np.column_stack((_.fittedvalues for _ in results))
#        return self.results.fittedvalues.reshape(-1, self.neqs, order='F')

    @cache_readonly
    def resid(self):
        results = self.results
        return np.column_stack((_.resid for _ in results))
#        return self._results.resid.reshape(-1,self.neqs,order='F')

    @cache_readonly
    def omega(self):
        resid = self.resid
        return np.dot(resid.T,resid)/(self.avobs - self.dfk)
#TODO: include dfk correction anywhere or not?  No small sample bias?

    @cache_readonly
    def llf(self):
        return self.model.loglike(self.params, self.omega)

#    @cache_readonly
#    def omega(self): # variance of residuals across equations
#        resid = self.resid
#        return np.dot(resid.T,resid)/(self.avobs - self.dfk)

#    @cache_readonly
#    def omega_beta_ols(self): # the covariance of each equation (check)
#        ncoefs = self.params.shape[1]
#        XTXinv = self._results.normalized_cov_params[:ncoefs,:ncoefs]
#        # above is iXX in old VAR
#        obols = map(np.multiply, [XTXinv]*self.neqs, np.diag(self.omega))
#        return np.asarray(obols)

#    @cache_readonly
#    def omega_beta_va(self):
#        return map(np.diag, self.omega_beta_ols)

#    @cache_readonly
#    def omega_beta_gls(self):
#        X = self.model.X
#        resid = self.resid
#        neqs = self.neqs
#        ncoefs = self.ncoefs
#        XTXinv = self._results.normalized_cov_params[:ncoefs,:ncoefs]
        # Get GLS Covariance
        # this is just a list of length nvars, with each
        # XeeX where e is the residuals for that equation
        # really just a scaling argument
#        XeeX = [chain_dot(X.T, resid[:,i][:,None], resid[:,i][:,None].T,
#            X) for i in range(neqs)]
#        obgls = np.array(map(chain_dot, [XTXinv]*neqs, XeeX,
#                [XTXinv]*neqs))
#        return obgls

#    @cache_readonly
#    def omega_beta_gls_va(self):
#        return map(np.diag, self.omega_beta_gls_va)

# the next three properties have rounding error stemming from fittedvalues
# dot vs matrix multiplication vs. old VAR, test with another package

#    @cache_readonly
#    def ssr(self):
#        return self.results.ssr # rss in old VAR

#    @cache_readonly
#    def root_MSE(self):
#        avobs = self.avobs
#        laglen = self.laglen
#        neqs = self.neqs
#        trendorder = self.trendorder
#        return np.sqrt(np.diag(self.omega*avobs/(avobs-neqs*laglen-
#            trendorder)))

    @cache_readonly
    def rsquared(self):
        results = self.results
        return np.vstack((_.rsquared for _ in results))

    @cache_readonly
    def bse(self):
        results = self.results
        return np.vstack((_.bse for _ in results))

    @cache_readonly
    def aic(self):
        #JP: same as self.detomega ?
        logdet = np_slogdet(self.omega)
        if logdet[0] == -1:
            raise ValueError("Omega matrix is not positive definite")
        elif logdet[0] == 0:
            raise ValueError("Omega matrix is singluar")
        else:
            logdet = logdet[1]
        neqs = self.neqs
        trendorder = self.trendorder
        return logdet+ (2/self.avobs) * (self.laglen*neqs**2+trendorder*neqs)

    @cache_readonly
    def bic(self):
        #JP: same as self.detomega ?
        logdet = np_slogdet(self.omega)
        if logdet[0] == -1:
            raise ValueError("Omega matrix is not positive definite")
        elif logdet[0] == 0:
            raise ValueError("Omega matrix is singluar")
        else:
            logdet = logdet[1]
        avobs = self.avobs
        neqs = self.neqs
        trendorder = self.trendorder
        return logdet+np.log(avobs)/avobs*(self.laglen*neqs**2 +
                neqs*trendorder)

    @cache_readonly
    def hqic(self):
        logdet = np_slogdet(self.omega)
        if logdet[0] == -1:
            raise ValueError("Omega matrix is not positive definite")
        elif logdet[0] == 0:
            raise ValueError("Omega matrix is singluar")
        else:
            logdet = logdet[1]
        avobs = self.avobs
        laglen = self.laglen
        neqs = self.neqs
        trendorder = self.trendorder
        return logdet + 2*np.log(np.log(avobs))/avobs * (laglen*neqs**2 +
                trendorder * neqs)
#TODO: do the above correctly handle extra exogenous variables?

    @cache_readonly
    def df_eq(self):
#TODO: change when we accept coefficient restrictions
        return self.ncoefs

    @cache_readonly
    def fpe(self):
        detomega = self.detomega
        avobs = self.avobs
        neqs = self.neqs
        laglen = self.laglen
        trendorder = self.trendorder
        return ((avobs+neqs*laglen+trendorder)/(avobs-neqs*laglen-
            trendorder))**neqs * detomega

    @cache_readonly
    def detomega(self):
        return np.linalg.det(self.omega)

#    @wrap
#    def wrap(self, attr, *args):
#        return self.__getattribute__(attr, *args)

    @cache_readonly
    def bse(self):
        return np.sqrt(np.diag(self.cov_params)).reshape(self.neqs, -1,
                order = 'F')

    @cache_readonly
    def z(self):
        return self.params/self.bse

    @cache_readonly
    def pvalues(self):
        return norm.sf(np.abs(self.z))*2

    @cache_readonly
    def cov_params(self):
        #NOTE: Cov(Vec(B)) = (Z'Z)^-1 kron Omega
        X = self.model.X
        return np.kron(np.linalg.inv(np.dot(X.T,X)), self.omega)
#TODO: this might need to be changed when order is changed and with exog

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

    def summary(self, endog_names=None, exog_names=None):
        """
        Summary of VAR model
        """
        import time
        from scikits.statsmodels.iolib import SimpleTable
        model = self.model

        if endog_names is None:
            endog_names = self.model.endog_names

        # take care of exogenous names
        if model.exog is not None and exog_names is None:
            exog_names = model.exog_names
        elif exog_names is not None:
            if len(exog_names) != model.exog.shape[1]:
                raise ValueError("The number of exog_names does not match the \
size of model.exog")
        else:
            exog_names = []

        lag_names = []
        # take care of lagged endogenous names
        laglen = self.laglen
        for i in range(1,laglen+1):
            for ename in endog_names:
                lag_names.append('L'+str(i)+'.'+ename)
        # put them together
        Xnames = exog_names + lag_names

        # handle the constant name
        trendorder = self.trendorder
        if trendorder != 0:
            Xnames.insert(0, 'const')
        if trendorder > 1:
            Xnames.insert(0, 'trend')
        if trendorder > 2:
            Xnames.insert(0, 'trend**2')
        Xnames *= self.neqs


        modeltype = model.__class__.__name__
        t = time.localtime()

        ncoefs = self.ncoefs #TODO: change when we allow coef restrictions
        part1_fmt = dict(
            data_fmts = ["%s"],
            empty_cell = '',
            colwidths = 15,
            colsep=' ',
            row_pre = '',
            row_post = '',
            table_dec_above='=',
            table_dec_below='',
            header_dec_below=None,
            header_fmt = '%s',
            stub_fmt = '%s',
            title_align='c',
            header_align = 'r',
            data_aligns = "r",
            stubs_align = "l",
            fmt = 'txt'
        )
        part2_fmt = dict(
            data_fmts = ["%#12.6g","%#12.6g","%#10.4g","%#5.4g"],
            empty_cell = '',
            colwidths = None,
            colsep='    ',
            row_pre = '',
            row_post = '',
            table_dec_above='-',
            table_dec_below='-',
            header_dec_below=None,
            header_fmt = '%s',
            stub_fmt = '%s',
            title_align='c',
            header_align = 'r',
            data_aligns = 'r',
            stubs_align = 'l',
            fmt = 'txt'
        )

        part3_fmt = dict(
            #data_fmts = ["%#12.6g","%#12.6g","%#10.4g","%#5.4g"],
            #data_fmts = ["%#10.4g","%#10.4g","%#10.4g","%#6.4g"],
            data_fmts = ["%#15.6F","%#15.6F","%#15.3F","%#14.3F"],
            empty_cell = '',
            #colwidths = 10,
            colsep='  ',
            row_pre = '',
            row_post = '',
            table_dec_above='=',
            table_dec_below='=',
            header_dec_below='-',
            header_fmt = '%s',
            stub_fmt = '%s',
            title_align='c',
            header_align = 'r',
            data_aligns = 'r',
            stubs_align = 'l',
            fmt = 'txt'
        )

        # Header information
        part1title = "Summary of Regression Results"
        part1data = [[modeltype],
                     ["OLS"], #TODO: change when fit methods change
                     [time.strftime("%a, %d, %b, %Y", t)],
                     [time.strftime("%H:%M:%S", t)]]
        part1header = None
        part1stubs = ('Model:',
                     'Method:',
                     'Date:',
                     'Time:')
        part1 = SimpleTable(part1data, part1header, part1stubs, title=
                part1title, txt_fmt=part1_fmt)

        #TODO: do we want individual statistics or should users just
        # use results if wanted?
        # Handle overall fit statistics
        part2Lstubs = ('No. of Equations:',
                       'Nobs:',
                       'Log likelihood:',
                       'AIC:')
        part2Rstubs = ('BIC:',
                       'HQIC:',
                       'FPE:',
                       'Det(Omega_mle):')
        part2Ldata = [[self.neqs],[self.nobs],[self.llf],[self.aic]]
        part2Rdata = [[self.bic],[self.hqic],[self.fpe],[self.detomega]]
        part2Lheader = None
        part2L = SimpleTable(part2Ldata, part2Lheader, part2Lstubs,
                txt_fmt = part2_fmt)
        part2R = SimpleTable(part2Rdata, part2Lheader, part2Rstubs,
                txt_fmt = part2_fmt)
        part2L.extend_right(part2R)

        # Handle coefficients
        part3data = []
        part3data = zip([self.params.ravel()[i] for i in range(len(Xnames))],
                [self.bse.ravel()[i] for i in range(len(Xnames))],
                [self.z.ravel()[i] for i in range(len(Xnames))],
                [self.pvalues.ravel()[i] for i in range(len(Xnames))])
        part3header = ('coefficient','std. error','z-stat','prob')
        part3stubs = Xnames
        part3 = SimpleTable(part3data, part3header, part3stubs, title=None,
                txt_fmt = part3_fmt)


        table = str(part1) +'\n'+str(part2L) + '\n' + str(part3)
        return table


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
        from copy import deepcopy

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
            self.IRF_attr['beta'] = deepcopy(self.ols_comp_results['beta'])
            self.ols()
            omega = deepcopy(self.ols_results['omega'])
            self.IRF_attr['omega'] = deepcopy(self.ols_results['omega'])
            self.setuseconst(1)
            self.data = data
            self.ols_comp()
        elif self.VAR_attr['useconst'] == 0:
            self.data = dmdata
            self.ols_comp()
            self.IRF_attr['beta'] = deepcopy(self.ols_comp_results['beta'])
            self.ols()
            omega = self.ols_results['omega']
            self.IRF_attr['omega'] = deepcopy(self.ols_results['omega'])
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
    import scikits.statsmodels.api as sm
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
    vr2 = VAR2(endog = data)
    dataset = sm.datasets.macrodata.load()
    data = dataset.data
    XX = data[['realinv','realgdp','realcons']].view((float,3))
    XX = np.diff(np.log(XX), axis=0)
    vrx = VAR(data=XX,laglen=2)
    vrx.ols() # fit
    for i,j in vrx.ols_results.items():
        setattr(vrx, i,j)
    vrx2 = VAR2(endog=XX)
    res = vrx2.fit(maxlag=2)
    vrx3 = VAR2(endog=XX)

    varx = VAR2(endog=XX, exog=np.diff(np.log(data['realgovt']), axis=0))
    resx = varx.fit(maxlag=2)

# some data for an example in Box Jenkins
    IBM = np.asarray([460,457,452,459,462,459,463,479,493,490.])
    w = np.diff(IBM)
    theta = .5
