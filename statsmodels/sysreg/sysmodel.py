import numpy as np
import statsmodels.tools.tools as tools 
import scipy.sparse as sparse
from scipy.linalg import block_diag
from statsmodels.base.model import LikelihoodModel, LikelihoodModelResults
from statsmodels.regression.linear_model import OLS
from statsmodels.compatnp.sparse import block_diag as sp_block_diag
from cStringIO import StringIO
from statsmodels.iolib import SimpleTable

class SysModel(LikelihoodModel):
    '''
    A multiple regressions model. The class SysModel itself is not to be used.

    SysModel lays out the methods and attributes expected of any subclass.

    Notations
    ---------
    neqs : number of equations
    nobs : number of observations (same for each equation)
    K_i : number of regressors in equation i including the intercept if one is
    included in the data.

    Parameters
    ----------
    sys : list of dict
        [eq_1,...,eq_neqs]
    
    eq_i['endog'] : ndarray, (nobs, 1) or (nobs,)
    eq_i['exog'] : ndarray, (nobs, K_i)

    Attributes
    ----------
    neqs : int
        Number of equations
    nobs : int
        Number of observations
    endog : ndarray, (nobs, neqs)
        LHS variables stacked so that each column is one endogenous variable.
    sp_exog : sparse matrix
        Contains a block diagonal sparse matrix of the design so that
        eq_i['exog'] are on the diagonal.
    df_model : ndarray, (neqs, 1)
        Model degrees of freedom of each equation. K_i - 1 where K_i is
        the number of regressors for equation i and one is subtracted
        for the constant.
    df_resid : ndarray, (neqs, 1)
        Residual degrees of freedom of each equation. Number of observations
        less the number of parameters.
    '''

    def __init__(self, sys, dfk=None):
        # Check if sys is correctly specified
        #issystem = isinstance(sys, (list, tuple)) and \
        #    all([isinstance(eq, dict) for eq in sys])
        #if not issystem:
        #    raise ValueError('systems must be list (or tuple) of dict')

        isregression = all(['endog' in eq for eq in sys]) and \
            all(['exog' in eq for eq in sys])
        if not isregression:
            raise ValueError('each equation of a system must have endog and \
                exog keys') 
        
        allnobs = [eq['endog'].shape[0] for eq in sys] + \
            [eq['exog'].shape[0] for eq in sys]
        isidnobs = len(set(allnobs)) == 1
        if not isidnobs:
            raise ValueError('each equation must have number of observations \
                be identical')

        # Others checks
        if not dfk in (None, 'dfk1', 'dfk2'):
            raise ValueError('dfk is not correctly specified')
        
        self.sys = sys
        self.neqs = len(sys)
        self.nobs = sys[0]['endog'].shape[0]
        self.k_exog_all = sum([eq['exog'].shape[1] for eq in sys])
        self.dfk = dfk

        # Degrees of Freedom
        df_model, df_resid = [], []
        for eq in sys:
            rank = tools.rank(eq['exog'])
            df_model.append(rank - 1)
            df_resid.append(self.nobs - rank)
        self.df_model, self.df_resid = np.asarray(df_model), np.asarray(df_resid)
        #TODO: check for singular exog here
 
        self.endog = np.column_stack((np.asarray(eq['endog']) for eq in sys))
        #self.exog = np.column_stack((np.asarray(eq['exog']) for eq in sys))
        #self.exog : ndarray, (nobs, sum(K_i))
        #RHS variables stacked so that each column is one regressor.
        self.sp_exog = sp_block_diag([np.asarray(eq['exog']) 
            for eq in sys])

        # Compute DoF corrections
        div_dfk1 = np.zeros((self.neqs, self.neqs))
        div_dfk2 = np.zeros((self.neqs, self.neqs))
        for i in range(self.neqs):
            for j in range(self.neqs):
                div_dfk1[i,j] = np.sqrt( (self.nobs - self.df_model[i] - 1) *
                                         (self.nobs - self.df_model[j] - 1) )
                div_dfk2[i,j] = self.nobs - np.max((self.df_model[i] + 1, 
                                                    self.df_model[j] + 1))
 
        self._div_dfk1 = div_dfk1
        self._div_dfk2 = div_dfk2

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

    def _compute_sigma(self, resids):
        '''
        Parameters
        ----------
        resids : ndarray, (nobs, neqs)
            Residuals for each equation stacked in column.
        '''
        s = np.dot(resids.T, resids)
        if self.dfk is None:
            return s / self.nobs
        elif self.dfk == 'dfk1':
            return s / self._div_dfk1
        else:
            return s / self._div_dfk2

class SysGLS(SysModel):
    '''
    Parameters
    ----------
    sys : list of dict
        cf. SysModel
    sigma : scalar or array
        `sigma` the contemporaneous covariance matrix.
        The default is None for no scaling (<=> OLS).  If `sigma` is a scalar, it is
        assumed that `sigma` is an (neqs, neqs) diagonal matrix with the given
        scalar, `sigma` as the value of each diagonal element.  If `sigma`
        is an neqs-length vector, then `sigma` is assumed to be a diagonal
        matrix with the given `sigma` on the diagonal (<=> WLS).
    dfk : None, 'dfk1', or 'dfk2'
        Default is None.  Correction for the degrees of freedom
        should be specified for small samples.  See the notes for more
        information.
    restrict_matrix : ndarray, (M, sum(K_i))
        The restriction matrix on parameters. M represents the number of linear
        constraints on parameters. See Notes.
    restrict_vect : ndarray, (M, 1)
        The RHS restriction vector. See Notes.

    Attributes
    ----------
    cholsigmainv : ndarray
        The transpose of the Cholesky decomposition of the pseudoinverse of
        the contemporaneous covariance matrix.
    wendog : ndarray, (neqs*nobs, 1)
        Endogenous variables whitened by cholsigmainv and stacked into a single
        column.
    wexog : ndarray
        Whitened exogenous variables sp_exog.
    pinv_wexog : ndarray
        Moore-Penrose pseudoinverse of `wexog`.
    normalized_cov_params : array

    Notes
    -----
    Linear restrictions on parameters are specified with the following equation:
        restrict_matrix * params = restrict_vect
    '''

    def __init__(self, sys, sigma=None, restrict_matrix=None, restrict_vect=None):
        super(SysGLS, self).__init__(sys)

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

        ## Handle restrictions
        self.isrestricted = not (restrict_matrix is None and restrict_vect is None)
        if self.isrestricted:
            if not(restrict_vect.shape[0] == restrict_matrix.shape[0]):
                raise ValueError('restrict_vect and restrict_matrix must have the \
                    same number of rows')
            self.nconstraints = restrict_vect.shape[0]
            if self.nconstraints >= self.k_exog_all:
                raise ValueError('total number of regressors must be greater than \
                    the number of constraints')
            self.restrict_matrix = restrict_matrix
            self.restrict_vect = restrict_vect

        self.initialize()

    def initialize(self):
        self.cholsigmainv = np.linalg.cholesky(np.linalg.pinv(self.sigma)).T
        self.wexog = self.whiten(self.sp_exog)
        self.wendog = self.whiten(self.endog.T.reshape(-1,1))
        self.pinv_wexog = np.linalg.pinv(self.wexog)
        
        if self.isrestricted:
            rwendog = np.zeros((self.k_exog_all + self.nconstraints,))
            rwendog[:self.k_exog_all] = np.squeeze(np.dot(self.wexog.T,self.wendog))
            rwendog[self.k_exog_all:] = self.restrict_vect
            self.rwendog = rwendog

            rwexog = np.zeros((self.k_exog_all + self.nconstraints,
                self.k_exog_all + self.nconstraints))
            rwexog[:self.k_exog_all, :self.k_exog_all] = np.dot(self.wexog.T, 
                    self.wexog)
            rwexog[:self.k_exog_all, self.k_exog_all:] = self.restrict_matrix.T
            rwexog[self.k_exog_all:, :self.k_exog_all] = self.restrict_matrix
            rwexog[self.k_exog_all:, self.k_exog_all:] = np.zeros((self.nconstraints,
                self.nconstraints))
            self.rwexog = rwexog

            pinv_rwexog = np.linalg.pinv(rwexog)
            self.pinv_rwexog = pinv_rwexog
    
    def whiten(self, X):
        '''
        SysGLS whiten method

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
        '''
        Notes
        -----
        This is a naive implementation that does not exploit the block
        diagonal structure. See [1] for better algorithms.
        [1] http://www.irisa.fr/aladin/wg-statlin/WORKSHOPS/RENNES02/SLIDES/Foschi.pdf
        '''
        if self.isrestricted:
            beta_lambda = np.dot(self.pinv_rwexog, self.rwendog)
            params = beta_lambda[:self.k_exog_all]
            normalized_cov_params = self.pinv_rwexog[:self.k_exog_all, :self.k_exog_all]
        else:
            params = np.squeeze(np.dot(self.pinv_wexog, self.wendog))
            normalized_cov_params = np.dot(self.pinv_wexog, self.pinv_wexog.T)
        
        return (params, normalized_cov_params)

    def fit(self, iterative=False, tol=1e-5, maxiter=100):
        """
        Full fit of the model.
        
        Parameters
        ----------
        iterative : bool
            If True the estimation procedure is iterated.
        tol : float
            Convergence threshold.
        maxiter : int
            Maximum number of iteration.

        Return
        ------
        A SysResults class instance.
        """
        res = self._compute_res()
        if not iterative:
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
 
class SysWLS(SysGLS):
    '''
    Parameters
    ----------
    weights : 1d array or scalar, None by default
        Variances of each equation. If weights is a scalar then homoscedasticity
        is assumed. Default is None and uses a feasible WLS.
    '''
    def __init__(self, sys, weights=None, dfk=None, restrict_matrix=None,
            restrict_vect=None):
        super(SysWLS, self).__init__(sys, sigma=None, restrict_matrix=
                restrict_matrix, restrict_vect=restrict_vect)

        self.dfk = dfk
        
        if weights is None:
            # Compute sigma by OLS equation by equation
            resids = []
            for eq in sys:
                res = OLS(eq['endog'], eq['exog']).fit()
                resids.append(res.resid)
            resids = np.column_stack(resids)
            sigma = self._compute_sigma(resids)
        else:
            weights = np.asarray(weights)
            # weights = scalar
            if weights.shape == ():
                sigma = np.diag(np.ones(self.neqs)*weights)
            # weights = 1d vector
            elif weights.ndim == 1 and weights.size == self.neqs:
                sigma = np.diag(weights)
            else:
                raise ValueError("weights is not correctly specified")

        self.sigma = sigma
        self.initialize()

    def _compute_sigma(self, resids):
        '''
        Parameters
        ----------
        resids : ndarray, (nobs, neqs)
            Residuals for each equation stacked in column.
        '''
        s = np.diag(np.diag(np.dot(resids.T, resids)))
        if self.dfk is None:
            return s / self.nobs
        elif self.dfk == 'dfk1':
            return s / self._div_dfk1
        else:
            return s / self._div_dfk2

class SysOLS(SysWLS):
    def __init__(self, sys, dfk=None, restrict_matrix=None, restrict_vect=None):
        super(SysOLS, self).__init__(sys, weights=1.0, dfk=dfk,
                restrict_matrix=restrict_matrix, restrict_vect=restrict_vect)

class SysSUR(SysGLS):
    def __init__(self, sys, dfk=None, restrict_matrix=None, restrict_vect=None):
        super(SysSUR, self).__init__(sys, sigma=None, 
                restrict_matrix=restrict_matrix, restrict_vect=restrict_vect)

        self.dfk = dfk

        # Compute sigma by OLS equation by equation
        resids = []
        for eq in sys:
            res = OLS(eq['endog'], eq['exog']).fit()
            resids.append(res.resid)
        resids = np.column_stack(resids)
        sigma = self._compute_sigma(resids)

        self.sigma = sigma
        self.initialize()
 
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
    '''
    Attributes
    ----------
    cov_resids_est : ndarray
        Residual covariance matrix used for estimation.
    cov_resids : ndarray
        Estimated residual covariance matrix with final residuals.
    '''
    def __init__(self, model, params, normalized_cov_params=None, scale=1.):
        super(SysResults, self).__init__(model, params, normalized_cov_params, 
                scale)
        self.cov_resids_est = model.sigma
        # Compute sigma with final residuals
        self.fittedvalues = model.predict(params=params, exog=None)
        self.resids = model.endog - self.fittedvalues.reshape(model.neqs,-1).T
        self.cov_resids = self._compute_sigma(self.resids)

        self.nobs = model.nobs
        self.df_resid = model.df_resid
        self.df_model = model.df_model
        self.ssr = np.array([sum(self.resids[:, eq] ** 2) 
            for eq in range(self.model.neqs)])
        self.mse = self.ssr / self.df_resid
        self.rmse = np.sqrt(self.mse)

        means = np.mean(model.endog, axis=0)
        self.sst = np.array([sum((model.endog[:,eq] - means[eq])**2)
            for eq in range(self.model.neqs)])
        self.rsquared = 1 - self.ssr / self.sst
        self.rsquared_adj = 1 - (1 - self.rsquared)*(float(self.model.nobs - 1) /
                self.model.df_resid)
        if hasattr(model, 'iterations'):
            self.iterations = model.iterations

    def summary(self, yname=None, xname=None, title=None):
        #TODO: handle variable names in SysModel
        return SysSummary(self, yname=yname, xname=xname, title=title)
    
    def _compute_sigma(self, resids):
        '''
        Parameters
        ----------
        resids : ndarray, (nobs, neqs)
            Residuals for each equation stacked in column.
        '''
        s = np.dot(resids.T, resids)
        if self.model.dfk is None:
            return s / self.model.nobs
        elif self.model.dfk == 'dfk1':
            return s / self.model._div_dfk1
        else:
            return s / self.model._div_dfk2

class SysSummary(object):
    default_fmt = dict(
        data_fmts = ["%#15.6F","%#15.6F","%#15.3F","%#14.3F"],
        empty_cell = '',
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

    part1_fmt = dict(default_fmt,
        data_fmts = ["%s"],
        colwidths = 15,
        colsep=' ',
        table_dec_below='',
        header_dec_below=None,
    )
    part2_fmt = dict(default_fmt,
        data_fmts = ["%d","%#12.6g","%#12.6g","%#16.6g","%#12.6g", "%#12.6g"],
    )

    def __init__(self, result, yname=None, xname=None, title=None):
        self.result = result
        if yname is None:
            yname = ['y%d' % (eq,) for eq in range(self.result.model.neqs)]
        if xname is None:
            xname = []
            for eq in range(self.result.model.neqs):
                name_eq = ['x%d_%d' % (eq, var)
                            for var in range(self.result.model.df_model[eq]+1)]
                xname += name_eq
        self.yname = yname
        self.xname = xname
        self.summary = self.make()

    def __repr__(self):
        return self.summary

    def make(self):
        """
        Summary of system model
        """
        buf = StringIO()
        
        print >> buf, self._header_table()
        print >> buf, self._stats_table()
        print >> buf, self._coef_table()
        print >> buf, self._resid_info()

        return buf.getvalue()

    def _header_table(self):
        import time
        t = time.localtime()

        result = self.result

        # Header information
        part1title = "Summary of System Regression Results"
        part1data = [[result.model.__class__.__name__[3:]],
                     [time.strftime("%a, %d, %b, %Y", t)],
                     [time.strftime("%H:%M:%S", t)],
                     [str(result.model.nobs)], 
                     [str(result.model.neqs)]]
        part1header = None
        part1stubs = ('Model:','Date:','Time:', '#observations:', '#equations:')
        part1 = SimpleTable(part1data, part1header, part1stubs,
                            title=part1title, txt_fmt=self.part1_fmt)
        buf = StringIO()
        print >> buf, str(part1)
        buf.write('\n')
        
        return buf.getvalue()

    def _stats_table(self):
        result = self.result
        neqs = result.model.neqs

        data = np.column_stack((result.df_resid, result.ssr, result.mse,
            result.rmse, result.rsquared, result.rsquared_adj))
        header = ('DoF', 'SSR', 'MSE', 'RMSE', 'R^2', 'Adj. R^2')

        buf = StringIO()
        table = SimpleTable(data, header, stubs=self.yname, 
                txt_fmt=self.part2_fmt)
        print >> buf, str(table)

        return buf.getvalue()

    def _coef_table(self):
        result = self.result
        neqs = result.model.neqs

        data = np.column_stack((result.params, result.bse, 
                                result.tvalues, result.pvalues,
                                result.conf_int()))
        header = ('coefficient','std. error','t-stat','p-value', 'conf int inf',
                  'conf int sup')

        buf = StringIO()
        start_index = 0
        for eq in range(neqs):
            offset = result.model.df_model[eq] + 1
            title = 'Results for equation %s' % (self.yname[eq],)
            table = SimpleTable(data[start_index:start_index+offset,:], header,
                                title=title, txt_fmt=self.default_fmt,
                                stubs=self.xname[start_index:start_index+offset])
            print >> buf, str(table)
            buf.write('\n')
            start_index += offset

        return buf.getvalue()

    def _resid_info(self):
        buf = StringIO()
        names = self.yname

        print >> buf, "Covariance matrix of residuals"
        from statsmodels.tsa.vector_ar.output import pprint_matrix
        print >> buf, pprint_matrix(self.result.cov_resids, names, names)

        return buf.getvalue()


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
    mod = SysSUR(sys)
    res = mod.fit()

    R = np.asarray([[0,1,2,0,0,1,3,0,0],[0,1,0,3,1,2,0,1,0]])
    q = np.asarray([0,1]) 
    modr = SysSUR(sys, restrict_matrix=R, restrict_vect=q)
    resr = modr.fit()

