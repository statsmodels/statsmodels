import numpy as np
from numpy.linalg import inv
#from scipy import optimize

from scipy.stats import t

from nipy.fixes.scipy.stats.models.contrast import ContrastResults
from nipy.fixes.scipy.stats.models.utils import recipr

class Model(object):
    """
    A (predictive) statistical model. The class Model itself does nothing
    but lays out the methods expected of any subclass.
    """

    def __init__(self):
        pass

    def initialize(self):
        """
        Initialize (possibly re-initialize) a Model instance. For
        instance, the design matrix of a linear model may change
        and some things must be recomputed.
        """
        raise NotImplementedError

    def fit(self):
        """
        Fit a model to data.
        """
        raise NotImplementedError

    def predict(self, design=None):
        """
        After a model has been fit, results are (assumed to be) stored
        in self.results, which itself should have a predict method.
        """
        self.results.predict(design)

    def view(self):
        """
        View results of a model.
        """
        raise NotImplementedError

class LikelihoodModel(Model):

    def logL(self, theta):
        """
        Log-likelihood of model.
        """
        raise NotImplementedError

    def score(self, theta):
        """
        Score function of model = gradient of logL with respect to
        theta.
        """
        raise NotImplementedError

    def information(self, theta):
        """
        Score function of model = - Hessian of logL with respect to
        theta.
        """
        raise NotImplementedError

    def newton(self, theta):
        raise NotImplementedError
#         def f(theta):
#             return -self.logL(theta)
#         self.results = optimize.fmin(f, theta)

class LikelihoodModelResults(object):
    ''' Class to contain results from likelihood models '''
    def __init__(self, beta, normalized_cov_beta=None, scale=1.):
        ''' Set up results structure
        beta     - parameter estimates from estimated model
        normalized_cov_beta -
           Normalized (before scaling) covariance of betas
        scale    - scalar

        normalized_cov_betas is also known as the hat matrix or H
        (Semiparametric regression, Ruppert, Wand, Carroll; CUP 2003)

        The covariance of betas is given by scale times
        normalized_cov_beta

        For (some subset of models) scale will typically be the
        mean square error from the estimated model (sigma^2)
        '''
        self.beta = beta
        self.normalized_cov_beta = normalized_cov_beta
        self.scale = scale

    def t(self, column=None):
        """
        Return the t-statistic for a given parameter estimate.

        Use Tcontrast for more complicated t-statistics.

        """

        if self.normalized_cov_beta is None:
            raise ValueError, 'need covariance of parameters for computing T statistics'

        if column is None:
            column = range(self.beta.shape[0])

        column = np.asarray(column)
        _beta = self.beta[column]
        _cov = self.cov_beta(column=column)
        if _cov.ndim == 2:
            _cov = np.diag(_cov)
        _t = _beta * recipr(np.sqrt(_cov))
        return _t

    def cov_beta(self, matrix=None, column=None, scale=None, other=None):
        """
        Returns the variance/covariance matrix of a linear contrast
        of the estimates of beta, multiplied by scale which
        will usually be an estimate of sigma^2.

        The covariance of
        interest is either specified as a (set of) column(s) or a matrix.
        """
        if self.normalized_cov_beta is None:
            raise ValueError, 'need covariance of parameters for computing (unnormalized) covariances'

        if scale is None:
            scale = self.scale

        if column is not None:
            column = np.asarray(column)
            if column.shape == ():
                return self.normalized_cov_beta[column, column] * scale
            else:
                return self.normalized_cov_beta[column][:,column] * scale

        elif matrix is not None:
            if other is None:
                other = matrix
            tmp = np.dot(matrix, np.dot(self.normalized_cov_beta, np.transpose(other)))
            return tmp * scale

        if matrix is None and column is None:
# need to generalize for the case when scale is not a scalar
# and we have robust estimates of \Omega the error covariance matrix
            if scale.size==1:
                scale=np.eye(len(self.resid))*scale
            return np.dot(np.dot(self.calc_beta, scale), self.calc_beta.T)

    def Tcontrast(self, matrix, t=True, sd=True, scale=None):
        """
        Compute a Tcontrast for a row vector matrix. To get the t-statistic
        for a single column, use the 't' method.
        """

        if self.normalized_cov_beta is None:
            raise ValueError, 'need covariance of parameters for computing T statistics'

        _t = _sd = None

        _effect = np.dot(matrix, self.beta)
        if sd:
            _sd = np.sqrt(self.cov_beta(matrix=matrix))
        if t:
            _t = _effect * recipr(_sd)
        return ContrastResults(effect=_effect, t=_t, sd=_sd, df_denom=self.df_resid)

    def Fcontrast(self, matrix, eff=True, t=True, sd=True, scale=None, invcov=None):
        """
        Compute an Fcontrast for a contrast matrix.

        Here, matrix M is assumed to be non-singular. More precisely,

        M pX pX' M'

        is assumed invertible. Here, pX is the generalized inverse of the
        design matrix of the model. There can be problems in non-OLS models
        where the rank of the covariance of the noise is not full.

        See the contrast module to see how to specify contrasts.
        In particular, the matrices from these contrasts will always be
        non-singular in the sense above.

        """

        if self.normalized_cov_beta is None:
            raise ValueError, 'need covariance of parameters for computing F statistics'

        cbeta = np.dot(matrix, self.beta)

        q = matrix.shape[0]
        if invcov is None:
            invcov = inv(self.cov_beta(matrix=matrix, scale=1.0))
        F = np.add.reduce(np.dot(invcov, cbeta) * cbeta, 0) * recipr((q * self.scale))
        return ContrastResults(F=F, df_denom=self.df_resid, df_num=invcov.shape[0])

    def conf_int(self, alpha=.05, cols=None):
        '''
        Returns the confidence interval of the specified beta estimates.

        Parameters
        ----------
        alpha : float, optional
            The `alpha` level for the confidence interval.
            ie., `alpha` = .05 returns a 95% confidence interval.
        cols : tuple, optional
            `cols` specifies which confidence intervals to return

        Returns : array
            Each item contains [lower, upper]

        Example
        -------
        >>>import numpy as np
        >>>from numpy.random import standard_normal as stan
        >>>import nipy.fixes.scipy.stats.models as SSM
        >>>x = np.hstack((stan((30,1)),stan((30,1)),stan((30,1))))
        >>>beta=np.array([3.25, 1.5, 7.0])
        >>>y = np.dot(x,beta) + stan((30))
        >>>model = SSM.regression.OLSModel(x, hascons=False).fit(y)
        >>>model.conf_int(cols=(1,2))

        Notes
        -----
        TODO:
        tails : string, optional
            `tails` can be "two", "upper", or "lower"
        '''
        if cols is None:
            lower = self.beta - t.ppf(1-alpha/2,self.df_resid) *\
                    np.diag(np.sqrt(self.cov_beta()))
            upper = self.beta + t.ppf(1-alpha/2,self.df_resid) *\
                    np.diag(np.sqrt(self.cov_beta()))
        else:
            lower=[]
            upper=[]
            for i in cols:
                lower.append(self.beta[i] - t.ppf(1-alpha/2,self.df_resid) *\
                    np.diag(np.sqrt(self.cov_beta()))[i])
                upper.append(self.beta[i] + t.ppf(1-alpha/2,self.df_resid) *\
                    np.diag(np.sqrt(self.cov_beta()))[i])
        return np.asarray(zip(lower,upper))

