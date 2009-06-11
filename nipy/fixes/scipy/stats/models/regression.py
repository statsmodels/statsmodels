"""
This module implements some standard regression models: OLS and WLS
models, as well as an AR(p) regression model.

Models are specified with a design matrix and are fit using their
'fit' method.

Subclasses that have more complicated covariance matrices
should write over the 'whiten' method as the fit method
prewhitens the response by calling 'whiten'.

General reference for regression models:

'Introduction to Linear Regression Analysis', Douglas C. Montgomery,
    Elizabeth A. Peck, G. Geoffrey Vining. Wiley, 2006.

"""

__docformat__ = 'restructuredtext en'

from string import join as sjoin
from csv import reader

import numpy as np
from scipy.linalg import norm, toeplitz

from nipy.fixes.scipy.stats.models.model import LikelihoodModel, \
     LikelihoodModelResults
from nipy.fixes.scipy.stats.models import utils

from scipy import stats
from scipy.stats.stats import ss

def read_design(desfile, delimiter=',', try_integer=True):
    """
    Return a record array with the design.
    The columns are first cast as numpy.float, if this fails, its
    dtype is unchanged.

    If try_integer is True and a given column can be cast as float,
    it is then tested to see if it can be cast as numpy.int.

    >>> design = [["id","age","gender"],[1,23.5,"male"],[2,24.1,"female"],[3,24.5,"male"]]
    >>> read_design(design)
    recarray([(1, 23.5, 'male'), (2, 24.100000000000001, 'female'),
    (3, 24.5, 'male')],
    dtype=[('id', '<i4'), ('age', '<f8'), ('gender', '|S6')])
    >>> design = [["id","age","gender"],[1,23.5,"male"],[2,24.1,"female"],[3.,24.5,"male"]]
    >>> read_design(design)
    recarray([(1, 23.5, 'male'), (2, 24.100000000000001, 'female'),
    (3, 24.5, 'male')],
    dtype=[('id', '<i4'), ('age', '<f8'), ('gender', '|S6')])
    >>> read_design(design, try_integer=False)
    recarray([(1.0, 23.5, 'male'), (2.0, 24.100000000000001, 'female'),
    (3.0, 24.5, 'male')],
    dtype=[('id', '<f8'), ('age', '<f8'), ('gender', '|S6')])
    >>>

    """

    if type(desfile) == type("string"):
        desfile = file(desfile)
        _reader = reader(desfile, delimiter=delimiter)
    else:
        _reader = iter(desfile)
    colnames = _reader.next()

    predesign = np.rec.fromrecords([row for row in _reader], names=colnames)

    # Try to typecast each column to float, then int

    dtypes = predesign.dtype.descr
    newdescr = []
    newdata = []
    for name, descr in dtypes:
        x = predesign[name]
        try:
            y = np.asarray(x.copy(), np.float) # cast as float
            if np.alltrue(np.equal(x, y)):
                if try_integer:
                    z = y.astype(np.int) # cast as int
                    if np.alltrue(np.equal(y, z)):
                        newdata.append(z)
                        newdescr.append(z.dtype.descr[0][1])
                    else:
                        newdata.append(y)
                        newdescr.append(y.dtype.descr[0][1])
                else:
                    newdata.append(y)
                    newdescr.append(y.dtype.descr[0][1])
        except:
            newdata.append(x)
            newdescr.append(descr)

    return np.rec.fromarrays(newdata, formats=sjoin(newdescr, ','), names=colnames)

#How to document a class?
#Docs are a little vague and there are no good examples
#Some of these attributes are most likely intended to be private I imagine
class OLSModel(LikelihoodModel):
    """
    A simple ordinary least squares model.

    Parameters
    ----------
        `design`: array-like
            This is your design matrix.  Data are assumed to be column ordered
            with observations in rows.
        `hascons`: boolean
            A  whether or not your data already contains a constant.
            By default hascons = True (this is so not to break the current
            behavior with the Formula framework).

    Methods
    -------
    model.logL(b=self.beta, Y)
        Returns the log-likelihood of the parameter estimates

        Parameters
        ----------
        b : array-like
            `b` is an array of parameter estimates the log-likelihood of which
            is to be tested.
        Y : array-like
            `Y` is the vector of dependent variables.
    model.__init___(design, hascons=True)
        Creates a `OLSModel` from a design.

    Attributes
    ----------
    design : ndarray
        This is the design, or X, matrix.
    wdesign : ndarray
        This is the whitened design matrix.
        design = wdesign by default for the OLSModel, though models that
        inherit from the OLSModel will whiten the design.
    calc_beta : ndarray
        This is the Moore-Penrose pseudoinverse of the whitened design matrix.
    normalized_cov_beta : ndarray
        np.dot(calc_beta, calc_beta.T)
    df_resid : integer
        Degrees of freedom of the residuals.
        Number of observations less the rank of the design.
    df_model : integer
        Degres of freedome of the model.
        The rank of the design.

    Examples
    --------
    >>> import numpy as N
    >>>
    >>> from nipy.fixes.scipy.stats.models.formula import Term, I
    >>> from nipy.fixes.scipy.stats.models.regression import OLSModel
    >>>
    >>> data={'Y':[1,3,4,5,2,3,4],
    ...       'X':range(1,8)}
    >>> f = term("X") + I
    >>> f.namespace = data
    >>>
    >>> model = OLSModel(f.design())
    >>> results = model.fit(data['Y'])
    >>>
    >>> results.beta
    array([ 0.25      ,  2.14285714])
    >>> results.t()
    array([ 0.98019606,  1.87867287])
    >>> print results.Tcontrast([0,1])
    <T contrast: effect=2.14285714286, sd=1.14062281591, t=1.87867287326, df_denom=5>
    >>> print results.Fcontrast(np.identity(2))
    <F contrast: F=19.4607843137, df_denom=5, df_num=2>
    """

    def __init__(self, design, hascons=True):
        super(OLSModel, self).__init__()
        self.initialize(design, hascons)

    def initialize(self, design, hascons):
# TODO: handle case for noconstant regression
        if hascons==True:
            self.design = design
        else:
            self.design = np.hstack((np.ones((design.shape[0], 1)), design))
        self.wdesign = self.whiten(self.design)
        self.calc_beta = np.linalg.pinv(self.wdesign)
        self.normalized_cov_beta = np.dot(self.calc_beta,
                                         np.transpose(self.calc_beta))
        self.df_resid = self.wdesign.shape[0] - utils.rank(self.design)
#       Below assumes that we will always have a constant for now
        self.df_model = utils.rank(self.design)-1

    def logL(self, b, Y):
        '''
        Returns the value of the loglikelihood function at b.

        Given the whitened design matrix, the loglikelihood is evaluated
        at the parameter vector `b` for the dependent variable `Y`.

        Parameters
        ----------
        `b` : array-like
            The parameter estimates.  Must be of length df_model.
        `Y` : ndarray
            The dependent variable.

        Returns
        -------
        The value of the loglikelihood function for an OLS Model.

        .. math:: \ell(\boldsymbol{y},\hat{\beta},\hat{\sigma})=
        -\frac{n}{2}(1+\log2\pi-\log n)-\frac{n}{2}\log\text{SSR}(\hat{\beta})
        '''
        n = self.wdesign.shape[0]
        return -n/2.*(1 + np.log(2*np.pi) - np.log*n) - \
                n/2.*np.log(ss(whiten(Y)-np.dot(self.wdesign,b)))

#   Note: why have a function that doesn't do anything? does it have to be here to be
#   overwritten?
#   Could this be replaced with the sandwich estimators
#   without writing a subclass?


    def whiten(self, Y):
        """
        OLS model whitener does nothing: returns Y.
        """
        return Y

#   Fixed, needed to return lfit
#   Is this lightweight function needed?
    def est_coef(self, Y):
        """
        Estimate coefficients using lstsq, returning fitted values, Y
        and coefficients, but initialize is not called so no
        psuedo-inverse is calculated.
        """
        Z = self.whiten(Y)

        lfit = RegressionResults(np.linalg.lstsq(self.wdesign, Z)[0], Y)
        lfit.predict = np.dot(self.design, lfit.beta)

        return lfit

    def fit(self, Y):
        """
        Full fit of the model including estimate of covariance matrix,
        (whitened) residuals and scale.

        """
        Z = self.whiten(Y)

        lfit = RegressionResults(np.dot(self.calc_beta, Z), Y,
                       normalized_cov_beta=self.normalized_cov_beta)
        lfit.predict = np.dot(self.design, lfit.beta)
        lfit.resid = Z - np.dot(self.wdesign, lfit.beta)
        lfit.scale = np.add.reduce(lfit.resid**2) / self.df_resid
        lfit.df_resid = self.df_resid
        lfit.df_model = self.df_model

        lfit.Z = Z
# presumably these will be reused somewhere, so this might not be the right place for them

        lfit.ESS = ss(lfit.predict - lfit.Z.mean())
        lfit.uTSS = ss(lfit.Z)
        lfit.uSSR = ss(lfit.resid)
        lfit.cTSS = ss(lfit.Z-lfit.Z.mean())
        lfit.SSR = ss(lfit.resid)

# Centered R2 for models with intercepts (as R does)
#        if hascons = True
        lfit.Rsq = 1 - lfit.SSR/lfit.cTSS                       # tested
#        else:
# Uncentered R2 for models without intercepts.
#        self.Rsq = 1 - self.SSR/self.uTSS
# R2 is uncentered like this, consider centered R2
        lfit.adjrsq = None
        lfit.MSE_model = lfit.ESS/lfit.df_model                 # tested
        lfit.MSE_resid = lfit.uSSR/lfit.df_resid                # tested
        lfit.MSE_total = lfit.uTSS/(lfit.df_model+lfit.df_resid)
        lfit.F = lfit.MSE_model/lfit.MSE_resid                  # tested
        lfit.F_p = stats.f.pdf(lfit.F, lfit.df_model, lfit.df_resid)
        lfit.bse = np.diag(np.sqrt(lfit.cov_beta()))
        return lfit

class ARModel(OLSModel):
    """
    A regression model with an AR(p) covariance structure.

    The linear autoregressive process of order p--AR(p)--is defined as:
        TODO

    Examples
    --------
    >>> import numpy as N
    >>> import numpy.random as R
    >>>
    >>> from nipy.fixes.scipy.stats.models.formula import Term, I
    >>> from nipy.fixes.scipy.stats.models.regression import ARModel
    >>>
    >>> data={'Y':[1,3,4,5,8,10,9],
    ...       'X':range(1,8)}
    >>> f = term("X") + I
    >>> f.namespace = data
    >>>
    >>> model = ARModel(f.design(), 2)
    >>> for i in range(6):
    ...     results = model.fit(data['Y'])
    ...     print "AR coefficients:", model.rho
    ...     rho, sigma = model.yule_walker(data["Y"] - results.predict)
    ...     model = ARModel(model.design, rho)
    ...
    AR coefficients: [ 0.  0.]
    AR coefficients: [-0.52571491 -0.84496178]
    AR coefficients: [-0.620642   -0.88654567]
    AR coefficients: [-0.61887622 -0.88137957]
    AR coefficients: [-0.61894058 -0.88152761]
    AR coefficients: [-0.61893842 -0.88152263]
    >>> results.beta
    array([ 1.58747943, -0.56145497])
    >>> results.t()
    array([ 30.796394  ,  -2.66543144])
    >>> print results.Tcontrast([0,1])
    <T contrast: effect=-0.561454972239, sd=0.210643186553, t=-2.66543144085, df_denom=5>
    >>> print results.Fcontrast(np.identity(2))
    <F contrast: F=2762.42812716, df_denom=5, df_num=2>
    >>>
    >>> model.rho = np.array([0,0])
    >>> model.iterative_fit(data['Y'], niter=3)
    >>> print model.rho
    [-0.61887622 -0.88137957]
    """
    def __init__(self, design, rho):
        if type(rho) is type(1):
            self.order = rho
            self.rho = np.zeros(self.order, np.float64)
        else:
            self.rho = np.squeeze(np.asarray(rho))
            if len(self.rho.shape) not in [0,1]:
                raise ValueError, "AR parameters must be a scalar or a vector"
            if self.rho.shape == ():
                self.rho.shape = (1,)
            self.order = self.rho.shape[0]
        super(ARModel, self).__init__(design)

    def iterative_fit(self, Y, niter=3):
        """
        Perform an iterative two-stage procedure to estimate AR(p)
        parameters and regression coefficients simultaneously.

        :Parameters:
            Y : TODO
                TODO
            niter : ``integer``
                the number of iterations
        """
        for i in range(niter):
            self.initialize(self.design)
            results = self.fit(Y)
            self.rho, _ = yule_walker(Y - results.predict,
                                      order=self.order, df=self.df)

    def whiten(self, X):
        """
        Whiten a series of columns according to an AR(p)
        covariance structure.

        :Parameters:
            X : TODO
                TODO
        """
        X = np.asarray(X, np.float64)
        _X = X.copy()
        for i in range(self.order):
            _X[(i+1):] = _X[(i+1):] - self.rho[i] * X[0:-(i+1)]
        return _X


def yule_walker(X, order=1, method="unbiased", df=None, inv=False):
    """
    Estimate AR(p) parameters from a sequence X using Yule-Walker equation.

    unbiased or maximum-likelihood estimator (mle)

    See, for example:

    http://en.wikipedia.org/wiki/Autoregressive_moving_average_model

    :Parameters:
        X : a 1d ndarray
        method : ``string``
               Method can be "unbiased" or "mle" and this determines
               denominator in estimate of autocorrelation function (ACF)
               at lag k. If "mle", the denominator is n=r.shape[0], if
               "unbiased" the denominator is n-k.
        df : ``integer``
               Specifies the degrees of freedom. If df is supplied,
               then it is assumed the X has df degrees of
               freedom rather than n.
    """

    method = str(method).lower()
    if method not in ["unbiased", "mle"]:
        raise ValueError, "ACF estimation method must be 'unbiased' \
        or 'MLE'"
    X = np.asarray(X, np.float64)
    X -= X.mean()
    n = df or X.shape[0]

    if method == "unbiased":
        denom = lambda k: n - k
    else:
        denom = lambda k: n

    if len(X.shape) != 1:
        raise ValueError, "expecting a vector to estimate AR parameters"
    r = np.zeros(order+1, np.float64)
    r[0] = (X**2).sum() / denom(0)
    for k in range(1,order+1):
        r[k] = (X[0:-k]*X[k:]).sum() / denom(k)
    R = toeplitz(r[:-1])

    rho = np.linalg.solve(R, r[1:])
    sigmasq = r[0] - (r[1:]*rho).sum()
    if inv == True:
        return rho, np.sqrt(sigmasq), np.linalg.inv(R)
    else:
        return rho, np.sqrt(sigmasq)

class WLSModel(OLSModel):
    """
    A regression model with diagonal but non-identity covariance
    structure. The weights are presumed to be
    (proportional to the) inverse of the
    variance of the observations.

    >>> import numpy as N
    >>>
    >>> from nipy.fixes.scipy.stats.models.formula import Term, I
    >>> from nipy.fixes.scipy.stats.models.regression import WLSModel
    >>>
    >>> data={'Y':[1,3,4,5,2,3,4],
    ...       'X':range(1,8)}
    >>> f = term("X") + I
    >>> f.namespace = data
    >>>
    >>> model = WLSModel(f.design(), weights=range(1,8))
    >>> results = model.fit(data['Y'])
    >>>
    >>> results.beta
    array([ 0.0952381 ,  2.91666667])
    >>> results.t()
    array([ 0.35684428,  2.0652652 ])
    >>> print results.Tcontrast([0,1])
    <T contrast: effect=2.91666666667, sd=1.41224801095, t=2.06526519708, df_denom=5>
    >>> print results.Fcontrast(np.identity(2))
    <F contrast: F=26.9986072423, df_denom=5, df_num=2>
    """
    def __init__(self, design, weights=1):
        weights = np.array(weights)
        if weights.shape == (): # scalar
            self.weights = weights
        else:
            design_rows = design.shape[0]
            if not(weights.shape[0] == design_rows and
                   weights.size == design_rows) :
                raise ValueError(
                    'Weights must be scalar or same length as design')
            self.weights = weights.reshape(design_rows)
        super(WLSModel, self).__init__(design)

    def whiten(self, X):
        """
        Whitener for WLS model, multiplies by sqrt(self.weights)
        """
        X = np.asarray(X, np.float64)

        if X.ndim == 1:
            return X * np.sqrt(self.weights)
        elif X.ndim == 2:
            c = np.sqrt(self.weights)
            v = np.zeros(X.shape, np.float64)
            for i in range(X.shape[1]):
                v[:,i] = X[:,i] * c
            return v

class RegressionResults(LikelihoodModelResults):
    """
    This class summarizes the fit of a linear regression model.

    It handles the output of contrasts, estimates of covariance, etc.
    """
# the init should contain all results needed in the other methods here
# and the expected "results" from running a fit
    def __init__(self, beta, Y, normalized_cov_beta=None, scale=1.):
        super(RegressionResults, self).__init__(beta,
                                                 normalized_cov_beta,
                                                 scale)
#Note: are the supers absolutely necessary? ping the tutors list?
        self.Y = Y


    def norm_resid(self):
        """
        Residuals, normalized to have unit length.

        Note: residuals are whitened residuals.

        Notes
        -----
        Is this supposed to return "stanardized residuals," residuals standardized
        to have mean zero and approximately unit variance?

        d_i = e_i/sqrt(MS_E)

        Where MS_E = SSE/(n - k)

        See: Montgomery and Peck 3.2.1 p. 68
             Davidson and MacKinnon 15.2 p 662

        """
        if not hasattr(self, 'resid'):
            raise ValueError, 'need normalized residuals to estimate standard deviation'

#        sdd = utils.recipr(self.sd) / np.sqrt(self.df)
#        return  self.resid * np.multiply.outer(np.ones(self.Y.shape[0]), sdd)
        return self.resid * utils.recipr(np.sqrt(self.scale))

# predict is a verb
# do the predicted values need to be done automatically, then?
# or should you give a predict method similar to STATA
    def predictors(self, design):
        """
        Return linear predictor values from a design matrix.
        """
        return np.dot(design, self.beta)

#    def Rsq(self, adjusted=False):
#        """
#        Return the R^2 value for each row of the response Y.
#
#        Notes
#        -----
#        Changed to the textbook definition of R^2.
#
#        See: Davidson and MacKinnon p 74
#        """
#        self.Ssq = np.std(self.Z,axis=0)**2
#        ratio = self.scale / self.Ssq
#        if not adjusted: ratio *= ((self.Y.shape[0] - 1) / self.df_resid)
#        return 1 - ratio
#        return 1 - np.add.reduce(self.resid**2)/np.add.reduce((self.Z-self.Z.mean())**2)


class GLSModel(OLSModel):

    """
    Generalized least squares model with a general covariance structure

    This should probably go into nipy.fixes.scipy.stats.models.regression

    """

    def __init__(self, design, sigma):
        self.cholsigmainv = np.linalg.cholesky(np.linalg.pinv(sigma)).T
        super(GLSModel, self).__init__(design)

    def whiten(self, Y):
        return np.dot(self.cholsigmainv, Y)


def isestimable(C, D):
    """
    From an q x p contrast matrix C and an n x p design matrix D, checks
    if the contrast C is estimable by looking at the rank of vstack([C,D]) and
    verifying it is the same as the rank of D.

    """
    if C.ndim == 1:
        C.shape = (C.shape[0], 1)
    new = np.vstack([C, D])
    if utils.rank(new) != utils.rank(D):
        return False
    return True
