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

import numpy.lib.recfunctions as nprf

def categorical(data, _column=None, _time=None):
    '''
    Returns an array changing categorical variables to dummy variables.

    Take a structured or record array and returns an array with categorical variables.

    Notes
    -----
    This returns a dummy variable for EVERY distinct string.  If noconsant
    then this is okay.  Otherwise, a "intercept" needs to be designated in regression.

    Returns the same array as it's given right now, consider returning a structured
    and plain ndarray (with names stripped, etc.)

    Where should categoricals home be?
    '''
    if _column==None:
    # this is the general case for when it's called and you just want to convert strings
    # which represent NOMINAL DATA only to dummies
        if not data.dtype.names and not data.mask.any():
# TODO: clean this up
            print data.dtype
            raise "There is not a categorical variable?"
            return data
        #if data.mask.any():            # this causes an error, and maybe should be a try?
        #    print "Masked arrays are not handled yet."
        #    return data

        elif data.dtype.names:  # this will catch both structured and record
                                # arrays, no other array could have string data!
                                # not sure about masked arrays yet
            for i in range(len(data.dtype)):
                if data.dtype[i].type is np.string_:
                    tmp_arr = np.unique(data[data.dtype.names[i]])
# changing the above line, makes it so strip doesn't work
                    tmp_dummy = (tmp_arr[:,np.newaxis]==data.field(i)).astype(float)
    # tmp_dummy is a number of dummies x number of observations array
                    data=nprf.drop_fields(data,data.dtype.names[i],usemask=False,
                                    asrecarray=True)
                    data=nprf.append_fields(data,tmp_arr.strip("\""), data=tmp_dummy,
                                        usemask=False, asrecarray=True)
            return data
    elif column and time:
        tmp_arr=np.unique(data[column])
# not finished



#def read_design(desfile, delimiter=',', try_integer=True):
#    """
#    Return a record array with the design.
#    The columns are first cast as numpy.float, if this fails, its
#    dtype is unchanged.

#    If try_integer is True and a given column can be cast as float,
#    it is then tested to see if it can be cast as numpy.int.

#    >>> design = [["id","age","gender"],[1,23.5,"male"],[2,24.1,"female"],[3,24.5,"male"]]
#    >>> read_design(design)
#    recarray([(1, 23.5, 'male'), (2, 24.100000000000001, 'female'),
#    (3, 24.5, 'male')],
#    dtype=[('id', '<i4'), ('age', '<f8'), ('gender', '|S6')])
#    >>> design = [["id","age","gender"],[1,23.5,"male"],[2,24.1,"female"],[3.,24.5,"male"]]
#    >>> read_design(design)
#    recarray([(1, 23.5, 'male'), (2, 24.100000000000001, 'female'),
#    (3, 24.5, 'male')],
#    dtype=[('id', '<i4'), ('age', '<f8'), ('gender', '|S6')])
#    >>> read_design(design, try_integer=False)
#    recarray([(1.0, 23.5, 'male'), (2.0, 24.100000000000001, 'female'),
#    (3.0, 24.5, 'male')],
#    dtype=[('id', '<f8'), ('age', '<f8'), ('gender', '|S6')])
#    >>>

#    """

#    if type(desfile) == type("string"):
#        desfile = file(desfile)
#        _reader = reader(desfile, delimiter=delimiter)
#    else:
#        _reader = iter(desfile)
#    colnames = _reader.next()

#    predesign = np.rec.fromrecords([row for row in _reader], names=colnames)

    # Try to typecast each column to float, then int

#    dtypes = predesign.dtype.descr
#    newdescr = []
#    newdata = []
   # for name, descr in dtypes:
#        x = predesign[name]
#        try:
#            y = np.asarray(x.copy(), np.float) # cast as float
#            if np.alltrue(np.equal(x, y)):
#                if try_integer:
#                    z = y.astype(np.int) # cast as int
#                    if np.alltrue(np.equal(y, z)):
#                        newdata.append(z)
#                        newdescr.append(z.dtype.descr[0][1])
#                    else:
#                        newdata.append(y)
#                        newdescr.append(y.dtype.descr[0][1])
#                else:
#                    newdata.append(y)
#                    newdescr.append(y.dtype.descr[0][1])
#        except:
#            newdata.append(x)
#            newdescr.append(descr)

#    return np.rec.fromarrays(newdata, formats=sjoin(newdescr, ','), names=colnames)

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

    def __init__(self, design, hascons=True, designtype=None):
# As options become longer will need to use *args or **kwargs
# **kwargs might make more sense if we're going to be mapping
# option to a function call
        super(OLSModel, self).__init__()
        self.initialize(design, hascons)

    def initialize(self, design, hascons=True, designtype=None):
# TODO: handle case for noconstant regression
        if hascons==True:
            self.design = design
        else:
            self.design = np.hstack((np.ones((design.shape[0], 1)), design))
        if designtype==None:
            pass
        elif designtype.lower()=="panel":
            categorical(design)
# TODO: this will handle nominal data
# TODO" need to handle ordinal data as well
        else:
            raise "designtype %s not understood." % datatype
        self.wdesign = self.whiten(self.design)
        self.calc_beta = np.linalg.pinv(self.wdesign)
        self.normalized_cov_beta = np.dot(self.calc_beta,
                                         np.transpose(self.calc_beta))
        self.df_resid = self.wdesign.shape[0] - utils.rank(self.design)
#       Below assumes that we will always have a constant for now
        self.df_model = utils.rank(self.design)-1

    def set_time(self, col):
        '''
        This allows you to set which column has the time variable
        for time fixed effects.
        '''
        self.design = categorical(self.design, col)

# resolve the wdesign vs design issue...consider changing the name
# design as well (!) no one in my program has ever heard of a
# design matrix

    def llf(self, b, Y):
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

        Notes
        -----
        The Likelihood Function is defined as
        .. math:: \ell(\boldsymbol{y},\hat{\beta},\hat{\sigma})=
        -\frac{n}{2}(1+\log2\pi-\log n)-\frac{n}{2}\log\text{SSR}(\hat{\beta})

        The AIC is defined as
        .. math:: \text{AIC}=\log\frac{SSR}{n}+\frac{2K}{n}

        The BIC (or Schwartz Criterion) is defined as
        .. math:: \text{BIC}=\log\frac{SSR}{n}+\frac{K}{n}\log n
        ..

        References
        ----------
        .. [1] W. Green.  "Econometric Analysis," 5th ed., Pearson, 2003.
        '''
        n = self.wdesign.shape[0]
        SSR = ss(Y - np.dot(self.wdesign,b))
        loglf = -n/2.*(1 + np.log(2*np.pi) - np.log(n)) - \
                n/2.*np.log(SSR)
        aic = np.log(SSR/n) + 2 * self.df_model/n
        bic = np.log(SSR/n) + self.df_model * np.log(n)/n
        return loglf,aic,bic

    def whiten(self, Y):
        """
        OLS model whitener does nothing: returns Y.
        """
        return Y

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

    def fit(self, Y, robust=None):
# I think it would also make sense (pracitcally if not from a programming standpoint!)
# to have an instance of OLSModel created when
# OLSModel(x).fit(y) is called.  What's the best way to do this?
        """
        Full fit of the model including estimate of covariance matrix,
        (whitened) residuals and scale.

        Parameters
        ----------
        Y : array-like
            The dependent variable for the Least Squares problem.
        robust : string, optional
            Estimation of the heteroskedasticity robust covariance matrix
            Values can be "HC0", "HC1", "HC2", or "HC3"


NOTE: Should these be defined here?
        Returns (does it actually "return" or is there a better term for this behavior?)
        --------
        adjRsq
            Adjusted R-squared
        AIC
            Akaike information criterion
        BIC
            Bayes information criterion
        bse
            The standard errors of the parameter estimates
        cTSS
            The centered total sum of squares
        df_resid
            Residual degrees of freedom
        df_model
            Model degress of freedom
        ESS
            Explained sum of squares
        F
            F-statistic
        F_p
            F-statistic p-value
        MSE_model
            Mean squared error the model
        MSE_resid
            Mean squared error of the residuals
        MSE_total
            Total mean squared error
        predict
            A postestimation function to predict the values for a given design
            model.predict(design)
        resid
            The residuals of the model.
        Rsq (need to be explicit about constant/noconstant)
            R-squared
*        scale
            A scale factor for the covariance matrix.
            Default value is SSR/(n-k)
            Otherwise, determined by the `robust` keyword
        SSR
            Sum of squared residuals
        uTSS
            Uncentered sum of squares
        Z
            The whitened dependent variable
        """
        Z = self.whiten(Y)
        lfit = RegressionResults(np.dot(self.calc_beta, Z), Y,
                       normalized_cov_beta=self.normalized_cov_beta)
        lfit.predict = np.dot(self.design, lfit.beta)
        lfit.resid = Z - np.dot(self.wdesign, lfit.beta)
        lfit.n = self.wdesign.shape[0]
        lfit.df_resid = self.df_resid
        lfit.df_model = self.df_model
        lfit.Z = Z
        lfit.calc_beta = self.calc_beta # needed for cov_beta()

        if robust is None: # usual estimate of omega (it's a scalar)npp
            lfit.scale = ss(lfit.resid) / self.df_resid
        elif robust=="HC0": # HC0 (White 1980)
            lfit.scale = np.diag(lfit.resid**2)
        elif robust=="HC1": # HC1-3 MacKinnon and White (1985)
            lfit.scale = lfit.n/(lfit.n-lfit.df_model)*(np.diag(lfit.resid**2))
        elif robust=="HC2":
            h=np.diag(np.dot(np.dot(self.wdesign,self.normalized_cov_beta),
                    self.wdesign.T))
            lfit.scale=np.diag(lfit.resid**2/(1-h))
# still unsure about wdesign vs design, would a weighted design be appropriate
# for het robust errors?
        elif robust=="HC3":
             h=np.diag(np.dot(np.dot(self.wdesign,self.normalized_cov_beta),
                    self.wdesign.T))
             lfit.scale=np.diag((lfit.resid/(1-h))**2)
        else:
            raise ValueError, "Robust option %s not understood" % robust

# presumably these will be reused somewhere, so this might not be the right place for them
# or they could just be overwritten by any OLSModel subclass
        lfit.ESS = ss(lfit.predict - lfit.Z.mean())
        lfit.uTSS = ss(lfit.Z)
        lfit.cTSS = ss(lfit.Z-lfit.Z.mean())
        lfit.SSR = ss(lfit.resid)
# Centered R2 for models with intercepts (as R does)
#        if hascons = True
        lfit.Rsq = 1 - lfit.SSR/lfit.cTSS                       # tested
#        else:
# Uncentered R2 for models without intercepts.
#        self.Rsq = 1 - self.SSR/self.uTSS
# R2 is uncentered like this, consider centered R2
# What about adjRsq for no constant regression?
        lfit.adjRsq = 1 - (lfit.n -1)/(lfit.n - lfit.df_model)*(1 - lfit.Rsq)
        lfit.MSE_model = lfit.ESS/lfit.df_model                 # tested
        lfit.MSE_resid = lfit.SSR/lfit.df_resid                # tested
        lfit.MSE_total = lfit.uTSS/(lfit.df_model+lfit.df_resid)
        lfit.F = lfit.MSE_model/lfit.MSE_resid                  # tested
        lfit.F_p = stats.f.pdf(lfit.F, lfit.df_model, lfit.df_resid)
        lfit.bse = np.diag(np.sqrt(lfit.cov_beta()))
        lfit.llf, lfit.aic, lfit.bic = self.llf(lfit.beta, Z)
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
        if not hasattr(secalf, 'resid'):
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
