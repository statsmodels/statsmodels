import numpy as np
from scipy import stats
from statsmodels.tools.decorators import cache_readonly
import statsmodels.base.model as base
from statsmodels.genmod.families import Family
from statsmodels.genmod.dependence_structures import VarStruct
import pandas
from patsy import dmatrices


#TODO should this inherit from something?
#TODO multinomial responses
class GEE:
    """Procedures for fitting marginal regression models to dependent data
    using Generalized Estimating Equations.

    References
    ----------
    KY Liang and S Zeger. "Longitudinal data analysis using generalized linear
    models". Biometrika (1986) 73 (1): 13-22.

    S Zeger and KY Liang. "Longitudinal Data Analysis for Discrete and
    Continuous Outcomes". Biometrics Vol. 42, No. 1 (Mar., 1986), pp. 121-130
    """
    

    def __init__(self, Y, X, Id, T=None, family=None, varstruct=None, ytype="interval"):
        """
        Parameters
        ----------
        Y : array-like
            1d array of endogenous response variable.
        X : array-like
            A nobs x k array where `nobs` is the number of observations and `k`
            is the number of regressors. An interecept is not included by default
            and should be added by the user. See `statsmodels.tools.add_constant`.
        Id : array-like
            A 1d array of length `nobs` containing the cluster labels.
        T : array-like
            1d array of time (or other index) values.
        family : family class instance
            The default is Gaussian.  To specify the binomial distribution
            family = sm.family.Binomial()
            Each family can take a link instance as an argument.  See
            statsmodels.family.family for more information.
        varstruct : VarStruct class instance
            The default is Independence.  To specify an exchangeable structure
            varstruct = sm.varstruct.Exchangeable()
            See statsmodels.varstruct.varstruct for more information.
        ytype : string
           Determines whether the response variable is analyzed as-is (ytype = 
           'interval'), or is recoded as binary indicators (ytype = 'ordinal' or
           'nominal').  Ordinal values are recoded as cumulative indicators I(Y > s),
           where s is one of the unique values of Y.  Nominal values are recoded as
           indicators I(Y = s).  For both ordinal and nominal values, each observed
           value is recoded as |S|-1 indicators, where S is the set of unique values of Y.
           No indicator is created for the greatest value in S.
        
        See also
        --------
        statsmodels.families.*

        Notes
        -----
        Only the following combinations make sense for family and link ::

                       + ident log logit probit cloglog pow opow nbinom loglog logc
          Gaussian     |   x    x                        x
          inv Gaussian |   x    x                        x
          binomial     |   x    x    x     x       x     x    x           x      x
          Poission     |   x    x                        x
          neg binomial |   x    x                        x          x
          gamma        |   x    x                        x

        Not all of these link functions are currently available.

        Endog and exog are references so that if the data they refer to are already
        arrays and these arrays are changed, endog and exog will change.

        """

        # Handle the ytype argument
        if ytype not in ("interval","ordinal","nominal"):
            raise ValueError("GEE: `ytype` must be one of 'interval', 'ordinal', or 'nominal'")

        # Handle the family argument
        if family is None:
            family = families.Gaussian()
        else:
            if not issubclass(family.__class__, Family):
                raise ValueError("GEE: `family` must be a genmod family instance")
        self.family = family
        
        # Handle the varstruct argument
        if varstruct is None:
            varstruct = dependence_structures.Independence()
        else:
            if not issubclass(varstruct.__class__, VarStruct):
                raise ValueError("GEE: `varstruct` must be a genmod varstruct instance")
        self.varstruct = varstruct

        if type(X) == pandas.DataFrame:
            self.xnames = X.columns
        else:
            self.xnames = ["v_%d" % (k+1) for k in range(X.shape[1])]

        if type(Y) == pandas.Series:
            self.yname = Y.name
        else:
            self.yname = "unnamed response"

        # Drop cases with missing data
        #TODO: Does this work for both numpy arrays and Pandas data frames/series?
        ii = pandas.notnull(Y) & pandas.notnull(X).all(1)
        if type(Y) == pandas.Series:
            Y = Y.loc[ii].values
        else:
            Y = Y[ii]
        if type(X) == pandas.DataFrame:
            X = X.loc[ii,:].as_matrix()
        else:
            X = X[ii,:]
        Id = Id[ii]
        if T is not None:
            T = T[ii]
        ixn = np.flatnonzero(ii)

        # Convert the data to the internal representation
        S = np.unique(Id)
        S.sort()
        Y1 = [list() for s in S]
        X1 = [list() for s in S]
        T1 = [list() for s in S]
        IX = [list() for s in S]
        for i in range(len(Y)):
            idx = int(Id[i])
            Y1[idx].append(Y[i])
            X1[idx].append(X[i])
            IX[idx].append(ixn[i])
            if T is not None:
                T1[idx].append(T[i])
        Y = [np.array(y) for y in Y1]
        X = [np.array(x) for x in X1]
        IX = [np.array(x) for x in IX]
        if T1 is not None:
            T = [np.array(t) for t in T1]

        # Save the row indices in the original data (prior to dropping missing and
        # prior to splitting into clusters) that correspond to the rows
        # of each element of Y and X.
        self.IX = IX

        # Need to do additional processing for categorical responses
        if ytype != "interval":
            self.Y_orig = Y
            self.X_orig = X
            Y,X,IY,BTW,nylevel = _setup_multicategorical(Y, X, ytype)
            self.nylevel = nylevel
            self.IY = IY
            self.BTW = BTW

        self.ytype = ytype
        self.Y = Y
        self.X = X
        self.family = family
        self.T = T

        # Some of the variance calculations require data or methods from the gee class.
        if ytype == "interval":
            self.varstruct.initialize(self)
        else:
            self.varstruct.initialize(self, IY, BTW)
            
        # Total sample size
        N = [len(y) for y in self.Y]
        self.nobs = sum(N)



    #TODO: merge with something in base?
    @classmethod
    def from_formula(cls, formula, groups, data, time=None, family=None, 
                     varstruct=None, ytype="interval"):
        """


        formula : string
            The formula for the marginal model
        
        groups : string
            The variable name that defines the group

        data : pandas.DataFrame
            A pandas data frame containing all the variables in formula
            and in groups

        """
        
        Y,X = dmatrices(formula, data, return_type="dataframe")
        Y = Y.iloc[:,0] # Convert to series

        T = None
        if time is not None:
            T = data[time]

        Id = data[groups]
        
        return GEE(Y, X, Id, T, family, varstruct, ytype) 



    def estimate_scale(self, beta):
        """
        Returns an estimate of the scale parameter `phi` at the given value
        of `beta`.
        """

        N = len(self.Y)
        nobs = self.nobs
        p = len(beta)

        mean = self.family.link.inverse
        varfunc = self.family.variance
        Y = self.Y
        X = self.X

        scale_inv,m = 0,0
        for i in range(N):

            if len(Y[i]) == 0:
                continue

            lp = np.dot(X[i], beta)
            E = mean(lp)

            S = np.sqrt(varfunc(E))
            resid = (self.Y[i] - E) / S

            n = len(resid)
            scale_inv += np.sum(resid**2)
            m += 0.5*n*(n-1)

        scale_inv /= (nobs-p)
        scale = 1 / scale_inv
        return scale


        
    def _beta_update(self, beta):
        """
        Returns a vector u based on the current regression
        coefficients beta such that beta + u is the next iterate when
        solving the score equations.
        """
        
        # Number of clusters
        N = len(self.Y)

        X = self.X
        Y = self.Y
        varstruct = self.varstruct
        
        mean = self.family.link.inverse
        mean_deriv = self.family.link.inverse_deriv
        varfunc = self.family.variance

        B,C = 0,0
        for i in range(N):

            if len(Y[i]) == 0:
                continue
            
            lp = np.dot(X[i], beta)
            E = mean(lp)
            Dt = X[i] * mean_deriv(lp)[:,None]
            D = Dt.T

            S = np.sqrt(varfunc(E))
            V,is_cor = self.varstruct.variance_matrix(E, i)
            if is_cor:
                V *= np.outer(S, S)

            VID = np.linalg.solve(V, D.T)
            B += np.dot(D, VID)

            R = Y[i] - E
            VIR = np.linalg.solve(V, R)
            C += np.dot(D, VIR)

        return np.linalg.solve(B, C)



    def _covmat(self, beta):
        """
        Returns the sampling covariance matrix of the regression parameters.
        """

        Y = self.Y
        X = self.X
        N = len(Y)
        
        mean = self.family.link.inverse
        mean_deriv = self.family.link.inverse_deriv
        varfunc = self.family.variance

        B,C = 0,0
        for i in range(N):

            if len(Y[i]) == 0:
                continue
            
            lp = np.dot(X[i], beta)
            E = mean(lp)
            Dt = X[i] * mean_deriv(lp)[:,None]
            D = Dt.T

            S = np.sqrt(varfunc(E))
            V,is_cor = self.varstruct.variance_matrix(E, i)
            if is_cor:
                V *= np.outer(S, S)

            VID = np.linalg.solve(V, D.T)
            B += np.dot(D, VID)

            R = Y[i] - E
            VIR = np.linalg.solve(V, R)
            DVIR = np.dot(D, VIR)
            C += np.outer(DVIR, DVIR)

        BI = np.linalg.inv(B)

        return np.dot(BI, np.dot(C, BI))



    def fit(self, maxit=100, ctol=1e-6, starting_beta=None):
        """
        Fits a GEE model.

        Parameters
        ----------
        maxit : integer
            The maximum number of iterations
        ctol : float
            The convergence criterion for stopping the Gauss-Seidel iterations
        starting_beta : array-like
            A vector of starting values for the regression coefficients


        Returns
        -------
        An instance of the GEEResults class

        """

        Y = self.Y
        X = self.X
        varstruct = self.varstruct
        p = X[0].shape[1]

        if starting_beta is None:

            if self.ytype == "interval":
                xnames1 = []
                beta = np.zeros(p, dtype=np.float64)
            else:
                xnames1 = ["cat_%d" % k for k in range(1,self.nylevel)]
                beta = _categorical_starting_values(self.Y_orig, self.X[0].shape[1],
                                                    self.nylevel, self.ytype)

            xnames1 += self.xnames
            beta = pandas.Series(beta, index=xnames1)

        else:
            if type(beta) == np.ndarray:
                ix = ["v%d" % k for k in range(1,len(beta)+1)]
                beta = pd.Series(starting_beta, index=ix)
            else:
                beta = starting_beta.copy()

        for iter in range(maxit):
            u = self._beta_update(beta)
            beta += u
            if np.sqrt(np.sum(u**2)) < ctol:
                break
            self._update_assoc(beta)
        
        bcov = self._covmat(beta)

        GR = GEEResults(self, beta, bcov)

        return GR


    def _update_assoc(self, beta):
        """
        Update the association parameters
        """

        self.varstruct.update(beta)



class GEEResults:


    def __init__(self, model, params, cov_params):

        self.model = model
        self.params = params
        self.cov_params = cov_params


    @cache_readonly
    def standard_errors(self):
        return np.sqrt(np.diag(self.cov_params))

    @cache_readonly
    def bse(self):
        return np.sqrt(np.diag(self.cov_params))

    @cache_readonly
    def tvalues(self):
        return self.params / self.standard_errors


    @cache_readonly
    def pvalues(self):
        dist = stats.norm
        return 2*dist.cdf(-np.abs(self.tvalues))


    @cache_readonly
    def resid(self):
        R = [y - np.dot(x, self.params) for x,y in zip(self.model.X, self.model.Y)]
        return R

        
    def predict(self, exog=None, link=False):

        if exog is None and link:
            F = [self.model.family.GetE(self.params, x) for x in self.model.X]
        elif exog is None and not link:
            F = [np.dot(x, self.params) for x in self.model.X]
        elif link:
            F = self.model.family.GetE(self.params, exog)
        elif not link:
            F = np.dot(exog, self.params)
            
        return F


    def conf_int(self, alpha=.05, cols=None):
        """
        Returns the confidence interval of the fitted parameters.
        
        Parameters
        ----------
        alpha : float, optional
             The `alpha` level for the confidence interval.
             i.e., The default `alpha` = .05 returns a 95% confidence interval.
        cols : array-like, optional
             `cols` specifies which confidence intervals to return

        Notes
        -----
        The confidence interval is based on the Gaussian distribution.
        """
        bse = self.standard_errors
        params = self.params
        dist = stats.norm
        q = dist.ppf(1 - alpha / 2)

        if cols is None:
            lower = self.params - q * bse
            upper = self.params + q * bse
        else:
            cols = np.asarray(cols)
            lower = params[cols] - q * bse[cols]
            upper = params[cols] + q * bse[cols]
        return np.asarray(zip(lower, upper))
    

    def summary(self, yname=None, xname=None, title=None, alpha=.05):
        """Summarize the Regression Results

        Parameters
        -----------
        yname : string, optional
            Default is `y`
        xname : list of strings, optional
            Default is `var_##` for ## in p the number of regressors
        title : string, optional
            Title for the top table. If not None, then this replaces the
            default title
        alpha : float
            significance level for the confidence intervals

        Returns
        -------
        smry : Summary instance
            this holds the summary tables and text, which can be printed or
            converted to various output formats.

        See Also
        --------
        statsmodels.iolib.summary.Summary : class to hold summary
            results

        """
 
        top_left = [('Dep. Variable:', None),
                    ('Model:', None),
                    ('Method:', ['Generalized Estimating Equations']),
                    ('Family:', [self.model.family.__class__.__name__]),
                    ('Dependence structure:', [self.model.varstruct.__class__.__name__]),
                    ('Response type:', [self.model.ytype.title()]),
                    ('Date:', None),
                    ('Time:', None),
        ]
        
        NY = [len(y) for y in self.model.Y]

        top_right = [('No. Observations:', [sum(NY)]),
                     ('No. clusters:', [len(self.model.Y)]),
                     ('Min. cluster size', [min(NY)]),
                     ('Max. cluster size', [max(NY)]),
                     ('Mean cluster size', ["%.1f" % np.mean(NY)]),
                 ]
        
        # The skew of the residuals
        R = np.concatenate(self.resid)
        skew1 = stats.skew(R)
        kurt1 = stats.kurtosis(R)
        V = [r - r.mean() for r in self.resid]
        V = np.concatenate(V)
        skew2 = stats.skew(V)
        kurt2 = stats.kurtosis(V)

        diagn_left = [('Skew:', ["%12.4f" % skew1]),
                      ('Centered skew:', ["%12.4f" % skew2])]

        diagn_right = [('Kurtosis:', ["%12.4f" % kurt1]),
                       ('Centered kurtosis:', ["%12.4f" % kurt2])
                   ]

        if title is None:
            title = self.model.__class__.__name__ + ' ' + "Regression Results"

        #create summary table instance
        from statsmodels.iolib.summary import Summary
        smry = Summary()
        smry.add_table_2cols(self, gleft=top_left, gright=top_right,
                          yname=self.model.yname, xname=xname, title=title)
        smry.add_table_params(self, yname=yname, xname=self.params.index.tolist(), 
                              alpha=alpha, use_t=True)

        smry.add_table_2cols(self, gleft=diagn_left, gright=diagn_right,
                          yname=yname, xname=xname,
                          title="")

        return smry






def _setup_multicategorical(Y, X, ytype):
    """Restructure nominal or ordinal multicategorical data as binary
    indicators so that they can be analysed using Generalized Estimating
    Equations.

    Nominal data are recoded as indicators.  Each element of Y is
    recoded as the sequence of |S|-1 indicators I(y = S[0]), ..., I(y
    = S[-1]), where S is the sorted list of unique values of Y
    (excluding the maximum value).  Also, the covariate vector is
    expanded by taking the Kronecker product of x with e_j, where e_y
    is the indicator vector with a 1 in position y.

    Ordinal data are recoded as cumulative indicators. Each element y
    of Y is recoded as |S|-1 indicators I(y > S[0]), ..., I(y > S[-1])
    where S is the sorted list of unique values of Y (excluding the
    maximum value).  Also, a vector e_y of |S| values is appended to
    the front of each covariate vector x, where e_y is the indicator
    vector with a 1 in position y.

    Arguments
    ---------
    Y: List
        A list of 1-dimensional NumPy arrays, giving the response
        values for the clusters
    X:  List
        A list of 2-dimensional NumPy arrays, giving the covariate
        data for the clusters.  X[i] should include an intercept
        for nominal data but no intercept should be included for
        ordinal data.
    ytype: string
        Either "ordinal" or "nominal"

    The number of rows of X[i] must equal the length of Y[i], and all
    the X[i] arrays should have the same number of columns.

    Returns:
    --------
    Y1:   Y recoded as described above
    X1:   X recoded as described above
    IY:   a list whose i^th element iy = IY[i] is a sequence of tuples
          (a,b), where Y[i][a:b] is the subvector of indicators derived
          from the same ordinal value 
    BTW   a list whose i^th element btw = BTW[i] is a map from cut-point
          pairs (c,c') to the indices of between-subject pairs derived
          from the given cut points
    """

    if ytype not in ("ordinal", "nominal"):
        raise ValueError("_setup_multicategorical: `ytype` must be either "
                         "'nominal' or 'categorical'") 

    # The unique outcomes
    YV = np.concatenate(Y)
    S = list(set(YV))
    S.sort()
    S = S[0:-1]

    ncut = len(S)

    # nominal=1, ordinal=0
    ytype_i = [0,1][ytype == "nominal"]

    Y1,X1,IY,BTW = [],[],[],[]
    for y,x in zip(Y,X): # Loop over clusters

        y1,x1,iy1 = [],[],[]
        jj = 0
        btw = {}

        for y2,x2 in zip(y,x): # Loop over data points within a cluster
            iy2 = []
            for js,s in enumerate(S):
                if ytype_i == 0:
                    y1.append(int(y2 > s))
                    x3 = np.concatenate((np.zeros(ncut, dtype=np.float64), x2))
                    x3[js] = 1
                else:
                    y1.append(int(y2 == s))
                    xx = np.zeros(ncut, dtype=np.float64)
                    xx[js] = 1
                    x3 = np.kronecker(xx, x3)
                x1.append(x3)
                iy2.append(jj)
                jj += 1
            iy1.append(iy2)
        Y1.append(np.array(y1))
        X1.append(np.array(x1))

        # Get a map from (c,c') tuples (pairs of points in S) to the
        # list of all index pairs corresponding to the tuple.
        btw = {}
        for i1,v1 in enumerate(iy1):
            for v2 in iy1[0:i1]:
                for j1,k1 in enumerate(v1):
                    for j2,k2 in enumerate(v2):
                        ii = [(j1,j2),(j2,j1)][j2<j1]
                        if ii not in btw:
                            btw[ii] = []
                        if j1 < j2:
                            btw[ii].append((k1,k2))
                        else:
                            btw[ii].append((k2,k1))
        for kk in btw.keys():
            btw[kk] = np.array(btw[kk])
        BTW.append(btw)

        # Convert from index list to slice endpoints
        iy1 = [(min(x),max(x)+1) for x in iy1]

        IY.append(iy1)


    return Y1,X1,IY,BTW,len(S)+1


def _categorical_starting_values(Y, q, nylevel, ytype):

    YV = np.concatenate(Y)
    S = list(set(YV))
    S.sort()
    S = S[0:-1]

    if ytype == "ordinal":
        Pr = np.array([np.mean(YV > s) for s in S])
        bl = np.log(Pr/(1-Pr))
        beta = np.concatenate((bl, np.zeros(q-nylevel+1)))
    elif ytype == "nominal":
        beta = np.zeros(X[0].shape[1], dtype=np.float64)

    return beta




