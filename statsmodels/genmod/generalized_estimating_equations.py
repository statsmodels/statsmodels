import numpy as np
from scipy import stats
from statsmodels.tools.decorators import cache_readonly
import statsmodels.base.model as base
from statsmodels.genmod.families import Family
from statsmodels.genmod.dependence_structures import VarStruct
import pandas


#TODO multinomial responses
class GEE(base.Model):
    """Procedures for fitting marginal regression models to dependent data
    using Generalized Estimating Equations.

    References
    ----------
    KY Liang and S Zeger. "Longitudinal data analysis using generalized linear
    models". Biometrika (1986) 73 (1): 13-22.

    S Zeger and KY Liang. "Longitudinal Data Analysis for Discrete and
    Continuous Outcomes". Biometrics Vol. 42, No. 1 (Mar., 1986), pp. 121-130

    Xu Guo and Wei Pan (2002). "Small sample performance of the score test in GEE".
    http://www.sph.umn.edu/faculty1/wp-content/uploads/2012/11/rr2002-013.pdf
    """
    

    def __init__(self, endog, exog, groups, time=None, family=None, 
                       varstruct=None, endog_type="interval", missing='none',
                       offset=None, constraint=None):
        """
        Parameters
        ----------
        endog : array-like
            1d array of endogenous response variable.
        exog : array-like
            A nobs x k array where `nobs` is the number of
            observations and `k` is the number of regressors. An
            interecept is not included by default and should be added
            by the user. See `statsmodels.tools.add_constant`.
        groups : array-like
            A 1d array of length `nobs` containing the cluster labels.
        time : array-like
            A 1d array of time (or other index) values.  This is only
            used if the dependence structure is Autoregressive
        family : family class instance
            The default is Gaussian.  To specify the binomial
            distribution family = sm.family.Binomial() Each family can
            take a link instance as an argument.  See
            statsmodels.family.family for more information.
        varstruct : VarStruct class instance
            The default is Independence.  To specify an exchangeable
            structure varstruct = sm.varstruct.Exchangeable() See
            statsmodels.varstruct.varstruct for more information.
        endog_type : string
           Determines whether the response variable is analyzed as-is
           (endog_type = 'interval'), or is recoded as binary
           indicators (endog_type = 'ordinal' or 'nominal').  Ordinal
           values are recoded as cumulative indicators I(endog > s),
           where s is one of the unique values of endog.  Nominal
           values are recoded as indicators I(endog = s).  For both
           ordinal and nominal values, each observed value is recoded
           as |S|-1 indicators, where S is the set of unique values of
           endog.  No indicator is created for the greatest value in
           S.
        offset : array-like
            An offset to be included in the fit.  If provided, must be an 
            array whose length is the number of rows in exog.
        constraint : (ndarray, ndarray)
           If provided, the constraint is a tuple (L, R) such that the
           model parameters are estimated under the constraint L *
           param = R, where  L is a q x p matrix and R is a q-dimensional
           vector.  If constraint is provided, a score test is
           performed to compare the constrained model to the
           unconstrained model.
        %(extra_params)s

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

        """ % {'extra_params' : base._missing_param_doc}

        # Pass groups and time so they are proceessed for missing data
        # along with endog and exog
        super(GEE, self).__init__(endog, exog, groups=groups, time=time,
                                  offset=offset, missing=missing)

        # Handle the endog_type argument
        if endog_type not in ("interval","ordinal","nominal"):
            raise ValueError("GEE: `endog_type` must be one of 'interval', 'ordinal', or 'nominal'")

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

        # Default offset
        if offset is None:
            offset = np.zeros(exog.shape[0], dtype=np.float64)

        # Default time
        if time is None:
            time = np.zeros(exog.shape[0], dtype=np.float64)

        # Handle the constraint
        self.constraint = None
        if constraint is not None:
            self.constraint = constraint
            if len(constraint) != 2:
                raise ValueError("GEE: `constraint` must be a 2-tuple.")
            L,R = tuple(constraint)
            if type(L) != np.ndarray:
                raise ValueError("GEE: `constraint[0]` must be a NumPy array.")
            print L.shape, exog.shape
            if L.shape[1] != exog.shape[1]:
                raise ValueError("GEE: incompatible constraint dimensions; the number of columns of `constraint[0]` must equal the number of columns of `exog`.") 
            if len(R) != L.shape[0]:
                raise ValueError("GEE: incompatible constraint dimensions; the length of `constraint[1]` must equal the number of rows of `constraint[0]`.")
            
            # The columns of L0 are an orthogonal basis for the
            # orthogonal complement to row(L), the columns of L1
            # are an orthogonal basis for row(L).  The columns of [L0,L1] 
            # are mutually orthogonal.             
            u,s,vt = np.linalg.svd(L.T, full_matrices=1)
            L0 = u[:,len(s):]
            L1 = u[:,0:len(s)]

            # param0 is one solution to the underdetermined system 
            # L * param = R.
            param0 = np.dot(L1, np.dot(vt, R) / s)

            # Reduce exog so that the constraint is satisfied, save
            # the full matrices for score testing.
            offset += np.dot(exog, param0)
            self.exog_preconstraint = exog.copy()
            L = np.hstack((L0,L1))
            exog_lintrans = np.dot(exog, L)
            exog = exog_lintrans[:,0:L0.shape[1]]

            self.constraint = self.constraint + (L0, param0,
                                                 exog_lintrans)

        # Convert to ndarrays
        if type(endog) == pandas.DataFrame:
            endog_nda = endog.iloc[:,0].values
        else:
            endog_nda = endog
        if type(exog) == pandas.DataFrame:
            exog_nda = exog.as_matrix()
        else:
            exog_nda = exog

        # Convert the data to the internal representation, which is a list
        # of arrays, corresponding to the clusters.
        S = np.unique(groups)
        S.sort()
        endog1 = [list() for s in S]
        exog1 = [list() for s in S]
        time1 = [list() for s in S]
        offset1 = [list() for s in S]
        exog_lintrans1 = [list() for s in S]
        row_indices = [list() for s in S]
        for i in range(len(endog_nda)):
            idx = int(groups[i])
            endog1[idx].append(endog_nda[i])
            exog1[idx].append(exog_nda[i,:])
            row_indices[idx].append(i)
            time1[idx].append(time[i])
            offset1[idx].append(offset[i])
            if constraint is not None:
                exog_lintrans1[idx].append(exog_lintrans[i,:])
        endog_li = [np.array(y) for y in endog1]
        exog_li = [np.array(x) for x in exog1]
        row_indices = [np.array(x) for x in row_indices]
        time_li = None
        time_li = [np.array(t) for t in time1]
        offset_li = [np.array(x) for x in offset1]
        if constraint is not None:
            exog_lintrans_li = [np.array(x) for x in exog_lintrans1]

        # Save the row indices in the original data (after dropping
        # missing but prior to splitting into clusters) that
        # correspond to the rows of endog and exog.
        self.row_indices = row_indices

        # Need to do additional processing for categorical responses
        if endog_type != "interval":
            endog_li,exog_li,offset_li,time_li,IY,BTW,nylevel =\
              _setup_multicategorical(endog_li, exog_li, offset_li, time_li, endog_type)
            self.nylevel = nylevel
            self.IY = IY
            self.BTW = BTW

        self.endog_type = endog_type
        self.endog = endog
        self.exog = exog
        self.endog_li = endog_li
        self.exog_li = exog_li
        self.family = family
        self.time = time
        self.time_li = time_li
        if constraint is not None:
            self.exog_lintrans_li = exog_lintrans_li

        self.offset = offset
        self.offset_li = offset_li

        # Some of the variance calculations require data or methods from 
        # the gee class.
        if endog_type == "interval":
            self.varstruct.initialize(self)
        else:
            self.varstruct.initialize(self, IY, BTW)
            
        # Total sample size
        N = [len(y) for y in self.endog_li]
        self.nobs = sum(N)


    def estimate_scale(self):
        """
        Returns an estimate of the scale parameter `phi` at the current
        parameter value.
        """

        endog = self.endog_li
        exog = self.exog_li
        offset = self.offset_li

        cached_means = self._cached_means

        num_clust = len(endog)
        nobs = self.nobs
        p = exog[0].shape[1]

        mean = self.family.link.inverse
        varfunc = self.family.variance

        scale_inv = 0.
        for i in range(num_clust):

            if len(endog[i]) == 0:
                continue

            E,lp = cached_means[i]

            S = np.sqrt(varfunc(E))
            resid = (endog[i] - offset[i] - E) / S

            n = len(resid)
            scale_inv += np.sum(resid**2)

        scale_inv /= (nobs-p)
        scale = 1 / scale_inv
        return scale


        
    def _beta_update(self):
        """
        Returns two values (u, score).  The vector u is the update
        vector such that params + u is the next iterate when solving
        the score equations.  The vector `score` is the current state of
        the score equations.
        """
        
        endog = self.endog_li
        exog = self.exog_li
        varstruct = self.varstruct

        # Number of clusters
        num_clust = len(endog)

        _cached_means = self._cached_means
        
        mean = self.family.link.inverse
        mean_deriv = self.family.link.inverse_deriv
        varfunc = self.family.variance

        B,score = 0,0
        for i in range(num_clust):

            if len(endog[i]) == 0:
                continue
            
            E,lp = _cached_means[i]

            Dt = exog[i] * mean_deriv(lp)[:,None]
            D = Dt.T

            S = np.sqrt(varfunc(E))
            V,is_cor = self.varstruct.variance_matrix(E, i)
            if is_cor:
                V *= np.outer(S, S)

            VID = np.linalg.solve(V, D.T)
            B += np.dot(D, VID)

            R = endog[i] - E
            VIR = np.linalg.solve(V, R)
            score += np.dot(D, VIR)

        u = np.linalg.solve(B, score)

        return u, score


    def _update_cached_means(self, beta):
        """
        _cached_means should always contain the most recent calculation of
        the cluster-wise mean vectors.  This function should be called
        every time the value of beta is changed, to keep the cached
        means up to date.
        """

        endog = self.endog_li
        exog = self.exog_li
        offset = self.offset_li
        num_clust = len(endog)
        
        mean = self.family.link.inverse

        self._cached_means = []

        for i in range(num_clust):

            if len(endog[i]) == 0:
                continue
            
            lp = offset[i] + np.dot(exog[i], beta)
            E = mean(lp)

            self._cached_means.append((E,lp))



    def _covmat(self):
        """
        Returns the sampling covariance matrix of the regression
        parameters and related quantities.

        Returns
        -------
        robust_covariance : array-like
           The robust, or sandwich estimate of the covariance, which is meaningful
           even if the working covariance structure is incorrectly specified.
        naive_covariance : array-like
           The model based estimate of the covariance, which is meaningful if
           the covariance structure is correctly specified.
        C : array-like
           The center matrix of the sandwich expression.
        """

        endog = self.endog_li
        exog = self.exog_li
        num_clust = len(endog)
        
        mean = self.family.link.inverse
        mean_deriv = self.family.link.inverse_deriv
        varfunc = self.family.variance
        _cached_means = self._cached_means

        B,C = 0,0
        for i in range(num_clust):

            if len(endog[i]) == 0:
                continue
            
            E,lp = _cached_means[i]

            Dt = exog[i] * mean_deriv(lp)[:,None]
            D = Dt.T

            S = np.sqrt(varfunc(E))
            V,is_cor = self.varstruct.variance_matrix(E, i)
            if is_cor:
                V *= np.outer(S, S)

            VID = np.linalg.solve(V, D.T)
            B += np.dot(D, VID)

            R = endog[i] - E
            VIR = np.linalg.solve(V, R)
            DVIR = np.dot(D, VIR)
            C += np.outer(DVIR, DVIR)

        scale = self.estimate_scale()

        naive_covariance = scale*np.linalg.inv(B)
        C /= scale**2
        robust_covariance = np.dot(naive_covariance, np.dot(C, naive_covariance))

        return robust_covariance, naive_covariance, C


    def predict(self, exog=None, linear=False):

        if exog is None and linear:
            F = [self.model.family.link(np.dot(self.params, x)) for x in self.model.exog]
        elif exog is None and not linear:
            F = [np.dot(x, self.params) for x in self.model.exog]
        elif linear:
            F = self.model.family.link(self.params, exog)
        elif not linear:
            F = np.dot(exog, self.params)
            
        return F




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

        endog = self.endog_li
        exog = self.exog_li
        varstruct = self.varstruct
        p = exog[0].shape[1]

        self.fit_history = {'params' : [],
                            'score_change' : []}

        if starting_beta is None:

            if self.endog_type == "interval":
                xnames1 = []
                beta = np.zeros(p, dtype=np.float64)
            else:
                xnames1 = ["cat_%d" % k for k in range(1,self.nylevel)]
                beta = _categorical_starting_values(self.endog, 
                                                    exog[0].shape[1],
                                                    self.nylevel, 
                                                    self.endog_type)

            xnames1 += self.exog_names
            if self.constraint is not None:
                xnames1 = [str(k) for k in range(len(beta))]
            beta = pandas.Series(beta, index=xnames1)

        else:
            if type(beta) == np.ndarray:
                ix = ["v%d" % k for k in range(1,len(beta)+1)]
                beta = pd.Series(starting_beta, index=ix)
            else:
                beta = starting_beta.copy()

        self._update_cached_means(beta)

        for iter in range(maxit):
            u,score = self._beta_update()
            beta += u
            self._update_cached_means(beta)
            sc = np.sqrt(np.sum(u**2))
            self.fit_history['params'].append(beta)
            if sc < ctol:
                break
            self._update_assoc(beta)

        bcov,ncov,C = self._covmat()

        # Expand the constraint and do the score test, if the constraint is
        # present.
        if self.constraint is not None:

            # The number of variables in the full model
            pb = self.exog_preconstraint.shape[1]
            beta0 = np.r_[beta, np.zeros(pb - len(beta))]

            # Get the score vector under the full model.
            save_exog_li = self.exog_li
            self.exog_li = self.exog_lintrans_li
            import copy
            save_cached_means = copy.deepcopy(self._cached_means)
            self._update_cached_means(beta0)
            _,score = self._beta_update()
            bcov1,ncov1,C = self._covmat()
            scale = self.estimate_scale()
            U2 = score[len(beta):] / scale

            A = np.linalg.inv(ncov1)

            B11 = C[0:p,0:p]
            B22 = C[p:,p:]
            B12 = C[0:p,p:]
            A11 = A[0:p,0:p]
            A22 = A[p:,p:]
            A12 = A[0:p,p:]

            score_cov = B22 - np.dot(A12.T, np.linalg.solve(A11, B12))
            score_cov -= np.dot(B12.T, np.linalg.solve(A11, A12))
            score_cov += np.dot(A12.T, np.dot(np.linalg.solve(A11, B11),
                                              np.linalg.solve(A11, A12)))

            from scipy.stats.distributions import chi2
            self.score_statistic = np.dot(U2, np.linalg.solve(score_cov, U2))
            self.score_df = len(U2)
            self.score_pvalue = 1 - chi2.cdf(self.score_statistic, self.score_df)

            # Reparameterize to the original coordinates
            L0 = self.constraint[2]
            param0 = self.constraint[3]
            beta = param0 + np.dot(L0, beta)
            bcov = np.dot(L0, np.dot(bcov, L0.T))

            self.exog_li = save_exog_li
            self._cached_means = save_cached_means

        GR = GEEResults(self, beta, bcov)

        GR.fit_history = self.fit_history

        return GR


    def _update_assoc(self, beta):
        """
        Update the association parameters
        """

        self.varstruct.update(beta)



class GEEResults(object):

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
        return self.model.endog - self.fittedvalues

    @cache_readonly
    def fittedvalues(self):
        return self.model.family.link.inverse(np.dot(self.model.exog, self.params))

        
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
                    ('Response type:', [self.model.endog_type.title()]),
                    ('Date:', None),
        ]
        
        NY = [len(y) for y in self.model.endog_li]

        top_right = [('No. Observations:', [sum(NY)]),
                     ('No. clusters:', [len(self.model.endog_li)]),
                     ('Min. cluster size', [min(NY)]),
                     ('Max. cluster size', [max(NY)]),
                     ('Mean cluster size', ["%.1f" % np.mean(NY)]),
                     ('No. iterations', ['%d' % len(self.fit_history)]),
                     ('Time:', None),
                 ]
        
        # The skew of the residuals
        R = self.resid
        skew1 = stats.skew(R)
        kurt1 = stats.kurtosis(R)
        V = R.copy() - R.mean()
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
                          yname=self.model.endog_names, xname=xname, title=title)
        smry.add_table_params(self, yname=yname, xname=self.params.index.tolist(), 
                              alpha=alpha, use_t=True)

        smry.add_table_2cols(self, gleft=diagn_left, gright=diagn_right,
                          yname=yname, xname=xname,
                          title="")

        return smry





def _setup_multicategorical(endog, exog, offset, time, endog_type):
    """Restructure nominal or ordinal multicategorical data as binary
    indicators so that they can be analysed using Generalized Estimating
    Equations.

    For nominal data, each element of endog is recoded as the sequence
    of |S|-1 indicators I(endog = S[0]), ..., I(endog = S[-1]), where
    S is the sorted list of unique values of endog (excluding the
    maximum value).

    For ordinal data, each element y of endog is recoded as |S|-1
    cumulative indicators I(endog > S[0]), ..., I(endog > S[-1]) where
    S is the sorted list of unique values of endog (excluding the
    maximum value).  

    For ordinal data, intercepts are constructed corresponding to the
    different levels of the outcome. For nominal data, the covariate
    vector is expanded so that different coefficients arise for each
    class.

    In both cases, the covariates in exog are copied over to all of
    the indicators derived from the same original value.

    Arguments
    ---------
    endog: List
        A list of 1-dimensional NumPy arrays, giving the response
        values for the clusters
    exog:  List
        A list of 2-dimensional NumPy arrays, giving the covariate
        data for the clusters.  exog[i] should include an intercept
        for nominal data but no intercept should be included for
        ordinal data.
    offset : List
        A list of 1-dimensional NumPy arrays containing the offset
        information
    time : List
        A list of 1-dimensional NumPy arrays containing time information
    endog_type: string
        Either "ordinal" or "nominal"

    The number of rows of exog[i] must equal the length of endog[i],
    and all the exog[i] arrays should have the same number of columns.

    Returns:
    --------
    endog1:   endog recoded as described above
    exog1:   exog recoded as described above
    offset1: offset expanded to fit the recoded data
    time1:   time expanded to fit the recoded data
    IY:   a list whose i^th element iy = IY[i] is a sequence of tuples
          (a,b), where endog[i][a:b] is the subvector of indicators derived
          from the same ordinal value 
    BTW   a list whose i^th element btw = BTW[i] is a map from cut-point
          pairs (c,c') to the indices of between-subject pairs derived
          from the given cut points

    """

    if endog_type not in ("ordinal", "nominal"):
        raise ValueError("_setup_multicategorical: `endog_type` must be either "
                         "'nominal' or 'categorical'") 

    # The unique outcomes, except the greatest one.
    YV = np.concatenate(endog)
    S = list(set(YV))
    S.sort()
    S = S[0:-1]

    ncut = len(S)

    # nominal=1, ordinal=0
    endog_type_i = [0,1][endog_type == "nominal"]

    endog1,exog1,offset1,time1,IY,BTW = [],[],[],[],[],[]
    for y,x,off,ti in zip(endog,exog,offset,time): # Loop over clusters

        y1,x1,off1,ti1,iy1 = [],[],[],[],[]
        jj = 0
        btw = {}

        for y2,x2,off2,ti2 in zip(y,x,off,ti): # Loop over data points within a cluster
            iy2 = []
            for js,s in enumerate(S): # Loop over thresholds for the indicators
                if endog_type_i == 0:
                    y1.append(int(y2 > s))
                    off1.append(off2)
                    ti1.append(ti2)
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
        endog1.append(np.array(y1))
        exog1.append(np.array(x1))
        offset1.append(np.array(off1))
        time1.append(np.array(ti1))

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

    return endog1,exog1,offset1,time1,IY,BTW,len(S)+1


def _categorical_starting_values(endog, q, nylevel, endog_type):

    S = list(set(endog))
    S.sort()
    S = S[0:-1]

    if endog_type == "ordinal":
        Pr = np.array([np.mean(endog > s) for s in S])
        bl = np.log(Pr/(1-Pr))
        beta = np.concatenate((bl, np.zeros(q-nylevel+1)))
    elif endog_type == "nominal":
        beta = np.zeros(exog[0].shape[1], dtype=np.float64)

    return beta




