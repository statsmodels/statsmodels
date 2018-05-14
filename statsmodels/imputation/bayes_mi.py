import numpy as np
from statsmodels.base.model import LikelihoodModelResults


class BayesGaussMI(object):
    """
    BayesGaussMI uses a Gaussian model to impute multivariate data.

    The approach is Bayesian.  The goal is to sample from the joint
    distribution of the mean vector, covariance matrix, and missing
    data values given the observed data values.  Conjugate priors for
    the population mean and covariance matrix are used.  Gibbs
    sampling is used to update the mean vector, covariance matrix, and
    missing data values in turn.  After burn-in, the imputed complete
    data sets from the Gibbs chain can be used in multiple imputation
    analyses (MI).

    Parameters
    ----------
    data : ndarray
        The array of data to be imputed.  Values in the array equal to
        NaN are imputed.
    mean_prior : ndarray, optional
        The covariance matrix of the Gaussian prior distribution for
        the mean vector.  If not provided, the identity matrix is
        used.
    cov_prior : ndarray, optional
        The center matrix for the inverse Wishart prior distribution
        for the covariance matrix.  If not provided, the identity
        matrix is used.
    cov_prior_df : positive float
        The degrees of freedom of the inverse Wishart prior
        distribution for the covariance matrix.  Defaults to 1.

    Returns:
    -------
    MIResults object

    Examples:
    ---------
    A basic example with OLS:

    >> def model_args(x):
           # Return endog, exog from x
           return (x[:, 0], x[:, 1:])
    >> imp = BayesGaussMI(x)
    >> mi = MI(imp, sm.OLS, model_args)
    """

    def __init__(self, data, mean_prior=None, cov_prior=None, cov_prior_df=1):

        data = np.asarray(data)
        self.data = data
        self.mask = np.isnan(data)
        self.nobs = self.mask.shape[0]
        self.nvar = self.mask.shape[1]

        # Identify all distinct missing data patterns
        z = 1 + np.log(1 + np.arange(self.mask.shape[1]))
        c = np.dot(self.mask, z)
        rowmap = {}
        for i, v in enumerate(c):
            if v == 0:
                # No missing values
                continue
            if v not in rowmap:
                rowmap[v] = []
            rowmap[v].append(i)
        self.patterns = [np.asarray(x) for x in rowmap.values()]

        # Simple starting values for mean and covariance
        p = self.data.shape[1]
        self.cov = np.eye(p)
        mean = []
        for i in range(p):
            v = self.data[:, i]
            v = v[np.isfinite(v)]
            if len(v) == 0:
                msg = "Column %d has no observed values" % i
                raise ValueError(msg)
            mean.append(v.mean())
        self.mean = np.asarray(mean)

        # Default covariance matrix of the (Gaussian) mean prior
        if mean_prior is None:
            mean_prior = np.eye(p)
        self.mean_prior = mean_prior

        # Default center matrix of the (inverse Wishart) covariance prior
        if cov_prior is None:
            cov_prior = np.eye(p)
        self.cov_prior = cov_prior

        # Degrees of freedom for the (inverse Wishart) covariance prior
        self.cov_prior_df = cov_prior_df

    def update(self):
        """
        Cycle through all Gibbs updates.
        """

        self.update_data()

        # Need to update data first
        self.update_mean()
        self.update_cov()

    def update_data(self):
        """
        Gibbs update of the missing data values.
        """

        for ix in self.patterns:

            i = ix[0]
            ix_miss = np.flatnonzero(self.mask[i, :])
            ix_obs = np.flatnonzero(~self.mask[i, :])

            mm = self.mean[ix_miss]
            mo = self.mean[ix_obs]

            voo = self.cov[ix_obs, :][:, ix_obs]
            vmm = self.cov[ix_miss, :][:, ix_miss]
            vmo = self.cov[ix_miss, :][:, ix_obs]

            r = self.data[ix, :][:, ix_obs] - mo
            cm = mm + np.dot(vmo, np.linalg.solve(voo, r.T)).T
            cv = vmm - np.dot(vmo, np.linalg.solve(voo, vmo.T))

            cs = np.linalg.cholesky(cv)
            u = np.random.normal(size=(len(ix), len(ix_miss)))
            self.data[np.ix_(ix, ix_miss)] = cm + np.dot(u, cs.T)

    def update_mean(self):
        """
        Gibbs update of the mean vector.

        Do not call until update_data has been called once.
        """
        # https://stats.stackexchange.com/questions/28744/multivariate-normal-posterior

        # Posterior covariance matrix of the mean
        cm = np.linalg.solve(self.cov/self.nobs + self.mean_prior,
                             self.mean_prior / self.nobs)
        cm = np.dot(self.cov, cm)

        # Posterior mean of the mean
        vm = np.linalg.solve(self.cov, self.data.sum(0))
        vm = np.dot(cm, vm)

        # Sample
        r = np.linalg.cholesky(cm)
        self.mean = vm + np.dot(r, np.random.normal(0, 1, self.nvar))

    def update_cov(self):
        """
        Gibbs update of the covariance matrix.

        Do not call until update_data has been called once.
        """
        # https://stats.stackexchange.com/questions/50844/estimating-the-covariance-posterior-distribution-of-a-multivariate-gaussian

        r = self.data - self.mean
        gr = np.dot(r.T, r)
        a = gr + self.cov_prior
        df = int(np.ceil(self.nobs + self.cov_prior_df))

        r = np.linalg.cholesky(np.linalg.inv(a))
        x = np.dot(np.random.normal(size=(df, self.nvar)), r.T)
        ma = np.dot(x.T, x)
        self.cov = np.linalg.inv(ma)


class MI(object):
    """
    MI performs multiple imputation using a provided imputer object.

    Parameters
    ----------
    imp : object
        An imputer class, such as BayesGaussMI.
    model : model class
        Any statsmodels model class.
    model_args : list-like, optional
        List of arguments to be passed to the model initializer.
        Must include endog, exog, and any other mandatory arguments
        for the model class.
    model_kwds : dict-like, optional
        Keyword arguments to be passed to the model initializer
    fit_args : list-like, optional
        List of arguments to be passed to the fit method
    fit_kwds : dict-like, optional
        Keyword arguments to be passed to the fit method
    xfunc : function mapping ndarray to ndarray
        A function that is applied to the complete data matrix
        prior to fitting the model
    burn : int
        Number of burn-in iterations
    nrep : int
        Number of imputed data sets to use in the analysis
    skip : int
        Number of Gibbs iterations to skip between successive
        mutiple imputation fits.

    Notes
    -----
    The imputer object must have an 'update' method, and a 'data'
    attribute that contains the current imputed dataset.

    xfunc can be used to introduce domain constraints, e.g. when
    imputing binary data the imputed continuous values can be rounded
    to 0/1.
    """

    def __init__(self, imp, model, model_args, model_kwds=None,
                 fit_args=None, fit_kwds=None, xfunc=None, burn=100,
                 nrep=20, skip=10):

        # The imputer
        self.imp = imp

        # The number of imputed data sets to skip between each imputed
        # data set tha that is used in the analysis.
        self.skip = skip

        # The model class
        self.model = model
        self.model_args = model_args

        if model_kwds is None:
            def f(x):
                return {}
            model_kwds = f
        self.model_kwds = model_kwds

        if fit_args is None:
            def f(x):
                return []
            fit_args = f
        self.fit_args = fit_args

        if fit_kwds is None:
            def f(x):
                return {}
            fit_kwds = f
        self.fit_kwds = fit_kwds

        self.xfunc = xfunc
        self.nrep = nrep
        self.skip = skip

        # Burn-in
        for k in range(burn):
            imp.update()

    def fit(self):
        """
        fit imputes datasets, fits models, and pools results.

        Returns
        -------
        A MIResults object.
        """

        par = []
        cov = []

        for k in range(self.nrep):

            for k in range(self.skip+1):
                self.imp.update()

            da = self.imp.data

            if self.xfunc is not None:
                da = self.xfunc(da)

            model = self.model(*self.model_args(da), **self.model_kwds(da))
            result = model.fit(*self.fit_args(da), **self.fit_kwds(da))

            par.append(result.params.copy())
            cov.append(result.cov_params().copy())

        params, cov_params, fmi = self._combine(par, cov)

        r = MIResults(self, model, params, cov_params)
        r.fmi = fmi
        return r

    def _combine(self, par, cov):
        # Helper function to apply "Rubin's combining rule"

        par = np.asarray(par)

        # Number of imputations
        m = par.shape[0]

        # Point estimate
        params = par.mean(0)

        # Within-imputation covariance
        wcov = sum(cov) / len(cov)

        # Between-imputation covariance
        bcov = np.cov(par.T)
        bcov = np.atleast_2d(bcov)

        # Overall covariance
        covp = wcov + (1 + 1/float(m))*bcov

        # Fraction of missing information
        fmi = (1 + 1/float(m)) * np.diag(bcov) / np.diag(covp)

        return params, covp, fmi


class MIResults(LikelihoodModelResults):
    """
    A results class for multiple imputation (MI).

    Parameters
    ----------
    mi : MI instance
        The MI object that produced the results
    model : instance of statsmodels model class
        This can be any instance from the multiple imputation runs.
        It is used to get class information, the specific parameter
        and data values are not used.
    params : array-like
        The overall multiple imputation parameter estimates.
    normalized_cov_params : array-like (2d)
        The overall variance covariance matrix of the estimates.
    """

    def __init__(self, mi, model, params, normalized_cov_params):

        super(MIResults, self).__init__(model, params, normalized_cov_params)
        self.mi = mi
        self._model = model

    def summary(self, title=None, alpha=.05):
        """
        Summarize the results of running multiple imputation.

        Parameters
        -----------
        title : string, optional
            Title for the top table. If not None, then this replaces
            the default title
        alpha : float
            Significance level for the confidence intervals

        Returns
        -------
        smry : Summary instance
            This holds the summary tables and text, which can be
            printed or converted to various output formats.
        """

        from statsmodels.iolib import summary2
        from statsmodels.compat.collections import OrderedDict

        smry = summary2.Summary()
        float_format = "%8.3f"

        info = OrderedDict()
        info["Method:"] = "MI"
        info["Model:"] = self.mi.model.__name__
        info["Dependent variable:"] = self._model.endog_names
        info["Sample size:"] = "%d" % self.mi.imp.data.shape[0]
        info["Num. imputations"] = "%d" % self.mi.nrep

        smry.add_dict(info, align='l', float_format=float_format)

        param = summary2.summary_params(self, alpha=alpha)
        param["FMI"] = self.fmi

        smry.add_df(param, float_format=float_format)
        smry.add_title(title=title, results=self)

        return smry
