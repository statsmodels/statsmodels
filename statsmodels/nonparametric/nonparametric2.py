"""
Multivariate Conditional and Unconditional Kernel Density Estimation
with Mixed Data Types

References
----------
[1] Racine, J., Li, Q. Nonparametric econometrics: theory and practice.
    Princeton University Press. (2007)
[2] Racine, Jeff. "Nonparametric Econometrics: A Primer," Foundation
    and Trends in Econometrics: Vol 3: No 1, pp1-88. (2008)
    http://dx.doi.org/10.1561/0800000009
[3] Racine, J., Li, Q. "Nonparametric Estimation of Distributions
    with Categorical and Continuous Data." Working Paper. (2000)
[4] Racine, J. Li, Q. "Kernel Estimation of Multivariate Conditional
    Distributions Annals of Economics and Finance 5, 211-235 (2004)
[5] Liu, R., Yang, L. "Kernel estimation of multivariate
    cumulative distribution function."
    Journal of Nonparametric Statistics (2008)
[6] Li, R., Ju, G. "Nonparametric Estimation of Multivariate CDF
    with Categorical and Continuous Data." Working Paper
[7] Li, Q., Racine, J. "Cross-validated local linear nonparametric
    regression" Statistica Sinica 14(2004), pp. 485-512
[8] Racine, J.: "Consistent Significance Testing for Nonparametric
        Regression" Journal of Business & Economics Statistics
[9] Racine, J., Hart, J., Li, Q., "Testing the Significance of
        Categorical Predictor Variables in Nonparametric Regression
        Models", 2006, Econometric Reviews 25, 523-544

"""

# TODO: make default behavior efficient=True above a certain n_obs


import copy

import numpy as np
from scipy import optimize
from scipy.stats.mstats import mquantiles

import kernels as kernels
import np_tools as tools

try:
    import joblib
    has_joblib = True
except ImportError:
    has_joblib = False


__all__ = ['KDE', 'ConditionalKDE', 'Reg', 'CensoredReg', 'EstimatorSettings']


def _compute_subset(class_type, data, bw, co, do, n_cvars, ix_ord,
                    ix_unord, n_sub, class_vars, randomize, bound):
    """"Compute bw on subset of data.

    Called from _GenericKDE._compute_efficient_*.

    Notes
    -----
    Needs to be outside the class in order for joblib to be able to pickle it.

    """
    if randomize:
        np.random.shuffle(data)
        sub_data = data[:n_sub, :]
    else:
        sub_data = data[bound[0]:bound[1], :]

    if class_type == 'KDE':
        var_type = class_vars[0]
        sub_model = KDE(sub_data, var_type, bw=bw,
                        defaults=EstimatorSettings(efficient=False))
    elif class_type == 'ConditionalKDE':
        k_dep, dep_type, indep_type = class_vars
        endog = sub_data[:, :k_dep]
        exog = sub_data[:, k_dep:]
        sub_model = ConditionalKDE(endog, exog, dep_type,
                                   indep_type, bw=bw,
                                   defaults=EstimatorSettings(efficient=False))
    elif class_type == 'Reg':
        var_type, K, reg_type = class_vars
        endog = tools.adjust_shape(sub_data[:, 0], 1)
        exog = tools.adjust_shape(sub_data[:, 1:], K)
        sub_model = Reg(endog=endog, exog=exog, reg_type=reg_type,
                        var_type=var_type, bw=bw,
                        defaults=EstimatorSettings(efficient=False))
    else:
        raise ValueError("Don't know what I am; aborting.")

    # Compute dispersion in next 7 lines
    if class_type == 'Reg':
        sub_data = sub_data[:, 1:]

    s1 = np.std(sub_data, axis=0)
    q75 = mquantiles(sub_data, 0.75, axis=0).data[0]
    q25 = mquantiles(sub_data, 0.25, axis=0).data[0]
    s2 = (q75 - q25) / 1.349  # IQR
    s = np.minimum(s1, s2)

    fct = s * n_sub**(-1. / (n_cvars + co))
    fct[ix_unord] = n_sub**(-2. / (n_cvars + do))
    fct[ix_ord] = n_sub**(-2. / (n_cvars + do))
    sample_scale_sub = sub_model.bw / fct  #TODO: check if correct
    bw_sub = sub_model.bw
    return sample_scale_sub, bw_sub


class _GenericKDE (object):
    """
    Generic KDE class with methods shared by both KDE and ConditionalKDE
    """
    def _compute_bw(self, bw):
        """
        Computes the bandwidth of the data.

        Parameters
        ----------
        bw: array_like or str
            If array_like: user-specified bandwidth.
            If a string, should be one of:

                - cv_ml: cross validation maximum likelihood
                - normal_reference: normal reference rule of thumb
                - cv_ls: cross validation least squares

        Notes
        -----
        The default values for bw is 'normal_reference'.
        """

        self.bw_func = dict(normal_reference=self._normal_reference,
                            cv_ml=self._cv_ml, cv_ls=self._cv_ls)
        if bw is None:
            bwfunc = self.bw_func['normal_reference']
            return bwfunc()

        if not isinstance(bw, basestring):
            self._bw_method = "user-specified"
            res = np.asarray(bw)
        else:
            # The user specified a bandwidth selection method
            self._bw_method = bw
            bwfunc = self.bw_func[bw]
            res = bwfunc()

        return res

    def _compute_dispersion(self, data):
        """
        Computes the measure of dispersion.

        The minimum of the standard deviation and interquartile range / 1.349

        References
        ----------
        See the user guide for the np package in R.
        In the notes on bwscaling option in npreg, npudens, npcdens there is
        a discussion on the measure of dispersion
        """
        if isinstance(self, Reg):
            data = data[:, 1:]

        s1 = np.std(data, axis=0)
        q75 = mquantiles(data, 0.75, axis=0).data[0]
        q25 = mquantiles(data, 0.25, axis=0).data[0]
        s2 = (q75 - q25) / 1.349
        return np.minimum(s1, s2)

    def _get_class_vars_type(self):
        """Helper method to be able to pass needed vars to _compute_subset."""
        if isinstance(self, KDE):
            class_type = 'KDE'
            class_vars = (self.var_type, )
        elif isinstance(self, ConditionalKDE):
            class_type = 'ConditionalKDE'
            class_vars = (self.k_dep, self.dep_type, self.indep_type)
        elif isinstance(self, Reg):
            class_type = 'Reg'
            class_vars = (self.var_type, self.K, self.reg_type)

        return class_type, class_vars

    def _compute_efficient(self, bw):
        """
        Computes the bandwidth by estimating the scaling factor (c)
        in n_res resamples of size ``n_sub`` (in `randomize` case), or by
        dividing ``nobs`` into as many ``n_sub`` blocks as needed (if
        `randomize` is False).

        References
        ----------
        See p.9 in socserv.mcmaster.ca/racine/np_faq.pdf
        """
        nobs = self.nobs
        n_sub = self.n_sub
        data = copy.deepcopy(self.data)
        n_cvars = self.data_type.count('c')
        co = 4  # 2*order of continuous kernel
        do = 4  # 2*order of discrete kernel
        _, ix_ord, ix_unord = tools._get_type_pos(self.data_type)

        # Define bounds for slicing the data
        if self.randomize:
            # randomize chooses blocks of size n_sub, independent of nobs
            bounds = [None] * self.n_res
        else:
            bounds = [(i * n_sub, (i+1) * n_sub) for i in range(nobs // n_sub)]
            if nobs % n_sub > 0:
                bounds.append((nobs - nobs % n_sub, nobs))

        n_blocks = self.n_res if self.randomize else len(bounds)
        sample_scale = np.empty((n_blocks, self.K))
        only_bw = np.empty((n_blocks, self.K))

        class_type, class_vars = self._get_class_vars_type()
        if has_joblib:
            # `res` is a list of tuples (sample_scale_sub, bw_sub)
            res = joblib.Parallel(n_jobs=self.n_jobs) \
                (joblib.delayed(_compute_subset) \
                (class_type, data, bw, co, do, n_cvars, ix_ord, ix_unord, \
                n_sub, class_vars, self.randomize, bounds[i]) \
                for i in range(n_blocks))
        else:
            res = []
            for i in xrange(n_blocks):
                res.append(_compute_subset(class_type, data, bw, co, do,
                                           n_cvars, ix_ord, ix_unord, n_sub,
                                           class_vars, self.randomize))

        for i in xrange(n_blocks):
            sample_scale[i, :] = res[i][0]
            only_bw[i, :] = res[i][1]

        s = self._compute_dispersion(data)
        order_func = np.median if self.return_median else np.mean
        m_scale = order_func(sample_scale, axis=0)
        # TODO: Check if 1/5 is correct in line below!
        bw = m_scale * s * nobs**(-1. / (n_cvars + co))
        bw[ix_ord] = m_scale[ix_ord] * nobs**(-2./ (n_cvars + do))
        bw[ix_unord] = m_scale[ix_unord] * nobs**(-2./ (n_cvars + do))

        if self.return_only_bw:
            bw = np.median(only_bw, axis=0)

        return bw

    def _set_defaults(self, defaults):
        """Sets the default values for the efficient estimation"""
        self.n_res = defaults.n_res
        self.n_sub = defaults.n_sub
        self.randomize = defaults.randomize
        self.return_median = defaults.return_median
        self.efficient = defaults.efficient
        self.return_only_bw = defaults.return_only_bw
        self.n_jobs = defaults.n_jobs

    def _normal_reference(self):
        """
        Returns Scott's normal reference rule of thumb bandwidth parameter.

        Notes
        -----
        See p.13 in [2] for an example and discussion.  The formula for the
        bandwidth is

        .. math:: h = 1.06n^{-1/(4+q)}

        where ``n`` is the number of observations and ``q`` is the number of
        variables.
        """
        X = np.std(self.data, axis=0)
        return 1.06 * X * self.nobs ** (- 1. / (4 + self.data.shape[1]))

    def _set_bw_bounds(self, bw):
        """
        Sets bandwidth lower bound to zero and for discrete values upper bound
        to 1.
        """
        bw[bw < 0] = 1e-10
        _, ix_ord, ix_unord = tools._get_type_pos(self.data_type)
        bw[ix_ord] = np.minimum(bw[ix_ord], 1.)
        bw[ix_unord] = np.minimum(bw[ix_unord], 1.)

        return bw

    def _cv_ml(self):
        """
        Returns the cross validation maximum likelihood bandwidth parameter.

        Notes
        -----
        For more details see p.16, 18, 27 in Ref. [1] (see module docstring).

        Returns the bandwidth estimate that maximizes the leave-out-out
        likelihood.  The leave-one-out log likelihood function is:

        .. math:: \ln L=\sum_{i=1}^{n}\ln f_{-i}(X_{i})

        The leave-one-out kernel estimator of :math:`f_{-i}` is:

        .. math:: f_{-i}(X_{i})=\frac{1}{(n-1)h}
                        \sum_{j=1,j\neq i}K_{h}(X_{i},X_{j})

        where :math:`K_{h}` represents the Generalized product kernel
        estimator:

        .. math:: K_{h}(X_{i},X_{j})=\prod_{s=1}^
                        {q}h_{s}^{-1}k\left(\frac{X_{is}-X_{js}}{h_{s}}\right)
        """
        # the initial value for the optimization is the normal_reference
        h0 = self._normal_reference()
        bw = optimize.fmin(self.loo_likelihood, x0=h0, args=(np.log, ),
                           maxiter=1e3, maxfun=1e3, disp=0, xtol=1e-3)
        bw = self._set_bw_bounds(bw)  # bound bw if necessary
        return bw

    def _cv_ls(self):
        """
        Returns the cross-validation least squares bandwidth parameter(s).

        Notes
        -----
        For more details see pp. 16, 27 in Ref. [1] (see module docstring).

        Returns the value of the bandwidth that maximizes the integrated mean
        square error between the estimated and actual distribution.  The
        integrated mean square error (IMSE) is given by:

        .. math:: \int\left[\hat{f}(x)-f(x)\right]^{2}dx

        This is the general formula for the IMSE.  The IMSE differs for
        conditional (ConditionalKDE) and unconditional (KDE) kernel density
        estimation.
        """
        h0 = self._normal_reference()
        bw = optimize.fmin(self.imse, x0=h0, maxiter=1e3, maxfun=1e3, disp=0,
                           xtol=1e-3)
        bw = self._set_bw_bounds(bw)  # bound bw if necessary
        return bw

    def loo_likelihood(self):
        raise NotImplementedError


class EstimatorSettings(object):
    """
    Object to specify settings for density estimation or regression.

    `EstimatorSettings` has several proporties related to how bandwidth
    estimation for the `KDE`, `ConditionalKde`, `Reg` and `CensoredReg`
    classes behaves.

    Parameters
    ----------
    efficient: bool, optional
        If True, the bandwidth estimation is to be performed
        efficiently -- by taking smaller sub-samples and estimating
        the scaling factor of each subsample.  This is useful for large
        samples (nobs >> 300) and/or multiple variables (K > 3).
        If False (default), all data is used at the same time.
    randomize: bool, optional
        If True, the bandwidth estimation is to be performed by
        taking `n_res` random resamples (with replacement) of size `n_sub` from
        the full sample.  If set to False (default), the estimation is
        performed by slicing the full sample in sub-samples of size `n_sub` so
        that all samples are used once.
    n_sub: int, optional
        Size of the sub-samples.  Default is 50.
    n_res: int, optional
        The number of random re-samples used to estimate the bandwidth.
        Only has an effect if ``randomize == True`.  Default value is 25.
    return_median: bool, optional
        If True (default), the estimator uses the median of all scaling factors
        for each sub-sample to estimate the bandwidth of the full sample.
        If False, the estimator uses the mean.
    return_only_bw: bool, optional
        If True, the estimator is to use the bandwidth and not the
        scaling factor.  This is *not* theoretically justified.
        Should be used only for experimenting.
    n_jobs : int, optional
        The number of jobs to use for parallel estimation with
        ``joblib.Parallel``.  Default is -1, meaning ``n_cores - 1``, with
        ``n_cores`` the number of available CPU cores.
        See the `joblib documentation
        <http://packages.python.org/joblib/parallel.html`_ for more details.

    Examples
    --------
    >>> settings = EstimatorSettings(randomize=True, n_jobs=3)
    >>> k_dens = KDE(data, var_type, defaults=settings)

    """
    def __init__(self, efficient=False, randomize=False, n_res=25, n_sub=50,
                 return_median=True, return_only_bw=False, n_jobs=-1):
        self.efficient = efficient
        self.randomize = randomize
        self.n_res = n_res
        self.n_sub = n_sub
        self.return_median = return_median
        self.return_only_bw = return_only_bw  # TODO: remove this?
        self.n_jobs = n_jobs


class KDE(_GenericKDE):
    """
    Unconditional Kernel Density Estimator

    Parameters
    ----------
    data: list of ndarrays or 2-D ndarray
        The training data for the Kernel Density Estimation, used to determine
        the bandwidth(s).  If a 2-D array, should be of shape
        (num_observations, num_variables).  If a list, each list element is a
        separate observation.
    var_type: str
        The type of the variables:

            c : Continuous
            u : Unordered (Discrete)
            o : Ordered (Discrete)

        The string should contain a type specifier for each variable, so for
        example ``var_type='ccuo'``.
    bw: array_like or str
        If an array, it is a fixed user-specified bandwidth.  If a string,
        should be one of:

            - normal_reference: normal reference rule of thumb (default)
            - cv_ml: cross validation maximum likelihood
            - cv_ls: cross validation least squares

    defaults: Instance of class EstimatorSettings
        The default values for the efficient bandwidth estimation

    Attributes
    ----------
    bw: array_like
        The bandwidth parameters.

    Methods
    -------
    pdf : the probability density function
    cdf : the cumulative distribution function
    imse : the integrated mean square error
    loo_likelihood : the leave one out likelihood

    Examples
    --------
    >>> from statsmodels.nonparametric import KDE
    >>> nobs = 300
    >>> np.random.seed(1234)  # Seed random generator
    >>> c1 = np.random.normal(size=(nobs,1))
    >>> c2 = np.random.normal(2, 1, size=(nobs,1))

    Estimate a bivariate distribution and display the bandwidth found:

    >>> dens_u = KDE(data=[c1,c2], var_type='cc', bw='normal_reference')
    >>> dens_u.bw
    array([ 0.39967419,  0.38423292])
    """
    def __init__(self, data, var_type, bw=None, defaults=EstimatorSettings()):
        self.var_type = var_type
        self.K = len(self.var_type)
        self.data = tools.adjust_shape(data, self.K)
        self.data_type = var_type
        self.nobs, self.K = np.shape(self.data)
        assert self.K == len(self.var_type)
        assert self.nobs > self.K  # Num of obs must be > than num of vars
        self._set_defaults(defaults)
        if not self.efficient:
            self.bw = self._compute_bw(bw)
        else:
            self.bw = self._compute_efficient(bw)

    def __repr__(self):
        """Provide something sane to print."""
        repr = "KDE instance\n"
        repr += "Number of variables: K = " + str(self.K) + "\n"
        repr += "Number of samples:   nobs = " + str(self.nobs) + "\n"
        repr += "Variable types:      " + self.var_type + "\n"
        repr += "BW selection method: " + self._bw_method + "\n"
        return repr

    def loo_likelihood(self, bw, func=lambda x: x):
        """
        Returns the leave-one-out likelihood function.

        The leave-one-out likelihood function for the unconditional KDE.

        Parameters
        ----------
        bw: array_like
            The value for the bandwidth parameter(s).
        func: function
            For the log likelihood should be ``numpy.log``.

        Notes
        -----
        The leave-one-out kernel estimator of :math:`f_{-i}` is:

        .. math:: f_{-i}(X_{i})=\frac{1}{(n-1)h}
                    \sum_{j=1,j\neq i}K_{h}(X_{i},X_{j})

        where :math:`K_{h}` represents the generalized product kernel
        estimator:

        .. math:: K_{h}(X_{i},X_{j}) =
            \prod_{s=1}^{q}h_{s}^{-1}k\left(\frac{X_{is}-X_{js}}{h_{s}}\right)
        """
        LOO = tools.LeaveOneOut(self.data)
        L = 0
        for i, X_not_i in enumerate(LOO):
            f_i = tools.gpke(bw, data=-X_not_i, data_predict=-self.data[i, :],
                             var_type=self.var_type)
            L += func(f_i)

        return -L

    def pdf(self, data_predict=None):
        """
        Evaluate the probability density function.

        Parameters
        ----------
        data_predict: array_like, optional
            Points to evaluate at.  If unspecified, the training data is used.

        Returns
        -------
        pdf_est: array_like
            Probability density function evaluated at `data_predict`.

        Notes
        -----
        The probability density is given by the generalized product kernel
        estimator:

        .. math:: K_{h}(X_{i},X_{j}) =
            \prod_{s=1}^{q}h_{s}^{-1}k\left(\frac{X_{is}-X_{js}}{h_{s}}\right)
        """
        if data_predict is None:
            data_predict = self.data
        else:
            data_predict = tools.adjust_shape(data_predict, self.K)

        pdf_est = []
        for i in xrange(np.shape(data_predict)[0]):
            pdf_est.append(tools.gpke(self.bw, data=self.data,
                           data_predict=data_predict[i, :],
                           var_type=self.var_type) / self.nobs)

        pdf_est = np.squeeze(pdf_est)
        return pdf_est

    def cdf(self, data_predict=None):
        """
        Evaluate the cumulative distribution function.

        Parameters
        ----------
        data_predict: array_like, optional
            Points to evaluate at.  If unspecified, the training data is used.

        Returns
        -------
        cdf_est: array_like
            The estimate of the cdf.

        Notes
        -----
        See http://en.wikipedia.org/wiki/Cumulative_distribution_function
        For more details on the estimation see Ref. [5] in module docstring.

        The multivariate CDF for mixed data (continuous and ordered/unordered
        discrete) is estimated by:

        ..math:: F(x^{c},x^{d})=n^{-1}\sum_{i=1}^{n}\left[G(
            \frac{x^{c}-X_{i}}{h})\sum_{u\leq x^{d}}L(X_{i}^{d},x_{i}^{d},
            \lambda)\right]

        where G() is the product kernel CDF estimator for the continuous
        and L() for the discrete variables.
        """
        if data_predict is None:
            data_predict = self.data
        else:
            data_predict = tools.adjust_shape(data_predict, self.K)

        cdf_est = []
        for i in xrange(np.shape(data_predict)[0]):
            cdf_est.append(tools.gpke(self.bw, data=self.data,
                                      data_predict=data_predict[i, :],
                                      var_type=self.var_type,
                                      ckertype="gaussian_cdf",
                                      ukertype="aitchisonaitken_cdf",
                                      okertype='wangryzin_cdf') / self.nobs)

        cdf_est = np.squeeze(cdf_est)
        return cdf_est

    def imse_orig(self, bw):
        """
        Returns the Integrated Mean Square Error for the unconditional KDE.

        Parameters
        ----------
        bw: array_like
            The bandwidth parameter(s).

        Returns
        ------
        CV: float
            The cross-validation objective function.

        Notes
        -----
        See p. 27 in [1]
        For details on how to handle the multivariate
        estimation with mixed data types see p.6 in [3]

        The formula for the cross-validation objective function is:

        .. math:: CV=\frac{1}{n^{2}}\sum_{i=1}^{n}\sum_{j=1}^{N}
            \bar{K}_{h}(X_{i},X_{j})-\frac{2}{n(n-1)}\sum_{i=1}^{n}
            \sum_{j=1,j\neq i}^{N}K_{h}(X_{i},X_{j})

        Where :math:`\bar{K}_{h}` is the multivariate product convolution
        kernel (consult [3] for mixed data types).
        """
        F = 0
        for i in range(self.nobs):
            k_bar_sum = tools.gpke(bw, data=-self.data,
                                   data_predict=-self.data[i, :],
                                   var_type=self.var_type,
                                   ckertype='gauss_convolution',
                                   okertype='wangryzin_convolution',
                                   ukertype='aitchisonaitken_convolution')
            F += k_bar_sum
        # there is a + because loo_likelihood returns the negative
        return (F / self.nobs**2 + self.loo_likelihood(bw) * \
                2 / ((self.nobs) * (self.nobs - 1)))

    def imse(self, bw):
        F = 0
        kertypes = dict(c=kernels.gaussian_convolution,
                        o=kernels.wang_ryzin_convolution,
                        u=kernels.aitchison_aitken_convolution)
        nobs = self.nobs
        data = -self.data
        var_type = self.var_type
        ix_cont = np.array([c == 'c' for c in var_type])
        _bw_cont_product = bw[ix_cont].prod()
        Kval = np.empty(data.shape)
        for i in range(nobs):
            for ii, vtype in enumerate(var_type):
                Kval[:, ii] = kertypes[vtype](bw[ii],
                                              data[:, ii],
                                              data[i, ii])

            dens = Kval.prod(axis=1) / _bw_cont_product
            k_bar_sum = dens.sum(axis=0)
            F += k_bar_sum

        kertypes = dict(c=kernels.gaussian,
                        o=kernels.wang_ryzin,
                        u=kernels.aitchison_aitken)
        LOO = tools.LeaveOneOut(self.data)
        L = 0
        Kval = np.empty((data.shape[0]-1, data.shape[1]))
        for i, X_not_i in enumerate(LOO):
            for ii, vtype in enumerate(var_type):
                Kval[:, ii] = kertypes[vtype](bw[ii],
                                              -X_not_i[:, ii],
                                              data[i, ii])
            dens = Kval.prod(axis=1) / _bw_cont_product
            L += dens.sum(axis=0)

        return (F / nobs**2 - 2 * L / (nobs * (nobs - 1)))


class ConditionalKDE(_GenericKDE):
    """
    Conditional Kernel Density Estimator.

    Calculates ``P(X_1,X_2,...X_n | Y_1,Y_2...Y_m) =
    P(X_1, X_2,...X_n, Y_1, Y_2,..., Y_m)/P(Y_1, Y_2,..., Y_m)``.
    The conditional density is by definition the ratio of the two unconditional
    densities, see [1]_.

    Parameters
    ----------
    endog: list of ndarrays or 2-D ndarray
        The training data for the dependent variables, used to determine
        the bandwidth(s).  If a 2-D array, should be of shape
        (num_observations, num_variables).  If a list, each list element is a
        separate observation.
    exog: list of ndarrays or 2-D ndarray
        The training data for the independent variable; same shape as `endog`.
    dep_type: str
        The type of the dependent variables:

            c : Continuous
            u : Unordered (Discrete)
            o : Ordered (Discrete)

        The string should contain a type specifier for each variable, so for
        example ``dep_type='ccuo'``.
    indep_type: str
        The type of the independent variables; specifed like `dep_type`.
    bw: array_like or str, optional
        If an array, it is a fixed user-specified bandwidth.  If a string,
        should be one of:

            - normal_reference: normal reference rule of thumb (default)
            - cv_ml: cross validation maximum likelihood
            - cv_ls: cross validation least squares

    defaults: Instance of class EstimatorSettings
        The default values for the efficient bandwidth estimation

    Attributes
    ---------
    bw: array_like
        The bandwidth parameters

    Methods
    -------
    pdf : the probability density function
    cdf : the cumulative distribution function
    imse : the integrated mean square error
    loo_likelihood : the leave one out likelihood

    References
    ----------
    .. [1] http://en.wikipedia.org/wiki/Conditional_probability_distribution

    Examples
    --------
    >>> nobs = 300
    >>> c1 = np.random.normal(size=(nobs,1))
    >>> c2 = np.random.normal(2,1,size=(nobs,1))

    >>> dens_c = ConditionalKDE(endog=[c1], exog=[c2], dep_type='c',
    ...               indep_type='c', bw='normal_reference')

    >>> print "The bandwidth is: ", dens_c.bw
    """

    def __init__(self, endog, exog, dep_type, indep_type, bw,
                 defaults=EstimatorSettings()):
        self.dep_type = dep_type
        self.indep_type = indep_type
        self.data_type = dep_type + indep_type
        self.k_dep = len(self.dep_type)
        self.k_indep = len(self.indep_type)
        self.endog = tools.adjust_shape(endog, self.k_dep)
        self.exog = tools.adjust_shape(exog, self.k_indep)
        self.nobs, self.k_dep = np.shape(self.endog)
        self.data = np.column_stack((self.endog, self.exog))
        self.K = np.shape(self.data)[1]
        assert len(self.dep_type) == self.k_dep
        assert len(self.indep_type) == np.shape(self.exog)[1]
        self._set_defaults(defaults)
        if not self.efficient:
            self.bw = self._compute_bw(bw)
        else:
            self.bw = self._compute_efficient(bw)

    def __repr__(self):
        """Provide something sane to print."""
        repr = "ConditionalKDE instance\n"
        repr += "Number of independent variables: k_indep = " + \
                str(self.k_indep) + "\n"
        repr += "Number of dependent variables: k_dep = " + \
                str(self.k_dep) + "\n"
        repr += "Number of observations: nobs = " + str(self.nobs) + "\n"
        repr += "Independent variable types:      " + self.indep_type + "\n"
        repr += "Dependent variable types:      " + self.dep_type + "\n"
        repr += "BW selection method: " + self._bw_method + "\n"
        return repr

    def loo_likelihood(self, bw, func=lambda x: x):
        """
        Returns the leave-one-out function for the data.

        Parameters
        ----------
        bw: array_like
            The bandwidth parameter(s).
        func: function f(x), optional
            Should be ``np.log`` for the log likelihood.
            Default is ``f(x) = x``.

        Returns
        -------
        L: float
            The value of the leave-one-out function for the data.

        Notes
        -----
        Similar to ``KDE.loo_likelihood`, but substitute
        ``f(x|y)=f(x,y)/f(y)`` for f(x).
        """
        yLOO = tools.LeaveOneOut(self.data)
        xLOO = tools.LeaveOneOut(self.exog).__iter__()
        L = 0
        for i, Y_j in enumerate(yLOO):
            X_not_i = xLOO.next()
            f_yx = tools.gpke(bw, data=-Y_j, data_predict=-self.data[i, :],
                              var_type=(self.dep_type + self.indep_type))
            f_x = tools.gpke(bw[self.k_dep:], data=-X_not_i,
                             data_predict=-self.exog[i, :],
                             var_type=self.indep_type)
            f_i = f_yx / f_x
            L += func(f_i)

        return - L

    def pdf(self, endog_predict=None, exog_predict=None):
        """
        Evaluate the probability density function.

        Parameters
        ----------
        endog_predict: array_like, optional
            Evaluation data for the dependent variables.  If unspecified, the
            training data is used.
        exog_predict: array_like, optional
            Evaluation data for the independent variables.

        Returns
        -------
        pdf: array_like
            The value of the probability density at `endog_predict` and `exog_predict`.

        Notes
        -----
        The formula for the conditional probability density is:

        .. math:: f(X|Y)=\frac{f(X,Y)}{f(Y)}

        with

        .. math:: f(X)=\prod_{s=1}^{q}h_{s}^{-1}k
                            \left(\frac{X_{is}-X_{js}}{h_{s}}\right)

        where :math:`k` is the appropriate kernel for each variable.
        """
        if endog_predict is None:
            endog_predict = self.endog
        else:
            endog_predict = tools.adjust_shape(endog_predict, self.k_dep)
        if exog_predict is None:
            exog_predict = self.exog
        else:
            exog_predict = tools.adjust_shape(exog_predict, self.k_indep)

        pdf_est = []
        data_predict = np.column_stack((endog_predict, exog_predict))
        for i in xrange(np.shape(data_predict)[0]):
            f_yx = tools.gpke(self.bw, data=self.data,
                              data_predict=data_predict[i, :],
                              var_type=(self.dep_type + self.indep_type))
            f_x = tools.gpke(self.bw[self.k_dep:], data=self.exog,
                             data_predict=exog_predict[i, :],
                             var_type=self.indep_type)
            pdf_est.append(f_yx / f_x)

        return np.squeeze(pdf_est)

    def cdf(self, endog_predict=None, exog_predict=None):
        """
        Cumulative distribution function for the conditional density.

        Parameters
        ----------
        endog_predict: array_like, optional
            The evaluation dependent variables at which the cdf is estimated.
            If not specified the training dependent variables are used.
        exog_predict: array_like, optional
            The evaluation independent variables at which the cdf is estimated.
            If not specified the training independent variables are used.

        Returns
        -------
        cdf_est: array_like
            The estimate of the cdf.

        Notes
        -----
        For more details on the estimation see [5], and p.181 in [1].

        The multivariate conditional CDF for mixed data (continuous and
        ordered/unordered discrete) is estimated by:

        ..math:: F(y|x)=\frac{n^{-1}\sum_{i=1}^{n}G(\frac{y-Y_{i}}{h_{0}})
                              W_{h}(X_{i},x)}{\widehat{\mu}(x)}

        where G() is the product kernel CDF estimator for the dependent (y)
        variable(s) and W() is the product kernel CDF estimator for the
        independent variable(s).
        """
        if endog_predict is None:
            endog_predict = self.endog
        else:
            endog_predict = tools.adjust_shape(endog_predict, self.k_dep)
        if exog_predict is None:
            exog_predict = self.exog
        else:
            exog_predict = tools.adjust_shape(exog_predict, self.k_indep)

        N_data_predict = np.shape(exog_predict)[0]
        cdf_est = np.empty(N_data_predict)
        for i in xrange(N_data_predict):
            mu_x = tools.gpke(self.bw[self.k_dep:], data=self.exog,
                              data_predict=exog_predict[i, :],
                              var_type=self.indep_type) / self.nobs
            mu_x = np.squeeze(mu_x)
            cdf_endog = tools.gpke(self.bw[0:self.k_dep], data=self.endog,
                                   data_predict=endog_predict[i, :],
                                   var_type=self.dep_type,
                                   ckertype="gaussian_cdf",
                                   ukertype="aitchisonaitken_cdf",
                                   okertype='wangryzin_cdf', tosum=False)

            cdf_exog = tools.gpke(self.bw[self.k_dep:], data=self.exog,
                                  data_predict=exog_predict[i, :],
                                  var_type=self.indep_type, tosum=False)
            S = (cdf_endog * cdf_exog).sum(axis=0)
            cdf_est[i] = S / (self.nobs * mu_x)

        return cdf_est

    def imse(self, bw):
        """
        The integrated mean square error for the conditional KDE.

        Parameters
        ----------
        bw: array_like
            The bandwidth parameter(s).

        Returns
        -------
        CV: float
            The cross-validation objective function.

        Notes
        -----
        For more details see pp. 156-166 in [1].
        For details on how to handle the mixed variable types see [3].

        The formula for the cross-validation objective function for mixed
        variable types is:

        .. math:: CV(h,\lambda)=\frac{1}{n}\sum_{l=1}^{n}
            \frac{G_{-l}(X_{l})}{\left[\mu_{-l}(X_{l})\right]^{2}}-
            \frac{2}{n}\sum_{l=1}^{n}\frac{f_{-l}(X_{l},Y_{l})}{\mu_{-l}(X_{l})}

        where

        .. math:: G_{-l}(X_{l}) = n^{-2}\sum_{i\neq l}\sum_{j\neq l}
                        K_{X_{i},X_{l}} K_{X_{j},X_{l}}K_{Y_{i},Y_{j}}^{(2)}

        where :math:`K_{X_{i},X_{l}}` is the multivariate product kernel and
        :math:`\mu_{-l}(X_{l})` is the leave-one-out estimator of the pdf.

        :math:`K_{Y_{i},Y_{j}}^{(2)}` is the convolution kernel.

        The value of the function is minimized by the ``_cv_ls`` method of the
        `_GenericKDE` class to return the bw estimates that minimize the
        distance between the estimated and "true" probability density.
        """
        zLOO = tools.LeaveOneOut(self.data)
        CV = 0
        nobs = float(self.nobs)
        expander = np.ones((self.nobs - 1, 1))
        for ii, Z in enumerate(zLOO):
            X = Z[:, self.k_dep:]
            Y = Z[:, :self.k_dep]
            Ye_L = np.kron(Y, expander)
            Ye_R = np.kron(expander, Y)
            Xe_L = np.kron(X, expander)
            Xe_R = np.kron(expander, X)
            K_Xi_Xl = tools.gpke(bw[self.k_dep:], data=Xe_L,
                                 data_predict=self.exog[ii, :],
                                 var_type=self.indep_type, tosum=False)
            K_Xj_Xl = tools.gpke(bw[self.k_dep:], data=Xe_R,
                                 data_predict=self.exog[ii, :],
                                 var_type=self.indep_type, tosum=False)
            K2_Yi_Yj = tools.gpke(bw[0:self.k_dep], data=Ye_L,
                                  data_predict=Ye_R, var_type=self.dep_type,
                                  ckertype='gauss_convolution',
                                  okertype='wangryzin_convolution',
                                  ukertype='aitchisonaitken_convolution',
                                  tosum=False)
            G = (K_Xi_Xl * K_Xj_Xl * K2_Yi_Yj).sum() / nobs**2
            f_X_Y = tools.gpke(bw, data=-Z, data_predict=-self.data[ii, :],
                               var_type=(self.dep_type + self.indep_type)) / \
                               nobs
            m_x = tools.gpke(bw[self.k_dep:], data=-X,
                             data_predict=-self.exog[ii, :],
                             var_type=self.indep_type) / nobs
            CV += (G / m_x ** 2) - 2 * (f_X_Y / m_x)

        return CV / nobs


class Reg(_GenericKDE):
    """
    Nonparametric Regression

    Calculates the condtional mean ``E[y|X]`` where ``y = g(X) + e``.

    Parameters
    ----------
    endog: list with one element which is array_like
        This is the dependent variable.
    exog: list
        The training data for the independent variable(s)
        Each element in the list is a separate variable
    dep_type: str
        The type of the dependent variable(s)::

            c: Continuous
            u: Unordered (Discrete)
            o: Ordered (Discrete)

    reg_type: str
        Type of regression estimator
        lc: Local Constant Estimator
        ll: Local Linear Estimator
    bw: array-like
        Either a user-specified bandwidth or the method for bandwidth
        selection.
        cv_ls: cross-validaton least squares
        aic: AIC Hurvich Estimator
    defaults: EstimatorSettings instance
        The default values for the efficient bandwidth estimation.

    Attributes
    ---------
    bw: array-like
        The bandwidth parameters.

    Methods
    -------
    r-squared(): Calculates the R-Squared for the model.
    mean(): Calculates the conditional mean.
    """

    def __init__(self, endog, exog, var_type, reg_type, bw='cv_ls',
                 defaults=EstimatorSettings()):
        self.var_type = var_type
        self.data_type = var_type
        self.reg_type = reg_type
        self.K = len(self.var_type)
        self.endog = tools.adjust_shape(endog, 1)
        self.exog = tools.adjust_shape(exog, self.K)
        self.data = np.column_stack((self.endog, self.exog))
        self.nobs = np.shape(self.exog)[0]
        self.bw_func = dict(cv_ls=self.cv_loo, aic=self.aic_hurvich)
        self.est = dict(lc=self._est_loc_constant, ll=self._est_loc_linear)
        self._set_defaults(defaults)
        if not self.efficient:
            self.bw = self.compute_reg_bw(bw)
        else:
            self.bw = self._compute_efficient(bw)

    def compute_reg_bw(self, bw):
        if not isinstance(bw, basestring):
            self._bw_method = "user-specified"
            return np.asarray(bw)
        else:
            # The user specified a bandwidth selection method e.g. 'cv_ls'
            self._bw_method = bw
            res = self.bw_func[bw]
            X = np.std(self.exog, axis=0)
            h0 = 1.06 * X * \
                 self.nobs ** (- 1. / (4 + np.size(self.exog, axis=1)))

        func = self.est[self.reg_type]
        return optimize.fmin(res, x0=h0, args=(func, ), maxiter=1e3,
                             maxfun=1e3, disp=0)

    def _est_loc_linear(self, bw, endog, exog, data_predict):
        """
        Local linear estimator of g(x) in the regression ``y = g(x) + e``.

        Parameters
        ----------
        bw: array_like
            Vector of bandwidth value(s).
        endog: 1D array_like
            The dependent variable.
        exog: 1D or 2D array_like
            The independent variable(s).
        data_predict: 1D array_like of length K, where K is the number of variables.
            The point at which the density is estimated.

        Returns
        -------
        D_x: array_like
            The value of the conditional mean at `data_predict`.

        Notes
        -----
        See p. 81 in [1] and p.38 in [2] for the formulas.
        Unlike other methods, this one requires that `data_predict` be 1D.
        """
        nobs, Qc = exog.shape
        Ker = tools.gpke(bw, data=exog, data_predict=data_predict,
                         var_type=self.var_type,
                         #ukertype='aitchison_aitken_reg',
                         #okertype='wangryzin_reg',
                         tosum=False) / float(nobs)
        # Create the matrix on p.492 in [7], after the multiplication w/ K_h,ij
        # See also p. 38 in [2]
        #ix_cont = np.arange(self.K)  # Use all vars instead of continuous only
        # Note: because ix_cont was defined here such that it selected all
        # columns, I removed the indexing with it from exog/data_predict.

        # Convert Ker to a 2-D array to make matrix operations below work
        Ker = Ker[:, np.newaxis]

        M12 = exog - data_predict
        M22 = np.dot(M12.T, M12 * Ker)
        M12 = (M12 * Ker).sum(axis=0)
        M = np.empty((Qc + 1, Qc + 1))
        M[0, 0] = Ker.sum()
        M[0, 1:] = M12
        M[1:, 0] = M12
        M[1:, 1:] = M22

        ker_endog = Ker * endog
        V = np.empty((Qc + 1, 1))
        V[0, 0] = ker_endog.sum()
        V[1:, 0] = ((exog - data_predict) * ker_endog).sum(axis=0)

        mean_mfx = np.dot(np.linalg.pinv(M), V)
        mean = mean_mfx[0]
        mfx = mean_mfx[1:, :]
        return mean, mfx

    def _est_loc_constant(self, bw, endog, exog, data_predict):
        """
        Local constant estimator of g(x) in the regression
        y = g(x) + e

        Parameters
        ----------
        bw: array_like
            Vector of bandwidth value(s)
        endog: 1D array_like
            The dependent variable
        exog: 1D or 2D array_like
            The independent variable(s)
        data_predict: 1D or 2D array_like
            The point(s) at which
            the density is estimated

        Returns
        -------
        G: array_like
            The value of the conditional mean at data_predict

        """
        KX = tools.gpke(bw, data=exog, data_predict=data_predict,
                        var_type=self.var_type,
                        #ukertype='aitchison_aitken_reg',
                        #okertype='wangryzin_reg',
                        tosum=False)
        KX = np.reshape(KX, np.shape(endog))
        G_numer = (KX * endog).sum(axis=0)
        G_denom = KX.sum(axis=0)
        G = G_numer / G_denom
        nobs, K = exog.shape
        f_x = G_denom / float(nobs)
        KX_c = tools.gpke(bw, data=exog, data_predict=data_predict,
                          var_type=self.var_type,
                          ckertype='d_gaussian',
                          #okertype='wangryzin_reg',
                          tosum=False)

        KX_c = KX_c[:, np.newaxis]
        d_mx = -(endog * KX_c).sum(axis=0) / float(nobs) #* np.prod(bw[:, ix_cont]))
        d_fx = -KX_c.sum(axis=0) / float(nobs) #* np.prod(bw[:, ix_cont]))
        B_x = d_mx / f_x - G * d_fx / f_x
        B_x = (G_numer * d_fx - G_denom * d_mx) / (G_denom**2)
        #B_x = (f_x * d_mx - m_x * d_fx) / (f_x ** 2)
        return G, B_x

    def aic_hurvich(self, bw, func=None):
        """
        Computes the AIC Hurvich criteria for the estimation of the bandwidth
        References
        ----------
        See ch.2 in [1]
        See p.35 in [2]
        """
        H = np.empty((self.nobs, self.nobs))
        for j in range(self.nobs):
            H[:, j] = tools.gpke(bw, data=self.exog, data_predict=self.exog[j,:],
                                 var_type=self.var_type, tosum=False)
        denom = H.sum(axis=1)
        H = H / denom
        gx = Reg(endog=self.endog, exog=self.exog, var_type=self.var_type,
                 reg_type=self.reg_type, bw=bw,
                 defaults=EstimatorSettings(efficient=False)).fit()[0]
        gx = np.reshape(gx, (self.nobs, 1))
        sigma = ((self.endog - gx)**2).sum(axis=0) / float(self.nobs)

        frac = (1 + np.trace(H) / float(self.nobs)) / \
               (1 - (np.trace(H) + 2) / float(self.nobs))
        #siga = np.dot(self.endog.T, (I - H).T)
        #sigb = np.dot((I - H), self.endog)
        #sigma = np.dot(siga, sigb) / float(self.nobs)
        aic = np.log(sigma) + frac

        return aic

    def cv_loo(self, bw, func):
        """
        The cross-validation function with leave-one-out
        estimator

        Parameters
        ----------
        bw: array_like
            Vector of bandwidth values
        func: callable function
            Returns the estimator of g(x).  Can be either ``_est_loc_constant``
            (local constant) or ``_est_loc_linear`` (local_linear).

        Returns
        -------
        L: float
            The value of the CV function

        Notes
        -----
        Calculates the cross-validation least-squares function. This function
        is minimized by compute_bw to calculate the optimal value of `bw`.

        For details see p.35 in [2]

        ..math:: CV(h)=n^{-1}\sum_{i=1}^{n}(Y_{i}-g_{-i}(X_{i}))^{2}

        where :math:`g_{-i}(X_{i})` is the leave-one-out estimator of g(X)
        and :math:`h` is the vector of bandwidths

        """
        LOO_X = tools.LeaveOneOut(self.exog)
        LOO_Y = tools.LeaveOneOut(self.endog).__iter__()
        L = 0
        for ii, X_not_i in enumerate(LOO_X):
            Y = LOO_Y.next()
            G = func(bw, endog=Y, exog=-X_not_i,
                     data_predict=-self.exog[ii, :])[0]
            L += (self.endog[ii] - G) ** 2

        # Note: There might be a way to vectorize this. See p.72 in [1]
        return L / self.nobs

    def r_squared(self):
        """
        Returns the R-Squared for the nonparametric regression

        Notes
        -----
        For more details see p.45 in [2]
        The R-Squared is calculated by:
        .. math:: R^{2}=\frac{\left[\sum_{i=1}^{n}
        (Y_{i}-\bar{y})(\hat{Y_{i}}-\bar{y}\right]^{2}}{\sum_{i=1}^{n}
        (Y_{i}-\bar{y})^{2}\sum_{i=1}^{n}(\hat{Y_{i}}-\bar{y})^{2}}

        where :math:`\hat{Y_{i}}` are the fitted values calculated in
        self.mean().
        """
        Y = np.squeeze(self.endog)
        Yhat = self.fit()[0]
        Y_bar = np.mean(Yhat)
        R2_numer = (((Y - Y_bar) * (Yhat - Y_bar)).sum())**2
        R2_denom = ((Y - Y_bar)**2).sum(axis=0) * \
                   ((Yhat - Y_bar)**2).sum(axis=0)
        return R2_numer / R2_denom

    def fit(self, data_predict=None):
        """
        Returns the marginal effects at the data_predict points
        """
        func = self.est[self.reg_type]
        if data_predict is None:
            data_predict = self.exog
        else:
            data_predict = tools.adjust_shape(data_predict, self.K)

        N_data_predict = np.shape(data_predict)[0]
        mean = np.empty((N_data_predict,))
        mfx = np.empty((N_data_predict, self.K))
        for i in xrange(N_data_predict):
            mean_mfx = func(self.bw, self.endog, self.exog,
                            data_predict=data_predict[i, :])
            mean[i] = mean_mfx[0]
            mfx_c = np.squeeze(mean_mfx[1])
            mfx[i, :] = mfx_c

        return mean, mfx

    def sig_test(self, var_pos, nboot=50, nested_res=25, pivot=False):
        """
        Significance test for the variables in the regression.

        Parameters
        ----------
        var_pos: tuple, list
            The position of the variable in exog to be tested

        Returns
        -------
        sig: str
            The level of significance:

                - * : at 90% confidence level
                - ** : at 95% confidence level
                - *** : at 99* confidence level
                - "Not Significant" : if not significant

        """
        var_pos = np.asarray(var_pos)
        ix_cont, ix_ord, ix_unord = tools._get_type_pos(self.var_type)
        if np.any(ix_cont[var_pos]):
            if np.any(ix_ord[var_pos]) or np.any(ix_unord[var_pos]):
                raise "Discrete variable in hypothesis. Must be continuous"

            Sig = TestRegCoefC(self, var_pos, nboot, nested_res, pivot)
        else:
            Sig = TestRegCoefD(self, var_pos, nboot)

        return Sig.sig

    def __repr__(self):
        """Provide something sane to print."""
        repr = "Reg instance\n"
        repr += "Number of variables: K = " + str(self.K) + "\n"
        repr += "Number of samples:   N = " + str(self.nobs) + "\n"
        repr += "Variable types:      " + self.var_type + "\n"
        repr += "BW selection method: " + self._bw_method + "\n"
        repr += "Estimator type: " + self.reg_type + "\n"
        return repr


class CensoredReg(Reg):
    """
    Nonparametric censored regression.

    Calculates the condtional mean ``E[y|X]`` where ``y = g(X) + e``,
    where y is left-censored.  Left censored variable Y is defined as
    ``Y = min {Y', L}`` where ``L`` is the value at which ``Y`` is censored
    and ``Y'`` is the true value of the variable.

    Parameters
    ----------
    endog: list with one element which is array_like
        This is the dependent variable.
    exog: list
        The training data for the independent variable(s)
        Each element in the list is a separate variable
    dep_type: str
        The type of the dependent variable(s)
        c: Continuous
        u: Unordered (Discrete)
        o: Ordered (Discrete)
    reg_type: str
        Type of regression estimator
        lc: Local Constant Estimator
        ll: Local Linear Estimator
    bw: array-like
        Either a user-specified bandwidth or
        the method for bandwidth selection.
        cv_ls: cross-validaton least squares
        aic: AIC Hurvich Estimator
    censor_val: Float
        Value at which the dependent variable is censored
    defaults: EstimatorSettings instance, optional
        The default values for the efficient bandwidth estimation

    Attributes
    ---------
    bw: array-like
        The bandwidth parameters

    Methods
    -------
    r-squared(): Calculates the R-Squared for the model
    mean(): Calculates the conditiona mean
    """

    def __init__(self, endog, exog, var_type, reg_type, bw='cv_ls',
                 censor_val=0, defaults=EstimatorSettings()):
        self.var_type = var_type
        self.data_type = var_type
        self.reg_type = reg_type
        self.K = len(self.var_type)
        self.endog = tools.adjust_shape(endog, 1)
        self.exog = tools.adjust_shape(exog, self.K)
        self.data = np.column_stack((self.endog, self.exog))
        self.nobs = np.shape(self.exog)[0]
        self.bw_func = dict(cv_ls=self.cv_loo, aic=self.aic_hurvich)
        self.est = dict(lc=self._est_loc_constant, ll=self._est_loc_linear)
        self._set_defaults(defaults)
        self.censor_val = censor_val
        if self.censor_val is not None:
            self.censored(censor_val)
        else:
            self.W_in = np.ones((self.nobs, 1))

        if not self.efficient:
            self.bw = self.compute_reg_bw(bw)
        else:
            self.bw = self._compute_efficient(bw)

    def censored(self, censor_val):
        # see pp. 341-344 in [1]
        self.d = (self.endog != censor_val) * 1.
        ix = np.argsort(np.squeeze(self.endog))
        self.endog = np.squeeze(self.endog[ix])
        self.endog = tools.adjust_shape(self.endog, 1)
        self.exog = np.squeeze(self.exog[ix])
        self.d = np.squeeze(self.d[ix])
        self.W_in = np.empty((self.nobs, 1))
        for i in xrange(1, self.nobs + 1):
            P=1
            for j in xrange(1, i):
                P *= ((self.nobs - j)/(float(self.nobs)-j+1))**self.d[j-1]
            self.W_in[i-1,0] = P * self.d[i-1] / (float(self.nobs) - i + 1 )

    def __repr__(self):
        """Provide something sane to print."""
        repr = "Reg instance\n"
        repr += "Number of variables: K = " + str(self.K) + "\n"
        repr += "Number of samples:   nobs = " + str(self.nobs) + "\n"
        repr += "Variable types:      " + self.var_type + "\n"
        repr += "BW selection method: " + self._bw_method + "\n"
        repr += "Estimator type: " + self.reg_type + "\n"
        return repr

    def _est_loc_linear(self, bw, endog, exog, data_predict, W):
        """
        Local linear estimator of g(x) in the regression ``y = g(x) + e``.

        Parameters
        ----------
        bw: array_like
            Vector of bandwidth value(s)
        endog: 1D array_like
            The dependent variable
        exog: 1D or 2D array_like
            The independent variable(s)
        data_predict: 1D array_like of length K, where K is
            the number of variables. The point at which
            the density is estimated

        Returns
        -------
        D_x: array_like
            The value of the conditional mean at data_predict

        Notes
        -----
        See p. 81 in [1] and p.38 in [2] for the formulas
        Unlike other methods, this one requires that data_predict be 1D
        """
        nobs, Qc = exog.shape
        Ker = tools.gpke(bw, data=exog, data_predict=data_predict,
                         var_type=self.var_type,
                         ukertype='aitchison_aitken_reg',
                         okertype='wangryzin_reg', tosum=False)
        # Create the matrix on p.492 in [7], after the multiplication w/ K_h,ij
        # See also p. 38 in [2]

        # Convert Ker to a 2-D array to make matrix operations below work
        Ker = Ker[:, np.newaxis]

        M12 = exog - data_predict
        M22 = np.dot(M12.T, M12 * Ker)
        M12 = (M12 * Ker).sum(axis=0)
        M = np.empty((Qc + 1, Qc + 1))
        M[0, 0] = Ker.sum()
        M[0, 1:] = M12
        M[1:, 0] = M12
        M[1:, 1:] = M22

        ker_endog = Ker * endog
        V = np.empty((Qc + 1, 1))
        V[0, 0] = ker_endog.sum()
        V[1:, 0] = ((exog - data_predict) * ker_endog).sum(axis=0)

        mean_mfx = np.dot(np.linalg.pinv(M), V)
        mean = mean_mfx[0]
        mfx = mean_mfx[1:, :]
        return mean, mfx


    def cv_loo(self, bw, func):
        """
        The cross-validation function with leave-one-out
        estimator

        Parameters
        ----------
        bw: array_like
            Vector of bandwidth values
        func: callable function
            Returns the estimator of g(x).
            Can be either ``_est_loc_constant`` (local constant) or
            ``_est_loc_linear`` (local_linear).

        Returns
        -------
        L: float
            The value of the CV function

        Notes
        -----
        Calculates the cross-validation least-squares
        function. This function is minimized by compute_bw
        to calculate the optimal value of bw

        For details see p.35 in [2]

        ..math:: CV(h)=n^{-1}\sum_{i=1}^{n}(Y_{i}-g_{-i}(X_{i}))^{2}

        where :math:`g_{-i}(X_{i})` is the leave-one-out estimator of g(X)
        and :math:`h` is the vector of bandwidths

        """
        LOO_X = tools.LeaveOneOut(self.exog)
        LOO_Y = tools.LeaveOneOut(self.endog).__iter__()
        LOO_W = tools.LeaveOneOut(self.W_in).__iter__()
        L = 0
        for ii, X_not_i in enumerate(LOO_X):
            Y = LOO_Y.next()
            w = LOO_W.next()
            G = func(bw, endog=Y, exog=-X_not_i,
                     data_predict=-self.exog[ii, :], W=w)[0]
            L += (self.endog[ii] - G) ** 2

        # Note: There might be a way to vectorize this. See p.72 in [1]
        return L / self.nobs

    def fit(self, data_predict=None):
        """
        Returns the marginal effects at the data_predict points.
        """
        func = self.est[self.reg_type]
        if data_predict is None:
            data_predict = self.exog
        else:
            data_predict = tools.adjust_shape(data_predict, self.K)

        N_data_predict = np.shape(data_predict)[0]
        mean = np.empty((N_data_predict,))
        mfx = np.empty((N_data_predict, self.K))
        for i in xrange(N_data_predict):
            mean_mfx = func(self.bw, self.endog, self.exog,
                            data_predict=data_predict[i, :],
                            W=self.W_in)
            mean[i] = mean_mfx[0]
            mfx_c = np.squeeze(mean_mfx[1])
            mfx[i, :] = mfx_c

        return mean, mfx


class TestRegCoefC(object):
    """
    Significance test for continuous variables in a nonparametric regression.

    The null hypothesis is ``dE(Y|X)/dX_not_i = 0``, the alternative hypothesis
    is ``dE(Y|X)/dX_not_i != 0``.

    Parameters
    ----------
    model: Reg instance
        This is the nonparametric regression model whose elements
        are tested for significance.
    test_vars: tuple, list of integers, array_like
        index of position of the continuous variables to be tested
        for significance. E.g. (1,3,5) jointly tests variables at
        position 1,3 and 5 for significance.
    nboot: int
        Number of bootstrap samples used to determine the distribution
        of the test statistic in a finite sample. Default is 400
    nested_res: int
        Number of nested resamples used to calculate lambda.
        Must enable the pivot option
    pivot: bool
        Pivot the test statistic by dividing by its standard error
        Significantly increases computational time. But pivot statistics
        have more desirable properties
        (See references)

    Attributes
    ----------
    sig: str
        The significance level of the variable(s) tested
        "Not Significant": Not significant at the 90% confidence level
                            Fails to reject the null
        "*": Significant at the 90% confidence level
        "**": Significant at the 95% confidence level
        "***": Significant at the 99% confidence level

    Notes
    -----
    This class allows testing of joint hypothesis as long as all variables
    are continuous.

    References
    ----------
    Racine, J.: "Consistent Significance Testing for Nonparametric Regression"
    Journal of Business \& Economics Statistics.

    Chapter 12 in [1].
    """
    # Significance of continuous vars in nonparametric regression
    # Racine: Consistent Significance Testing for Nonparametric Regression
    # Journal of Business & Economics Statistics
    def __init__(self, model, test_vars, nboot=400, nested_res=400,
                 pivot=False):
        self.nboot = nboot
        self.nres = nested_res
        self.test_vars = test_vars
        self.model = model
        self.bw = model.bw
        self.var_type = model.var_type
        self.K = len(self.var_type)
        self.endog = model.endog
        self.exog = model.exog
        self.gx = model.est[model.reg_type]
        self.test_vars = test_vars
        self.pivot = pivot
        self.run()

    def run(self):
        self.test_stat = self._compute_test_stat(self.endog, self.exog)
        self.sig = self._compute_sig()

    def _compute_test_stat(self, Y, X):
        """
        Computes the test statistic.  See p.371 in [8].
        """
        lam = self._compute_lambda(Y, X)
        t = lam
        if self.pivot:
            se_lam = self._compute_se_lambda(Y, X)
            t = lam / float(se_lam)

        return t

    def _compute_lambda(self, Y, X):
        """Computes only lambda -- the main part of the test statistic"""
        n = np.shape(X)[0]
        Y = tools.adjust_shape(Y, 1)
        X = tools.adjust_shape(X, self.K)
        b = Reg(Y, X, self.var_type, self.model.reg_type, self.bw,
                        defaults = EstimatorSettings(efficient=False)).fit()[1]

        b = b[:, self.test_vars]
        b = np.reshape(b, (n, len(self.test_vars)))
        #fct = np.std(b)  # Pivot the statistic by dividing by SE
        fct = 1.  # Don't Pivot -- Bootstrapping works better if Pivot
        lam = ((b / fct) ** 2).sum() / float(n)
        return lam

    def _compute_se_lambda(self, Y, X):
        """
        Calculates the SE of lambda by nested resampling
        Used to pivot the statistic.
        Bootstrapping works better with estimating pivotal statistics
        but slows down computation significantly.
        """
        n = np.shape(Y)[0]
        lam = np.empty(shape=(self.nres, ))
        for i in xrange(self.nres):
            ind = np.random.random_integers(0, n-1, size=(n,1))
            Y1 = Y[ind, 0]
            X1 = X[ind, 0:]
            lam[i] = self._compute_lambda(Y1, X1)

        se_lambda = np.std(lam)
        return se_lambda

    def _compute_sig(self):
        """
        Computes the significance value for the variable(s) tested.

        The empirical distribution of the test statistic is obtained through
        bootstrapping the sample.  The null hypothesis is rejected if the test
        statistic is larger than the 90, 95, 99 percentiles.
        """
        t_dist = np.empty(shape=(self.nboot, ))
        Y = self.endog
        X = copy.deepcopy(self.exog)
        n = np.shape(Y)[0]

        X[:, self.test_vars] = np.mean(X[:, self.test_vars], axis=0)
        # Calculate the restricted mean. See p. 372 in [8]
        M = Reg(Y, X, self.var_type, self.model.reg_type, self.bw,
                defaults = EstimatorSettings(efficient=False)).fit()[0]
        M = np.reshape(M, (n, 1))
        e = Y - M
        e = e - np.mean(e)  # recenter residuals
        for i in xrange(self.nboot):
            ind = np.random.random_integers(0, n-1, size=(n,1))
            e_boot = e[ind, 0]
            Y_boot = M + e_boot
            t_dist[i] = self._compute_test_stat(Y_boot, self.exog)

        sig = "Not Significant"
        if self.test_stat > mquantiles(t_dist, 0.9):
            sig = "*"
        if self.test_stat > mquantiles(t_dist, 0.95):
            sig = "**"
        if self.test_stat > mquantiles(t_dist, 0.99):
            sig = "***"

        return sig


class TestRegCoefD(TestRegCoefC):
    """
    Significance test for the categorical variables in a nonparametric
    regression.

    Parameters
    ----------
    model: Instance of Reg class
        This is the nonparametric regression model whose elements
        are tested for significance.
    test_vars: tuple, list of one element
        index of position of the discrete variable to be tested
        for significance. E.g. (3) tests variable at
        position 3 for significance.
    nboot: int
        Number of bootstrap samples used to determine the distribution
        of the test statistic in a finite sample. Default is 400

    Attributes
    ----------
    sig: str
        The significance level of the variable(s) tested
        "Not Significant": Not significant at the 90% confidence level
                            Fails to reject the null
        "*": Significant at the 90% confidence level
        "**": Significant at the 95% confidence level
        "***": Significant at the 99% confidence level

    Notes
    -----
    This class currently doesn't allow joint hypothesis.
    Only one variable can be tested at a time

    References
    ----------
    See [9] and chapter 12 in [1].
    """

    def _compute_test_stat(self, Y, X):
        """Computes the test statistic"""

        dom_x = np.sort(np.unique(self.exog[:, self.test_vars]))

        n = np.shape(X)[0]
        model = Reg(Y, X, self.var_type, self.model.reg_type, self.bw,
                        defaults = EstimatorSettings(efficient=False))
        X1 = copy.deepcopy(X)
        X1[:, self.test_vars] = 0

        m0 = model.fit(data_predict=X1)[0]
        m0 = np.reshape(m0, (n, 1))
        I = np.zeros((n, 1))
        for i in dom_x[1:] :
            X1[:, self.test_vars] = i
            m1 = model.fit(data_predict=X1)[0]
            m1 = np.reshape(m1, (n, 1))
            I += (m1 - m0) ** 2

        I = I.sum(axis=0) / float(n)
        return I

    def _compute_sig(self):
        """Calculates the significance level of the variable tested"""

        m = self._est_cond_mean()
        Y = self.endog
        X = self.exog
        n = np.shape(X)[0]
        u = Y - m
        u = u - np.mean(u)  # center
        fct1 = (1 - 5**0.5) / 2.
        fct2 = (1 + 5**0.5) / 2.
        u1 = fct1 * u
        u2 = fct2 * u
        r = fct2 / (5 ** 0.5)
        I_dist = np.empty((self.nboot,1))
        for j in xrange(self.nboot):
            u_boot = copy.deepcopy(u2)

            prob = np.random.uniform(0,1, size = (n,1))
            ind = prob < r
            u_boot[ind] = u1[ind]
            Y_boot = m + u_boot
            I_dist[j] = self._compute_test_stat(Y_boot, X)

        sig = "Not Significant"
        if self.test_stat > mquantiles(I_dist, 0.9):
            sig = "*"
        if self.test_stat > mquantiles(I_dist, 0.95):
            sig = "**"
        if self.test_stat > mquantiles(I_dist, 0.99):
            sig = "***"

        return sig

    def _est_cond_mean(self):
        """
        Calculates the expected conditional mean
        m(X, Z=l) for all possible l
        """
        self.dom_x = np.sort(np.unique(self.exog[:, self.test_vars]))
        X = copy.deepcopy(self.exog)
        m=0
        for i in self.dom_x:
            X[:, self.test_vars]  = i
            m += self.model.fit(data_predict = X)[0]

        m = m / float(len(self.dom_x))
        m = np.reshape(m, (np.shape(self.exog)[0], 1))
        return m


class TestFForm(object):
    """
    Nonparametric test for functional form.

    Parameters
    ----------
    endog: list
        Dependent variable (training set)
    exog: list of array_like objects
        The independent (right-hand-side) variables
    bw: array_like, str
        Bandwidths for exog or specify method for bandwidth selection
    fform: function
        The functional form ``y = g(b, x)`` to be tested. Takes as inputs
        the RHS variables `exog` and the coefficients ``b`` (betas)
        and returns a fitted ``y_hat``.
    var_type: str
        The type of the independent `exog` variables:

            - c: continuous
            - o: ordered
            - u: unordered

    estimator: function
        Must return the estimated coefficients b (betas). Takes as inputs
        ``(endog, exog)``.  E.g. least square estimator::

            lambda (x,y): np.dot(np.pinv(np.dot(x.T, x)), np.dot(x.T, y))

    References
    ----------
    See Racine, J.: "Consistent Significance Testing for Nonparametric
    Regression" Journal of Business \& Economics Statistics.

    See chapter 12 in [1]  pp. 355-357.

    """
    def __init__(self, endog, exog, bw, var_type, fform, estimator):
        self.endog = endog
        self.exog = exog
        self.fform = fform
        self.estimator = estimator
        self.bw = KDE(exog, bw=bw, var_type=var_type).bw
        self.sig = self._compute_sig()

    def _compute_sig(self):
        Y = self.endog
        X = self.exog
        b = self.estimator(Y, X)
        m = self.fform(X, b)
        n = np.shape(X)[0]
        u = Y - m
        u = u - np.mean(u)  # center residuals
        self.test_stat = self._compute_test_stat(u)
        fct1 = (1 - 5**0.5) / 2.
        fct2 = (1 + 5**0.5) / 2.
        u1 = fct1 * u
        u2 = fct2 * u
        r = fct2 / (5 ** 0.5)
        I_dist = np.empty((self.nboot,1))
        for j in xrange(self.nboot):
            u_boot = copy.deepcopy(u2)

            prob = np.random.uniform(0,1, size = (n,1))
            ind = prob < r
            u_boot[ind] = u1[ind]
            Y_boot = m + u_boot
            b_hat = self.estimator(Y_boot, X)
            m_hat = self.fform(X, b_hat)
            u_boot_hat = Y_boot - m_hat
            I_dist[j] = self._compute_test_stat(u_boot_hat)

        sig = "Not Significant"
        if self.test_stat > mquantiles(I_dist, 0.9):
            sig = "*"
        if self.test_stat > mquantiles(I_dist, 0.95):
            sig = "**"
        if self.test_stat > mquantiles(I_dist, 0.99):
            sig = "***"
        return sig

    def _compute_test_stat(self, u):
        n = np.shape(u)[0]
        XLOO = tools.LeaveOneOut(self.exog)
        uLOO = tools.LeaveOneOut(u).__iter__()
        I = 0
        S2 = 0
        for i, X_not_i in enumerate(XLOO):
            u_j = uLOO.next()
            # See Bootstrapping procedure on p. 357 in [1]
            K = tools.gpke(self.bw, data=-X_not_i, data_predict=-X_not_i[i, :],
                           var_type=self.var_type, tosum=False)
            f_i = u[i] * u_j * K
            I += f_i  # See eq. 12.7 on p. 355 in [1]
            S2 += f_i ** 2  # See Theorem 12.1 on p.356 in [1]

        I *= 1. / (n * (n - 1))
        ix_cont = tools._get_type_pos(self.var_type)[0]
        hp = self.bw[ix_cont].prod()
        S2 *= 2 * hp / (n * (n - 1))
        T = n * I * np.sqrt(hp / S2)
        return T


class SingleIndexModel(Reg):
    """
    Single index semiparametric model ``y = g(X * b) + e``.

    Parameters
    ----------
    endog: array_like
        The dependent variable
    exog: array_like
        The independent variable(s)
    var_type: str
        The type of variables in X:

            - c: continuous
            - o: ordered
            - u: unordered

    Attributes
    ----------
    b: array_like
        The linear coefficients b (betas)
    bw: array_like
        Bandwidths

    Methods
    -------
    fit(): Computes the fitted values ``E[Y|X] = g(X * b)``
           and the marginal effects ``dY/dX``.

    References
    ----------
    See chapter on semiparametric models in [1]

    Notes
    -----
    This model resembles the binary choice models. The user knows
    that X and b interact linearly, but ``g(X * b)`` is unknown.
    In the parametric binary choice models the user usually assumes
    some distribution of g() such as normal or logistic.

    """
    def __init__(self, endog, exog, var_type):
        self.var_type = var_type
        self.K = len(var_type)
        self.endog = tools.adjust_shape(endog, 1)
        self.exog = tools.adjust_shape(exog, self.K)
        self.nobs = np.shape(self.exog)[0]
        self.data_type = self.var_type
        self.func = self._est_loc_linear

        self.b, self.bw = self._est_b_bw()

    def _est_b_bw(self):
        params0 = np.random.uniform(size=(2*self.K, ))
        b_bw = optimize.fmin(self.cv_loo, params0, disp=0)
        b = b_bw[0:self.K]
        bw = b_bw[self.K:]
        bw = self._set_bw_bounds(bw)
        return b, bw

    def cv_loo(self, params):
        # See p. 254 in Textbook
        params = np.asarray(params)
        b = params[0 : self.K]
        bw = params[self.K:]
        LOO_X = tools.LeaveOneOut(self.exog)
        LOO_Y = tools.LeaveOneOut(self.endog).__iter__()
        L = 0
        for i, X_not_i in enumerate(LOO_X):
            Y = LOO_Y.next()
            G = self.func(bw, endog=Y, exog=-b*X_not_i,
                          data_predict=-b*self.exog[i, :])[0]
            L += (self.endog[i] - G) ** 2

        # Note: There might be a way to vectorize this. See p.72 in [1]
        return L / self.nobs

    def fit(self, data_predict=None):
        if data_predict is None:
            data_predict = self.exog
        else:
            data_predict = tools.adjust_shape(data_predict, self.K)

        N_data_predict = np.shape(data_predict)[0]
        mean = np.empty((N_data_predict,))
        mfx = np.empty((N_data_predict, self.K))
        for i in xrange(N_data_predict):
            mean_mfx = self.func(self.bw, self.endog,
                                 self.b * self.exog,
                                 data_predict=self.b * data_predict[i, :])
            mean[i] = mean_mfx[0]
            mfx_c = np.squeeze(mean_mfx[1])
            mfx[i, :] = mfx_c

        return mean, mfx

    def __repr__(self):
        """Provide something sane to print."""
        repr = "Single Index Model \n"
        repr += "Number of variables: K = " + str(self.K) + "\n"
        repr += "Number of samples:   nobs = " + str(self.nobs) + "\n"
        repr += "Variable types:      " + self.var_type + "\n"
        repr += "BW selection method: cv_ls" + "\n"
        repr += "Estimator type: local constant" + "\n"
        return repr


class SemiLinear(Reg):
    """
    Semiparametric partially linear model, ``Y = Xb + g(Z) + e``.

    Parameters
    ----------
    endog: array_like
        The dependent variable
    exog: array_like
        The linear component in the regression
    exog_nonparametric: array_like
        The nonparametric component in the regression
    var_type: str
        The type of the variables in the nonparametric component;

            - c: continuous
            - o: ordered
            - u: unordered

    l_K: int
        The number of the variables that comprise the linear component.

    Attributes
    ----------
    bw: array_like
        Bandwidths for the nonparametric component exog_nonparametric
    b: array_like
        Coefficients in the linear component

    Methods
    -------
    fit(): Returns the fitted mean and marginal effects dy/dz

    Notes
    -----
    This model uses only the local constant regression estimator

    References
    ----------
    See chapter on Semiparametric Models in [1]
    """

    def __init__(self, endog, exog, exog_nonparametric, var_type, l_K):
        self.endog = tools.adjust_shape(endog, 1)
        self.exog = tools.adjust_shape(exog, l_K)
        self.K = len(var_type)
        self.exog_nonparametric = tools.adjust_shape(exog_nonparametric, self.K)
        self.l_K = l_K
        self.nobs = np.shape(self.exog)[0]
        self.var_type = var_type
        self.data_type = self.var_type
        self.func = self._est_loc_linear

        self.b, self.bw = self._est_b_bw()

    def _est_b_bw(self):
        """
        Computes the (beta) coefficients and the bandwidths.

        Minimizes ``cv_loo`` with respect to ``b`` and ``bw``.
        """
        params0 = np.random.uniform(size=(self.l_K + self.K, ))
        b_bw = optimize.fmin(self.cv_loo, params0, disp=0)
        b = b_bw[0 : self.l_K]
        bw = b_bw[self.l_K:]
        #bw = self._set_bw_bounds(np.asarray(bw))
        return b, bw

    def cv_loo(self, params):
        """
        Similar to the cross validation leave-one-out estimator.

        Modified to reflect the linear components.

        Parameters
        ----------
        params: array_like
            Vector consisting of the coefficients (b) and the bandwidths (bw).
            The first ``l_K`` elements are the coefficients.

        Returns
        -------
        L: float
            The value of the objective function

        References
        ----------
        See p.254 in [1]
        """
        params = np.asarray(params)
        b = params[0 : self.l_K]
        bw = params[self.l_K:]
        LOO_X = tools.LeaveOneOut(self.exog)
        LOO_Y = tools.LeaveOneOut(self.endog).__iter__()
        LOO_Z = tools.LeaveOneOut(self.exog_nonparametric).__iter__()
        Xb = b * self.exog
        L = 0
        for ii, X_not_i in enumerate(LOO_X):
            Y = LOO_Y.next()
            Z = LOO_Z.next()
            Xb_j = b * X_not_i
            Yx = Y - Xb_j
            G = self.func(bw, endog=Yx, exog=-Z,
                          data_predict=-self.exog_nonparametric[ii, :])[0]
            lt = Xb[ii, :].sum()  # linear term
            L += (self.endog[ii] - lt - G) ** 2

        return L

    def fit(self, exog_predict=None, exog_nonparametric_predict=None):
        """Computes fitted values and marginal effects"""

        if exog_predict is None:
            exog_predict = self.exog
        else:
            exog_predict = tools.adjust_shape(exog_predict, self.l_K)

        if exog_nonparametric_predict is None:
            exog_nonparametric_predict = self.exog_nonparametric
        else:
            exog_nonparametric_predict = tools.adjust_shape(exog_predict, self.K)

        N_data_predict = np.shape(exog_nonparametric_predict)[0]
        mean = np.empty((N_data_predict,))
        mfx = np.empty((N_data_predict, self.K))
        Y = self.endog - self.b * exog_predict
        for i in xrange(N_data_predict):
            mean_mfx = self.func(self.bw, Y, self.exog_nonparametric,
                                 data_predict=exog_predict[i, :])
            mean[i] = mean_mfx[0]
            mfx_c = np.squeeze(mean_mfx[1])
            mfx[i, :] = mfx_c

        return mean, mfx

    def __repr__(self):
        """Provide something sane to print."""
        repr = "Semiparamatric Partially Linear Model \n"
        repr += "Number of variables: K = " + str(self.K) + "\n"
        repr += "Number of samples:   N = " + str(self.nobs) + "\n"
        repr += "Variable types:      " + self.var_type + "\n"
        repr += "BW selection method: cv_ls" + "\n"
        repr += "Estimator type: local constant" + "\n"
        return repr
