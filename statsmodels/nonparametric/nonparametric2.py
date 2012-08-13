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
"""

import numpy as np
from scipy import optimize
import np_tools as tools


__all__ = ['UKDE', 'CKDE']


class _GenericKDE (object):
    """
    Generic KDE class with methods shared by both UKDE and CKDE
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

    def _normal_reference(self):
        """
        Returns Scott's normal reference rule of thumb bandwidth parameter.

        Notes
        -----
        See p.13 in [2] for an example and discussion.  The formula for the
        bandwidth is

        .. math:: h = 1.06n^{-1/(4+q)}

        where :math:`n` is the number of observations and :math:`q` is the
        number of variables.
        """
        c = 1.06
        X = np.std(self.all_vars, axis=0)
        return c * X * self.N ** (- 1. / (4 + np.size(self.all_vars, axis=1)))

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
                           maxiter=1e3, maxfun=1e3, disp=0)
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
        conditional (CKDE) and unconditional (UKDE) kernel density estimation.
        """
        h0 = self._normal_reference()
        bw = optimize.fmin(self.imse, x0=h0, maxiter=1e3, maxfun=1e3, disp=0)
        return np.abs(bw)

    def loo_likelihood(self):
        raise NotImplementedError


class UKDE(_GenericKDE):
    """
    Unconditional Kernel Density Estimator

    Parameters
    ----------
    tdat: list of ndarrays or 2-D ndarray
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
    >>> from statsmodels.nonparametric import UKDE
    >>> N = 300
    >>> np.random.seed(1234)  # Seed random generator
    >>> c1 = np.random.normal(size=(N,1))
    >>> c2 = np.random.normal(2, 1, size=(N,1))

    Estimate a bivariate distribution and display the bandwidth found:

    >>> dens_u = UKDE(tdat=[c1,c2], var_type='cc', bw='normal_reference')
    >>> dens_u.bw
    array([ 0.39967419,  0.38423292])
    """
    def __init__(self, tdat, var_type, bw=None):
        self.var_type = var_type
        self.K = len(self.var_type)
        self.tdat = tools.adjust_shape(tdat, self.K)
        self.all_vars = self.tdat
        self.N, self.K = np.shape(self.tdat)
        assert self.K == len(self.var_type)
        assert self.N > self.K  # Num of obs must be > than num of vars
        self.bw = self._compute_bw(bw)

    def __repr__(self):
        """Provide something sane to print."""
        repr = "UKDE instance\n"
        repr += "Number of variables: K = " + str(self.K) + "\n"
        repr += "Number of samples:   N = " + str(self.N) + "\n"
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
            For the log likelihood should be numpy.log

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
        LOO = tools.LeaveOneOut(self.tdat)
        L = 0
        for i, X_j in enumerate(LOO):
            f_i = tools.gpke(bw, tdat=-X_j, edat=-self.tdat[i, :],
                             var_type=self.var_type)
            L += func(f_i)

        return -L

    def pdf(self, edat=None):
        """
        Evaluate the probability density function.

        Parameters
        ----------
        edat: array_like, optional
            Points to evaluate at.  If unspecified, the training data is used.

        Returns
        -------
        pdf_est: array_like
            Probability density function evaluated at `edat`.

        Notes
        -----
        The probability density is given by the generalized product kernel
        estimator:

        .. math:: K_{h}(X_{i},X_{j}) =
            \prod_{s=1}^{q}h_{s}^{-1}k\left(\frac{X_{is}-X_{js}}{h_{s}}\right)
        """
        if edat is None:
            edat = self.tdat
        else:
            edat = tools.adjust_shape(edat, self.K)

        pdf_est = []
        N_edat = np.shape(edat)[0]
        for i in xrange(N_edat):
            pdf_est.append(tools.gpke(self.bw, tdat=self.tdat, edat=edat[i, :],
                          var_type=self.var_type) / self.N)

        pdf_est = np.squeeze(pdf_est)
        return pdf_est

    def cdf(self, edat=None):
        """
        Evaluate the cumulative distribution function.

        Parameters
        ----------
        edat: array_like, optional
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
        if edat is None:
            edat = self.tdat
        else:
            edat = tools.adjust_shape(edat, self.K)

        N_edat = np.shape(edat)[0]
        cdf_est = []
        for i in xrange(N_edat):
            cdf_est.append(tools.gpke(self.bw, tdat=self.tdat,
                             edat=edat[i, :], var_type=self.var_type,
                             ckertype="gaussian_cdf",
                             ukertype="aitchisonaitken_cdf",
                             okertype='wangryzin_cdf') / self.N)

        cdf_est = np.squeeze(cdf_est)
        return cdf_est

    def imse(self, bw):
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
        for i in range(self.N):
            k_bar_sum = tools.gpke(bw, tdat=-self.tdat, edat=-self.tdat[i, :],
                                   var_type=self.var_type,
                                   ckertype='gauss_convolution',
                                   okertype='wangryzin_convolution',
                                   ukertype='aitchisonaitken_convolution')
            F += k_bar_sum
        # there is a + because loo_likelihood returns the negative
        return (F / (self.N ** 2) + self.loo_likelihood(bw) *\
                2 / ((self.N) * (self.N - 1)))


class CKDE(_GenericKDE):
    """
    Conditional Kernel Density Estimator.

    Calculates ``P(X_1,X_2,...X_n | Y_1,Y_2...Y_m) =
    P(X_1, X_2,...X_n, Y_1, Y_2,..., Y_m)/P(Y_1, Y_2,..., Y_m)``.
    The conditional density is by definition the ratio of the two unconditional
    densities, see [1]_.

    Parameters
    ----------
    tydat: list of ndarrays or 2-D ndarray
        The training data for the dependent variables, used to determine
        the bandwidth(s).  If a 2-D array, should be of shape
        (num_observations, num_variables).  If a list, each list element is a
        separate observation.
    txdat: list of ndarrays or 2-D ndarray
        The training data for the independent variable; same shape as `tydat`.
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
    >>> N = 300
    >>> c1 = np.random.normal(size=(N,1))
    >>> c2 = np.random.normal(2,1,size=(N,1))

    >>> dens_c = CKDE(tydat=[c1], txdat=[c2], dep_type='c',
    ...               indep_type='c', bwmethod='normal_reference')

    >>> print "The bandwidth is: ", dens_c.bw
    """
    def __init__(self, tydat, txdat, dep_type, indep_type, bw=None):
        self.dep_type = dep_type
        self.indep_type = indep_type
        self.K_dep = len(self.dep_type)
        self.K_indep = len(self.indep_type)
        self.tydat = tools.adjust_shape(tydat, self.K_dep)
        self.txdat = tools.adjust_shape(txdat, self.K_indep)
        self.N, self.K_dep = np.shape(self.tydat)
        self.all_vars = np.concatenate((self.tydat, self.txdat), axis=1)
        assert len(self.dep_type) == self.K_dep
        assert len(self.indep_type) == np.shape(self.txdat)[1]
        self.bw = self._compute_bw(bw)

    def __repr__(self):
        """Provide something sane to print."""
        repr = "CKDE instance\n"
        repr += "Number of independent variables: K_indep = " + \
                str(self.K_indep) + "\n"
        repr += "Number of dependent variables: K_dep = " + \
                str(self.K_dep) + "\n"
        repr += "Number of obs:ervation   N = " + str(self.N) + "\n"
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
        Similar to ``UKDE.loo_likelihood`, but substitute
        ``f(x|y)=f(x,y)/f(y)`` for f(x).
        """
        yLOO = tools.LeaveOneOut(self.all_vars)
        xLOO = tools.LeaveOneOut(self.txdat).__iter__()
        L = 0
        for i, Y_j in enumerate(yLOO):
            X_j = xLOO.next()
            f_yx = tools.gpke(bw, tdat=-Y_j, edat=-self.all_vars[i, :],
                              var_type=(self.dep_type + self.indep_type))
            f_x = tools.gpke(bw[self.K_dep::], tdat=-X_j,
                             edat=-self.txdat[i, :], var_type=self.indep_type)
            f_i = f_yx / f_x
            L += func(f_i)
        return - L

    def pdf(self, eydat=None, exdat=None):
        """
        Evaluate the probability density function.

        Parameters
        ----------
        eydat: array_like, optional
            Evaluation data for the dependent variables.  If unspecified, the
            training data is used.
        exdat: array_like, optional
            Evaluation data for the independent variables.

        Returns
        -------
        pdf: array_like
            The value of the probability density at `eydat` and `exdat`.

        Notes
        -----
        The formula for the conditional probability density is:

        .. math:: f(X|Y)=\frac{f(X,Y)}{f(Y)}

        with

        .. math:: f(X)=\prod_{s=1}^{q}h_{s}^{-1}k
                            \left(\frac{X_{is}-X_{js}}{h_{s}}\right)

        where :math:`k` is the appropriate kernel for each variable.
        """
        if eydat is None:
            eydat = self.tydat
        else:
            eydat = tools.adjust_shape(eydat, self.K_dep)
        if exdat is None:
            exdat = self.txdat
        else:
            exdat = tools.adjust_shape(exdat, self.K_indep)

        pdf_est = []
        edat = np.concatenate((eydat, exdat), axis=1)
        N_edat = np.shape(edat)[0]
        for i in xrange(N_edat):
            f_yx = tools.gpke(self.bw, tdat=self.all_vars, edat=edat[i, :],
                              var_type=(self.dep_type + self.indep_type))
            f_x = tools.gpke(self.bw[self.K_dep::], tdat=self.txdat,
                             edat=exdat[i, :], var_type=self.indep_type)
            pdf_est.append(f_yx / f_x)

        return np.squeeze(pdf_est)

    def cdf(self, eydat=None, exdat=None):
        """
        Cumulative distribution function for the conditional density.

        Parameters
        ----------
        eydat: array_like, optional
            The evaluation dependent variables at which the cdf is estimated.
            If not specified the training dependent variables are used.
        exdat: array_like, optional
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
        if eydat is None:
            eydat = self.tydat
        else:
            eydat = tools.adjust_shape(eydat, self.K_dep)
        if exdat is None:
            exdat = self.txdat
        else:
            exdat = tools.adjust_shape(exdat, self.K_indep)

        N_edat = np.shape(exdat)[0]
        cdf_est = np.empty(N_edat)
        for i in xrange(N_edat):
            mu_x = tools.gpke(self.bw[self.K_dep::], tdat=self.txdat,
                              edat=exdat[i, :], var_type=self.indep_type) / self.N
            mu_x = np.squeeze(mu_x)
            G_y = tools.gpke(self.bw[0:self.K_dep], tdat=self.tydat,
                             edat=eydat[i, :], var_type=self.dep_type,
                             ckertype="gaussian_cdf",
                             ukertype="aitchisonaitken_cdf",
                             okertype='wangryzin_cdf', tosum=False)

            W_x = tools.gpke(self.bw[self.K_dep::], tdat=self.txdat,
                         edat=exdat[i, :], var_type=self.indep_type,
                         tosum=False)
            S = np.sum(G_y * W_x, axis=0)
            cdf_est[i] = S / (self.N * mu_x)

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
            The cross-validation objective function

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
        zLOO = tools.LeaveOneOut(self.all_vars)
        CV = 0
        for l, Z in enumerate(zLOO):
            X = Z[:, self.K_dep::]
            Y = Z[:, 0:self.K_dep]
            Expander = np.ones((self.N - 1, 1))
            Ye_L = np.kron(Y, Expander)
            Ye_R = np.kron(Expander, Y)
            Xe_L = np.kron(X, Expander)
            Xe_R = np.kron(Expander, X)
            K_Xi_Xl = tools.gpke(bw[self.K_dep::], tdat=Xe_L,
                                 edat=self.txdat[l, :],
                                 var_type=self.indep_type, tosum=False)
            K_Xj_Xl = tools.gpke(bw[self.K_dep::], tdat=Xe_R,
                                 edat=self.txdat[l, :],
                                 var_type=self.indep_type, tosum=False)
            K2_Yi_Yj = tools.gpke(bw[0:self.K_dep], tdat=Ye_L,
                                  edat=Ye_R, var_type=self.dep_type,
                                  ckertype='gauss_convolution',
                                  okertype='wangryzin_convolution',
                                  ukertype='aitchisonaitken_convolution',
                                  tosum=False)
            G = np.sum(K_Xi_Xl * K_Xj_Xl * K2_Yi_Yj)
            G = G / self.N ** 2
            f_X_Y = tools.gpke(bw, tdat=-Z, edat=-self.all_vars[l, :],
                               var_type=(self.dep_type +
                                         self.indep_type)) / float(self.N)
            m_x = tools.gpke(bw[self.K_dep::], tdat=-X, edat=-self.txdat[l, :],
                             var_type=self.indep_type) / float(self.N)
            CV += (G / m_x ** 2) - 2 * (f_X_Y / m_x)

        return CV / float(self.N)
