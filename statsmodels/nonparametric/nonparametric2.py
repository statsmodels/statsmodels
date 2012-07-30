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
"""

import numpy as np
from scipy import integrate, stats
import np_tools as tools
import scipy.optimize as opt
import KernelFunctions as kf

__all__ = ['UKDE', 'CKDE', 'Reg']


class GenericKDE (object):
    """
    Generic KDE class with methods shared by both UKDE and CKDE
    """
    def compute_bw(self, bw):
        """
        Computes the bandwidth of the data

        Parameters
        ----------
        bw: array-like or string
            If array-like: user-specified bandwidth.
            If string:
            cv_ml: cross validation maximum likelihood
            normal_reference: normal reference rule of thumb
            cv_ls: cross validation least squares

        Notes
        ----------
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
            # The user specified a bandwidth selection
            # method e.g. 'normal-reference'
            self._bw_method = bw
            bwfunc = self.bw_func[bw]
            res = bwfunc()
        return res

    def _normal_reference(self):
        """
        Returns Scott's normal reference rule of thumb bandwidth parameter

        Notes
        -----
        See p.13 in [2] for en example and discussion

        The formula for the bandwidth is

        .. math:: h = 1.06n^{-1/(4+q)}

        where :math:`n` is the number of observations and :math:`q`
        is the number of variables

        """
        c = 1.06
        X = np.std(self.all_vars, axis=0)
        return c * X * self.N ** (- 1. / (4 + np.size(self.all_vars, axis=1)))

    def _cv_ml(self):
        """
        Returns the cross validation maximum likelihood bandwidth parameter

        Notes
        -----
        For more details see p.16, 18, 27 in [1]

        Returns the bandwidth estimate that maximizes the
        leave-out-out likelihood
        The leave-one-out log likelihood function is:

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
        bw = opt.fmin(self.loo_likelihood, x0=h0, args=(np.log, ),
                      maxiter=1e3, maxfun=1e3, disp=0)
        return bw

    def _cv_ls(self):
        """
        Returns the cross-validation least squares bandwidth parameter(s)

        Notes
        -----
        For more details see pp. 16, 27 in [1]

        Returns the value of the bandwidth that maximizes
        the integrated mean square error
        between the estimated and actual distribution.
        The integrated mean square error is given by:

        .. math:: \int\left[\hat{f}(x)-f(x)\right]^{2}dx

        This is the general formula for the imse.
        The imse differes for conditional (CKDE) and
        unconditional (UKDE) kernel density estimation.
        """
        h0 = self._normal_reference()
        bw = opt.fmin(self.imse, x0=h0, maxiter=1e3,
                      maxfun=1e3, disp=0)
        return np.abs(bw)

    def loo_likelihood(self):
        pass


class UKDE(GenericKDE):
    """
    Unconditional Kernel Density Estimator

    Parameters
    ----------
    tdat: list
        The training data for the Kernel Density Estimation.
        Each element of the list
        is an array-like seperate variable

    var_type: str
        The type of the variables
        c: Continuous
        u: Unordered (Discrete)
        o: Ordered (Discrete)

    bw: array-like
        Either a user-specified bandwidth.
        The method for bandwidth selection.
        cv_ml: cross validation maximum likelihood
        normal_reference: normal reference rule of thumb
        cv_ls: cross validation least squares

    Attributes
    ----------
    bw: array-like
        The bandwidth parameters.

    Methods
    -------
    pdf(): the probability density function
    cdf(): the cumulative distribution function
    imse(): the integrated mean square error
    loo_likelihood(): the leave one out likelihood

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

        #self.tdat = np.column_stack(tdat)
        self.var_type = var_type
        self.K = len(self.var_type)
        self.tdat = tools.adjust_shape(tdat, self.K)
        self.all_vars = self.tdat
        self.N, self.K = np.shape(self.tdat)
        assert self.K == len(self.var_type)
        assert self.N > self.K  # Num of obs must be > than num of vars
        self.bw = self.compute_bw(bw)

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
        Returns the leave-one-out likelihood function

        The leave-one-out likelihood function for the unconditional KDE

        Parameters
        ----------
        bw: array-like
            The value for the bandwdith parameter(s)
        func: function
            For the log likelihood should be numpy.log

        Notes
        -----
        The leave-one-out kernel estimator of :math:`f_{-i}` is:

        .. math:: f_{-i}(X_{i})=\frac{1}{(n-1)h}
        \sum_{j=1,j\neq i}K_{h}(X_{i},X_{j})

        where :math:`K_{h}` represents the
        Generalized product kernel estimator:

        .. math:: K_{h}(X_{i},X_{j})=
        \prod_{s=1}^{q}h_{s}^{-1}k\left(\frac{X_{is}-X_{js}}{h_{s}}\right)
        """

        LOO = tools.LeaveOneOut(self.tdat)
        i = 0
        L = 0
        for X_j in LOO:
            f_i = tools.gpke(bw, tdat=-X_j, edat=-self.tdat[i, :],
                             var_type=self.var_type)
            i += 1
            L += func(f_i)
        return -L

    def pdf(self, edat=None):
        """
        Probability density function at edat

        Parameters
        ----------
        edat: array-like
            Evaluation data.
            If unspecified, the training data is used

        Returns
        -----------
        pdf_est: array-like
            Probability density function evaluated at edat
        ix: array-like, optional
            The index of sorting if issorted=True
        Notes
        -----
        The probability density is given by the generalized
        product kernel estimator:

        .. math:: K_{h}(X_{i},X_{j})=
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
        Cumulative distribution function

        Parameters
        ----------
        edat: array_like
            The evaluation points at which the cdf is estimated
            If not specified the default value is the training
            data tdat
        Returns
        -------
        cdf_est: array_like
            The estimate of the cdf

        Notes
        -----
        See http://en.wikipedia.org/wiki/Cumulative_distribution_function
        For more details on the estimation see [5]

        The multivariate CDF for mixed data
        (continuous and ordered/unordered discrete) is estimated by:

        ..math:: F(x^{c},x^{d})=n^{-1}\sum_{i=1}^{n}\left[G(
        \frac{x^{c}-X_{i}}{h})\sum_{u\leq x^{d}}L(X_{i}^{d},x_{i}^{d},
        \lambda)\right]

        where G() is the product kernel CDF estimator for the continuous
        variables and L() is for the discrete
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
        Returns the Integrated Mean Square Error for the unconditional UKDE

        Parameters
        ----------
        bw: array-like
            The bandwidth parameter(s)
        Returns
        ------
        CV: float
            The cross-validation objective function

        Notes
        -----
        See p. 27 in [1]
        For details on how to handle the multivariate
        estimation with mixed data types see p.6 in [3]

        The formula for the cross-validation objective function is:

        .. math:: CV=\frac{1}{n^{2}}\sum_{i=1}^{n}\sum_{j=1}^{N}
        \bar{K}_{h}(X_{i},X_{j})-\frac{2}{n(n-1)}\sum_{i=1}^{n}
        \sum_{j=1,j\neq i}^{N}K_{h}(X_{i},X_{j})

        Where :math:`\bar{K}_{h}` is the multivariate
        product convolution kernel
        (consult [3] for mixed data types)
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


class CKDE(GenericKDE):
    """
    Conditional Kernel Density Estimator

    Calculates P(X_1,X_2,...X_n | Y_1,Y_2...Y_m) =
    P(X_1, X_2,...X_n, Y_1, Y_2,..., Y_m)/P(Y_1, Y_2,..., Y_m)
    The conditional density is by definition the ratio of
    the two unconditional densities
    http://en.wikipedia.org/wiki/Conditional_probability_distribution

    Parameters
    ----------
    tydat: list
        The training data for the dependent variable.
        Each element of the list is a seperate variable
    txdat: list
        The training data for the independent variable
    dep_type: str
        The type of the dependent variables
        c: Continuous
        u: Unordered (Discrete)
        o: Ordered (Discrete)
    indep_type: str
        The type of the independent variables
        same as dep_type

    bw: array-like
        Either a user-specified bandwidth.
        The method for bandwidth selection.
        cv_ml: cross validation maximum likelihood
        normal_reference: normal reference rule of thumb
        cv_ls: cross validation least squares

    Attributes
    ---------
    bw: array-like
        The bandwidth parameters

    Methods
    -------
    pdf(): the probability density function

    Example
    --------
    import numpy as np
    N=300
    c1=np.random.normal(size=(N,1))
    c2=np.random.normal(2,1,size=(N,1))

    dens_c=UKDE(tydat=[c1],txdat=[c2],dep_type='c',
    indep_type='c',bwmethod='normal_reference')

    print "The bandwdith is: ", dens_c.bw
    """

    def __init__(self, tydat, txdat, dep_type, indep_type, bw):

#        self.tydat = np.column_stack(tydat)
#        self.txdat = np.column_stack(txdat)
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
        self.bw = self.compute_bw(bw)

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
        Returns the leave-one-out function for the data

        Parameters
        ----------
        bw: array-like
            The bandwidth parameter(s)
        func: function
            Should be np.log for the log likelihood

        Returns
        -------
        L: float
            The value of the leave-one-out function for the data

        Notes
        -----
        Similar to the loo_likelihood in GenericKDE but
        substitute for f(x), f(x|y)=f(x,y)/f(y)
        """
        yLOO = tools.LeaveOneOut(self.all_vars)
        xLOO = tools.LeaveOneOut(self.txdat).__iter__()
        i = 0
        L = 0
        for Y_j in yLOO:
            X_j = xLOO.next()
            f_yx = tools.gpke(bw, tdat=-Y_j, edat=-self.all_vars[i, :],
                              var_type=(self.dep_type + self.indep_type))
            f_x = tools.gpke(bw[self.K_dep::], tdat=-X_j,
                             edat=-self.txdat[i, :], var_type=self.indep_type)
            f_i = f_yx / f_x
            i += 1
            L += func(f_i)
        return - L

    def pdf(self, eydat=None, exdat=None):
        """
        Returns the probability density function

        Parameters
        ----------
        eydat: array-like
            Evaluation data for the dependent variables.
            If unspecified, the training data is used
        exdat: array-like
            Evaluation data for the independent variables

        Returns
        -------
        pdf: array-like
            The value of the probability density at eydat and exdat

        Notes
        -----
        The formula for the conditional probability density is:

        .. math:: f(X|Y)=\frac{f(X,Y)}{f(Y)}

        with

        .. math:: f(X)=\prod_{s=1}^{q}h_{s}^{-1}k\left
        (\frac{X_{is}-X_{js}}{h_{s}}\right)

        where :math:`k` is the appropriate kernel for each variable
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
        Cumulative distribution function for
        the conditional density

        Parameters
        ----------
        eydat: array_like
            The evaluation dependent variables at which the cdf is estimated
            If not specified the default value is the training dependent
            variables
        exdat: array_like
            The evaluation independent variables at which the cdf is estimated
            If not specified the default value is the training independent
            variables

        Returns
        -------
        cdf_est: array_like
            The estimate of the cdf

        Notes
        -----
        See http://en.wikipedia.org/wiki/Cumulative_distribution_function
        For more details on the estimation see [5] and p.181 in [1]

        The multivariate conditional CDF for mixed data
        (continuous and ordered/unordered discrete) is estimated by:

        ..math:: F(y|x)=\frac{n^{-1}\sum_{i=1}^{n}G(\frac{y-Y_{i}}{h_{0}})
        W_{h}(X_{i},x)}{\widehat{\mu}(x)}

        where G() is the product kernel CDF
        estimator for the dependent (y) variable
        and W() is the product kernel CDF estimator
        for the independent variable(s)
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
                         edat=exdat[i, :], var_type=self.indep_type, tosum=False)
            S = np.sum(G_y * W_x, axis=0)
            cdf_est[i] = S / (self.N * mu_x)
        return cdf_est

    def imse(self, bw):
        """
        The integrated mean square error for the conditional KDE

        Parameters
        ----------
        bw: array-like
            The bandwidth parameter(s)

        Returns
        -------
        CV: float
            The cross-validation objective function

        Notes
        -----

        For more details see pp. 156-166 in [1]
        For details on how to handel the mixed variable types see [3]

        The formula for the cross-validation objective
        function for mixed variable types is:

        .. math:: CV(h,\lambda)=\frac{1}{n}\sum_{l=1}^{n}
        \frac{G_{-l}(X_{l})}{\left[\mu_{-l}(X_{l})\right]^{2}}-
        \frac{2}{n}\sum_{l=1}^{n}\frac{f_{-l}(X_{l},Y_{l})}{\mu_{-l}(X_{l})}

        where

        .. math:: G_{-l}(X_{l})=
        n^{-2}\sum_{i\neq l}\sum_{j\neq l}K_{X_{i},X_{l}}
        K_{X_{j},X_{l}}K_{Y_{i},Y_{j}}^{(2)}

        where :math:`K_{X_{i},X_{l}}` is the multivariate product kernel
        and :math:`\mu_{-l}(X_{l})` is
        the leave-one-out estimator of the pdf

        :math:`K_{Y_{i},Y_{j}}^{(2)}` is the convolution kernel

        The value of the function is minimized by _cv_ls method of the
        GenericKDE class to return the bw estimates that minimize
        the distance between the estimated and "true" probability density
        """

        zLOO = tools.LeaveOneOut(self.all_vars)
        l = 0
        CV = 0
        for Z in zLOO:
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
            l += 1
            CV += (G / m_x ** 2) - 2 * (f_X_Y / m_x)
        return CV / float(self.N)


class Reg(object):
    """
    Nonparametric Regression

    Calculates the condtional mean E[y|X] where y = g(X) + e

    Parameters
    ----------
    tydat: list with one element which is array_like
        This is the dependent variable.
    txdat: list
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

    Attributes
    ---------
    bw: array-like
        The bandwidth parameters

    Methods
    -------
    r-squared(): Calculates the R-Squared for the model
    mean(): Calculates the conditiona mean
    """

    def __init__(self, tydat, txdat, var_type, reg_type, bw='cv_ls'):

        #self.tydat = np.column_stack(tydat)
        #self.txdat = np.column_stack(txdat)
        self.var_type = var_type
        self.reg_type = reg_type
        self.K = len(self.var_type)
        self.tydat = tools.adjust_shape(tydat, 1)
        self.txdat = tools.adjust_shape(txdat, self.K)
        self.N = np.shape(self.txdat)[0] 
        self.bw_func = dict(cv_ls=self.cv_loo, aic=self.aic_hurvich)
        self.est = dict(lc=self.g_lc, ll=self.g_ll)
        self.bw = self.compute_bw(bw)

    def __repr__(self):
        """Provide something sane to print."""
        repr = "Reg instance\n"
        repr += "Number of variables: K = " + str(self.K) + "\n"
        repr += "Number of samples:   N = " + str(self.N) + "\n"
        repr += "Variable types:      " + self.var_type + "\n"
        repr += "BW selection method: " + self._bw_method + "\n"
        repr += "Estimator type: " + self.reg_type + "\n"
        return repr

    def g_ll(self, bw, tydat, txdat, edat):
        """
        Local linear estimator of g(x) in the regression
        y = g(x) + e

        Parameters
        ----------
        bw: array_like
            Vector of bandwidth value(s)
        tydat: 1D array_like
            The dependent variable
        txdat: 1D or 2D array_like
            The independent variable(s)
        edat: 1D array_like of length K, where K is
            the number of variables. The point at which
            the density is estimated

        Returns
        -------
        D_x: array_like
            The value of the conditional mean at edat

        Notes
        -----
        See p. 81 in [1] and p.38 in [2] for the formulas
        Unlike other methods, this one requires that edat be 1D
        """

        Ker = tools.gpke(bw, tdat=txdat, edat=edat, var_type=self.var_type,
                            ukertype='aitchison_aitken_reg',
                            okertype='wangryzin_reg', 
                            tosum=False)
        # Create the matrix on p.492 in [7], after the multiplication w/ K_h,ij
        # See also p. 38 in [2]
        iscontinuous = tools._get_type_pos(self.var_type)[0]
        iscontinuous = xrange(self.K)  # Use all vars instead of continuous only
        Ker = np.reshape(Ker, np.shape(tydat))  # FIXME: try to remove for speed
        N, Qc = np.shape(txdat[:, iscontinuous])
        Ker = Ker / float(N)
        L = 0
        R = 0
        M12 = (txdat[:, iscontinuous] - edat[:, iscontinuous])
        M22 = np.dot(M12.T, M12 * Ker)
        M22 = np.reshape(M22, (Qc, Qc))
        M12 = np.sum(M12 * Ker , axis=0)
        M12 = np.reshape(M12, (1, Qc))
        M21 = M12.T
        M11 = np.sum(np.ones((N,1)) * Ker, axis=0)
        M11 = np.reshape(M11, (1,1))
        M_1 = np.concatenate((M11, M12), axis=1)
        M_2 = np.concatenate((M21, M22), axis=1)
        M = np.concatenate((M_1, M_2), axis=0)
        V1 = np.sum(np.ones((N,1)) * Ker * tydat, axis=0)
        V2 = (txdat[:, iscontinuous] - edat[:, iscontinuous])
        V2 = np.sum(V2 * Ker * tydat , axis=0)
        V1 = np.reshape(V1, (1,1))
        V2 = np.reshape(V2, (Qc, 1))

        V = np.concatenate((V1, V2), axis=0)
        assert np.shape(M) == (Qc + 1, Qc + 1)
        assert np.shape(V) == (Qc + 1, 1)
        mean_mfx = np.dot(np.linalg.pinv(M), V)
        mean = mean_mfx[0]
        mfx = mean_mfx[1::, :]
        return mean, mfx

    def g_lc(self, bw, tydat, txdat, edat):
        """
        Local constant estimator of g(x) in the regression
        y = g(x) + e

        Parameters
        ----------
        bw: array_like
            Vector of bandwidth value(s)
        tydat: 1D array_like
            The dependent variable
        txdat: 1D or 2D array_like
            The independent variable(s)
        edat: 1D or 2D array_like
            The point(s) at which
            the density is estimated

        Returns
        -------
        G: array_like
            The value of the conditional mean at edat

        """
        KX = tools.gpke(bw, tdat=txdat, edat=edat,
                        var_type=self.var_type,
                            #ukertype='aitchison_aitken_reg',
                            #okertype='wangryzin_reg', 
                            tosum=False)
        KX = np.reshape(KX, np.shape(tydat))
        G_numer = np.sum(tydat * KX, axis=0)
        G_denom = np.sum(tools.gpke(bw, tdat=txdat, edat=edat,
                                    var_type=self.var_type,
                            #ukertype='aitchison_aitken_reg',
                            #okertype='wangryzin_reg', 
                            tosum=False), axis=0)
        G = G_numer / G_denom
        B_x = np.ones((self.K))
        N, K = np.shape(txdat)
        f_x = np.sum(KX, axis=0) / float(N)
        iscontinuous = tools._get_type_pos(self.var_type)[0]
        txdat_c = txdat[:, iscontinuous]
        edat_c = edat[:, iscontinuous]
        #Kc = len(edat_c)
        #KX_c = tools.gpke(bw[:, iscontinuous], tdat=txdat_c, edat=edat_c,
        #                var_type='c' * Kc,
        #                    ckertype='d_gaussian',
        #                    tosum=False)
        KX_c = tools.gpke(bw, tdat=txdat, edat=edat,
                        var_type=self.var_type,
                            ckertype='d_gaussian',
                            #okertype='wangryzin_reg', 
                            tosum=False)

        KX_c = np.reshape(KX_c, (N, 1))
        d_mx = - np.sum(tydat * KX_c, axis=0) / float(N) #* np.prod(bw[:, iscontinuous]))
        d_fx = - np.sum(KX_c, axis=0) / float(N) #* np.prod(bw[:, iscontinuous]))
        B_x = d_mx / f_x - G * d_fx / f_x
        m_x = G_numer
        B_x = (G_numer * d_fx - G_denom * d_mx)/(G_denom**2)
        #B_x = (f_x * d_mx - m_x * d_fx) / (f_x ** 2)
        return G, B_x

    def aic_hurvich(self, bw, func=None):
        print "running"
        H = np.empty((self.N, self.N))
        for i in range(self.N):
            for j in range(self.N):
                txdat=np.reshape(self.txdat[i, :], (1, self.K))
                exdat = np.reshape(self.txdat[j,:], (1, self.K))
                H[i,j] = tools.gpke(bw, tdat=txdat, edat=exdat,
                                var_type=self.var_type,
                                #ukertype='aitchison_aitken_reg',
                                #okertype='wangryzin_reg', 
                                tosum=True)
                denom = tools.gpke(bw, tdat=-self.txdat, edat=-txdat,
                                var_type=self.var_type,
                                #ukertype='aitchison_aitken_reg',
                                #okertype='wangryzin_reg', 
                                tosum=True)
                H[i,j] = H[i,j]/denom
 
        I = np.eye(self.N)
        sig = np.dot(np.dot(self.tydat.T,(I - H).T), (I - H)) * self.tydat / float(self.N)
        frac = (1 + np.trace(H)/float(self.N)) / (1 - (np.trace(H) +2)/float(self.N))
        aic = np.log(sig) + frac

    def aic_hurvich_fast(self, bw, func=None):
        H = np.empty((self.N, self.N))
        for j in range(self.N):
            H[:, j] = tools.gpke(bw, tdat=self.txdat, edat=self.txdat[j,:],
                            var_type=self.var_type, tosum=False)
        
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
            Can be either g_lc(local constant) or g_ll(local_linear)

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
        #print "Running"
        LOO_X = tools.LeaveOneOut(self.txdat)
        LOO_Y = tools.LeaveOneOut(self.tydat).__iter__()
        i = 0
        L = 0
        for X_j in LOO_X:
            Y = LOO_Y.next()
            G = func(bw, tydat=Y, txdat=-X_j, edat=-self.txdat[i, :])[0]
            L += (self.tydat[i] - G) ** 2
            i += 1
        # Note: There might be a way to vectorize this. See p.72 in [1]
        return L / self.N

    def compute_bw(self, bw):
        if not isinstance(bw, basestring):
            self._bw_method = "user-specified"
            return np.asarray(bw)
        else:
            # The user specified a bandwidth selection
            # method e.g. 'cv_ls'
            self._bw_method = bw
            res = self.bw_func[bw]
            X = np.std(self.txdat, axis=0)
            h0 = 1.06 * X * \
                 self.N ** (- 1. / (4 + np.size(self.txdat, axis=1)))
        func = self.est[self.reg_type]
        return opt.fmin(res, x0=h0, args=(func, ), maxiter=1e3,
                      maxfun=1e3, disp=0)

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

        where :math:`\hat{Y_{i}}` are the
        fitted values calculated in self.mean()
        """
        Y = np.squeeze(self.tydat)
        Yhat = self.fit()[0]
        Y_bar = np.mean(Yhat)
        R2_numer = (np.sum((Y - Y_bar) * (Yhat - Y_bar)) ** 2)
        R2_denom = np.sum((Y - Y_bar) ** 2, axis=0) * \
                   np.sum((Yhat - Y_bar) ** 2, axis=0)
        return R2_numer / R2_denom

    def fit(self, edat=None):
        """
        Returns the marginal effects at the edat points
        """
        func = self.est[self.reg_type]
        if edat is None:
            edat = self.txdat
        else:
            edat = tools.adjust_shape(edat, self.K)
        N_edat = np.shape(edat)[0]
        mean = np.empty((N_edat,))
        mfx = np.empty((N_edat, self.K))
        for i in xrange(N_edat):
            mean_mfx = func(self.bw, self.tydat, self.txdat, edat=edat[i, :])
            mean[i] = mean_mfx[0]
            mfx_c = np.squeeze(mean_mfx[1])
            mfx[i, :] = mfx_c
        return mean, mfx
