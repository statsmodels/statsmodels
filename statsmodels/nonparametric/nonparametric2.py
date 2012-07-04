"""
Multivariate Conditional and Unconditional Kernel Density Estimation
with Mixed Data Types

References
----------
[1] Racine, J., Li, Q. Nonparametric econometrics: theory and practice. Princeton
    University Press. (2007)
[2] Racine, Jeff. "Nonparametric Econometrics: A Primer," Foundation and
    Trends in Econometrics: Vol 3: No 1, pp1-88. (2008)
    http://dx.doi.org/10.1561/0800000009
[3] Racine, J., Li, Q. "Nonparametric Estimation of Distributions with Categorical
    and Continuous Data." Working Paper. (2000)
[4] Racine, J. Li, Q. "Kernel Estimation of Multivariate Conditional Distributions
    Annals of Economics and Finance 5, 211-235 (2004)

"""

import numpy as np
from scipy import integrate, stats
import np_tools as tools
import scipy.optimize as opt


__all__ = ['UKDE', 'CKDE']


class Generic_KDE ():
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

##        if type(bw) != str:  # The user provided an actual bandwidth estimate
##            return np.asarray(bw)
##        else:  # The user specified a bandwidth selection method e.g. 'normal-reference'
##            res = bwfunc()
        if not isinstance(bw, basestring):
            # The user provided an actual bandwidth estimate
            # TODO: would be good if the user could provide a function here
            # that uses tdat/N/K, instead of just a result.
            self._bw_method = "user-specified"
            res = np.asarray(bw)
        else:
            # The user specified a bandwidth selection method e.g. 'normal-reference'
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
        
        where :math:`n` is the number of observations and :math:`q` is the number of
        variables

        
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
        
        Returns the bandwidth estimate that maximizes the leave-out-out likelihood
        The leave-one-out log likelihood function is:

        .. math:: \ln L=\sum_{i=1}^{n}\ln f_{-i}(X_{i})

        The leave-one-out kernel estimator of :math:`f_{-i}` is:
        
        .. math:: f_{-i}(X_{i})=\frac{1}{(n-1)h}\sum_{j=1,j\neq i}K_{h}(X_{i},X_{j})

        where :math:`K_{h}` represents the Generalized product kernel estimator:

        .. math:: K_{h}(X_{i},X_{j})=\prod_{s=1}^{q}h_{s}^{-1}k\left(\frac{X_{is}-X_{js}}{h_{s}}\right)
        
        """
        h0 = self._normal_reference()  # the initial value for the optimization is the normal_reference
        bw = opt.fmin(self.loo_likelihood, x0=h0, args=(np.log, ),
                      maxiter=1e3, maxfun=1e3, disp=0)
        return bw
    def _cv_ls (self):
        """
        Returns the cross-validation least squares bandwidth parameter(s)

        Notes
        -----
        For more details see pp. 16, 27 in [1]

        Returns the value of the bandwidth that maximizes the integrated mean square error
        between the estimated and actual distribution.
        The integrated mean square error is given by:
        
        .. math:: \int\left[\hat{f}(x)-f(x)\right]^{2}dx

        This is the general formula for the IMSE. The IMSE differes for conditional (CKDE) and
        unconditional (UKDE) kernel density estimation.
        
        """
        h0 = self._normal_reference()
        bw = opt.fmin(self.IMSE, x0=h0, maxiter=1e3,
                      maxfun=1e3, disp=0)
        return np.abs(bw)  # Getting correct but negative values for bw. from time to time . Why?

    def loo_likelihood(self):
        pass


class UKDE(Generic_KDE):
    """
    Unconditional Kernel Density Estimator

    Parameters
    ----------
    tdat: list
        The training data for the Kernel Density Estimation. Each element of the list
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
    ---------
    bw: array-like
        The bandwidth parameters.

    Methods
    -------
    pdf(): the probability density function
    cdf(): the cumulative distribution function
    IMSE(): the integrated mean square error
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

        self.tdat = np.column_stack(tdat)
        self.all_vars = self.tdat
        self.N, self.K = np.shape(self.tdat)
        self.var_type = var_type
        assert self.K == len(self.var_type)
        assert self.N > self.K  # Num of obs must be > than num of vars
        self.bw = self.compute_bw(bw)

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
        
        .. math:: f_{-i}(X_{i})=\frac{1}{(n-1)h}\sum_{j=1,j\neq i}K_{h}(X_{i},X_{j})

        where :math:`K_{h}` represents the Generalized product kernel estimator:

        .. math:: K_{h}(X_{i},X_{j})=\prod_{s=1}^{q}h_{s}^{-1}k\left(\frac{X_{is}-X_{js}}{h_{s}}\right)
        
        """
        LOO = tools.LeaveOneOut(self.tdat)
        i = 0
        L = 0
        for X_j in LOO:
            f_i = tools.GPKE(bw, tdat=-X_j, edat=-self.tdat[i, :],
                             var_type=self.var_type)
            i += 1
            L += func(f_i)
        return -L

    def pdf(self, edat=None, issorted=False):
        """
        Probability density function at edat

        Parameters
        ----------
        edat: array-like
            Evaluation data.
            If unspecified, the training data is used
        issorted: Boolean
            User specifies whether to return a sorted array of the pdf

        Returns
        -----------
        pdf_est: array-like
            Probability density function evaluated at edat
        ix: array-like, optional
            The index of sorting if issorted=True
        Notes
        -----
        The probability density is given by the generalized product kernel estimator
        
        .. math:: K_{h}(X_{i},X_{j})=\prod_{s=1}^{q}h_{s}^{-1}k\left(\frac{X_{is}-X_{js}}{h_{s}}\right)
        
        """
        if edat is None:
            edat = self.tdat
        
        pdf_est = tools.GPKE(self.bw, tdat=self.tdat, edat=edat,
                          var_type=self.var_type) / self.N
        pdf_est = np.squeeze(pdf_est)
##        if issorted:
##            ix = np.argsort(np.squeeze(edat))
##            pdf_est = pdf_est[ix]
##            return pdf_est, ix
        
        return pdf_est
    def IMSE(self, bw):
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
        For details on how to handle the multivariate estimation with mixed data types see p.6 in [3]

        The formula for the cross-validation objective function is:
        
        .. math:: CV=\frac{1}{n^{2}}\sum_{i=1}^{n}\sum_{j=1}^{N}\bar{K}_{h}(X_{i},X_{j})-\frac{2}{n(n-1)}\sum_{i=1}^{n}\sum_{j=1,j\neq i}^{N}K_{h}(X_{i},X_{j})

        Where :math:`\bar{K}_{h}` is the multivariate product convolution kernel (consult [3] for mixed data types)
        """
        
        #print "running..."
        F = 0
        for i in range(self.N):
            k_bar_sum = tools.GPKE(bw, tdat=-self.tdat, edat=-self.tdat[i, :],
                                    var_type=self.var_type, ckertype='gauss_convolution', okertype='wangryzin_convolution',
                                    ukertype='aitchisonaitken_convolution')
            F += k_bar_sum
        # there is a + because loo_likelihood returns the negative
        return (F / (self.N ** 2) + self.loo_likelihood(bw) * 2 / ((self.N) * (self.N - 1)))

    def cdf(self, val):
        """
        Returns the cumulative distribution function evaluated at val

        Currently only works with up to three continuous variables

        Parameters
        ----------
        val: list of floats
            The values at which the cdf is estimated
            
        Returns
        -------
        cdf: array
            The estimate of the the cdf evaluated at the values specified in val
            
        """
        # TODO: 1)Include option to fix certain variables
        #       2)Handle ordered variables
        #       3)Handle more than 3 continuous (with warning for speed)
        #       4)Hande unordered variables by fixing


        if self.K == 1:
            func = tools.IntegrateSingle
            n_edat = np.size(val)
        elif self.K == 2:
            func = tools.IntegrateDbl
        elif self.K == 3:
            func = tools.IntegrateTrpl
        return func(val, self.pdf)

    def __repr__(self):
        """Provide something sane to print."""
        repr = "UKDE instance\n"
        repr += "Number of variables: K = " + str(self.K) + "\n"
        repr += "Number of samples:   N = " + str(self.N) + "\n"
        repr += "Variable types:      " + self.var_type + "\n"
        repr += "BW selection method: " + self._bw_method + "\n"
        return repr

class CKDE(Generic_KDE):
    """
    Conditional Kernel Density Estimator

    Calculates P(X_1,X_2,...X_n | Y_1,Y_2...Y_m) = P(X_1, X_2,...X_n, Y_1, Y_2,..., Y_m)/P(Y_1, Y_2,..., Y_m)
    The conditional density is by definition the ratio of the two unconditional densities
    http://en.wikipedia.org/wiki/Conditional_probability_distribution

    Parameters
    ----------
    tydat: list
        The training data for the dependent variable. Each element of the list
        is a seperate variable
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

    dens_c=UKDE(tydat=[c1],txdat=[c2],dep_type='c',indep_type='c',bwmethod='normal_reference')

    print "The bandwdith is: ", dens_c.bw
    """

    def __init__(self, tydat, txdat, dep_type, indep_type, bw):

        self.tydat = np.column_stack(tydat)
        self.txdat = np.column_stack(txdat)
        self.N, self.K_dep = np.shape(self.tydat)
        self.K_indep = np.shape(self.txdat)[1]
        self.all_vars = np.concatenate((self.tydat, self.txdat), axis=1)
        self.dep_type = dep_type
        self.indep_type = indep_type
        assert len(self.dep_type) == self.K_dep
        assert len(self.indep_type) == self.K_indep
        self.bw = self.compute_bw(bw)

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
        Similar to the loo_likelihood in Generic_KDE but substitute for f(x), f(x|y)=f(x,y)/f(y)
        """
        yLOO = tools.LeaveOneOut(self.all_vars)
        xLOO = tools.LeaveOneOut(self.txdat).__iter__()
        i = 0
        L = 0
        for Y_j in yLOO:
            X_j = xLOO.next()
            f_yx = tools.GPKE(bw, tdat=-Y_j, edat=-self.all_vars[i, :],
                              var_type=(self.dep_type + self.indep_type))
            f_x = tools.GPKE(bw[self.K_dep::], tdat=-X_j,
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

        .. math:: f(X)=\prod_{s=1}^{q}h_{s}^{-1}k\left(\frac{X_{is}-X_{js}}{h_{s}}\right)

        where :math:`k` is the appropriate kernel for each variable
        """

        if eydat is None:
            eydat = self.all_vars
        if exdat is None:
            exdat = self.txdat
        f_yx = tools.GPKE(self.bw, tdat=np.concatenate((self.tydat, self.txdat), axis=1),
                          edat=eydat, var_type=(self.dep_type + self.indep_type))
        f_x = tools.GPKE(self.bw[self.K_dep::], tdat=self.txdat,
                         edat=exdat, var_type=self.indep_type)
        return (f_yx / f_x)

    def IMSE(self, bw):
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

        The formula for the cross-validation objective function for mixed variable types is:

        .. math:: CV(h,\lambda)=\frac{1}{n}\sum_{l=1}^{n}\frac{G_{-l}(X_{l})}{\left[\mu_{-l}(X_{l})\right]^{2}}-\frac{2}{n}\sum_{l=1}^{n}\frac{f_{-l}(X_{l},Y_{l})}{\mu_{-l}(X_{l})}

        where

        .. math:: G_{-l}(X_{l})=n^{-2}\sum_{i\neq l}\sum_{j\neq l}K_{X_{i},X_{l}}K_{X_{j},X_{l}}K_{Y_{i},Y_{j}}^{(2)}

        where :math:`K_{X_{i},X_{l}}` is the multivariate product kernel and :math:`\mu_{-l}(X_{l})` is
        the leave-one-out estimator of the pdf

        :math:`K_{Y_{i},Y_{j}}^{(2)}` is the convolution kernel

        The value of the function is minimized by _cv_ls method of the Generic_KDE class to return the bw estimates that minimize
        the distance between the estimated and "true" probability density
        
        """
        #print "Starting"
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
            K_Xi_Xl = tools.PKE(bw[self.K_dep::], tdat=Xe_L,
                                  edat=self.txdat[l, :], var_type=self.indep_type)
            K_Xj_Xl = tools.PKE(bw[self.K_dep::], tdat=Xe_R,
                                  edat=self.txdat[l, :], var_type=self.indep_type)
            K2_Yi_Yj = tools.PKE(bw[0:self.K_dep], tdat=Ye_L,
                                   edat=Ye_R, var_type=self.dep_type,
                             ckertype='gauss_convolution', okertype='wangryzin_convolution',
                                   ukertype='aitchisonaitken_convolution')
            G = np.sum(K_Xi_Xl * K_Xj_Xl * K2_Yi_Yj)
            G = G / self.N ** 2
            f_X_Y = tools.GPKE(bw, tdat=-Z, edat=-self.all_vars[l, :],
                               var_type=(self.dep_type + self.indep_type)) / float(self.N)
            m_x = tools.GPKE(bw[self.K_dep::], tdat=-X, edat=-self.txdat[l, :],
                             var_type=self.indep_type) / float(self.N)
            l += 1
            CV += (G / m_x ** 2) - 2 * (f_X_Y / m_x)
        return CV / float(self.N)

class Reg (object):
    def __init__(self, tydat, txdat, var_type, bw):

        self.tydat = np.column_stack(tydat)
        
        self.txdat = np.column_stack(txdat)

        self.N, self.K = np.shape(self.txdat)
        self.var_type = var_type
        self.bw_func = dict(cv_lc=self.CV_LC, aic=self.AIC_Hurvich)
        self.bw = self.compute_bw(bw)
        
    def Cond_Mean(self, edat=None):
        if edat is None:
            edat = self.txdat
        # The numerator in the formula is:
        G_numer = np.sum(self.tydat * tools.GPKE_Reg(bw, tdat=self.txdat, edat=edat, var_type=self.var_type))
        # The denominator in the formula is:
        G_denom = np.sum(tools.GPKE_Reg(bw, tdat=self.txdat, edat=edat, var_type=self.var_type))
        # The conditional mean is:
        G = G_numer / G_denom
        return G
    
    def CV_LC(self, bw):
        LOO_X = tools.LeaveOneOut(self.txdat)
        LOO_Y = tools.LeaveOneOut(self.tydat).__iter__()
        i = 0
        L = 0
        for X_j in LOO_X:
            #print "running"
            Y = LOO_Y.next()
            G_numer = np.sum(Y * tools.GPKE_Reg(bw, tdat=-X_j, edat=-self.txdat[i, :],
                             var_type=self.var_type))
            G_denom = np.sum(tools.GPKE_Reg(bw, tdat=-X_j, edat=-self.txdat[i, :],
                             var_type=self.var_type))
            G = G_numer / G_denom
            L += (self.tydat[i] - G)**2
            i += 1
        # Note: There might be a way to vectorize this. See p.72 in [1]
        return L / self.N     
    def AIC_Hurvich(self, bw):
        pass
    
    def compute_bw(self,bw):
        if not isinstance(bw, basestring):
            # The user provided an actual bandwidth estimate
            # TODO: would be good if the user could provide a function here
            # that uses tdat/N/K, instead of just a result.
            self._bw_method = "user-specified"
            res = np.asarray(bw)
        else:
            # The user specified a bandwidth selection method e.g. 'normal-reference'
            self._bw_method = bw
            bwfunc = self.bw_func[bw]
            X = np.std(self.txdat, axis=0)
            h0 = 1.06 * X * self.N ** (- 1. / (4 + np.size(self.txdat, axis=1)))
            res = bwfunc
        return opt.fmin(res, x0=h0, maxiter=1e3,
                      maxfun=1e3, disp=0)
       
    
