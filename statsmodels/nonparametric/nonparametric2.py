"""
Multivariate Conditional and Unconditional Kernel Density Estimation
with Mixed Data Types

References
----------
Racine, Jeff. (2008) "Nonparametric Econometrics: A Primer," Foundation and
    Trends in Econometrics: Vol 3: No 1, pp1-88.
    http://dx.doi.org/10.1561/0800000009

"""

import numpy as np
from scipy import integrate, stats
import np_tools as tools
import scipy.optimize as opt



class Generic_KDE ():
    # Generic KDE class with methods shared by both conditional and unconditional kernel density estimators
    
    def compute_bw(self, bw):
        """
        Returns the bandwidth of the data

        Parameters
        ----------
        bw: array-like
            User-specified bandwidth.
        bwmethod: str
            The method for bandwidth selection.
            cv_ml: cross validation maximum likelihood
            normal_reference: normal reference rule of thumb
        Notes
        ----------
        The default values for bw and bwmethod are None. The user must specify either a value for bw
        or bwmethod but not both. 
        """

        
        self.bw_func = dict(normal_reference = self._normal_reference, cv_ml = self._cv_ml, cv_ls = self._cv_ls)
        if bw is None:
            bwfunc = self.bw_func['normal_reference']
            return bwfunc()
         
        if type(bw) != str:  # The user provided an actual bandwidth estimate
            return np.asarray(bw)
        else: # The user specified a bandwidth selection method e.g. 'normal-reference'
            bwfunc = self.bw_func[bw]
            return bwfunc()
    
    def _normal_reference(self):
        """
        Returns Scott's normal reference rule of thumb bandwidth parameter
        """
        c = 1.06        
        X = np.std(self.all_vars, axis=0)       
        return c*X*self.N**(-1./(4+np.size(self.all_vars, axis=1)))
    
    def _cv_ml (self):
        """
        Returns the cross validation maximum likelihood bandwidth parameter
        """
        
        h0 = self._normal_reference() # the initial value for the optimization is the normal_reference
        bw = opt.fmin(self.loo_likelihood, x0 = h0, args = (np.log,), maxiter = 1e3, maxfun = 1e3,disp = 0)
        return bw

#TODO: Add the least squares cross validation bandwidth method
    def _cv_ls (self):
        h0 = self._normal_reference()
        bw = opt.fmin(self.IMSE, x0 = h0, maxiter = 1e3, maxfun = 1e3,disp = 0)
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
        is a seperate variable
    var_type: str
        The type of the variables
        c: Continuous
        u: Unordered (Discrete)
        o: Ordered (Discrete)
    
    bw: array-like
        User-specified bandwidth.
    bwmethod: str
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

    dens_u=UKDE(tdat=[c1,c2],var_type='cc',bwmethod='normal_reference')

    print "The bandwdith is: ", dens_u.bw
    """
    def __init__(self, tdat, var_type, bw = None):
        
        self.tdat = np.column_stack(tdat)
        self.all_vars = self.tdat
        self.N,self.K = np.shape(self.tdat)
        self.var_type = var_type
        assert self.K == len(self.var_type)
        self.bw = self.compute_bw(bw)
             

    def loo_likelihood(self, bw, func = lambda x:x):
        """
        Returns the leave-one-out likelihood for the data
        Parameters
        ----------
        bw: array-like
            The value for the bandwdith parameters
        """
        LOO = tools.LeaveOneOut(self.tdat)
        i = 0
        L = 0
        for X_j in LOO:
            f_i = tools.GPKE(bw, tdat = -X_j, edat = -self.tdat[i, :], var_type = self.var_type)          
            i += 1
            L += func(f_i)       
        return -L
    
    def pdf(self, edat = None):
        """
        Returns the probability density function

        Parameters
        ----------
        edat: array-like
            Evaluation data.
            If unspecified, the training data is used
        """
        if edat is None: edat = self.tdat
        return tools.GPKE(self.bw, tdat = self.tdat, edat = edat, var_type = self.var_type)/self.N
    
    def IMSE(self, bw):
        """
        Returns the First term from the Integrated Mean Square Error
        Integrate [ f_hat(x) - f(x)]^2 dx
        """
        F=0
        for i in range(self.N):
            k_bar_sum = tools.GPKE (bw, tdat = -self.tdat, edat = -self.tdat[i, :],
                                    var_type = self.var_type, ckertype = 'gauss_convolution', okertype = 'wangryzin_convolution')
            F += k_bar_sum
        return (F/(self.N**2) + self.loo_likelihood(bw)*2/((self.N)*(self.N - 1)))  # there is a + because loo_likelihood returns the negative
        
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
    
    bw: array-like
        User-specified bandwidth.
    bwmethod: str
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

    def __init__ (self, tydat, txdat, dep_type, indep_type, bw):
            
        self.tydat = np.column_stack(tydat)
        self.txdat = np.column_stack(txdat)
        self.N,self.K_dep = np.shape(self.tydat)
        self.K_indep = np.shape(self.txdat)[1]
        self.all_vars = np.concatenate((self.tydat, self.txdat), axis = 1)
        self.dep_type = dep_type; self.indep_type = indep_type
        assert len(self.dep_type) == self.K_dep
        assert len(self.indep_type) == self.K_indep
        
        self.bw = self.compute_bw(bw)
        
    def loo_likelihood(self, bw, func = lambda x:x):
        """
        Returns the leave-one-out likelihood for the data
        """
        yLOO = tools.LeaveOneOut(self.all_vars)
        xLOO = tools.LeaveOneOut(self.txdat).__iter__()
        i = 0
        L = 0
        for Y_j in yLOO:
            X_j = xLOO.next()
            f_yx = tools.GPKE(bw, tdat = -Y_j, edat=-self.all_vars[i,:], var_type = (self.dep_type + self.indep_type))
            f_x = tools.GPKE(bw[self.K_dep::], tdat = -X_j, edat=-self.txdat[i, :], var_type = self.indep_type)
            f_i = f_yx/f_x
            i += 1
            L += func(f_i)       
        return -L
    
    def pdf(self,eydat = None, exdat = None):
        """
        Returns the probability density function

        Parameters
        ----------
        eydat: array-like
            Evaluation data for the dependent variables.
            If unspecified, the training data is used
        exdat: array-like
            Evaluation data for the independent variables
        """

        if eydat is None: eydat = self.all_vars
        if exdat is None: exdat = self.txdat
        
        f_yx = tools.GPKE(self.bw,tdat=np.concatenate((self.tydat, self.txdat), axis=1), edat = eydat, var_type = (self.dep_type + self.indep_type))
        f_x = tools.GPKE(self.bw[self.K_dep::], tdat = self.txdat, edat = exdat, var_type = self.indep_type)
        return (f_yx/f_x)

