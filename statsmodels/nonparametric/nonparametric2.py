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
import KernelFunctions as kf
import scipy.optimize as opt

kernel_func=dict(wangryzin = kf.WangRyzin, aitchisonaitken = kf.AitchisonAitken,
                 epanechnikov = kf.Epanechnikov, gaussian = kf.Gaussian)


class LeaveOneOut(object):
    # Written by Skipper
    """
    Generator to give leave one out views on X

    Parameters
    ----------
    X : array-like
        2d array

    Examples
    --------
    >>> X = np.random.normal(0,1,[10,2])
    >>> loo = LeaveOneOut(X)
    >>> for x in loo:
    ...    print x

    Notes
    -----
    A little lighter weight than sklearn LOO. We don't need test index.
    Also passes views on X, not the index.
    """
    def __init__(self, X):
        self.X = np.asarray(X)

    def __iter__(self):
        X = self.X
        N,K = np.shape(X)
        for i in xrange(N):
            index = np.ones([N, K], dtype=np.bool)
            index[i, :] = False
            yield X[index].reshape([N - 1,K])


def GPKE(bw,tdat,edat,var_type,ckertype='gaussian',okertype='wangryzin',ukertype='aitchisonaitken'):
    """
    Returns the non-normalized Generalized Product Kernel Estimator
    
    Parameters
    ----------
    bw: array-like
        The user-specified bandwdith parameters
    tdat: 1D or 2d array
        The training data
    edat: 1d array
        The evaluation points at which the kernel estimation is performed
    var_type: str
        The variable type (continuous, ordered, unordered)
    ckertype: str
        The kernel used for the continuous variables
    okertype: str
        The kernel used for the ordered discrete variables
    ukertype: str
        The kernel used for the unordered discrete variables
        
    """
    var_type = np.asarray(list(var_type))
    iscontinuous = np.where(var_type=='c')[0]
    isordered = np.where(var_type == 'o')[0]
    isunordered = np.where(var_type == 'u')[0]
    
    if tdat.ndim > 1:
        N,K = np.shape(tdat)
    else:
        K = 1
        N = np.shape(tdat)[0]
        tdat = tdat.reshape([N,K])
    
    if edat.ndim > 1:
        N_edat = np.shape(edat)[0]
    else:
        N_edat = 1
        edat = edat.reshape([N_edat, K])
    
    bw = np.reshape(np.asarray(bw), (K,))  #must remain 1-D for indexing to work
    dens = np.empty([N_edat, 1])
       
    for i in xrange(N_edat):
        
        Kval = np.concatenate((
        kernel_func[ckertype](bw[iscontinuous], tdat[:, iscontinuous], edat[i, iscontinuous]),
        kernel_func[okertype](bw[isordered], tdat[:, isordered],edat[i, isordered]),
        kernel_func[ukertype](bw[isunordered], tdat[:, isunordered], edat[i, isunordered])
        ), axis=1)
        
        dens[i] = np.sum(np.prod(Kval,axis=1))*1./(np.prod(bw[iscontinuous]))
    return dens

class Generic_KDE ():
    # Generic KDE class with methods shared by both conditional and unconditional kernel density estimators
    
    def get_bw(self,bw,bwmethod):
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
        The default values for bw and bwmethod are False. The user must specify either a value for bw
        or bwmethod but not both. 
        """
#TODO: Combine into only one parameter bw: array-like, str
        
        self.bw_func = dict(normal_reference = self.normal_reference, cv_ml = self.cv_ml)
        assert (bw != False or bwmethod != False) # either bw or bwmethod should be input by the user
        
        if bw != False:
            return np.asarray(bw)
        if bwmethod != False:
            self.bwmethod = bwmethod
            bwfunc = self.bw_func[bwmethod]
            
            return bwfunc()
    
    def normal_reference(self):
        """
        Returns the normal reference rule of thumb bandwidth parameter
        """
        c = 1.06        
        X = np.std(self.all_vars, axis=0)       
        return c*X*self.N**(-1./(4+np.size(self.all_vars, axis=1)))
    
    def cv_ml (self):
        """
        Returns the cross validation maximum likelihood bandwidth parameter
        """
        
        h0 = self.normal_reference() # the initial value for the optimization is the normal_reference
        bw = opt.fmin(self.loo_likelihood, x0 = h0, maxiter = 1e3, maxfun = 1e3,disp = 0)
        return bw

#TODO: Add the least squares cross validation bandwidth method
    def cv_ls (self):
        pass
    
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
    def __init__(self, tdat, var_type, bw = False, bwmethod = False):
        for var in tdat:
            var = np.asarray(var)
            var = var.reshape([len(var), 1])
        
        self.tdat = np.concatenate(tdat, axis=1)
        self.all_vars = self.tdat
        self.N,self.K = np.shape(self.tdat)
        self.var_type = var_type
        
        self.bw = self.get_bw(bw, bwmethod)
             

    def loo_likelihood(self,bw):
        """
        Returns the leave-one-out likelihood for the data
        Parameters
        ----------
        bw: array-like
            The value for the bandwdith parameters
        """
        LOO = LeaveOneOut(self.tdat)
        i = 0
        L = 0
        for X_j in LOO:
            f_i = GPKE(bw, tdat = -X_j, edat = -self.tdat[i, :], var_type=self.var_type)/(self.N-1)           
            i += 1
            L += np.log(f_i)       
        return -L
    
    def pdf(self,edat=False):
        """
        Returns the probability density function

        Parameters
        ----------
        edat: array-like
            Evaluation data.
            If unspecified, the training data is used
        """
        if edat==False: edat = self.tdat
        return GPKE(self.bw,tdat=self.tdat,edat=edat,var_type=self.var_type)/self.N

class CKDE(Generic_KDE):
    """
    Conditional Kernel Density Estimator

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

    def __init__ (self, tydat, txdat, dep_type, indep_type, bw=False, bwmethod=False):
        for var in tydat:
            var = np.asarray(var)
            var = var.reshape([len(var), 1])
        for var in txdat:
            var = np.asarray(var)
            var = var.reshape([len(var), 1])
            
        self.tydat = np.concatenate(tydat,axis=1)
        self.txdat = np.concatenate(txdat,axis=1)
        self.N,self.K_dep = np.shape(self.tydat)
        self.K_indep = np.shape(self.txdat)[1]
        self.all_vars = np.concatenate((self.tydat,self.txdat),axis=1)
        self.dep_type = dep_type; self.indep_type = indep_type
        
        self.bw=self.get_bw(bw,bwmethod)
    def loo_likelihood(self,bw):
        """
        Returns the leave-one-out likelihood for the data
        """
       
        
        yLOO = LeaveOneOut(self.all_vars)
        xLOO = LeaveOneOut(self.txdat).__iter__()
        i = 0
        L = 0
        for Y_j in yLOO:
            X_j = xLOO.next()
            f_yx = GPKE(bw, tdat = -Y_j, edat=-self.all_vars[i,:], var_type = (self.dep_type + self.indep_type))
            f_x = GPKE(bw[self.K_dep::], tdat = -X_j, edat=-self.txdat[i,:], var_type = self.indep_type)
            f_i = f_yx/f_x
            i += 1
            L += np.log(f_i)       
        return -L
    
    def pdf(self,eydat=False,exdat=False):
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

        if eydat==False: eydat = self.all_vars
        if exdat==False: exdat = self.txdat
        
        f_yx = GPKE(self.bw,tdat=np.concatenate((self.tydat, self.txdat), axis=1), edat = eydat, var_type=(self.dep_type + self.indep_type))
        f_x = GPKE(self.bw[self.K_dep::], tdat = self.txdat, edat = exdat, var_type = self.indep_type)
        return (f_yx/f_x)

