# -*- coding: utf-8 -*-

from __future__ import division
from __future__ import print_function
from ._gpa_rotation import oblimin_objective, orthomax_objective, CF_objective
from ._gpa_rotation import ff_partial_target, ff_target
from ._gpa_rotation import vgQ_partial_target, vgQ_target
from ._gpa_rotation import rotateA, GPA
import numpy as np
import unittest

__all__=[]

def rotate_factors(A, method, *method_args, **algorithm_kwargs):
    r"""
    Subroutine for orthogonal and oblique rotation of the matrix :math:`A`.
    For orthogonal rotations :math:`A` is rotated to :math:`L` according to
    
    .. math::
        L =  AT,
        
    where :math:`T` is an orthogonal matrix. And, for oblique rotations
    :math:`A` is rotated to :math:`L` according to
    
    .. math::
        L =  A(T^*)^{-1},
    where :math:`T` is a normal matrix.
    
    Methods
    -------
    What follows is a list of available methods. Depending on the method
    additional argument are required and different algorithms
    are available. The algorithm_kwargs are additional keyword arguments
    passed to the selected algorithm (see the parameters section).
    Unless stated otherwise, only the gpa and
    gpa_der_free algorithm are available.
    
    Below,
    
    * :math:`L` is a :math:`p\times k` matrix;
    * :math:`N` is :math:`k\times k` matrix with zeros on the diagonal and ones elsewhere;
    * :math:`M` is :math:`p\times p` matrix with zeros on the diagonal and ones elsewhere;
    * :math:`C` is a :math:`p\times p` matrix with elements equal to :math:`1/p`;
    * :math:`(X,Y)=\operatorname{Tr}(X^*Y)` is the Frobenius norm;
    * :math:`\circ` is the element-wise product or Hadamard product.
    
    oblimin : orthogonal or oblique rotation that minimizes
        .. math::
            \phi(L) = \frac{1}{4}(L\circ L,(I-\gamma C)(L\circ L)N).
            
        For orthogonal rotations:

        * :math:`\gamma=0` corresponds to quartimax,
        * :math:`\gamma=\frac{1}{2}` corresponds to biquartimax,
        * :math:`\gamma=1` corresponds to varimax,
        * :math:`\gamma=\frac{1}{p}` corresponds to equamax.
        For oblique rotations rotations:
    
        * :math:`\gamma=0` corresponds to quartimin,
        * :math:`\gamma=\frac{1}{2}` corresponds to biquartimin.
        
        method_args:
        
        gamma : float
            oblimin family parameter
        rotation_method : string
            should be one of {orthogonal, oblique}
    
    orthomax : orthogonal rotation that minimizes
        .. math::
            \phi(L) = -\frac{1}{4}(L\circ L,(I-\gamma C)(L\circ L)),
            
        where :math:`0\leq\gamma\leq1`. The orthomax family is equivalent to 
        the oblimin family (when restricted to orthogonal rotations). Furthermore,

        * :math:`\gamma=0` corresponds to quartimax,
        * :math:`\gamma=\frac{1}{2}` corresponds to biquartimax,
        * :math:`\gamma=1` corresponds to varimax,
        * :math:`\gamma=\frac{1}{p}` corresponds to equamax.
        
        method_args:
        
        gamma : float (between 0 and 1)
            orthomax family parameter
    
    CF : Crawford-Ferguson family for orthogonal and oblique rotation wich minimizes:
        .. math::
            \phi(L) =\frac{1-\kappa}{4} (L\circ L,(L\circ L)N)
                      -\frac{1}{4}(L\circ L,M(L\circ L)),
                      
        where :math:`0\leq\kappa\leq1`. For orthogonal rotations the oblimin
        (and orthomax) family of rotations is equivalent to the Crawford-Ferguson family.
        To be more precise:
    
        * :math:`\kappa=0` corresponds to quartimax,
        * :math:`\kappa=\frac{1}{p}` corresponds to varimax,
        * :math:`\kappa=\frac{k-1}{p+k-2}` corresponds to parsimax,
        * :math:`\kappa=1` corresponds to factor parsimony.
        
        method_args:
        
        kappa : float (between 0 and 1)
            Crawford-Ferguson family parameter
        rotation_method : string
            should be one of {orthogonal, oblique}
    
    quartimax : orthogonal rotation method
        minimizes the orthomax objective with :math:`\gamma=0`
        
    biquartimax : orthogonal rotation method
        minimizes the orthomax objective with :math:`\gamma=\frac{1}{2}`
        
    varimax : orthogonal rotation method
        minimizes the orthomax objective with :math:`\gamma=1`
        
    equamax : orthogonal rotation method
        minimizes the orthomax objective with :math:`\gamma=\frac{1}{p}`
        
    parsimax : orthogonal rotation method
        minimizes the Crawford-Ferguson family objective with :math:`\kappa=\frac{k-1}{p+k-2}`
        
    parsimony : orthogonal rotation method
        minimizes the Crawford-Ferguson family objective with :math:`\kappa=1`
    
    quartimin : oblique rotation method that minimizes
        minimizes the oblimin objective with :math:`\gamma=0`      

    quartimin : oblique rotation method that minimizes
        minimizes the oblimin objective with :math:`\gamma=\frac{1}{2}`   
        
    target : orthogonal or oblique rotation that rotates towards a target matrix :math:`H` by minimizing the objective
        .. math::
            \phi(L) =\frac{1}{2}\|L-H\|^2.
        
        method_args:
        
        H : numpy matrix
            target matrix
        rotation_method : string
            should be one of {orthogonal, oblique}

        For orthogonal rotations the algorithm can be set to analytic in which case
        the following keyword arguments are available:
        
        full_rank : boolean (default False)
            if set to true full rank is assumed    

    partial_target : orthogonal (default) or oblique rotation that partially rotates
        towards a target matrix :math:`H` by minimizing the objective:
        
        .. math::
            \phi(L) =\frac{1}{2}\|W\circ(L-H)\|^2.
        
        method_args:
        
        H : numpy matrix
            target matrix
        W : numpy matrix (default matrix with equal weight one for all entries)
            matrix with weights, entries can either be one or zero
    
    Parameters
    ---------
    A : numpy matrix (default None)
        non rotated factors
    method : string
        should be one of the methods listed above
    method_args : list
        additional arguments that should be provided with each method
    algorithm_kwargs : dictionary
        algorithm : string (default gpa)
            should be one of:
            
            * 'gpa': a numerical method
            * 'gpa_der_free': a derivative free numerical method
            * 'analytic' : an analytic method
        Depending on the algorithm, there are algorithm specific keyword
        arguments. For the gpa and gpa_der_free, the following
        keyword arguments are available:
        
        max_tries : integer (default 501)
            maximum number of iterations
        tol : float
            stop criterion, algorithm stops if Frobenius norm of gradient is
            smaller then tol
        For analytic, the supporeted arguments depend on the method, see above.
            
        See the lower level functions for more details.
        
    Returns
    -------
    The tuple :math:`(L,T)`
    
    Examples
    -------
    >>> A = np.random.randn(8,2)
    >>> L, T = rotate_factors(A,'varimax')
    >>> np.allclose(L,A.dot(T))
    >>> L, T = rotate_factors(A,'orthomax',0.5)
    >>> np.allclose(L,A.dot(T))
    >>> L, T = rotate_factors(A,'quartimin',0.5)
    >>> np.allclose(L,A.dot(np.linalg.inv(T.T)))
    """
    if 'algorithm' in algorithm_kwargs:
        algorithm = algorithm_kwargs['algorithm']
        algorithm_kwargs.pop('algorithm')
    else:
        algorithm = 'gpa'
    assert not ('rotation_method' in algorithm_kwargs), 'rotation_method cannot be provided as keyword argument'
    L=None
    T=None
    ff=None
    vgQ=None
    p,k = A.shape
    #set ff or vgQ to appropriate objective function, compute solution using recursion or analytically compute solution
    if method == 'orthomax':
        assert len(method_args)==1, 'Only %s family parameter should be provided' % method
        rotation_method='orthogonal'
        gamma = method_args[0]
        if algorithm =='gpa':
            vgQ=lambda L=None, A=None, T=None: orthomax_objective(L=L,A=A,T=T,
                                                                     gamma=gamma,
                                                                     return_gradient=True)
        elif algorithm =='gpa_der_free':
            ff = lambda L=None, A=None, T=None: orthomax_objective(L=L,A=A,T=T,
                                                                      gamma=gamma,
                                                                      return_gradient=False)
        else:
            raise ValueError('Algorithm %s is not possible for %s rotation' % (algorithm, method))
    elif method == 'oblimin':
        assert len(method_args)==2, 'Both %s family parameter and rotation_method should be provided' % method
        rotation_method=method_args[1]
        assert rotation_method in ['orthogonal','oblique'], 'rotation_method should be one of {orthogonal, oblique}'
        gamma = method_args[0]
        if algorithm =='gpa':
            vgQ=lambda L=None, A=None, T=None: oblimin_objective(L=L,A=A,T=T,
                                                                    gamma=gamma,
                                                                    return_gradient=True)
        elif algorithm =='gpa_der_free':
            ff = lambda L=None, A=None, T=None: oblimin_objective(L=L,A=A,T=T,
                                                                     gamma=gamma,
                                                                     rotation_method=rotation_method,
                                                                     return_gradient=False)
        else:
            raise ValueError('Algorithm %s is not possible for %s rotation' % (algorithm, method))
    elif method == 'CF':
        assert len(method_args)==2, 'Both %s family parameter and rotation_method should be provided' % method
        rotation_method=method_args[1]
        assert rotation_method in ['orthogonal','oblique'], 'rotation_method should be one of {orthogonal, oblique}'
        kappa = method_args[0]
        if algorithm =='gpa':
            vgQ=lambda L=None, A=None, T=None: CF_objective(L=L,A=A,T=T,
                                                               kappa=kappa,
                                                               rotation_method=rotation_method,
                                                               return_gradient=True)
        elif algorithm =='gpa_der_free':
            ff = lambda L=None, A=None, T=None: CF_objective(L=L,A=A,T=T,
                                                                kappa=kappa,
                                                                rotation_method=rotation_method,
                                                                return_gradient=False)
        else:
            raise ValueError('Algorithm %s is not possible for %s rotation' % (algorithm, method))
    elif method == 'quartimax':
        return rotate_factors(A, 'orthomax', 0, **algorithm_kwargs)
    elif method == 'biquartimax':
        return rotate_factors(A, 'orthomax', 0.5, **algorithm_kwargs)
    elif method == 'varimax':
        return rotate_factors(A, 'orthomax', 1, **algorithm_kwargs)
    elif method == 'equamax':
        return rotate_factors(A, 'orthomax', 1/p, **algorithm_kwargs)
    elif method == 'parsimax':
        return rotate_factors(A, 'CF', (k-1)/(p+k-2), 'orthogonal', **algorithm_kwargs)    
    elif method == 'parsimony':
        return rotate_factors(A, 'CF', 1, 'orthogonal', **algorithm_kwargs)    
    elif method == 'quartimin':
        return rotate_factors(A, 'oblimin', 0, 'oblique', **algorithm_kwargs)
    elif method == 'biquartimin':
        return rotate_factors(A, 'oblimin', 0.5, 'oblique', **algorithm_kwargs)
    elif method == 'target':
        assert len(method_args)==2, 'only the rotation target and orthogonal/oblique should be provide for %s rotation' % method
        H=method_args[0]
        rotation_method=method_args[1]
        assert rotation_method in ['orthogonal','oblique'], 'rotation_method should be one of {orthogonal, oblique}'
        if algorithm =='gpa':
            vgQ=lambda L=None, A=None, T=None: vgQ_target(H,L=L,A=A,T=T,rotation_method=rotation_method)
        elif algorithm =='gpa_der_free':
            ff = lambda L=None, A=None, T=None: ff_target(H,L=L,A=A,T=T,rotation_method=rotation_method)
        elif algorithm =='analytic':
            assert rotation_method == 'orthogonal', 'For analytic %s rotation only orthogonal rotation is supported'
            T= ar.target_rotation(A,H,**algorithm_kwargs)
        else:
            raise ValueError('Algorithm %s is not possible for %s rotation' % (algorithm, method))
    elif method == 'partial_target':
        assert len(method_args)==2, '2 additional arguments are expected for %s rotation' % method
        H=method_args[0]
        W=method_args[1]
        rotation_method='orthogonal'
        if algorithm =='gpa':
            vgQ=lambda L=None, A=None, T=None: vgQ_partial_target(H,W=W,L=L,A=A,T=T)
        elif algorithm =='gpa_der_free':
            ff = lambda L=None, A=None, T=None: ff_partial_target(H,W=W,L=L,A=A,T=T)
        else:
            raise ValueError('Algorithm %s is not possible for %s rotation' % (algorithm, method))
    else:
        raise ValueError('Invalid method')
    #compute L and T if not already done
    if T is None:
        L, phi, T, table = GPA(A, vgQ=vgQ, ff=ff, rotation_method=rotation_method, **algorithm_kwargs)
    if L is None:
        assert T is not None, 'Cannot compute L without T'
        L=rotateA(A,T,rotation_method=rotation_method)
    #return
    return L, T

class unittests(unittest.TestCase):
    
    @staticmethod    
    def str2matrix(A):
        A=A.lstrip().rstrip().split('\n')
        A=np.array([row.split() for row in A]).astype(np.float)
        return A
        
    def get_A(self):
        return self.str2matrix("""
         .830 -.396
         .818 -.469
         .777 -.470
         .798 -.401
         .786  .500
         .672  .458
         .594  .444
         .647  .333
        """)
        
    def get_H(self):
        return  self.str2matrix("""
          .8 -.3
          .8 -.4
          .7 -.4
          .9 -.4
          .8  .5
          .6  .4
          .5  .4
          .6  .3
        """)

    def get_W(self):
        return self.str2matrix("""
        1 0
        0 1
        0 0
        1 1
        1 0
        1 0
        0 1
        1 0
        """)

        
    def _test_template(self, method,*method_args, **algorithms):
        A=self.get_A()
        algorithm1= 'gpa' if 'algorithm1' not in algorithms else algorithms['algorithm1']
        algorithm2= 'gpa_der_free' if 'algorithm`' not in algorithms else algorithms['algorithm1']
        L1, T1 = rotate_factors(A,method,*method_args, algorithm = algorithm1)        
        L2, T2 = rotate_factors(A,method,*method_args, algorithm = algorithm2)
        self.assertTrue(np.allclose(L1, L2, atol=1e-5))
        self.assertTrue(np.allclose(T1, T2, atol=1e-5))
    
    def test_methods(self):
        """
        Quartimax derivative free example
        http://www.stat.ucla.edu/research/gpa
        """
        #orthomax, oblimin and CF are tested indirectly
        methods=['quartimin', 'biquartimin',
                 'quartimax', 'biquartimax', 'varimax', 'equamax', 'parsimax', 'parsimony',
                 'target', 'partial_target']
        for method in methods:
            method_args=[]
            if method == 'target':
                method_args=[self.get_H(),'orthogonal']
                self._test_template(method, *method_args)
                method_args=[self.get_H(),'oblique']
                self._test_template(method, *method_args)
                method_args=[self.get_H(),'orthogonal']
                self._test_template(method, *method_args, algorithm2='analytic')
            elif method == 'partial_target':
                method_args=[self.get_H(), self.get_W()]
            self._test_template(method, *method_args)
    
if __name__ == '__main__': # run only if this file is run directly and not when imported
    run_unit_tests=True
    test_only = list() # if list is empty then test all
    #test_only.append('test_orthogonal_target')
    if run_unit_tests:
        if len(test_only) > 0:
            suite = unittest.TestSuite()
            for ut in test_only:
                suite.addTest(unittests(ut))
            unittest.TextTestRunner().run(suite)
        else:
            #unittest.main()
            suite = unittest.TestLoader().loadTestsFromTestCase(unittests)
            unittest.TextTestRunner(verbosity=2).run(suite)
    else: # run a basic example
        A = np.random.randn(8,2)
        L, T = rotate_factors(A,'varimax')
        L, T = rotate_factors(A,'orthomax',0.5)
        L, T = rotate_factors(A,'quartimin')