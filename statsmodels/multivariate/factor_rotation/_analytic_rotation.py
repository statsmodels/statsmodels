# -*- coding: utf-8 -*-
"""
This file contains analytic implementations of rotation methods.
"""

from __future__ import division
from __future__ import print_function

import numpy as np
import scipy as sp
import scipy.linalg
import unittest

def target_rotation(A, H, full_rank = False):
    r"""
    Analytically performs orthogonal rotations towards a target matrix,
    i.e., we minimize:
        
    .. math::
        \phi(L) =\frac{1}{2}\|AT-H\|^2.
        
    where :math:`T` is an orthogonal matrix. This problem is also known as
    an orthogonal Procrustes problem.
        
    Under the assumption that :math:`A^*H` has full rank, the analytical
    solution :math:`T` is given by:
    
    .. math::
        T = (A^*HH^*A)^{-\frac{1}{2}}A^*H,
		
    see Green (1952). In other cases the solution is given by :math:`T = UV`,
    where :math:`U` and :math:`V` result from the singular value decomposition
    of :math:`A^*H`:
        
    .. math::
        A^*H = U\Sigma V,
    
    see Schonemann (1966).
    
    Parametes
    ---------
    A : numpy matrix (default None)
        non rotated factors
    H : numpy matrix
        target matrix
    full_rank : boolean (default FAlse)
        if set to true full rank is assumed
        
    Returns
    -------
    The matrix :math:`T`.
    
    References
    ----------
    [1] Green (1952, Psychometrika) - The orthogonal approximation of an
    oblique structure in factor analysis
	
    [2] Schonemann (1966) - A generalized solution of the orthogonal
    procrustes problem
	
    [3] Gower, Dijksterhuis (2004) - Procustes problems
    """
    ATH=A.T.dot(H)
    if full_rank or np.linalg.matrix_rank(ATH)==A.shape[1]:
        T = sp.linalg.fractional_matrix_power(ATH.dot(ATH.T),-1/2).dot(ATH)
    else:
        U,D,V=np.linalg.svd(ATH,full_matrices=False)
        T=U.dot(V)
    return T

def procrustes(A, H):
    r"""
    Analytically solves the following Procrustes problem:
        
    .. math::
        \phi(L) =\frac{1}{2}\|AT-H\|^2.
        
    (With no further conditions on :math:`H`)

    Under the assumption that :math:`A^*H` has full rank, the analytical
    solution :math:`T` is given by:
    
    .. math::
        T = (A^*HH^*A)^{-\frac{1}{2}}A^*H,
		
    see Navarra, Simoncini (2010).
    
    Parametes
    ---------
    A : numpy matrix
        non rotated factors
    H : numpy matrix
        target matrix
    full_rank : boolean (default False)
        if set to true full rank is assumed
        
    Returns
    -------
    The matrix :math:`T`.
    
    References
    ----------
    [1] Navarra, Simoncini (2010) - A guide to emprirical orthogonal functions
    for climate data analysis
    """
    return np.linalg.inv(A.T.dot(A)).dot(A.T).dot(H);

def promax(A,k=2):
    r"""
    Performs promax rotation of the matrix :math:`A`.
    
    This method was not very clear to me from the literature, this implementation
    is as I understand it should work.
    
    Promax rotation is performed in the following steps:
    
    * Deterine varimax rotated patterns :math:`V`.
    
    * Construct a rotation target matrix :math:`|V_{ij}|^k/V_{ij}
    
    * Perform procrustes rotation towards the target to obtain T
    
    * Determine the patterns
    
    First, varimax rotation a target matrix :math:`H` is determined with orthogonal varimax rotation.
    Then, oblique target rotation is performed towards the target.
    
    Parameters
    ---------
    A : numpy matrix
        non rotated factors
    k : float
        parameter, should be positive
    
    References
    ----------
    [1] Browne (2001) - An overview of analytic rotation in exploratory factor analysis
    
    [2] Navarra, Simoncini (2010) - A guide to emprirical orthogonal functions
    for climate data analysis
    """
    assert k>0
    #define rotation target using varimax rotation:
    from ._wrappers import rotate_factors
    V, T = rotate_factors(A,'varimax')
    H = np.abs(V)**k/V
    #solve procrustes problem
    S=procrustes(A,H) #np.linalg.inv(A.T.dot(A)).dot(A.T).dot(H);
    #normalize
    d=np.sqrt(np.diag(np.linalg.inv(S.T.dot(S))));
    D=np.diag(d)
    T=np.linalg.inv(S.dot(D)).T
    #return
    return A.dot(T), T

class unittests(unittest.TestCase):
    
    @staticmethod    
    def str2matrix(A):
        A=A.lstrip().rstrip().split('\n')
        A=np.array([row.split() for row in A]).astype(np.float)
        return A
        
    def test_target_rotation(self):
        """
        Rotation towards target matrix example
        http://www.stat.ucla.edu/research/gpa
        """
        A= self.str2matrix("""
         .830 -.396
         .818 -.469
         .777 -.470
         .798 -.401
         .786  .500
         .672  .458
         .594  .444
         .647  .333
        """)
        H= self.str2matrix("""
          .8 -.3
          .8 -.4
          .7 -.4
          .9 -.4
          .8  .5
          .6  .4
          .5  .4
          .6  .3
        """)
        T = target_rotation(A,H)
        L = A.dot(T)
        L_required = self.str2matrix("""
        0.84168  -0.37053
        0.83191  -0.44386
        0.79096  -0.44611
        0.80985  -0.37650
        0.77040   0.52371
        0.65774   0.47826
        0.58020   0.46189
        0.63656   0.35255
        """)
        self.assertTrue(np.allclose(L,L_required,atol=1e-05))
        T = target_rotation(A,H,full_rank=True)
        L = A.dot(T)
        self.assertTrue(np.allclose(L,L_required,atol=1e-05))

    def test_orthogonal_target(self):
        """
        Rotation towards target matrix example
        http://www.stat.ucla.edu/research/gpa
        """
        import _gpa_rotation as gr
        A= self.str2matrix("""
         .830 -.396
         .818 -.469
         .777 -.470
         .798 -.401
         .786  .500
         .672  .458
         .594  .444
         .647  .333
        """)
        H= self.str2matrix("""
          .8 -.3
          .8 -.4
          .7 -.4
          .9 -.4
          .8  .5
          .6  .4
          .5  .4
          .6  .3
        """)
        vgQ = lambda L=None, A=None, T=None: gr.vgQ_target(H,L=L,A=A,T=T)
        L, phi, T, table = gr.GPA(A, vgQ=vgQ, rotation_method='orthogonal')
        T_analytic = target_rotation(A,H)
        self.assertTrue(np.allclose(T,T_analytic,atol=1e-05))


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
        A= unittests.str2matrix("""
         .830 -.396
         .818 -.469
         .777 -.470
         .798 -.401
         .786  .500
         .672  .458
         .594  .444
         .647  .333
        """)
        H= unittests.str2matrix("""
          .8 -.3
          .8 -.4
          .7 -.4
          .9 -.4
          .8  .5
          .6  .4
          .5  .4
          .6  .3
        """)
        T = target_rotation(A,H)
        print(T)
        L, T = promax(A)
