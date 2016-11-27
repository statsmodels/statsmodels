# -*- coding: utf-8 -*-
"""
This file contains a Python version of the gradient projection rotation
algorithms (GPA) developed by Bernaards, C.A. and Jennrich, R.I.
The code is based on code developed Bernaards, C.A. and Jennrich, R.I.
and is ported and made available with permission of the authors.

References
----------
[1] Bernaards, C.A. and Jennrich, R.I. (2005) Gradient Projection Algorithms and Software for Arbitrary Rotation Criteria in Factor Analysis. Educational and Psychological Measurement, 65 (5), 676-696.

[2] Jennrich, R.I. (2001). A simple general procedure for orthogonal rotation. Psychometrika, 66, 289-306.

[3] Jennrich, R.I. (2002). A simple general method for oblique rotation. Psychometrika, 67, 7-19.

[4] http://www.stat.ucla.edu/research/gpa/matlab.net

[5] http://www.stat.ucla.edu/research/gpa/GPderfree.txt
"""

from __future__ import division
import numpy as np
import unittest

def GPA(A, ff=None, vgQ=None, T=None, max_tries=501,
       rotation_method = 'orthogonal', tol=1e-5):
    """
    The gradient projection algorithm (GPA) minimizes a target function
    :math:`\phi(L)`, where :math:`L` is a matrix with rotated factors.

    For orthogonal rotation methods :math:`L=AT`, where :math:`T` is an
    orthogonal matrix. For oblique rotation matrices :math:`L=A(T^*)^{-1}`,
    where :math:`T` is a normal matrix, i.e., :math:`TT^*=T^*T`. Oblique
    rotations relax the orthogonality constraint in order to gain simplicity
    in the interpretation.

    Parameters
    ----------
    A : numpy matrix
        non rotated factors
    T : numpy matrix (default identity matrix)
        initial guess of rotation matrix
    ff : function (defualt None)
        criterion :math:`\phi` to optimize. Should have A, T, L as keyword arguments
        and mapping to a float. Only used (and required) if vgQ is not provided.
    vgQ : function (defualt None)
        criterion :math:`\phi` to optimize and its derivative. Should have  A, T, L as
        keyword arguments and mapping to a tuple containing a
        float and vector. Can be omitted if ff is provided.
    max_tries : integer (default 501)
        maximum number of iterations
    rotation_method : string
        should be one of {orthogonal, oblique}
    tol : float
        stop criterion, algorithm stops if Frobenius norm of gradient is smaller
        then tol
    """
    #pre processing
    if rotation_method not in ['orthogonal', 'oblique']:
        raise ValueError('rotation_method should be one of {orthogonal, oblique}')
    if vgQ is None:
        if ff is None:
            raise ValueError('ff should be provided if vgQ is not')
        derivative_free=True
        Gff = lambda x: Gf(x, lambda y: ff(T=y,A=A,L=None))
    else:
        derivative_free=False
    if T is None:
        T=np.eye(A.shape[1])
    #pre processing for iteration
    al=1
    table=[]
    #pre processing for iteration: initialize f and G
    if derivative_free:
        f=ff(T=T,A=A,L=None)
        G=Gff(T)
    elif rotation_method == 'orthogonal': # and not derivative_free
        L= A.dot(T)
        f,Gq = vgQ(L=L)
        G=(A.T).dot(Gq)
    else: #i.e. rotation_method == 'oblique' and not derivative_free
        Ti=np.linalg.inv(T)
        L= A.dot(Ti.T)
        f,Gq = vgQ(L=L)
        G=-((L.T).dot(Gq).dot(Ti)).T
    #iteration
    for i_try in range(0,max_tries):
        #determine Gp
        if rotation_method == 'orthogonal':
            M=(T.T).dot(G)
            S=(M+M.T)/2
            Gp=G-T.dot(S)
        else: #i.e. if rotation_method == 'oblique':
            Gp=G-T.dot(np.diag(np.sum(T*G,axis=0)))
        s=np.linalg.norm(Gp,'fro');
        table.append([i_try, f, np.log10(s), al])
        #if we are close stop
        if s < tol: break
        #update T
        al=2*al
        for i in range(11):
            #determine Tt
            X=T-al*Gp
            if rotation_method == 'orthogonal':
                U,D,V=np.linalg.svd(X,full_matrices=False)
                Tt=U.dot(V)
            else: #i.e. if rotation_method == 'oblique':
                v=1/np.sqrt(np.sum(X**2,axis=0))
                Tt=X.dot(np.diag(v))
            #calculate objective using Tt
            if derivative_free:
                ft=ff(T=Tt,A=A,L=None)
            elif rotation_method == 'orthogonal': # and not derivative_free
                L=A.dot(Tt)
                ft,Gq=vgQ(L=L);
            else: #i.e. rotation_method == 'oblique' and not derivative_free
                Ti=np.linalg.inv(Tt)
                L= A.dot(Ti.T)
                ft,Gq = vgQ(L=L)
            #if sufficient improvement in objective -> use this T
            if ft<f-.5*s**2*al: break
            al=al/2
        #post processing for next iteration
        T=Tt
        f=ft
        if derivative_free:
            G=Gff(T)
        elif rotation_method == 'orthogonal': # and not derivative_free
            G=(A.T).dot(Gq)
        else: #i.e. rotation_method == 'oblique' and not derivative_free
            G=-((L.T).dot(Gq).dot(Ti)).T
    #post processing
    Th=T
    Lh = rotateA(A,T,rotation_method=rotation_method)
    Phi = (T.T).dot(T)
    return Lh, Phi, Th, table

def Gf(T, ff):
    """
    Subroutine for the gradient of f using numerical derivatives.
    """
    k=T.shape[0]
    ep = 1e-4
    G=np.zeros((k,k))
    for r in range(k):
        for s in range(k):
            dT=np.zeros((k,k))
            dT[r,s]=ep
            G[r,s]=(ff(T+dT)-ff(T-dT))/(2*ep);
    return G

def rotateA(A, T, rotation_method='orthogonal'):
    r"""
    For orthogonal rotation methods :math:`L=AT`, where :math:`T` is an
    orthogonal matrix. For oblique rotation matrices :math:`L=A(T^*)^{-1}`,
    where :math:`T` is a normal matrix, i.e., :math:`TT^*=T^*T`. Oblique
    rotations relax the orthogonality constraint in order to gain simplicity
    in the interpretation.
    """
    if rotation_method == 'orthogonal':
        L=A.dot(T)
    elif rotation_method == 'oblique':
        L=A.dot(np.linalg.inv(T.T))
    else: #i.e. if rotation_method == 'oblique':
        raise ValueError('rotation_method should be one of {orthogonal, oblique}')
    return L

def oblimin_objective(L=None, A=None, T=None, gamma=0,
                      rotation_method='orthogonal',
                      return_gradient=True):
    r"""
    Objective function for the oblimin family for orthogonal or
    oblique rotation wich minimizes:

    .. math::
        \phi(L) = \frac{1}{4}(L\circ L,(I-\gamma C)(L\circ L)N),

    where :math:`L` is a :math:`p\times k` matrix, :math:`N` is :math:`k\times k`
    matrix with zeros on the diagonal and ones elsewhere, :math:`C` is a
    :math:`p\times p` matrix with elements equal to :math:`1/p`,
    :math:`(X,Y)=\operatorname{Tr}(X^*Y)` is the Frobenius norm and :math:`\circ`
    is the element-wise product or Hadamard product.

    The gradient is given by

    .. math::
        L\circ\left[(I-\gamma C) (L \circ L)N\right].

    Either :math:`L` should be provided or :math:`A` and :math:`T` should be provided.

    For orthogonal rotations :math:`L` satisfies

    .. math::
        L =  AT,

    where :math:`T` is an orthogonal matrix. For oblique rotations :math:`L` satisfies

    .. math::
        L =  A(T^*)^{-1},

    where :math:`T` is a normal matrix.

    The oblimin family is parametrized by the parameter :math:`\gamma`. For orthogonal
    rotations:

    * :math:`\gamma=0` corresponds to quartimax,
    * :math:`\gamma=\frac{1}{2}` corresponds to biquartimax,
    * :math:`\gamma=1` corresponds to varimax,
    * :math:`\gamma=\frac{1}{p}` corresponds to equamax.
    For oblique rotations rotations:

    * :math:`\gamma=0` corresponds to quartimin,
    * :math:`\gamma=\frac{1}{2}` corresponds to biquartimin.

    Parametes
    ---------
    L : numpy matrix (default None)
        rotated factors, i.e., :math:`L=A(T^*)^{-1}=AT`
    A : numpy matrix (default None)
        non rotated factors
    T : numpy matrix (default None)
        rotation matrix
    gamma : float (default 0)
        a parameter
    rotation_method : string
        should be one of {orthogonal, oblique}
    return_gradient : boolean (default True)
        toggles return of gradient
    """
    if L is None:
        assert(A is not None and T is not None)
        L=rotateA(A,T,rotation_method=rotation_method)
    p,k = L.shape
    L2=L**2
    N=np.ones((k,k))-np.eye(k)
    if np.isclose(gamma,0):
        X=L2.dot(N)
    else:
        C=np.ones((p,p))/p
        X=(np.eye(p)-gamma*C).dot(L2).dot(N)
    phi=np.sum(L2*X)/4
    if return_gradient:
        Gphi=L*X
        return phi, Gphi
    else:
        return phi

def orthomax_objective(L=None, A=None, T=None, gamma=0, return_gradient=True):
    r"""
    Objective function for the orthomax family for orthogonal
    rotation wich minimizes the following objective:

    .. math::
        \phi(L) = -\frac{1}{4}(L\circ L,(I-\gamma C)(L\circ L)),

    where :math:`0\leq\gamma\leq1`, :math:`L` is a :math:`p\times k` matrix,
    :math:`C` is a  :math:`p\times p` matrix with elements equal to :math:`1/p`,
    :math:`(X,Y)=\operatorname{Tr}(X^*Y)` is the Frobenius norm and :math:`\circ`
    is the element-wise product or Hadamard product.

    Either :math:`L` should be provided or :math:`A` and :math:`T` should be provided.

    For orthogonal rotations :math:`L` satisfies

    .. math::
        L =  AT,

    where :math:`T` is an orthogonal matrix.

    The orthomax family is parametrized by the parameter :math:`\gamma`:

    * :math:`\gamma=0` corresponds to quartimax,
    * :math:`\gamma=\frac{1}{2}` corresponds to biquartimax,
    * :math:`\gamma=1` corresponds to varimax,
    * :math:`\gamma=\frac{1}{p}` corresponds to equamax.

    Parametes
    ---------
    L : numpy matrix (default None)
        rotated factors, i.e., :math:`L=A(T^*)^{-1}=AT`
    A : numpy matrix (default None)
        non rotated factors
    T : numpy matrix (default None)
        rotation matrix
    gamma : float (default 0)
        a parameter
    return_gradient : boolean (default True)
        toggles return of gradient
    """
    assert 0<=gamma<=1, "Gamma should be between 0 and 1"
    if L is None:
        assert(A is not None and T is not None)
        L=rotateA(A,T,rotation_method='orthogonal')
    p,k = L.shape
    L2=L**2
    if np.isclose(gamma,0):
        X=L2
    else:
        C=np.ones((p,p))/p
        X=(np.eye(p)-gamma*C).dot(L2)
    phi=-np.sum(L2*X)/4
    if return_gradient:
        Gphi=-L*X
        return phi, Gphi
    else:
        return phi

def CF_objective(L=None, A=None, T=None, kappa=0,
                 rotation_method='orthogonal',
                 return_gradient=True):
    r"""
    Objective function for the Crawford-Ferguson family for orthogonal
    and oblique rotation wich minimizes the following objective:

    .. math::
        \phi(L) =\frac{1-\kappa}{4} (L\circ L,(L\circ L)N)
                  -\frac{1}{4}(L\circ L,M(L\circ L)),

    where :math:`0\leq\kappa\leq1`, :math:`L` is a :math:`p\times k` matrix,
    :math:`N` is :math:`k\times k` matrix with zeros on the diagonal and ones elsewhere,
    :math:`M` is :math:`p\times p` matrix with zeros on the diagonal and ones elsewhere
    :math:`(X,Y)=\operatorname{Tr}(X^*Y)` is the Frobenius norm and :math:`\circ`
    is the element-wise product or Hadamard product.

    The gradient is given by

    .. math::
       d\phi(L) = (1-\kappa) L\circ\left[(L\circ L)N\right]
                   -\kappa L\circ \left[M(L\circ L)\right].

    Either :math:`L` should be provided or :math:`A` and :math:`T` should be provided.

    For orthogonal rotations :math:`L` satisfies

    .. math::
        L =  AT,

    where :math:`T` is an orthogonal matrix. For oblique rotations :math:`L` satisfies

    .. math::
        L =  A(T^*)^{-1},

    where :math:`T` is a normal matrix.

    For orthogonal rotations the oblimin (and orthomax) family of rotations is
    equivalent to the Crawford-Ferguson family. To be more precise:

    * :math:`\kappa=0` corresponds to quartimax,
    * :math:`\kappa=\frac{1}{p}` corresponds to variamx,
    * :math:`\kappa=\frac{k-1}{p+k-2}` corresponds to parsimax,
    * :math:`\kappa=1` corresponds to factor parsimony.

    Parametes
    ---------
    L : numpy matrix (default None)
        rotated factors, i.e., :math:`L=A(T^*)^{-1}=AT`
    A : numpy matrix (default None)
        non rotated factors
    T : numpy matrix (default None)
        rotation matrix
    gamma : float (default 0)
        a parameter
    rotation_method : string
        should be one of {orthogonal, oblique}
    return_gradient : boolean (default True)
        toggles return of gradient
    """
    assert 0<=kappa<=1, "Kappa should be between 0 and 1"
    if L is None:
        assert(A is not None and T is not None)
        L=rotateA(A,T,rotation_method=rotation_method)
    p,k = L.shape
    L2=L**2
    X=None
    if not np.isclose(kappa,1):
        N=np.ones((k,k))-np.eye(k)
        X=(1-kappa)*L2.dot(N)
    if not np.isclose(kappa,0):
        M=np.ones((p,p))-np.eye(p)
        if X is None:
            X=kappa*M.dot(L2)
        else:
            X+=kappa*M.dot(L2)
    phi=np.sum(L2*X)/4
    if return_gradient:
        Gphi=L*X
        return phi, Gphi
    else:
        return phi

def vgQ_target(H, L=None, A=None, T=None, rotation_method='orthogonal'):
    r"""
    Subroutine for the value of vgQ using orthogonal or oblique rotation towards a target matrix,
    i.e., we minimize:

    .. math::
        \phi(L) =\frac{1}{2}\|L-H\|^2

    and the gradient is given by

    .. math::
        d\phi(L)=L-H.

    Either :math:`L` should be provided or :math:`A` and :math:`T` should be provided.

    For orthogonal rotations :math:`L` satisfies

    .. math::
        L =  AT,

    where :math:`T` is an orthogonal matrix. For oblique rotations :math:`L` satisfies

    .. math::
        L =  A(T^*)^{-1},

    where :math:`T` is a normal matrix.

    Parametes
    ---------
    H : numpy matrix
        target matrix
    L : numpy matrix (default None)
        rotated factors, i.e., :math:`L=A(T^*)^{-1}=AT`
    A : numpy matrix (default None)
        non rotated factors
    T : numpy matrix (default None)
        rotation matrix
    rotation_method : string
        should be one of {orthogonal, oblique}
    """
    if L is None:
        assert(A is not None and T is not None)
        L=rotateA(A,T,rotation_method=rotation_method)
    q=np.linalg.norm(L-H,'fro')**2
    Gq=2*(L-H);
    return q, Gq

def ff_target(H, L=None, A=None, T=None, rotation_method='orthogonal'):
    r"""
    Subroutine for the value of f using (orthogonal or oblique) rotation towards a target matrix,
    i.e., we minimize:

    .. math::
        \phi(L) =\frac{1}{2}\|L-H\|^2.

    Either :math:`L` should be provided or :math:`A` and :math:`T` should be
    provided. For orthogonal rotations :math:`L` satisfies

    .. math::
        L =  AT,

    where :math:`T` is an orthogonal matrix. For oblique rotations :math:`L` satisfies

    .. math::
        L =  A(T^*)^{-1},

    where :math:`T` is a normal matrix.

    Parametes
    ---------
    H : numpy matrix
        target matrix
    L : numpy matrix (default None)
        rotated factors, i.e., :math:`L=A(T^*)^{-1}=AT`
    A : numpy matrix (default None)
        non rotated factors
    T : numpy matrix (default None)
        rotation matrix
    rotation_method : string
        should be one of {orthogonal, oblique}
    """
    if L is None:
        assert(A is not None and T is not None)
        L=rotateA(A,T,rotation_method=rotation_method)
    return np.linalg.norm(L-H,'fro')**2

def vgQ_partial_target(H, W=None, L=None, A=None, T=None):
    r"""
    Subroutine for the value of vgQ using orthogonal rotation towards a partial
    target matrix, i.e., we minimize:

    .. math::
        \phi(L) =\frac{1}{2}\|W\circ(L-H)\|^2,

    where :math:`\circ` is the element-wise product or Hadamard product and :math:`W`
    is a matrix whose entries can only be one or zero. The gradient is given by

    .. math::
        d\phi(L)=W\circ(L-H).

    Either :math:`L` should be provided or :math:`A` and :math:`T` should be provided.

    For orthogonal rotations :math:`L` satisfies

    .. math::
        L =  AT,

    where :math:`T` is an orthogonal matrix.

    Parametes
    ---------
    H : numpy matrix
        target matrix
    W : numpy matrix (default matrix with equal weight one for all entries)
        matrix with weights, entries can either be one or zero
    L : numpy matrix (default None)
        rotated factors, i.e., :math:`L=A(T^*)^{-1}=AT`
    A : numpy matrix (default None)
        non rotated factors
    T : numpy matrix (default None)
        rotation matrix
    """
    if W is None:
        return vgQ_target(H, L=L, A=A, T=T)
    if L is None:
        assert(A is not None and T is not None)
        L=rotateA(A,T,rotation_method='orthogonal')
    q=np.linalg.norm(W*(L-H),'fro')**2
    Gq=2*W*(L-H)
    return q, Gq

def ff_partial_target(H, W=None, L=None, A=None, T=None):
    r"""
    Subroutine for the value of vgQ using orthogonal rotation towards a partial
    target matrix, i.e., we minimize:

    .. math::
        \phi(L) =\frac{1}{2}\|W\circ(L-H)\|^2,

    where :math:`\circ` is the element-wise product or Hadamard product and :math:`W`
    is a matrix whose entries can only be one or zero. Either :math:`L` should be
    provided or :math:`A` and :math:`T` should be provided.

    For orthogonal rotations :math:`L` satisfies

    .. math::
        L =  AT,

    where :math:`T` is an orthogonal matrix.

    Parametes
    ---------
    H : numpy matrix
        target matrix
    W : numpy matrix (default matrix with equal weight one for all entries)
        matrix with weights, entries can either be one or zero
    L : numpy matrix (default None)
        rotated factors, i.e., :math:`L=A(T^*)^{-1}=AT`
    A : numpy matrix (default None)
        non rotated factors
    T : numpy matrix (default None)
        rotation matrix
    """
    if W is None:
        return ff_target(H, L=L, A=A, T=T)
    if L is None:
        assert(A is not None and T is not None)
        L=rotateA(A,T,rotation_method='orthogonal')
    q=np.linalg.norm(W*(L-H),'fro')**2
    return q




class unittests(unittest.TestCase):

    @staticmethod
    def str2matrix(A):
        A=A.lstrip().rstrip().split('\n')
        A=np.array([row.split() for row in A]).astype(np.float)
        return A

    @classmethod
    def get_A(cls):
        return cls.str2matrix("""
         .830 -.396
         .818 -.469
         .777 -.470
         .798 -.401
         .786  .500
         .672  .458
         .594  .444
         .647  .333
        """)

    @classmethod
    def get_quartimin_example(cls):
        A = cls.get_A()
        table_required = cls.str2matrix("""
          0.00000    0.42806   -0.46393    1.00000
        1.00000    0.41311   -0.57313    0.25000
        2.00000    0.38238   -0.36652    0.50000
        3.00000    0.31850   -0.21011    0.50000
        4.00000    0.20937   -0.13838    0.50000
        5.00000    0.12379   -0.35583    0.25000
        6.00000    0.04289   -0.53244    0.50000
        7.00000    0.01098   -0.86649    0.50000
        8.00000    0.00566   -1.65798    0.50000
        9.00000    0.00558   -2.13212    0.25000
       10.00000    0.00557   -2.49020    0.25000
       11.00000    0.00557   -2.84585    0.25000
       12.00000    0.00557   -3.20320    0.25000
       13.00000    0.00557   -3.56143    0.25000
       14.00000    0.00557   -3.92005    0.25000
       15.00000    0.00557   -4.27885    0.25000
       16.00000    0.00557   -4.63772    0.25000
       17.00000    0.00557   -4.99663    0.25000
       18.00000    0.00557   -5.35555    0.25000
        """)
        L_required = cls.str2matrix("""
       0.891822   0.056015
       0.953680  -0.023246
       0.929150  -0.046503
       0.876683   0.033658
       0.013701   0.925000
      -0.017265   0.821253
      -0.052445   0.764953
       0.085890   0.683115
        """)
        return A, table_required, L_required

    @classmethod
    def get_biquartimin_example(cls):
        A = cls.get_A()
        table_required = cls.str2matrix("""
            0.00000    0.21632   -0.54955    1.00000
            1.00000    0.19519   -0.46174    0.50000
            2.00000    0.09479   -0.16365    1.00000
            3.00000   -0.06302   -0.32096    0.50000
            4.00000   -0.21304   -0.46562    1.00000
            5.00000   -0.33199   -0.33287    1.00000
            6.00000   -0.35108   -0.63990    0.12500
            7.00000   -0.35543   -1.20916    0.12500
            8.00000   -0.35568   -2.61213    0.12500
            9.00000   -0.35568   -2.97910    0.06250
           10.00000   -0.35568   -3.32645    0.06250
           11.00000   -0.35568   -3.66021    0.06250
           12.00000   -0.35568   -3.98564    0.06250
           13.00000   -0.35568   -4.30635    0.06250
           14.00000   -0.35568   -4.62451    0.06250
           15.00000   -0.35568   -4.94133    0.06250
           16.00000   -0.35568   -5.25745    0.06250
        """)
        L_required = cls.str2matrix("""
           1.01753  -0.13657
           1.11338  -0.24643
           1.09200  -0.26890
           1.00676  -0.16010
          -0.26534   1.11371
          -0.26972   0.99553
          -0.29341   0.93561
          -0.10806   0.80513
        """)
        return A, table_required, L_required

    @classmethod
    def get_biquartimin_example_derivative_free(cls):
        A = cls.get_A()
        table_required = cls.str2matrix("""
            0.00000    0.21632   -0.54955    1.00000
            1.00000    0.19519   -0.46174    0.50000
            2.00000    0.09479   -0.16365    1.00000
            3.00000   -0.06302   -0.32096    0.50000
            4.00000   -0.21304   -0.46562    1.00000
            5.00000   -0.33199   -0.33287    1.00000
            6.00000   -0.35108   -0.63990    0.12500
            7.00000   -0.35543   -1.20916    0.12500
            8.00000   -0.35568   -2.61213    0.12500
            9.00000   -0.35568   -2.97910    0.06250
           10.00000   -0.35568   -3.32645    0.06250
           11.00000   -0.35568   -3.66021    0.06250
           12.00000   -0.35568   -3.98564    0.06250
           13.00000   -0.35568   -4.30634    0.06250
           14.00000   -0.35568   -4.62451    0.06250
           15.00000   -0.35568   -4.94133    0.06250
           16.00000   -0.35568   -6.32435    0.12500
        """)
        L_required = cls.str2matrix("""
           1.01753  -0.13657
           1.11338  -0.24643
           1.09200  -0.26890
           1.00676  -0.16010
          -0.26534   1.11371
          -0.26972   0.99553
          -0.29342   0.93561
          -0.10806   0.80513
        """)
        return A, table_required, L_required

    @classmethod
    def get_quartimax_example_derivative_free(cls):
        A = cls.get_A()
        table_required = cls.str2matrix("""
        0.00000   -0.72073   -0.65498    1.00000
        1.00000   -0.88561   -0.34614    2.00000
        2.00000   -1.01992   -1.07152    1.00000
        3.00000   -1.02237   -1.51373    0.50000
        4.00000   -1.02269   -1.96205    0.50000
        5.00000   -1.02273   -2.41116    0.50000
        6.00000   -1.02273   -2.86037    0.50000
        7.00000   -1.02273   -3.30959    0.50000
        8.00000   -1.02273   -3.75881    0.50000
        9.00000   -1.02273   -4.20804    0.50000
       10.00000   -1.02273   -4.65726    0.50000
       11.00000   -1.02273   -5.10648    0.50000
        """)
        L_required = cls.str2matrix("""
       0.89876   0.19482
       0.93394   0.12974
       0.90213   0.10386
       0.87651   0.17128
       0.31558   0.87647
       0.25113   0.77349
       0.19801   0.71468
       0.30786   0.65933
        """)
        return A, table_required, L_required

    def test_orthomax(self):
        """
        Quartimax example
        http://www.stat.ucla.edu/research/gpa
        """
        A=self.get_A()
        vgQ = lambda L=None, A=None, T=None: orthomax_objective(L=L, A=A, T=T, gamma=0,
               return_gradient=True)
        L, phi, T, table = GPA(A, vgQ=vgQ, rotation_method='orthogonal')
        table_required = self.str2matrix("""
         0.00000   -0.72073   -0.65498    1.00000
         1.00000   -0.88561   -0.34614    2.00000
         2.00000   -1.01992   -1.07152    1.00000
         3.00000   -1.02237   -1.51373    0.50000
         4.00000   -1.02269   -1.96205    0.50000
         5.00000   -1.02273   -2.41116    0.50000
         6.00000   -1.02273   -2.86037    0.50000
         7.00000   -1.02273   -3.30959    0.50000
         8.00000   -1.02273   -3.75881    0.50000
         9.00000   -1.02273   -4.20804    0.50000
        10.00000   -1.02273   -4.65726    0.50000
        11.00000   -1.02273   -5.10648    0.50000
        """)
        L_required = self.str2matrix("""
        0.89876   0.19482
        0.93394   0.12974
        0.90213   0.10386
        0.87651   0.17128
        0.31558   0.87647
        0.25113   0.77349
        0.19801   0.71468
        0.30786   0.65933
        """)
        self.assertTrue(np.allclose(table,table_required,atol=1e-05))
        self.assertTrue(np.allclose(L,L_required,atol=1e-05))
        #oblimin criterion gives same result
        vgQ = lambda L=None, A=None, T=None: oblimin_objective(L=L, A=A, T=T, gamma=0,
                      rotation_method='orthogonal', return_gradient=True)
        L_oblimin, phi2, T2, table2 = GPA(A, vgQ=vgQ, rotation_method='orthogonal')
        self.assertTrue(np.allclose(L,L_oblimin,atol=1e-05))
        #derivative free quartimax
        A, table_required, L_required = self.get_quartimax_example_derivative_free()
        ff = lambda L=None, A=None, T=None: orthomax_objective(L=L, A=A, T=T, gamma=0, return_gradient=False)
        L, phi, T, table = GPA(A, ff=ff, rotation_method='orthogonal')
        self.assertTrue(np.allclose(table,table_required,atol=1e-05))
        self.assertTrue(np.allclose(L,L_required,atol=1e-05))

    def test_equivalence_orthomax_oblimin(self):
        """
        These criteria should be equivalent when restricted to orthogonal rotation.
        See Hartman 1976 page 299.
        """
        A=self.get_A()
        gamma=0 #quartimax
        vgQ = lambda L=None, A=None, T=None: orthomax_objective(L=L, A=A, T=T, gamma=gamma,
               return_gradient=True)
        L_orthomax, phi, T, table = GPA(A, vgQ=vgQ, rotation_method='orthogonal')
        vgQ = lambda L=None, A=None, T=None: oblimin_objective(L=L, A=A, T=T, gamma=gamma,
                      rotation_method='orthogonal', return_gradient=True)
        L_oblimin, phi2, T2, table2 = GPA(A, vgQ=vgQ, rotation_method='orthogonal')
        self.assertTrue(np.allclose(L_orthomax,L_oblimin,atol=1e-05))
        gamma=1 #varimax
        vgQ = lambda L=None, A=None, T=None: orthomax_objective(L=L, A=A, T=T, gamma=gamma,
               return_gradient=True)
        L_orthomax, phi, T, table = GPA(A, vgQ=vgQ, rotation_method='orthogonal')
        vgQ = lambda L=None, A=None, T=None: oblimin_objective(L=L, A=A, T=T, gamma=gamma,
                      rotation_method='orthogonal', return_gradient=True)
        L_oblimin, phi2, T2, table2 = GPA(A, vgQ=vgQ, rotation_method='orthogonal')
        self.assertTrue(np.allclose(L_orthomax,L_oblimin,atol=1e-05))

    def test_orthogonal_target(self):
        """
        Rotation towards target matrix example
        http://www.stat.ucla.edu/research/gpa
        """
        A=self.get_A()
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
        vgQ = lambda L=None, A=None, T=None: vgQ_target(H,L=L,A=A,T=T)
        L, phi, T, table = GPA(A, vgQ=vgQ, rotation_method='orthogonal')
        table_required = self.str2matrix("""
        0.00000   0.05925  -0.61244   1.00000
        1.00000   0.05444  -1.14701   0.12500
        2.00000   0.05403  -1.68194   0.12500
        3.00000   0.05399  -2.21689   0.12500
        4.00000   0.05399  -2.75185   0.12500
        5.00000   0.05399  -3.28681   0.12500
        6.00000   0.05399  -3.82176   0.12500
        7.00000   0.05399  -4.35672   0.12500
        8.00000   0.05399  -4.89168   0.12500
        9.00000   0.05399  -5.42664   0.12500
        """)
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
        self.assertTrue(np.allclose(table,table_required,atol=1e-05))
        self.assertTrue(np.allclose(L,L_required,atol=1e-05))
        ff = lambda L=None, A=None, T=None: ff_target(H,L=L,A=A,T=T)
        L2, phi, T2, table = GPA(A, ff=ff, rotation_method='orthogonal')
        self.assertTrue(np.allclose(L,L2,atol=1e-05))
        self.assertTrue(np.allclose(T,T2,atol=1e-05))
        vgQ = lambda L=None, A=None, T=None: vgQ_target(H,L=L,A=A,T=T, rotation_method='oblique')
        L, phi, T, table = GPA(A, vgQ=vgQ, rotation_method='oblique')
        ff = lambda L=None, A=None, T=None: ff_target(H,L=L,A=A,T=T, rotation_method='oblique')
        L2, phi, T2, table = GPA(A, ff=ff, rotation_method='oblique')
        self.assertTrue(np.allclose(L,L2,atol=1e-05))
        self.assertTrue(np.allclose(T,T2,atol=1e-05))

    def test_orthogonal_partial_target(self):
        """
        Rotation towards target matrix example
        http://www.stat.ucla.edu/research/gpa
        """
        A=self.get_A()
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
        W= self.str2matrix("""
        1 0
        0 1
        0 0
        1 1
        1 0
        1 0
        0 1
        1 0
        """)
        vgQ = lambda L=None, A=None, T=None: vgQ_partial_target(H, W, L=L,A=A,T=T)
        L, phi, T, table = GPA(A, vgQ=vgQ, rotation_method='orthogonal')
        table_required = self.str2matrix("""
         0.00000    0.02559   -0.84194    1.00000
         1.00000    0.02203   -1.27116    0.25000
         2.00000    0.02154   -1.71198    0.25000
         3.00000    0.02148   -2.15713    0.25000
         4.00000    0.02147   -2.60385    0.25000
         5.00000    0.02147   -3.05114    0.25000
         6.00000    0.02147   -3.49863    0.25000
         7.00000    0.02147   -3.94619    0.25000
         8.00000    0.02147   -4.39377    0.25000
         9.00000    0.02147   -4.84137    0.25000
        10.00000    0.02147   -5.28897    0.25000
        """)
        L_required = self.str2matrix("""
        0.84526  -0.36228
        0.83621  -0.43571
        0.79528  -0.43836
        0.81349  -0.36857
        0.76525   0.53122
        0.65303   0.48467
        0.57565   0.46754
        0.63308   0.35876
        """)
        self.assertTrue(np.allclose(table,table_required,atol=1e-05))
        self.assertTrue(np.allclose(L,L_required,atol=1e-05))
        ff = lambda L=None, A=None, T=None: ff_partial_target(H,W,L=L,A=A,T=T)
        L2, phi, T2, table = GPA(A, ff=ff, rotation_method='orthogonal')
        self.assertTrue(np.allclose(L,L2,atol=1e-05))
        self.assertTrue(np.allclose(T,T2,atol=1e-05))

    def test_oblimin(self):
        #quartimin
        A, table_required, L_required = self.get_quartimin_example()
        vgQ = lambda L=None, A=None, T=None: oblimin_objective(L=L, A=A, T=T, gamma=0, rotation_method='oblique')
        L, phi, T, table = GPA(A, vgQ=vgQ, rotation_method='oblique')
        self.assertTrue(np.allclose(table,table_required,atol=1e-05))
        self.assertTrue(np.allclose(L,L_required,atol=1e-05))
        #quartimin derivative free
        ff = lambda L=None, A=None, T=None: oblimin_objective(L=L, A=A, T=T, gamma=0, rotation_method='oblique', return_gradient=False)
        L, phi, T, table = GPA(A, ff=ff, rotation_method='oblique')
        self.assertTrue(np.allclose(L,L_required,atol=1e-05))
        self.assertTrue(np.allclose(table,table_required,atol=1e-05))
        #biquartimin
        A, table_required, L_required = self.get_biquartimin_example()
        vgQ = lambda L=None, A=None, T=None: oblimin_objective(L=L, A=A, T=T, gamma=1/2, rotation_method='oblique')
        L, phi, T, table = GPA(A, vgQ=vgQ, rotation_method='oblique')
        self.assertTrue(np.allclose(table,table_required,atol=1e-05))
        self.assertTrue(np.allclose(L,L_required,atol=1e-05))
        #quartimin derivative free
        A, table_required, L_required = self.get_biquartimin_example_derivative_free()
        ff = lambda L=None, A=None, T=None: oblimin_objective(L=L, A=A, T=T, gamma=1/2, rotation_method='oblique', return_gradient=False)
        L, phi, T, table = GPA(A, ff=ff, rotation_method='oblique')
        self.assertTrue(np.allclose(L,L_required,atol=1e-05))
        self.assertTrue(np.allclose(table,table_required,atol=1e-05))

    def test_CF(self):
        #quartimax
        A, table_required, L_required = self.get_quartimax_example_derivative_free()
        vgQ = lambda L=None, A=None, T=None: CF_objective(L=L, A=A, T=T,
                                                          kappa=0,
                                                          rotation_method='orthogonal',
                                                          return_gradient=True)
        L, phi, T, table = GPA(A, vgQ=vgQ, rotation_method='orthogonal')
        self.assertTrue(np.allclose(L,L_required,atol=1e-05))
        #quartimax derivative free
        ff = lambda L=None, A=None, T=None: CF_objective(L=L, A=A, T=T,
                                                         kappa=0,
                                                         rotation_method='orthogonal',
                                                         return_gradient=False)
        L, phi, T, table = GPA(A, ff=ff, rotation_method='orthogonal')
        self.assertTrue(np.allclose(L,L_required,atol=1e-05))
        #varimax
        p,k=A.shape
        vgQ = lambda L=None, A=None, T=None: orthomax_objective(L=L, A=A, T=T,
                                                                gamma=1,
                                                                return_gradient=True)
        L_vm, phi, T, table = GPA(A, vgQ=vgQ, rotation_method='orthogonal')
        vgQ = lambda L=None, A=None, T=None: CF_objective(L=L, A=A, T=T,
                                                          kappa=1/p,
                                                          rotation_method='orthogonal',
                                                          return_gradient=True)
        L_CF, phi, T, table = GPA(A, vgQ=vgQ, rotation_method='orthogonal')
        ff = lambda L=None, A=None, T=None: CF_objective(L=L, A=A, T=T,
                                                         kappa=1/p,
                                                         rotation_method='orthogonal',
                                                         return_gradient=False)
        L_CF_df, phi, T, table = GPA(A, ff=ff, rotation_method='orthogonal')
        self.assertTrue(np.allclose(L_vm,L_CF,atol=1e-05))
        self.assertTrue(np.allclose(L_CF,L_CF_df,atol=1e-05))

if __name__ == '__main__': # run only if this file is run directly and not when imported
    run_unit_tests=True
    test_only = list() # if list is empty then test all
    #test_only.append('test_CF')
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
        L, phi, T, table = GPA(A, vgQ=vgQ_quartimax, rotation_method='orthogonal')
