import copy

import numpy as np
from numpy.linalg import pinv
from statsmodels.sandbox import utils_old as utils

class ContrastResults(object):
    """
    Results from looking at a particular contrast of coefficients in
    a parametric model. The class does nothing, it is a container
    for the results from T and F contrasts.
    """

    def __init__(self, t=None, F=None, sd=None, effect=None, df_denom=None,
                 df_num=None):
        if F is not None:
            self.F = F
            self.df_denom = df_denom
            self.df_num = df_num
        else:
            self.t = t
            self.sd = sd
            self.effect = effect
            self.df_denom = df_denom

    def __array__(self):
        if hasattr(self, "F"):
            return self.F
        else:
            return self.t

    def __str__(self):
        if hasattr(self, 'F'):
            return '<F contrast: F=%s, df_denom=%d, df_num=%d>' % \
                   (repr(self.F), self.df_denom, self.df_num)
        else:
            return '<T contrast: effect=%s, sd=%s, t=%s, df_denom=%d>' % \
                   (repr(self.effect), repr(self.sd), repr(self.t), self.df_denom)


class Contrast(object):
    """
    This class is used to construct contrast matrices in regression models.
    They are specified by a (term, formula) pair.

    The term, T,  is a linear combination of columns of the design
    matrix D=formula(). The matrix attribute is
    a contrast matrix C so that

    colspan(dot(D, C)) = colspan(dot(D, dot(pinv(D), T)))

    where pinv(D) is the generalized inverse of D. Further, the matrix

    Tnew = dot(C, D)

    is full rank. The rank attribute is the rank of

    dot(D, dot(pinv(D), T))

    In a regression model, the contrast tests that E(dot(Tnew, Y)) = 0
    for each column of Tnew.

    """

    def __init__(self, term, formula, name=''):
        self.term = term
        self.formula = formula
        if name is '':
            self.name = str(term)
        else:
            self.name = name

    def __str__(self):
        return '<contrast:%s>' % \
               repr({'term':str(self.term), 'formula':str(self.formula)})

    def compute_matrix(self, *args, **kw):
        """
        Construct a contrast matrix C so that

        colspan(dot(D, C)) = colspan(dot(D, dot(pinv(D), T)))

        where pinv(D) is the generalized inverse of D=self.D=self.formula().

        If the design, self.D is already set,
        then evaldesign can be set to False.
        """

        t = copy.copy(self.term)
        t.namespace = self.formula.namespace
        T = np.transpose(np.array(t(*args, **kw)))

        if T.ndim == 1:
            T.shape = (T.shape[0], 1)

        self.T = utils.clean0(T)

        self.D = self.formula.design(*args, **kw)

        self._matrix = contrastfromcols(self.T, self.D)
        try:
            self.rank = self.matrix.shape[1]
        except:
            self.rank = 1

    def _get_matrix(self):
        """
        This will fail if the formula needs arguments to construct
        the design.
        """
        if not hasattr(self, "_matrix"):
            self.compute_matrix()
        return self._matrix
    matrix = property(_get_matrix)

def contrastfromcols(L, D, pseudo=None):
    """
    From an n x p design matrix D and a matrix L, tries
    to determine a p x q contrast matrix C which
    determines a contrast of full rank, i.e. the
    n x q matrix

    dot(transpose(C), pinv(D))

    is full rank.

    L must satisfy either L.shape[0] == n or L.shape[1] == p.

    If L.shape[0] == n, then L is thought of as representing
    columns in the column space of D.

    If L.shape[1] == p, then L is thought of as what is known
    as a contrast matrix. In this case, this function returns an estimable
    contrast corresponding to the dot(D, L.T)

    Note that this always produces a meaningful contrast, not always
    with the intended properties because q is always non-zero unless
    L is identically 0. That is, it produces a contrast that spans
    the column space of L (after projection onto the column space of D).

    """

    L = np.asarray(L)
    D = np.asarray(D)

    n, p = D.shape

    if L.shape[0] != n and L.shape[1] != p:
        raise ValueError('shape of L and D mismatched')

    if pseudo is None:
        pseudo = pinv(D)

    if L.shape[0] == n:
        C = np.dot(pseudo, L).T
    else:
        C = L
        C = np.dot(pseudo, np.dot(D, C.T)).T

    Lp = np.dot(D, C.T)

    if len(Lp.shape) == 1:
        Lp.shape = (n, 1)

    if utils.rank(Lp) != Lp.shape[1]:
        Lp = utils.fullrank(Lp)
        C = np.dot(pseudo, Lp).T

    return np.squeeze(C)
