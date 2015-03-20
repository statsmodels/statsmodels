from __future__ import print_function, absolute_import, division
import numpy as np
from ..compat.python import range
from scipy import optimize
from ._grid_interpolation import GridInterpolator


class LeaveOneOut(object):
    """
    Implement full LeaveOneOut method.

    Parameters
    ----------
    data: tuple of ndarray
        Data to process.
    is_sel: tuple of bool
        Of same length as data, indicate for each array if they are to be filtered or not.
    npts: int
        Number of points in the dataset
    """
    def __init__(self, data, is_sel, npts):
        self.data = data
        self.is_sel = is_sel
        self.npts = npts

    @property
    def nb_tests(self):
        return self.npts

    def __iter__(self):
        data = self.data
        is_sel = self.is_sel
        sel = np.ones((self.npts,), dtype=bool)
        n = self.npts
        for i in range(n):
            sel[i] = False
            yield (i, tuple(d[sel] if is_sel[i] else d for i, d in enumerate(data)))
            sel[i] = True


class LeaveOneOutSampling(object):
    """
    Implement a heureustic for the LeaveOneOut method in which only a sampling of the data are left out.

    Parameters
    ----------
    data: tuple of ndarray
        Data to process.
    is_sel: tuple of bool
        Of same length as data, indicate for each array if they are to be filtered or not.
    npts: int
        Number of points in the dataset
    sampling: int
        Number of points to sample. Must be striclty less than npts.
    """
    def __init__(self, data, is_sel, npts, sampling):
        self.data = data
        self.is_sel = is_sel
        self.npts = npts
        self.sampling = sampling
        self.ks = np.random.permutation(npts)[:sampling]

    @property
    def nb_tests(self):
        return self.sampling

    def __iter__(self):
        data = self.data
        is_sel = self.is_sel
        n = self.npts
        sel = np.ones((n,), dtype=bool)
        for k in self.ks:
            sel[k] = False
            yield (k, tuple(d[sel] if is_sel[i] else d for i, d in enumerate(data)))
            sel[k] = True


class LeaveKOutFolding(object):
    """
    Implement a variation on Leave-One-Out where the dataset is tiled into k folds. At each iteration, one fold is used
    for testing, while the others are used for fitting.

    Parameters
    ----------
    data: tuple of ndarray
        Data to process.
    is_sel: tuple of bool
        Of same length as data, indicate for each array if they are to be filtered or not.
    npts: int
        Number of points in the dataset
    folding: int
        Number of folds to use.
    repeats: int
        Number of repeats for the folding.
    """
    def __init__(self, data, is_sel, npts, folding, repeats):
        self.data = data
        self.is_sel = is_sel
        self.npts = npts
        self.folding = folding
        self.repeats = repeats
        fold_size = npts // folding
        rem = npts % folding
        folds = [None] * (folding*repeats)
        for n in range(repeats):
            idx = np.random.permutation(npts)
            cur_idx = 0
            for i in range(rem):
                end_idx = cur_idx + fold_size + 1
                folds[n*folding + i] = idx[cur_idx:end_idx]
                cur_idx = end_idx

            for i in range(rem, folding):
                end_idx = cur_idx + fold_size
                folds[n*folding + i] = idx[cur_idx:end_idx]
                cur_idx = end_idx

        self.folds = folds

    @property
    def nb_tests(self):
        return self.npts

    def __iter__(self):
        data = self.data
        is_sel = self.is_sel
        n = self.npts
        sel = np.ones((n,), dtype=bool)
        for f in self.folds:
            sel[f] = False
            yield (f, tuple(d[sel] if is_sel[i] else d for i, d in enumerate(data)))
            sel[f] = True


def leave_some_out(exog, *data, **kwords):
    '''
    This function selected between the various LeaveOut objects.

    Each object will take a list of arrays. The first array must be the exogeneous dataset. Other arrays are either
    "scalar" or arrays. If they are scalars, they will be passed along at each yield, if they are arrays, they must have
    the same length as the exogeneous dataset and they will be filtered in the same way.

    The object can be iterated on and return a tuples whose first element is the index(es) of the element(s) left out,
    and the second element is a tuple of same size of `data` with what is to be used for the arrays.

    If no parameter is specified beside the data, then an exhaustive leave-one-out is performed. 'sampling' and
    'folding' parameters are exclusive and cannot be both specified.

    Parameters
    ----------
    exog: ndarray
        1D or 2D array with the data to fit. The first dimension is the number of points in the dataset.
    *data: tuple
        Other arrays or values to select for. If the value doesn't have the same length as exog, then it will be sent
        as-is all the times. Otherwise, it will be selected like exog.
    sampling: int
        Instead of an exhaustive leave-one-out, a random sub-sample is iterated over
    folding: int
        The exogeneous dataset is split into k groups of same length. For each iteration, (k-1) groups are used for
        fitting and the last one is used for testing.
    repeats: int
        Together with `folding`, `repeat` indicates we will use more than one folding.
    '''
    sampling = kwords.get('sampling', None)
    folding = kwords.get('folding', None)
    repeats = int(kwords.get('repeats', 1))
    if sampling is not None and folding is not None:
        raise ValueError("You can only specify one of 'folding' or 'sampling'")
    data = (exog,) + data
    npts = exog.shape[0]
    is_sel = [d.ndim > 0 and d.shape[0] == npts for d in data]
    if sampling is not None and sampling > npts:
        sampling = None

    if sampling is not None:
        return LeaveOneOutSampling(data, is_sel, npts, sampling)
    if folding is not None:
        return LeaveKOutFolding(data, is_sel, npts, folding, repeats)
    return LeaveOneOut(data, is_sel, npts)


class ContinuousIMSE(object):
    """
    Compute the integrated mean square error for continuous axes.

    Notes
    -----
    We need to check how different it would be for discrete axes.
    """
    def __init__(self, model, initial_method=None, grid_size=None, use_grid=False, **loo_args):
        from . import bandwidths
        test_model = model.copy()
        if initial_method is None:
            test_model.bandwidth = bandwidths.Multivariate()
        else:
            test_model.bandwidth = initial_method
        test_est = test_model.fit()
        #print("Initial bandwidth: {0}".format(test_est.bandwidth))

        LOO_model = model.copy()
        LOO_model.bandwidth = test_est.bandwidth
        LOO_est = LOO_model.fit()

        self.LOO = leave_some_out(test_est.exog, test_est.weights, test_est.adjust, **loo_args)
        self.bw_min = test_est.bandwidth * 1e-3
        self.test_est = test_est
        self.LOO_est = LOO_est
        self.grid_size = grid_size
        self.use_grid = use_grid

    @property
    def init_bandwidth(self):
        return self.test_est.bandwidth

    def __call__(self, bw):
        if np.any(bw <= self.bw_min):
            return np.inf
        LOO_est = self.LOO_est
        test_est = self.test_est

        LOO_est.bandwidth = test_est.bandwidth = bw
        exog = test_est.exog
        Fx, Fy = test_est.grid(N=self.grid_size)
        F = Fx.integrate(Fy ** 2)
        L = 0
        use_grid = self.use_grid
        interp = None
        for i, (Xi, Wi, Li) in self.LOO:
            LOO_est.update_inputs(Xi, Wi, Li)
            if use_grid:
                gr, pdf = LOO_est.grid(N=self.grid_size)
                interp = GridInterpolator(gr, pdf)
                vals = interp(exog[i])
            else:
                vals = LOO_est.pdf(exog[i])
            L += np.sum(vals)
        return F - 2 * L / self.LOO.nb_tests


class lsq_crossvalidation(object):
    r"""
    Implement the Cross-Validation Least Square bandwidth estimation method.

    Notes
    -----
    For more details see pp. 16, 27 in Ref. [1] (see module docstring).

    Returns the value of the bandwidth that maximizes the integrated mean
    square error between the estimated and actual distribution.  The
    integrated mean square error (IMSE) is given by:

    .. math:: \int\left[\hat{f}(x)-f(x)\right]^{2}dx

    This is the general formula for the IMSE.

    Parameters
    ----------
    imse: class
        Class from which the Integrated Mean Square Error object is created.. If not provided, it will use :py:class:`ContinuousIMSE`
    imse_args: dictionary
        Arguments for the creation of the IMSE object.
    """

    def __init__(self, imse=None, imse_args={}):
        if imse is None:
            self.imse = ContinuousIMSE
        else:
            self.imse = imse
        self.imse_args = imse_args

    def __call__(self, model):
        imse = self.imse(model, **self.imse_args)
        res = optimize.minimize(imse, x0=imse.init_bandwidth, tol=1e-3, options=dict(maxiter=1e3), method='Nelder-Mead')
        if not res.success:
            print("Error, could not find minimum: '{0}'".format(res.message))
            return imse.init_bandwidth
        return res.x
