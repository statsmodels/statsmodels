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


class CVFunc(object):
    """
    Base class for cross-validation functions.

    This class setup the 'leave some out' objects. The derived class should
    implement the __call__ method.

    Parameters
    ----------
    model: :py:class:`.kde.KDE`
        Model to be fitted
    initial_method: callable or value
        Initial value for the bandwidth
    grid_size: int or tuple of int
        Size of the grid to use to compute the square of the estimated distribution
    use_grid: bool
        If True, instead of evaluating the function at the points needed for
        the cross-validation, the points will be estimated by linear
        interpolation on a grid the same size as the one used for the
        distribution estimation. This is only useful if folding is used, as
        many points need to be evaluated each time.
    lso_args: dict
        Argument forwardede to the :py:func:`leave_some_out` function

    Attributes
    ----------
    test_model : object
        Model used to estimate the current fit
    test_est : object
        Fitted test model
    LSO_model : object
        Copy of the model, modified for the needs of the current optimization
    LSO_est : object
        Fitted LSO model
    bw_min : float
        Minimum reasonable bandwidth (default: 1e-3*initial bandwidth)
    grid_size : int or list of int
        Copy of the `grid_size` argument
    use_grid: bool
        Copy of the `use_grid` argument
    """
    def __init__(self, model, initial_method=None, grid_size=None, use_grid=False, **lso_args):
        from . import bandwidths
        test_model = model.copy()
        if initial_method is None:
            test_model.bandwidth = bandwidths.Multivariate()
        else:
            test_model.bandwidth = initial_method
        test_est = test_model.fit()

        LSO_model = model.copy()
        LSO_model.bandwidth = test_est.bandwidth
        LSO_est = LSO_model.fit()

        self.LSO = leave_some_out(test_est.exog, test_est.weights, test_est.adjust, **lso_args)
        bw = np.asarray(test_est.bandwidth)
        self._is_cov = bw.ndim == 2
        if self._is_cov:
            ndim = bw.shape[0]
            self.ndim = ndim
            triu = np.triu_indices(ndim, 1)
            self._triu = triu
            self._size_triu = len(triu[0])
            self._init_bandwidth = np.concatenate([bw[triu], bw.diagonal()])
        else:
            self.ndim = 1
            self._init_bandwidth = bw
        self.bw_min = (bw*1e-4).min()
        self.test_est = test_est
        self.LSO_est = LSO_est
        self.grid_size = grid_size
        self.use_grid = use_grid

        self._bw_shape = bw.shape

    @property
    def init_bandwidth(self):
        """
        Initial bandwidth
        """
        return self._init_bandwidth

    @property
    def bw_shape(self):
        '''
        Shape of the bandwidth
        '''
        return self._bw_shape

    def bandwidth(self, bw):
        '''
        Create the correct bandwidth from the estimated parameters
        '''
        if self._is_cov:
            C = np.diag(bw[-self.ndim:])
            T = np.zeros_like(C)
            T[self._triu] = bw[:self._size_triu]
            return C + T + T.T
        return bw

    def __call__(self, bw):
        bw = np.asarray(bw)
        if np.any(bw <= self.bw_min):
            return np.inf
        return self.value(self.bandwidth(bw))

    def value(self, bw):
        """
        Return the quantity to be minimise.

        Derived classes will need to override this method.
        """
        raise NotImplementedError()


class CV_IMSE(CVFunc):
    """
    Compute the integrated mean square error by cross-validation

    Parameters
    ----------
    model : :py:class:`.kde.KDE`
        Model to be fitted
    initial_method : callable or value
        Initial value for the bandwidth
    grid_size : int or tuple of int
        Size of the grid to use to compute the square of the estimated distribution
    use_grid : bool
        If True, instead of evaluating the function at the points needed for
        the cross-validation, the points will be estimated by linear
        interpolation on a grid the same size as the one used for the
        distribution estimation. This is only useful if folding is used, as
        many points need to be evaluated each time.
    lso_args : dict
        Argument forwardede to the :py:func:`leave_some_out` function
    """

    def value(self, bw):
        if np.any(bw <= self.bw_min):
            return np.inf
        bw = bw.reshape(self.bw_shape)
        LSO_est = self.LSO_est
        test_est = self.test_est

        LSO_est.bandwidth = test_est.bandwidth = bw
        exog = test_est.exog
        Fx, Fy = test_est.grid(N=self.grid_size)
        F = Fx.integrate(Fy ** 2)
        L = 0
        use_grid = self.use_grid
        interp = None
        for i, (Xi, Wi, Li) in self.LSO:
            LSO_est.update_inputs(Xi, Wi, Li)
            if use_grid:
                gr, pdf = LSO_est.grid(N=self.grid_size)
                interp = GridInterpolator(gr, pdf)
                vals = interp(exog[i])
            else:
                vals = LSO_est.pdf(exog[i])
            L += np.sum(vals)
        return F - 2 * L / self.LSO.nb_tests


class CV_LogLikelihood(CVFunc):
    """
    Compute the log-likelihood of the data by cross-validation

    Parameters
    ----------
    model : :py:class:`.kde.KDE`
        Model to be fitted
    initial_method : callable or value
        Initial value for the bandwidth
    grid_size : int or tuple of int
        Size of the grid to use to compute the square of the estimated distribution
    use_grid : bool
        If True, instead of evaluating the function at the points needed for
        the cross-validation, the points will be estimated by linear
        interpolation on a grid the same size as the one used for the
        distribution estimation. This is only useful if folding is used, as
        many points need to be evaluated each time.
    lso_args : dict
        Argument forwardede to the :py:func:`leave_some_out` function

    Notes
    -----

    The function returned is actually the opposite of the log likelihood, so
    the cross validation function will compute the maximum of the
    log-likelihood.
    """

    def value(self, bw):
        if np.any(bw <= self.bw_min):
            return np.inf
        bw = bw.reshape(self.bw_shape)
        LSO_est = self.LSO_est
        test_est = self.test_est

        LSO_est.bandwidth = bw
        exog = test_est.exog
        L = 0
        use_grid = self.use_grid
        interp = None
        for i, (Xi, Wi, Li) in self.LSO:
            LSO_est.update_inputs(Xi, Wi, Li)
            if use_grid:
                gr, pdf = LSO_est.grid(N=self.grid_size)
                interp = GridInterpolator(gr, pdf)
                vals = interp(exog[i])
            else:
                vals = LSO_est.pdf(exog[i])
            L -= np.sum(np.log(vals))
        return L


class crossvalidation(object):
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
    func: callable
        Function that will create the minimiser.
    func_args: tuple
        Positional arguments for the creation of the function object.
    func_kwargs: dictionary
        Named arguments for the creation of the function object.

    Notes
    -----

    The call creating the minimiser is::

        func(model, *func_args, **func_kwargs)

    The returned callable is passed to the :py:func:`scipy.optimize.minimize`
    function. It must also have a `init_bandwidth` attribute giving a first
    estimate of the bandwidth.
    """

    def __init__(self, func=CV_LogLikelihood, *func_args, **func_kwargs):
        self.func = func
        self.func_args = func_args
        self.func_kwargs = func_kwargs

    def __call__(self, model):
        func = self.func(model, *self.func_args, **self.func_kwargs)
        res = optimize.minimize(func, x0=func.init_bandwidth, tol=1e-3, method='Nelder-Mead')
        if not res.success:
            print("Error, could not find minimum: '{0}'".format(res.message))
            return func.init_bandwidth
        return func.bandwidth(res.x)
