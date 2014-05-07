import numpy as np
from scipy.optimize import brentq
import copy
import pandas as pd
from statsmodels.graphics import utils as gutils

class SolutionPathResults(object):
    """
    A class representing a fitted solution path that is obtained using
    regularized regression.

    Parameters:
    -----------
    model : object
        The model for which the solution path is to be constructed.
    wt_vec : array-like
        The penalty weights for the model coefficients are
        obtained by multiplying this vector by a scalar
        penalty weight.
    pwts : array-like
        The sequence of penalty weights that determine where the
        solution path is calculated
    ix_nonzero : list of arrays
        The indices of the positions where the estimated coefficients
        are nonzero.
    nonzero_params : list of arrays
        The values of the nonzero estimated coefficients.

    Notes:
    ------
    `pwts`, `ix_nonzero`, `ix_zero`, and `nonzero_params` all have
    the same number of elements.  Corresponding positions in these
    lists correspond to a single regularized fit using the given
    penalty weight.

    Extra parameters may not be regularized (this depends on how
    fit_regularized is implemented in a particular model).  They will
    still be included on the solution path.
    """

    def __init__(self, model, wt_vec, pwts, ix_nonzero,
                 nonzero_params, **fit_args):
        self.model = model
        self.pwts = pwts
        self.ix_nonzero = [np.asarray(x) for x in ix_nonzero]
        self.nonzero_params = nonzero_params
        self.fit_args = fit_args
        self.wt_vec = wt_vec

        self.num_nonzero = np.asarray([len(x) for x in ix_nonzero])

        ix_zero = []
        for ix in ix_nonzero:
            ii = np.asarray([i for i in range(model.nparams)
                             if i not in ix])
            ix_zero.append(ii)
        self.ix_zero = ix_zero

        self._unpack()

    def _unpack(self):

        d = self.model.nparams
        params_unpacked = []
        for i in range(len(self.pwts)):
            x = np.zeros(d, dtype=np.float64)
            x[self.ix_nonzero[i]] = self.nonzero_params[i]
            params_unpacked.append(x)
        self.params_unpacked = np.asarray(params_unpacked)

    def refit(self, nvar):
        """
        Returns a results object by calling the `fit` method of the
        model object, restricting the fit to a given number of
        variables.

        Parameters:
        -----------
        nvar : integer
            The number of variables in the fitted model.

        Returns:
        --------
        A model results instance, or None if no model with the
        desired number of variables was found.
        """

        # Find the model that gives us the desired number of
        # variables.  Choose the least penalized model if there are
        # several.
        ii = np.flatnonzero(self.num_nonzero == nvar)
        if len(ii) == 0:
            return None
        ii = min(ii)

        # We need the variable names to see which variables are still
        # in the model
        ix = self.ix_nonzero[ii]
        if isinstance(self.model.exog, pd.DataFrame):
            new_exog = self.model.exog.iloc[:,ix]
        else:
            new_exog = pd.DataFrame(self.model.exog[:,ix])
            new_exog.columns = [self.model.exog_names[i] for i in ix]

        model1 = self.model.__class__(self.model.endog, new_exog)

        return model1.fit(**self.fit_args)

    def plot(self, ax=None):
        """
        Make a plot of the solution paths.

        Parameters:
        -----------
        ax : Matplotlib axes instance
          An axes on which to draw the graph.  If None, new figure and
          axes objects are created.

        Returns:
        --------
        fig : Figure
            The figure given by `ax.Figure`, or a new instance.
        """

        if ax is None:
            fig, ax = gutils.create_mpl_ax(ax)
        else:
            fig = ax.get_figure()

        for k in range(self.params_unpacked.shape[1]):
            ax.plot(self.pwts, self.params_unpacked[:,k], '-',
                    color='grey')

        ax.set_xlabel("Penalty weight")
        ax.set_ylabel("Coefficient")

        return fig



class SolutionPath(object):
    """
    Calculate the solution path for a regularized regression.  This is
    a utility class used by base.model.solution_path to compute the
    solution path.

    Parameters:
    -----------
    model : model class
        The model model class for which the solution path will
        be obtained.
    maxvar : integer
        The maximum number of variables with nonzero coefficients
        (see notes)
    wt_vec : array-like
        The penalty weights (per-parameter) are proportional to
        the values in this vector.
    param_threshold : float
        Coefficients smaller than this number in absolute value
        are treated as zero
    start_pwt : float
        The smallest penalty weight that is considered.
    no_selection: bool
        If true, the penalty function does not produce exact
        zero coefficients.  In this case, the method does not
        try to find the points where the coefficients become
        zero.
    xtol : float
       Try to resolve the points where coefficients become
       zero to this level of accuracy.
    fit_params : keyword parameters
        Additional keyword parameters passed to fit_regularized.

    Notes:
    ------
    The returned solution path may contain some models with more than
    `maxvar` variables, but only the solution path for models with up
    to `maxvar` variables is thoroughly explored.

    In some cases two variables may enter/exit the model
    simultaneously, so some model sizes may not be present in the
    solution path.
    """

    def __init__(self, model, maxvar=None, wt_vec=None, param_threshold=1e-4,
                 start_pwt=0., no_selection=False, xtol=1.,
                 **fit_params):

        self.model = model
        self.maxvar = maxvar if maxvar is not None else model.nparams
        self.param_threshold = param_threshold
        self.xtol = xtol
        self.start_pwt = start_pwt

        if wt_vec is not None:
            if len(wt_vec) != model.nparams:
                raise ValueError("len(wt_vec) must equal the number of parameters")
        else:
            wt_vec = np.ones(model.nparams, dtype=np.float64)
        self.wt_vec = wt_vec

        self.fit_params = fit_params

        self.weights = []
        self.ix_nonzero = []
        self.ix_zero = []
        self.params = []
        self.num_nonzero = []

        if no_selection:
            self.no_selection_paths()
        else:
            self.initialize()
            self.refine()

    def weights_less_complex(self, nvar):
        """
        Return all the weights on the path corresponding to models
        with fewer than `nvar` variables.
        """

        ii = np.flatnonzero(np.asarray(self.num_nonzero) < nvar)
        return np.asarray(self.weights)[ii]

    def weights_more_complex(self, nvar):
        """
        Return all the weights on the path corresponding to models
        with more than `nvar` variables.
        """

        ii = np.flatnonzero(np.asarray(self.num_nonzero) > nvar)
        return np.asarray(self.weights)[ii]

    def weights_as_complex(self, nvar):
        """
        Return all the weights on the path corresponding to models
        with exactly `nvar` variables.
        """

        ii = np.flatnonzero(np.asarray(self.num_nonzero) == nvar)
        return np.asarray(self.weights)[ii]

    def add_point(self, pen_wt, params):
        """
        Add one point to the solution path.

        Parameters:
        -----------
        pen_wt : float
            The penalty coefficient
        params : array-like
            The model coefficients

        Returns:
        --------
        The number of nonzero coefficients in the added point.
        """

        paramsa = np.abs(params)
        ix0 = np.flatnonzero(paramsa <= self.param_threshold)
        ix1 = np.flatnonzero(paramsa > self.param_threshold)
        ii = np.searchsorted(self.weights, pen_wt)
        self.weights.insert(ii, pen_wt)
        self.ix_zero.insert(ii, ix0)
        self.ix_nonzero.insert(ii, ix1)
        self.params.insert(ii, params[ix1])
        self.num_nonzero.insert(ii, len(ix1))
        return len(ix1)

    def fit_regularized(self, pen_wt):
        """
        Call the model's fit regularized using a given level of
        penalty weighting.  Also, attempt a 'warm start' using
        starting values obtaine from other regularized fits if
        possible.

        Parameters:
        -----------
        pen_wt : float
            The multiplier of the penalty weight vector

        Returns:
        --------
        A vector of coefficient estimates.
        """

        if len(self.weights) > 0:
            wa = np.asarray(self.weights)
            ii = np.argmin(np.abs(wa - pen_wt))
            start_params = np.zeros(self.model.nparams,
                                    dtype=np.float64)
            start_params[self.ix_nonzero[ii]] = self.params[ii]
        else:
            start_params = None

        mdf = self.model.fit_regularized(alpha=pen_wt*self.wt_vec,
                start_params=start_params, **self.fit_params)

        return mdf.params

    def initialize(self):
        """
        Add points to the solution path until each possible model size
        is represented at least once, if possible.

        Notes:
        ------
        In some degenerate cases, two variables may enter at exactly
        the same penalty weight, so some model sizes are not
        achievable.
        """

        # This will be the most complex model on the path
        start = self.fit_regularized(self.start_pwt)
        self.add_point(self.start_pwt, start)

        # Increase the tuning parameter until all coefficients are
        # zero
        pen_wt = 1.
        n_unpenalized = sum(self.wt_vec == 0)
        while pen_wt < 1e10:
            params = self.fit_regularized(pen_wt)
            nvar = self.add_point(pen_wt, params)
            if nvar == n_unpenalized:
                break
            pen_wt *= 2.
        if pen_wt >= 1e10:
            ix = np.flatnonzero(np.abs(params) > self.param_threshold)
            raise RuntimeError("Unable to shrink coefficients %s to zero." % str(ix))

        # Fill in the gaps; nv is the number of variables with nonzero
        # coefficients
        for nv in range(1, self.maxvar+1):

            # Check for models that have the desired number of
            # variables
            if nv in self.num_nonzero:
                continue

            # Try to find a penalty weight that gives the desired
            # model size
            while True:
                w0 = max(self.weights_more_complex(nv))
                w2 = min(self.weights_less_complex(nv))

                # It may not be possible to obtain certain model sizes
                if w2 - w0 < 1e-4:
                    break

                pen_wt = (w0 + w2) / 2.
                params = self.fit_regularized(pen_wt)
                nvar = self.add_point(pen_wt, params)
                if nvar == nv:
                    break


    def refine(self):
        """
        Extend the solution path so that the minimal possible weight
        (within `xtol`) having each model size is included.
        """

        for i in range(0, self.maxvar+1):

            w0 = self.weights_more_complex(i)
            w1 = self.weights_as_complex(i)

            if len(w0) == 0 or len(w1) == 0:
                continue

            # Don't try too hard to explore any non-monotonic regions.
            if max(w0) > min(w1):
                continue

            def f(pen_wt):
                # Don't fit again if the weight is within xtol of a
                # previous fit.
                aw = np.asarray(self.weights)
                ii = np.argmin(np.abs(aw - pen_wt))
                if abs(pen_wt - self.weights[ii]) < self.xtol:
                    nvar = self.num_nonzero[ii]
                else:
                    params = self.fit_regularized(pen_wt)
                    nvar = self.add_point(pen_wt, params)
                return nvar - (i + 0.5)

            # Non-monotonicity
            w0 = max(w0)
            w1 = min(w1)
            if f(w0) * f(w1) > 1e-5:
                continue

            brentq(f, w0, w1, xtol=self.xtol)

    def no_selection_paths(self):
        """
        Construct the path using a smooth penalty that does not
        produce coefficient estimates that are exactly zero.
        """

        # This will be the most complex model on the path
        start = self.fit_regularized(self.start_pwt)
        self.add_point(self.start_pwt, start)

        # Increase the tuning parameter until all coefficients are
        # zero
        pen_wt = 1.
        n_unpenalized = sum(pen_wt == 0)
        while pen_wt < 1e10:
            params = self.fit_regularized(pen_wt)
            nvar = self.add_point(pen_wt, params)
            if nvar == n_unpenalized:
                break
            pen_wt *= 2.
        if pen_wt >= 1e10:
            ix = np.flatnonzero(np.abs(params) > self.param_threshold)
            raise RuntimeError("Unable to shrink coefficients %s sufficiently close to zero." % str(ix))
