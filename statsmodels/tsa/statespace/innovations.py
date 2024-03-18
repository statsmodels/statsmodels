import numpy as np
from numpy.linalg import LinAlgError, eigvals

from statsmodels.tsa.statespace.mlemodel import MLEModel


class InnnovationsMLEModel(MLEModel):
    r"""
    Innovations State space model for maximum likelihood estimation

    Parameters
    ----------
    endog : array_like
        The observed time-series process :math:`y`
    k_states : int
        The dimension of the unobserved state process. Do not include the hidden state.
    exog : array_like, optional
        Array of exogenous regressors, shaped nobs x k. Default is no
        exogenous regressors.
    dates : array_like of datetime, optional
        An array-like object of datetime objects. If a Pandas object is given
        for endog, it is assumed to have a DateIndex.
    freq : str, optional
        The frequency of the time-series. A Pandas offset or 'B', 'D', 'W',
        'M', 'A', or 'Q'. This is optional if dates are given.
    **kwargs
        Keyword arguments may be used to provide default values for state space
        matrices or for Kalman filtering options. See `Representation`, and
        `KalmanFilter` for more details.

    Attributes
    ----------
    ssm : statsmodels.tsa.statespace.kalman_filter.KalmanFilter
        Underlying state space representation.

    See Also
    --------
    statsmodels.tsa.statespace.mlemodel.MLEResults
    statsmodels.tsa.statespace.kalman_filter.KalmanFilter
    statsmodels.tsa.statespace.representation.Representation

    Notes
    -----
    This class wraps the innovations state space model (also known as SSOE, Single Source Of Errors)
    with Kalman filtering to add in functionality for maximum likelihood estimation.

    Considering innovations state space
    .. math::
        y_t & = w \alpha_{t-1} + \varepsilon_t \\
        \alpha_{t} & = F \alpha_{t-1} + g \varepsilon_t \\

    and the linear time-discrete state space formula with filtering timing
    (also known as MSOE, Multiple Source Of Errors)
    .. math::
        y_t & = Z \alpha_{t+1} \\
        \alpha_{t+1} & = T \alpha_t + R \eta_t \\

    we wrap the SSOE formula into MSOE form.
    .. math::
        y_t & = \begin{bmatrix} w & 1 \end{bmatrix} \alpha_t \\
        \alpha_{t+1} & = \begin{bmatrix} \boldsymbol{0}_k & 0 \\ F & g \end{bmatrix}
        \alpha_t + \begin{bmatrix} \boldsymbol{0}_k \\ 1 \end{bmatrix} \eta_t \\

    """
    def __init__(self, endog, k_states, **kwargs):
        # enforce posdef to be 1
        kwargs.update(k_posdef=1)
        # hidden states
        k_states += 1
        super().__init__(endog, k_states, **kwargs)
        # single error form
        self.ssm["design", 0, -1] = 1
        # for cached error
        self.ssm["selection", -1, 0] = 1

    def __setitem__(self, key, value):

        _type = type(key)
        if _type is str:
            if key == "cov":
                self.ssm["state_cov", :, :] = value
            elif key == "F":
                self.ssm["transition", :-1, :-1] = value
            elif key == "g":
                self.ssm["transition", -1, :-1] = value
            elif key == "w":
                self.ssm["design", :, :-1] = value
            else:
                return super().__setitem__(key, value)
        elif _type is tuple:
            name, slice_ = key[0], key[1:]
            if name == "cov":
                self.ssm["state_cov", :, :] = value
            # TODO: very careful for out of range
            elif name == "F":
                self.ssm["transition", slice_[0], slice_[1]] = value
            elif name == "g":
                a, b = slice_
                if b == 0 or b is None:
                    b = -1
                if isinstance(a, slice) and a.stop is None:
                    a.stop = -1
                elif a is None:
                    a = slice(stop=-1)
                self.ssm["transition", a, b] = value
            elif name == "w":
                self.ssm["design", slice_[0], slice_[1]] = value
            else:
                return super().__setitem__(name, slice_)

    def __getitem__(self, key):

        _type = type(key)
        if _type is str:
            if key == "g":
                return self.ssm["transition", :-1, -1:]
            elif key == "F":
                return self.ssm["transition", :-1, :-1]
            elif key == "w":
                return self.ssm["design", :, :-1]
            elif key == "cov":
                return self.ssm["state_cov", :, :]
            else:
                return super().__getitem__(key)
        elif _type is tuple:
            name, slice_ = key[0], key[1:]
            if name == "g":
                return self.ssm["transition", :-1, -1:][slice_]
            elif name == "F":
                return self.ssm["transition", :-1, :-1][slice_]
            elif name == "w":
                return self.ssm["design", :, :-1][slice_]
            elif name == "cov":
                return self.ssm["state_cov", :, :][slice_]
            else:
                return super().__getitem__(key)

    @property
    def D(self):
        return self["F"] - self["g"] @ self["w"]

    def _check_admissible(self, D=None):
        if D is None:
            D = self.D
        try:
            eig = eigvals(D)
        except LinAlgError:
            return False
        return np.all(np.abs(eig) < 1.01)
