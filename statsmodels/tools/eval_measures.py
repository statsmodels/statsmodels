# -*- coding: utf-8 -*-
"""some measures for evaluation of prediction, tests and model selection

Created on Tue Nov 08 15:23:20 2011

Author: Josef Perktold
License: BSD-3

"""


import numpy as np

def mse(x1, x2, axis=0):
    '''mean squared error

    Parameters
    ----------
    x1, x2 : array_like
       The performance measure depends on the difference between these two
       arrays.
    axis : int
       axis along which the summary statistic is calculated

    Returns
    -------
    mse : ndarray or float
       mean squared error along given axis.

    Notes
    -----
    If ``x1`` and ``x2`` have different shapes, then they need to broadcast.
    This uses ``numpy.asanyarray`` to convert the input. Whether this is the
    desired result or not depends on the array subclass, for example
    numpy matrices will silently produce an incorrect result.

    '''
    x1 = np.asanyarray(x1)
    x2 = np.asanyarray(x2)
    return np.mean((x1-x2)**2, axis=axis)

def rmse(x1, x2, axis=0):
    '''root mean squared error

    Parameters
    ----------
    x1, x2 : array_like
       The performance measure depends on the difference between these two
       arrays.
    axis : int
       axis along which the summary statistic is calculated

    Returns
    -------
    rmse : ndarray or float
       root mean squared error along given axis.

    Notes
    -----
    If ``x1`` and ``x2`` have different shapes, then they need to broadcast.
    This uses ``numpy.asanyarray`` to convert the input. Whether this is the
    desired result or not depends on the array subclass, for example
    numpy matrices will silently produce an incorrect result.

    '''
    x1 = np.asanyarray(x1)
    x2 = np.asanyarray(x2)
    return np.sqrt(mse(x1, x2, axis=axis))

def maxabs(x1, x2, axis=0):
    '''maximum absolute error

    Parameters
    ----------
    x1, x2 : array_like
       The performance measure depends on the difference between these two
       arrays.
    axis : int
       axis along which the summary statistic is calculated

    Returns
    -------
    maxabs : ndarray or float
       maximum absolute difference along given axis.

    Notes
    -----
    If ``x1`` and ``x2`` have different shapes, then they need to broadcast.
    This uses ``numpy.asanyarray`` to convert the input. Whether this is the
    desired result or not depends on the array subclass.

    '''
    x1 = np.asanyarray(x1)
    x2 = np.asanyarray(x2)
    return np.max(np.abs(x1-x2), axis=axis)

def meanabs(x1, x2, axis=0):
    '''mean absolute error

    Parameters
    ----------
    x1, x2 : array_like
       The performance measure depends on the difference between these two
       arrays.
    axis : int
       axis along which the summary statistic is calculated

    Returns
    -------
    meanabs : ndarray or float
       mean absolute difference along given axis.

    Notes
    -----
    If ``x1`` and ``x2`` have different shapes, then they need to broadcast.
    This uses ``numpy.asanyarray`` to convert the input. Whether this is the
    desired result or not depends on the array subclass.

    '''
    x1 = np.asanyarray(x1)
    x2 = np.asanyarray(x2)
    return np.mean(np.abs(x1-x2), axis=axis)

def medianabs(x1, x2, axis=0):
    '''median absolute error

    Parameters
    ----------
    x1, x2 : array_like
       The performance measure depends on the difference between these two
       arrays.
    axis : int
       axis along which the summary statistic is calculated

    Returns
    -------
    medianabs : ndarray or float
       median absolute difference along given axis.

    Notes
    -----
    If ``x1`` and ``x2`` have different shapes, then they need to broadcast.
    This uses ``numpy.asanyarray`` to convert the input. Whether this is the
    desired result or not depends on the array subclass.

    '''
    x1 = np.asanyarray(x1)
    x2 = np.asanyarray(x2)
    return np.median(np.abs(x1-x2), axis=axis)

def bias(x1, x2, axis=0):
    '''bias, mean error

    Parameters
    ----------
    x1, x2 : array_like
       The performance measure depends on the difference between these two
       arrays.
    axis : int
       axis along which the summary statistic is calculated

    Returns
    -------
    bias : ndarray or float
       bias, or mean difference along given axis.

    Notes
    -----
    If ``x1`` and ``x2`` have different shapes, then they need to broadcast.
    This uses ``numpy.asanyarray`` to convert the input. Whether this is the
    desired result or not depends on the array subclass.

    '''
    x1 = np.asanyarray(x1)
    x2 = np.asanyarray(x2)
    return np.mean(x1-x2, axis=axis)

def medianbias(x1, x2, axis=0):
    '''median bias, median error

    Parameters
    ----------
    x1, x2 : array_like
       The performance measure depends on the difference between these two
       arrays.
    axis : int
       axis along which the summary statistic is calculated

    Returns
    -------
    medianbias : ndarray or float
       median bias, or median difference along given axis.

    Notes
    -----
    If ``x1`` and ``x2`` have different shapes, then they need to broadcast.
    This uses ``numpy.asanyarray`` to convert the input. Whether this is the
    desired result or not depends on the array subclass.

    '''
    x1 = np.asanyarray(x1)
    x2 = np.asanyarray(x2)
    return np.median(x1-x2, axis=axis)

def vare(x1, x2, ddof=0, axis=0):
    '''variance of error

    Parameters
    ----------
    x1, x2 : array_like
       The performance measure depends on the difference between these two
       arrays.
    axis : int
       axis along which the summary statistic is calculated

    Returns
    -------
    vare : ndarray or float
       variance of difference along given axis.

    Notes
    -----
    If ``x1`` and ``x2`` have different shapes, then they need to broadcast.
    This uses ``numpy.asanyarray`` to convert the input. Whether this is the
    desired result or not depends on the array subclass.

    '''
    x1 = np.asanyarray(x1)
    x2 = np.asanyarray(x2)
    return np.var(x1-x2, ddof=ddof, axis=axis)

def stde(x1, x2, ddof=0, axis=0):
    '''standard deviation of error

    Parameters
    ----------
    x1, x2 : array_like
       The performance measure depends on the difference between these two
       arrays.
    axis : int
       axis along which the summary statistic is calculated

    Returns
    -------
    stde : ndarray or float
       standard deviation of difference along given axis.

    Notes
    -----
    If ``x1`` and ``x2`` have different shapes, then they need to broadcast.
    This uses ``numpy.asanyarray`` to convert the input. Whether this is the
    desired result or not depends on the array subclass.

    '''
    x1 = np.asanyarray(x1)
    x2 = np.asanyarray(x2)
    return np.std(x1-x2, ddof=ddof, axis=axis)

def iqr(x1, x2, axis=0):
    '''interquartile range of error

    rounded index, no interpolations

    this could use newer numpy function instead

    Parameters
    ----------
    x1, x2 : array_like
       The performance measure depends on the difference between these two
       arrays.
    axis : int
       axis along which the summary statistic is calculated

    Returns
    -------
    mse : ndarray or float
       mean squared error along given axis.

    Notes
    -----
    If ``x1`` and ``x2`` have different shapes, then they need to broadcast.

    This uses ``numpy.asarray`` to convert the input, in contrast to the other
    functions in this category.

    '''
    x1 = np.asarray(x1)
    x2 = np.asarray(x2)
    if axis is None:
        x1 = np.ravel(x1)
        x2 = np.ravel(x2)
        axis = 0
    xdiff = np.sort(x1 - x2)
    nobs = x1.shape[axis]
    idx = np.round((nobs-1) * np.array([0.25, 0.75])).astype(int)
    sl = [slice(None)] * xdiff.ndim
    sl[axis] = idx
    iqr = np.diff(xdiff[sl], axis=axis)
    iqr = np.squeeze(iqr) #drop reduced dimension
    if iqr.size == 1:
        return iqr #[0]
    else:
        return iqr


# Information Criteria
#---------------------

def aic(llf, nobs, df_modelwc):
    '''Akaike information criterion

    Parameters
    ----------
    llf : float
        value of the loglikelihood
    nobs : int
        number of observations
    df_modelwc : int
        number of parameters including constant

    Returns
    -------
    aic : float
        information criterion

    References
    ----------
    http://en.wikipedia.org/wiki/Akaike_information_criterion

    '''
    return -2. * llf + 2. * df_modelwc

def aicc(llf, nobs, df_modelwc):
    '''Akaike information criterion (AIC) with small sample correction

    Parameters
    ----------
    llf : float
        value of the loglikelihood
    nobs : int
        number of observations
    df_modelwc : int
        number of parameters including constant

    Returns
    -------
    aicc : float
        information criterion

    References
    ----------
    http://en.wikipedia.org/wiki/Akaike_information_criterion#AICc

    '''
    return -2. * llf + 2. * df_modelwc * nobs / (nobs - df_modelwc - 1.)
    #float division

def bic(llf, nobs, df_modelwc):
    '''Bayesian information criterion (BIC) or Schwarz criterion

    Parameters
    ----------
    llf : float
        value of the loglikelihood
    nobs : int
        number of observations
    df_modelwc : int
        number of parameters including constant

    Returns
    -------
    bic : float
        information criterion

    References
    ----------
    http://en.wikipedia.org/wiki/Bayesian_information_criterion

    '''
    return -2. * llf + np.log(nobs) * df_modelwc

def hqic(llf, nobs, df_modelwc):
    '''Hannan-Quinn information criterion (HQC)

    Parameters
    ----------
    llf : float
        value of the loglikelihood
    nobs : int
        number of observations
    df_modelwc : int
        number of parameters including constant

    Returns
    -------
    hqic : float
        information criterion

    References
    ----------
    Wikipedia doesn't say much

    '''
    return -2. * llf + 2 * np.log(np.log(nobs)) * df_modelwc


#IC based on residual sigma

def aic_sigma(sigma2, nobs, df_modelwc, islog=False):
    '''Akaike information criterion

    Parameters
    ----------
    sigma2 : float
        estimate of the residual variance or determinant of Sigma_hat in the
        multivariate case. If islog is true, then it is assumed that sigma
        is already log-ed, for example logdetSigma.
    nobs : int
        number of observations
    df_modelwc : int
        number of parameters including constant

    Returns
    -------
    aic : float
        information criterion

    Notes
    -----
    A constant has been dropped in comparison to the loglikelihood base
    information criteria. The information criteria should be used to compare
    only comparable models.

    For example, AIC is defined in terms of the loglikelihood as

    :math:`-2 llf + 2 k`

    in terms of :math:`\hat{\sigma}^2`

    :math:`log(\hat{\sigma}^2) + 2 k / n`

    in terms of the determinant of :math:`\hat{\Sigma}`

    :math:`log(\|\hat{\Sigma}\|) + 2 k / n`

    Note: In our definition we do not divide by n in the log-likelihood
    version.

    TODO: Latex math

    reference for example lecture notes by Herman Bierens

    See Also
    --------

    References
    ----------
    http://en.wikipedia.org/wiki/Akaike_information_criterion

    '''
    if not islog:
        sigma2 = np.log(sigma2)
    return sigma2 + aic(0, nobs, df_modelwc) / nobs


def aicc_sigma(sigma2, nobs, df_modelwc, islog=False):
    '''Akaike information criterion (AIC) with small sample correction

    Parameters
    ----------
    sigma2 : float
        estimate of the residual variance or determinant of Sigma_hat in the
        multivariate case. If islog is true, then it is assumed that sigma
        is already log-ed, for example logdetSigma.
    nobs : int
        number of observations
    df_modelwc : int
        number of parameters including constant

    Returns
    -------
    aicc : float
        information criterion

    Notes
    -----
    A constant has been dropped in comparison to the loglikelihood base
    information criteria. These should be used to compare for comparable models.

    References
    ----------
    http://en.wikipedia.org/wiki/Akaike_information_criterion#AICc

    '''
    if not islog:
        sigma2 = np.log(sigma2)
    return sigma2 + aicc(0, nobs, df_modelwc) / nobs
    #float division


def bic_sigma(sigma2, nobs, df_modelwc, islog=False):
    '''Bayesian information criterion (BIC) or Schwarz criterion

    Parameters
    ----------
    sigma2 : float
        estimate of the residual variance or determinant of Sigma_hat in the
        multivariate case. If islog is true, then it is assumed that sigma
        is already log-ed, for example logdetSigma.
    nobs : int
        number of observations
    df_modelwc : int
        number of parameters including constant

    Returns
    -------
    bic : float
        information criterion

    Notes
    -----
    A constant has been dropped in comparison to the loglikelihood base
    information criteria. These should be used to compare for comparable models.

    References
    ----------
    http://en.wikipedia.org/wiki/Bayesian_information_criterion

    '''
    if not islog:
        sigma2 = np.log(sigma2)
    return sigma2 + bic(0, nobs, df_modelwc) / nobs


def hqic_sigma(sigma2, nobs, df_modelwc, islog=False):
    '''Hannan-Quinn information criterion (HQC)

    Parameters
    ----------
    sigma2 : float
        estimate of the residual variance or determinant of Sigma_hat in the
        multivariate case. If islog is true, then it is assumed that sigma
        is already log-ed, for example logdetSigma.
    nobs : int
        number of observations
    df_modelwc : int
        number of parameters including constant

    Returns
    -------
    hqic : float
        information criterion

    Notes
    -----
    A constant has been dropped in comparison to the loglikelihood base
    information criteria. These should be used to compare for comparable models.

    References
    ----------
    xxx

    '''
    if not islog:
        sigma2 = np.log(sigma2)
    return sigma2 + hqic(0, nobs, df_modelwc) / nobs


#from var_model.py, VAR only? separates neqs and k_vars per equation
#def fpe_sigma():
#    ((nobs + self.df_model) / self.df_resid) ** neqs * np.exp(ld)

# Evaluation measures for classification models

def _pred_table_to_cells(pred_table, true):
    """
    Returns true positive, false positive, false negative, true negative
    """
    pred_table = np.asarray(pred_table)

    if true == 0:
        tp, fp, fn, tn = pred_table.flatten()
    elif true == 1:
        tn, fn, fp, tp = pred_table.flatten()
    else:  # pragma: no cover
        raise ValueError("Only true == [0, 1] supported.")

    return tp, fp, fn, tn


def precision(pred_table, true=1):
    """
    Computes classification precision given prediction table.

    Binary classification only.

    Parameters
    ----------
    pred_table : array-like
        2 x 2 prediction table (aka confusion matrix). The actual class is
        assumed to be the rows. The predicted class is in the columns.
        The "True" category is in the column given by ``true''
    true : int {0, 1}
        The column which contains the "True" category.

    Returns
    -------
    precision : float
        The precision of the classification

    Notes
    -----
    Precision is analagous to (absence of) Type I errors. The probability
    that a randomly selected document is classified correctly. I.e., there are
    no false negatives. Sometimes called positive predictive value.
    """
    tp, fp, fn, tn = _pred_table_to_cells(pred_table, true)

    try:
        return tp / (tp + fp)
    except ZeroDivisionError:
        return np.nan


def recall(pred_table, true=1):
    """
    Computes the classification recall (or sensitivity) of prediction table.

    Binary classification only.

    Parameters
    ----------
    pred_table : array-like
        2 x 2 prediction table (aka confusion matrix). The actual class is
        assumed to be the rows. The predicted class is in the columns.
        The "True" category is in the column given by ``true''
    true : int {0, 1}
        The column which contains the "True" category.

    Returns
    -------
    recall : float
        The recall of the classification

    Notes
    -----

    Computes

    .. math::

       tp / (tp + fn)

    Analagous to (absence of) Type II errors. Out of all the ones that are
    true, how many did you predict as true. I.e., with a recall of 1 there
    are no false positives. Also called the true positive rate.
    """
    tp, fp, fn, tn = _pred_table_to_cells(pred_table, true)

    try:
        return tp / (tp + fn)
    except ZeroDivisionError:
        return np.nan

sensitivity = recall


def specificity(pred_table, true=1):
    """
    Computes the classification specificity of prediction table.

    Binary classification only.

    Parameters
    ----------
    pred_table : array-like
        2 x 2 prediction table (aka confusion matrix). The actual class is
        assumed to be the rows. The predicted class is in the columns.
        The "True" category is in the column given by ``true''
    true : int {0, 1}
        The column which contains the "True" category.

    Returns
    -------
    specificity : float
        The specificity of the classification

    Notes
    -----

    Computes

    .. math::

       tn / (tn + fp)

    Analagous to (absence of) type I errors. Out of all the ones that are
    negative, how many did you predict as negative. I.e., with a specificty
    of 1 there are no false positives. Sometimes called the true negative
    rate.
    """
    if true == 1:
        true = 0
    elif true == 0:
        true = 1
    return recall(pred_table, true)


def accuracy(pred_table, true=1):
    """
    Compuates the classification accuracy of a prediction table.

    Binary classification only.

    Parameters
    ----------
    pred_table : array-like
        2 x 2 prediction table (aka confusion matrix). The actual class is
        assumed to be the rows. The predicted class is in the columns.
        The "True" category is in the column given by ``true''
    true : int {0, 1}
        The column which contains the "True" category.

    Returns
    -------
    accuracy : float
        The accuracy of the classification

    Notes
    -----

    Computes

    .. math::

       (tp + tn) / (tp + tn + fp + fn)

    Sometimes called Rand Accuracy.
    """
    tp, fp, fn, tn = _pred_table_to_cells(pred_table, true)
    if tp == tn == fp == fn == 0:  # pragma: no cover
        raise ValueError("Prediction table cannot be all zero")
    return (tp + tn) / (tp + tn + fp + fn)


def fscore_measure(pred_table, b=1, true=1):
    """
    Return the F-b measure for a prediction table

    Binary classification only.

    Parameters
    ----------
    pred_table : array-like
        2 x 2 prediction table (aka confusion matrix). The actual class is
        assumed to be the rows. The predicted class is in the columns.
        The "True" category is in the column given by ``true''
    b : float
        b should be positive. It indicates the how much more importance is
        attached to recall than precision. The default is 1, giving the
        traditional F-measure. For example, if a user attached half as much
        importance to recall as precision, b = .5. If twice as much, b = 2.
    true : int {0, 1}
        The column which contains the "True" category.

    Notes
    -----
    Computes

    .. math::

       (1 + b^2)*(precision * recall)/(b^2 * precision + recall)
    """
    r = recall(pred_table, true)
    p = precision(pred_table, true)
    return (1 + b**2) * r*p/(b**2 * p + r)


__all__ = [maxabs, meanabs, medianabs, medianbias, mse, rmse, stde, vare,
           aic, aic_sigma, aicc, aicc_sigma, bias, bic, bic_sigma,
           hqic, hqic_sigma, iqr, precision, recall, accuracy, fscore_measure,
           specificity, sensitivity]
