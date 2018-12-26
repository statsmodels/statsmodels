"""
Automatic model selection for Exponential Smoothing models.

Author: Chad Fulton, Abhijeet Panda
License: BSD-3
"""
import warnings
from collections import OrderedDict

import numpy as np

from statsmodels.tsa.base.tsa_model import TimeSeriesModel
from statsmodels.tsa.holtwinters import ExponentialSmoothing


def evaluate(endog, measure, trend, damped, seasonal, seasonal_periods,
             alpha=None, beta=None, gamma=None, phi=None,
             fit_kwargs=None, **spec):
    r"""
    Evaluate a given ExponentialSmoothing model specification

    Parameters
    ----------
    endog : array_like
        The observed time-series process :math:`y`
    measure : {'aic', 'bic', 'hqic'}
        Which information criteria to use for model evaluation.
    trend : str or None
        Trend specifications to consider; valid values include
        `[None, 'add', 'mul']`.
    damped : bool or None
        Whether the trend component is damped.
    seasonal : str or None
        Seasonal specifications to consider; valid values include
        `[None, 'add', 'mul']`.
    seasonal_periods : int, optional
        The period of the seasonal component, if any. Default is None.
    alpha : float, optional
        Value at which to fix the level smoothing parameter. Default is to
        estimate this parameter.
    beta : float, optional
        Value at which to fix the slope smoothing parameter. Default is to
        estimate this parameter.
    gamma : float, optional
        Value at which to fix the seasoanl smoothing parameter. Default is to
        estimate this parameter.
    phi : float, optional
        Value at which to fix the slope damping parameter. Default is to
        estimate this parameter.
    fit_kwargs : dict, optional
        Keyword arguments that were passed to the call to `fit` for each
        evaluated model.
    **spec
        Additional keyword arguments to be passed to the each evaluted model.

    Returns
    -------
    results : HoltWintersResults or None
        Results from the fitted model, if available.
    value : float
        Value of the information criteria.
    exception : Exception or None
        Exception raised during model fitting, if any.
    warnings : list
        List of warnings issued during model fitting. May be empty.
    """
    warning = []
    exception = None
    try:
        with warnings.catch_warnings(record=True) as warning:
            mod = ExponentialSmoothing(
                endog, trend=trend, damped=damped, seasonal=seasonal,
                seasonal_periods=seasonal_periods, **spec)
            fit_kwargs = {} if fit_kwargs is None else fit_kwargs
            # fit_kwargs.setdefault('disp', False)
            res = mod.fit(smoothing_level=alpha, smoothing_slope=beta,
                          smoothing_seasonal=gamma, damping_slope=phi,
                          **fit_kwargs)
            value = getattr(res, measure)
    except Exception as e:
        res = None
        value = np.nan
        exception = e
        raise

    return res, value, exception, warning


class Specification(object):
    r"""
    Selection search over exponential smoothing models.

    Parameters
    ----------
    trend : str or None
        Trend specifications to consider; valid values include
        `[None, 'add', 'mul']`.
    damped : bool or None
        Whether the trend component is damped.
    seasonal : str or None
        Seasonal specifications to consider; valid values include
        `[None, 'add', 'mul']`.
    seasonal_periods : int, optional
        The period of the seasonal component, if any. Default is None.
    alpha : float, optional
        Value at which to fix the level smoothing parameter. Default is to
        estimate this parameter.
    beta : float, optional
        Value at which to fix the slope smoothing parameter. Default is to
        estimate this parameter.
    gamma : float, optional
        Value at which to fix the seasoanl smoothing parameter. Default is to
        estimate this parameter.
    phi : float, optional
        Value at which to fix the slope damping parameter. Default is to
        estimate this parameter.
    fit_kwargs : dict, optional
        Keyword arguments that were passed to the call to `fit` for each
        evaluated model.
    **spec
        Additional keyword arguments to be passed to the each evaluted model.
    """
    def __init__(self, trend, damped, seasonal, seasonal_periods, alpha=None,
                 beta=None, gamma=None, phi=None, fit_kwargs=None, **spec):
        self.trend = trend
        self.damped = damped
        self.seasonal = seasonal
        self.seasonal_periods = seasonal_periods

        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.phi = phi

        self.fit_kwargs = fit_kwargs
        self.spec = spec

    def __call__(self, endog):
        """
        Construct a model using the specification information

        Parameters
        ----------
        endog : array_like
            An observed time-series process :math:`y`.

        Returns
        -------
        model : ExponentialSmoothing
            An ExponentialSmoothing model instance.
        """
        return ExponentialSmoothing(
                endog, trend=self.trend, damped=self.damped,
                seasonal=self.seasonal, seasonal_periods=self.seasonal_periods,
                **self.spec)

    def __str__(self):
        components = []
        if self.trend is not None:
            components.append('trend="%s"' % self.trend)
            if self.damped is not None:
                components.append('damped=%s' % self.damped)
        if self.seasonal is not None and self.seasonal > 1:
            components.append('seasonal="%s"' % self.seasonal)
            components.append('seasonal_periods=%d' % self.seasonal_periods)

        if len(self.spec) > 0:
            components.append('**%s' % str(self.spec))

        return ('ExponentialSmoothing(endog, %s)' % ', '.join(components))


class EvaluatedSpecification(object):
    """
    Evaluated exponential smoothing specification

    Parameters
    ----------
    endog : array_like
        The observed time-series process :math:`y`
    specification : Specification
        Unobserved components specification object.
    measure : {'aic', 'bic', 'hqic'}
        Which information criteria was used for model evaluation.
    value : float
        Value of the information criteria.
    exception : Exception or None
        Exception raised during model fitting, if any.
    warnings : list
        List of warnings issued during model fitting. May be empty.
    selected : bool, optional
        Whether or not the model was selected as having the optimal information
        criteria.
    results : TimeSeriesModelResults, optional
        Results object from model fitting.
    """
    def __init__(self, endog, specification, measure, value, exception,
                 warning, selected=None, results=None):
        self.endog = endog
        self.specification = specification
        self.measure = measure
        self.value = value
        self.exception = exception
        self.warnings = warning
        self.selected = selected
        self.results = results

    @property
    def status(self):
        return 0 if self.exception is None and len(self.warnings) == 0 else 1

    def model(self, endog=None):
        if endog is None:
            endog = self.endog
        return self.specification(endog)

    def __str__(self):
        result = '%s=%.2f' % (self.measure, self.value)
        if self.selected is not None:
            result += (
                ', %s' % ('selected' if self.selected else 'not selected'))
        return '%s[%s]' % (self.specification, result)


class EvaluationResults(OrderedDict):
    """
    ExponentialSmoothing specification search results

    Parameters
    ----------
    endog : array_like
        The observed time-series process :math:`y`
    measure : {'aic', 'bic', 'hqic'}
        Which information criteria was used for model evaluation.
    results : dict of EvaluatedSpecification
        Dictionary with values containing the evaluated models to be considered
        in the specification search.
    alpha : float, optional
        Value at which to fix the level smoothing parameter. Default is to
        estimate this parameter.
    beta : float, optional
        Value at which to fix the slope smoothing parameter. Default is to
        estimate this parameter.
    gamma : float, optional
        Value at which to fix the seasoanl smoothing parameter. Default is to
        estimate this parameter.
    phi : float, optional
        Value at which to fix the slope damping parameter. Default is to
        estimate this parameter.
    fit_kwargs : dict, optional
        Keyword arguments that were passed to the call to `fit` for each
        evaluated model.
    **spec
        Additional keyword arguments to be passed to the each evaluted model.
    """
    def __init__(self, endog, measure, results, alpha=None, beta=None,
                 gamma=None, phi=None, fit_kwargs=None, **spec):
        self.endog = endog
        self.measure = measure

        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.phi = phi

        self.fit_kwargs = fit_kwargs
        self.spec = spec

        # Need to do this for `Summary` to work properly
        self.model = TimeSeriesModel(self.endog)
        self.model.nobs = len(self.model.endog)
        self.params = []

        def sort_value(x):
            return x.value
        
        super(EvaluationResults, self).__init__(
            zip(range(len(results)), sorted(results.values(), key=sort_value)))

        i = 0
        for key, result in self.items():
            result.selected = (i == 0)
            if result.selected:
                self.value = result.value
            i += 1

        self.selected = self[0]

    def summary(self):
        """
        Summarize the specification search

        Returns
        -------
        summary : Summary instance
            This holds the summary table and text, which can be printed or
            converted to various output formats.

        See Also
        --------
        statsmodels.iolib.summary.Summary
        """
        from statsmodels.iolib.summary import Summary

        # Model specification results
        model = self.model
        title = 'Specification Search Results'

        # Information about the endog
        start = 0
        if model._index_dates:
            ix = model._index
            d = ix[start]
            sample = ['%02d-%02d-%02d' % (d.month, d.day, d.year)]
            d = ix[-1]
            sample += ['- ' + '%02d-%02d-%02d' % (d.month, d.day, d.year)]
        else:
            sample = [str(start), ' - ' + str(model.nobs)]

        # Create the tables
        model_name = 'ExponentialSmoothing'

        top_left = [('Dep. Variable:', None)]
        top_left.append(('Model:', [model_name]))
        top_left += [
            ('Date:', None),
            ('Time:', None),
            ('Sample:', [sample[0]]),
            ('', [sample[1]])
        ]

        measure = self.measure.upper()
        top_right = [
            ('No. Observations:', [model.nobs]),
            ('No. Specifications:', [len(self)]),
            ('Measure', [measure]),
            ('Minimum %s' % measure, ["%#5.3f" % self.value]),
        ]

        # - Models table -----------------------------------------------------
        trend = []
        damped = []
        seasonal = []
        measure = []
        for res in self.values():
            spec = res.specification

            trend.append(spec.trend)
            damped.append(spec.damped)
            seasonal.append(spec.seasonal)
            measure.append(res.value)

        # Add the model specification information
        param_header = []
        params_data = []
        if np.any(trend):
            param_header.append('trend')
            params_data.append(trend)
            if np.any(damped):
                param_header.append('damped')
                params_data.append(damped)
        if np.any(seasonal):
            param_header.append('seasonal')
            params_data.append(seasonal)

        # Add the specified measure
        param_header.append(self.measure)
        params_data.append(['%#6.3f' % x for x in measure])

        # Add the other measures, if applicable
        for other_measure in ['aic', 'bic']:
            if other_measure == self.measure:
                continue
            param_header.append(other_measure)
            params_data.append(['%#6.2f' % getattr(x.results, other_measure)
                                for x in self.values()
                                if x.results is not None])

        # Add the number of parameters, if applicable
        param_header.append('k_params')

        params_data.append([
            '%d' % np.sum(x.results.params_formatted['optimized'])
            for x in self.values() if x.results is not None])

        # Add in the status
        param_header.append('warnings/exception')
        params_data.append(['%s' % 'No' if x.status == 0 else 'Yes'
                            for x in self.values()])

        params_data = list(zip(*params_data))
        params_stubs = ['%d ' % x for x in range(len(params_data))]

        from statsmodels.iolib.table import SimpleTable
        models_table = SimpleTable(params_data,
                                      param_header,
                                      params_stubs,
                                      title='Evaluated models',
                                      )

        # - Summary ----------------------------------------------------------

        summary = Summary()
        summary.add_table_2cols(self, gleft=top_left, gright=top_right,
                                title=title)
        summary.tables.append(models_table)

        fixed_params = []
        if self.alpha is not None:
            fixed_params.append(('smoothing_level', self.alpha))
        if self.beta is not None:
            fixed_params.append(('smoothing_slope', self.beta))
        if self.gamma is not None:
            fixed_params.append(('smoothing_seasonal', self.gamma))
        if self.phi is not None:
            fixed_params.append(('damping_slope', self.phi))
        if len(fixed_params) > 0:
            etext = ["    {0}: {1}".format(key, value)
                     for key, value in fixed_params]
            etext.insert(0, "Fixed parameters:")
            summary.add_extra_txt(etext)

        if len(self.spec) > 0:
            etext = ["    {0}: {1}".format(key, value)
                     for key, value in self.spec.items()]
            etext.insert(0, "Additional specification variables:")
            summary.add_extra_txt(etext)

        return summary


def specification_search(endog, measure='aic', trend=None, damped=None,
                         seasonal=None, seasonal_periods=1, additive_only=None,
                         options=None, alpha=None, beta=None, gamma=None,
                         phi=None, fit_kwargs=None, store_results=True,
                         store_results_data=True, **spec):
    r"""
    Selection search over exponential smoothing models.

    Parameters
    ----------
    endog : list
        The observed time-series process :math:`y`
    measure : {'aic', 'bic', 'hqic'}
        Which information criteria to use for model evaluation.
        Default is 'aic'.
    trend : str or list of str or None, optional
        Trend specifications to consider; valid values include
        `[None, 'add', 'mul']`.
    damped : bool or list of bool, optional
        Damped trend specifications to consider. Either a single
        specification boolean or a list of specification booleans.
    seasonal_periods : int, optional
        The period of the seasonal component, if any. Default is None.
    additive_only : boolean, optional
        Whether or not to only consider additive trend and seasonal models.
        Cannot be used in combination with `trend` or `seasonal`.
    alpha : float, optional
        Value at which to fix the level smoothing parameter. Default is to
        estimate this parameter.
    beta : float, optional
        Value at which to fix the slope smoothing parameter. Default is to
        estimate this parameter.
    gamma : float, optional
        Value at which to fix the seasoanl smoothing parameter. Default is to
        estimate this parameter.
    phi : float, optional
        Value at which to fix the slope damping parameter. Default is to
        estimate this parameter.
    options : iterable of iterables, optional
        List of model specifications to consider. Each specification should be
        an iterable of the form `(trend, damped, seasonal)`.
    fit_kwargs : dict, optional
        Keyword arguments to be passed to each call to `fit`.
    store_results : bool, optional
        Whether or not to save the results objects associated with each
        evaluated model. Default is True.
    store_results_data ; bool, optional
        Whether or not to save all results data. If False, the `remove_data`
        method is called on each results object. Default is False. Setting this
        to True will result in increased memory usage.
    **spec
        Additional arguments to be passed to `ExponentialSmoothing`. No model
        specification is made on these arguments - all evaluated models will be
        passed these arguments.

    Returns
    -------
    EvaluationResults

    Notes
    -----
    Status : Work In Progress.

    Example
    -------
    from statsmodels import datasets
    macrodata = datasets.macrodata.load_pandas().data
    macrodata.index = pd.PeriodIndex(start='1959Q1', end='2009Q3', freq='Q')
    search_results = exponentialsmoothing.specification_search(macrodata.infl)
    print(search_results.summary())
    selected_results = search_results.selected.results
    """
    # Validate input
    if (additive_only is not None and
            (trend is not None or seasonal is not None)):
        raise ValueError('Cannot specify `additive_only` in combination with'
                         ' `trend` or `seasonal`.')
    if seasonal is not None and seasonal_periods is None:
        raise ValueError('When argument `seasonal` is given,'
                         ' `seasonal_periods` must also be provided.')
    some_spec_given = (trend is not None or damped is not None or
                       seasonal is not None or additive_only is not None)
    if options is not None and some_spec_given:
        raise ValueError('Cannot specify individual specifications (`trend`,'
                         ' `damped`, etc.) in combination with `options`.')

    # Set default values
    additive_only = True if additive_only is None else False
    if trend is None:
        trend = [None, 'add'] if additive_only else [None, 'add', 'mul']
    if damped is None:
        damped = [False, True]
    seasonal_periods = 1 if seasonal_periods is None else seasonal_periods
    if seasonal is None and seasonal_periods > 1:
        seasonal = [None, 'add'] if additive_only else [None, 'add', 'mul']

    # Construct the grid
    if options is not None:
        grid = [[np.array(options[j][i]) for j in range(len(options))]
                for i in range(len(options[0]))]
    else:
        grid = np.meshgrid(trend, damped, seasonal)

    # Perform selection exercise
    evaluated = {}
    for key in np.nditer(grid, flags=['refs_ok']):
        key = tuple(map(np.asscalar, key))
        (trend, damped, seasonal) = key

        if trend is None and damped:
            continue

        res, value, exception, warning = evaluate(
            endog, measure, trend, damped, seasonal, seasonal_periods,
            alpha, beta, gamma, phi, fit_kwargs=fit_kwargs, **spec)
        if not store_results:
            res = None
        # TODO: the result class currently doesn't support this:
        # if res is not None and not store_results_data:
        #     res.remove_data()
        specification = Specification(
            trend, damped, seasonal, seasonal_periods, alpha, beta, gamma, phi,
            fit_kwargs=fit_kwargs, **spec)
        evaluated[key] = EvaluatedSpecification(
            endog, specification, measure, value, exception, warning,
            selected=None, results=res)

    return EvaluationResults(endog, measure, evaluated, alpha, beta, gamma,
                             phi, fit_kwargs=fit_kwargs, **spec)


def auto(endog, measure='aic', trend=None, damped=None,
         seasonal=None, seasonal_periods=1, additive_only=None,
         options=None, alpha=None, beta=None, gamma=None,
         phi=None, fit_kwargs=None, **spec):
    r"""
    Selection search over exponential smoothing models.

    Parameters
    ----------
    endog : list
        The observed time-series process :math:`y`
    measure : {'aic', 'bic', 'hqic'}
        Which information criteria to use for model evaluation.
        Default is 'aic'.
    trend : str or list of str or None, optional
        Trend specifications to consider; value values include
        `[None, 'add', 'mul']`.
    damped : bool or list of bool, optional
        Damped trend specifications to consider. Either a single
        specification boolean or a list of specification booleans.
    seasonal_periods : int, optional
        The period of the seasonal component, if any. Default is None.
    additive_only : boolean, optional
        Whether or not to only consider additive trend and seasonal models.
        Cannot be used in combination with `trend` or `seasonal`.
    alpha : float, optional
        Value at which to fix the level smoothing parameter. Default is to
        estimate this parameter.
    beta : float, optional
        Value at which to fix the slope smoothing parameter. Default is to
        estimate this parameter.
    gamma : float, optional
        Value at which to fix the seasoanl smoothing parameter. Default is to
        estimate this parameter.
    phi : float, optional
        Value at which to fix the slope damping parameter. Default is to
        estimate this parameter.
    options : iterable of iterables, optional
        List of model specifications to consider. Each specification should be
        an iterable of the form `(trend, damped, seasonal)`.
    fit_kwargs : dict, optional
        Keyword arguments to be passed to each call to `fit`.
    store_results : bool, optional
        Whether or not to save the results objects associated with each
        evaluated model. Default is True.
    store_results_data ; bool, optional
        Whether or not to save all results data. If False, the `remove_data`
        method is called on each results object. Default is False. Setting this
        to True will result in increased memory usage.
    **spec
        Additional arguments to be passed to `ExponentialSmoothing`. No model
        specification is made on these arguments - all evaluated models will be
        passed these arguments.

    Returns
    -------
    HoltWintersResults

    Notes
    -----
    Status : Work In Progress.

    Example
    -------
    from statsmodels import datasets
    macrodata = datasets.macrodata.load_pandas().data
    macrodata.index = pd.PeriodIndex(start='1959Q1', end='2009Q3', freq='Q')
    mod = exponentialsmoothing.auto(macrodata['infl'])
    """
    res = specification_search(
        endog, measure=measure, trend=trend, damped=damped, seasonal=seasonal,
        seasonal_periods=seasonal_periods, additive_only=additive_only,
        options=options, alpha=alpha, beta=beta, gamma=gamma, phi=phi,
        fit_kwargs=fit_kwargs, store_results=False, store_results_data=False,
        **spec)

    fit_kwargs = {} if fit_kwargs is None else fit_kwargs
    return res[0].model().fit(**fit_kwargs)
