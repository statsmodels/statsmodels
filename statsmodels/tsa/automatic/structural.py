"""
Automatic model selection for SARIMAX models.

Author: Chad Fulton
License: BSD-3
"""
import warnings
from collections import OrderedDict

import numpy as np

from statsmodels.tsa.base.tsa_model import TimeSeriesModel
from statsmodels.tsa.statespace.structural import UnobservedComponents


levels = {
    'stationary': ['irregular', 'fixed intercept', 'deterministic constant'],
    'trend_stationary': ['fixed slope', 'deterministic trend',
                         'local linear deterministic trend'],
    'I(1)': ['local level', 'random walk', 'random walk with drift'],
    'I(2)': ['local linear trend', 'smooth trend', 'random trend']
}
levels['I(0)'] = levels['stationary']


def evaluate(endog, measure, level, seasonal, cycle, autoregressive,
             stochastic_seasonal, stochastic_cycle, damped_cycle,
             fit_kwargs=None, **spec):
    r"""
    Evaluate a given UnobservedComponents model specification

    Parameters
    ----------
    endog : list
        The observed time-series process :math:`y`
    measure : {'aic', 'bic', 'hqic'}
        Which information criteria to use for model evaluation.
    level : str
        Level specification string.
    seasonal : int or None
        The period of the seasonal component or None.
    cycle : bool
        Whether or not to include a cycle component.
    autoregressive : int or list of int, optional
        The order of the autoregressive component or None.
    stochastic_seasonal : bool or iterable of bool, optional
        Whether or not each seasonal component(s) is (are) stochastic.
    stochastic_cycle : bool or iterable of bool, optional
        Whether or not any cycle component is stochastic.
    damped_cycle : bool or iterable of bool, optional
        Whether or not the cycle component is damped.
    fit_kwargs : dict, optional
        Keyword arguments to be passed to each call to `fit`.
    **spec
        Additional arguments to be passed to `UnobservedComponents` (for
        example `exog`). No model specification is made on these arguments -
        all evaluated models will be passed these arguments.

    Returns
    -------
    results : UnobservedComponentsResults or None
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
            mod = UnobservedComponents(
                endog, level, seasonal=seasonal, cycle=cycle,
                autoregressive=autoregressive,
                stochastic_seasonal=stochastic_seasonal,
                stochastic_cycle=stochastic_cycle, damped_cycle=damped_cycle,
                **spec)
            fit_kwargs = {} if fit_kwargs is None else fit_kwargs
            fit_kwargs.setdefault('disp', False)
            res = mod.fit(**fit_kwargs)
            value = getattr(res, measure)
    except Exception as e:
        res = None
        value = np.nan
        exception = e

    return res, value, exception, warning


class Specification(object):
    r"""
    Unobserved components specification

    Parameters
    ----------
    endog : list
        The observed time-series process :math:`y`
    measure : {'aic', 'bic', 'hqic'}
        Which information criteria to use for model evaluation.
        Default is 'aic'.
    level : str
        Level specification string.
    seasonal : int or None
        The period of the seasonal component or None.
    cycle : bool
        Whether or not to include a cycle component.
    autoregressive : int or list of int, optional
        The order of the autoregressive component or None.
    stochastic_seasonal : bool or iterable of bool, optional
        Whether or not each seasonal component(s) is (are) stochastic.
    stochastic_cycle : bool or iterable of bool, optional
        Whether or not any cycle component is stochastic.
    damped_cycle : bool or iterable of bool, optional
        Whether or not the cycle component is damped.
    fit_kwargs : dict, optional
        Keyword arguments to be passed to each call to `fit`.
    **spec
        Additional arguments to be passed to `UnobservedComponents` (for
        example `exog`). No model specification is made on these arguments -
        all evaluated models will be passed these arguments.
    """
    def __init__(self, level, seasonal, cycle, autoregressive,
                 stochastic_seasonal, stochastic_cycle, damped_cycle,
                 fit_kwargs=None, **spec):
        self.level = level
        self.seasonal = seasonal
        self.cycle = cycle
        self.autoregressive = autoregressive
        self.stochastic_seasonal = stochastic_seasonal
        self.stochastic_cycle = stochastic_cycle
        self.damped_cycle = damped_cycle
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
        model : UnobservedComponents
            An UnobservedComponents model instance.
        """
        return UnobservedComponents(
                endog, self.level, seasonal=self.seasonal, cycle=self.cycle,
                autoregressive=self.autoregressive,
                stochastic_seasonal=self.stochastic_seasonal,
                stochastic_cycle=self.stochastic_cycle,
                damped_cycle=self.damped_cycle, **self.spec)

    def __str__(self):
        components = ['"%s"' % self.level]
        if self.seasonal is not None and self.seasonal > 1:
            components.append('seasonal=%d' % self.seasonal)
        if self.cycle:
            components.append('cycle=True')
        if self.autoregressive is not None and self.autoregressive > 0:
            components.append('autoregressive=%d' % self.autoregressive)
        if self.seasonal is not None and self.stochastic_seasonal:
            components.append('stochastic_seasonal=True')
        if self.cycle:
            if self.stochastic_cycle:
                components.append('stochastic_cycle=True')
            if self.damped_cycle:
                components.append('damped_cycle=True')

        if len(self.spec) > 0:
            components.append('**%s' % str(self.spec))

        return ('UnobservedComponents(endog, %s)' % ', '.join(components))


class EvaluatedSpecification(object):
    """
    Evaluated UnobservedComponents specification

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
    UnobservedComponents specification search results

    Parameters
    ----------
    endog : array_like
        The observed time-series process :math:`y`
    measure : {'aic', 'bic', 'hqic'}
        Which information criteria was used for model evaluation.
    results : dict of EvaluatedSpecification
        Dictionary with values containing the evaluated models to be considered
        in the specification search.
    fit_kwargs : dict, optional
        Keyword arguments that were passed to the call to `fit` for each
        evaluated model.
    **spec
        Additional keyword arguments to be passed to the each evaluted model.
    """
    def __init__(self, endog, measure, results, fit_kwargs=None, **spec):
        self.endog = endog
        self.measure = measure

        # Need to do this for `Summary` to work properly
        self.model = TimeSeriesModel(self.endog)
        self.model.nobs = len(self.model.endog)
        self.params = []
        self.fit_kwargs = fit_kwargs
        self.spec = spec

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
        model_name = 'UnobservedComponents'

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
        level = []
        seasonal = []
        cycle = []
        autoregressive = []
        stochastic_seasonal = []
        stochastic_cycle = []
        damped_cycle = []
        measure = []
        for res in self.values():
            spec = res.specification

            level.append(spec.level)
            seasonal.append(spec.seasonal)
            cycle.append(spec.cycle)
            autoregressive.append(spec.autoregressive)
            stochastic_seasonal.append(spec.stochastic_seasonal)
            stochastic_cycle.append(spec.stochastic_cycle)
            damped_cycle.append(spec.damped_cycle)
            measure.append(res.value)

        # Add the model specification information
        param_header = ['level']
        params_data = [level]
        if np.any(seasonal):
            param_header.append('seasonal')
            params_data.append(seasonal)
        if np.any(cycle):
            param_header.append('cycle')
            params_data.append(cycle)
        if np.any(autoregressive):
            param_header.append('autoregressive')
            params_data.append(autoregressive)
        if np.any(seasonal) and np.any(stochastic_seasonal):
            param_header.append('stochastic_seasonal')
            params_data.append(stochastic_seasonal)
        if np.any(cycle):
            if np.any(stochastic_cycle):
                param_header.append('stochastic_cycle')
                params_data.append(stochastic_cycle)
            if np.any(damped_cycle):
                param_header.append('damped_cycle')
                params_data.append(damped_cycle)

        # Add the specified measure
        param_header.append(self.measure)
        params_data.append(['%#6.3f' % x for x in measure])

        # Add the other measures, if applicable
        for other_measure in ['aic', 'bic', 'hqic']:
            if other_measure == self.measure:
                continue
            param_header.append(other_measure)
            params_data.append(['%#6.2f' % getattr(x.results, other_measure, None)
                                for x in self.values()
                                if x.results is not None])

        # Add the number of parameters, if applicable
        param_header.append('k_params')
        params_data.append(['%d' % len(getattr(x.results, 'params'))
                            for x in self.values()
                            if x.results is not None])

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

        if len(self.spec) > 0:
            etext = ["    {0}: {1}".format(key, value)
                     for key, value in self.spec.items()]
            etext.insert(0, "Additional specification variables:")
            summary.add_extra_txt(etext)

        return summary


def specification_search(endog, measure='aic', level=None, seasonal=None,
                         cycle=None, autoregressive=None,
                         stochastic_seasonal=None, stochastic_cycle=None,
                         damped_cycle=None, allow_cycle=None,
                         max_autoregressive=None, options=None,
                         fit_kwargs=None, store_results=True,
                         store_results_data=False, **spec):
    r"""
    Selection search over unobserved components models.

    Parameters
    ----------
    endog : list
        The observed time-series process :math:`y`
    measure : {'aic', 'bic', 'hqic'}
        Which information criteria to use for model evaluation.
        Default is 'aic'.
    level : str or iterable of str
        Level specifications to consider. Either a single specification string,
        a list of specification strings, or a specification grouping.
        Specification strings are described in the `UnobservedComponents`
        documentation, and the groupings are ['stationary', 'trend_stationary',
        'I(1)', 'I(2)']. Default is 'I(1)' (i.e. models with one order of
        integration).
    seasonal : int or None, optional
        The period of the seasonal component, if any. Default is None.
    cycle : bool or iterable of bool, optional
        Cycle specifications to consider. Either a single specification
        boolean or a list of specification booleans.
    autoregressive : int or list of int, optional
        Autoregressive orders to consider. Either a single order or a list of
        orders to consider.
    stochastic_seasonal : bool or iterable of bool, optional
        Stochastic seasonal specifications to consider. Either a single
        specification boolean or a list of specification booleans.
    stochastic_cycle : bool or iterable of bool, optional
        Stochastic cycle specifications to consider. Either a single
        specification boolean or a list of specification booleans.
    damped_cycle : bool or iterable of bool, optional
        Damped cycle specifications to consider. Either a single
        specification boolean or a list of specification booleans.
    allow_cycle : bool, optional
        Flag to indicate that a cycle should be considered. Default is False.
        Cannot be used in combination with the `cycle` argument.
    max_autoregressive : iterable, optional
        Maximum order to consider for the autoregressive component. If this is
        set, then all orders up to `max_order` are evaluated. Default is 0.
    options : iterable of iterables, optional
        List of model specifications to consider. Each specification should be
        an iterable of the form `(level, seasonal, cycle, autoregressive,
        stochastic_seasonal, stochastic_cycle, damped_cycle)`.
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
        Additional arguments to be passed to `UnobservedComponents` (for
        example `exog`). No model specification is made on these arguments -
        all evaluated models will be passed these arguments.

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
    search_results = structural.specification_search(macrodata['infl'], d=0)
    print(search_results.summary())
    selected_results = search_results.selected.results
    """
    # Validate input
    measure = measure.lower()
    if measure not in ['aic', 'bic', 'hqic']:
        raise ValueError('Invalid measure')

    some_specs_given = (level is not None or seasonal is not None or
                        cycle is not None or autoregressive is not None or
                        stochastic_seasonal is not None or
                        stochastic_cycle is not None or
                        damped_cycle is not None)

    if cycle is not None and allow_cycle is not None:
        raise ValueError('Cannot specify `cycle` in combination with'
                         ' `allow_cycle`.')

    if autoregressive is not None and max_autoregressive is not None:
        raise ValueError('Cannot specify `autoregressive` in combination with'
                         ' `max_autoregressive`.')

    if options is not None and some_specs_given:
        raise ValueError('Cannot specify individual specification data'
                         ' in combination with `options`.')

    # Set default values
    level = 'I(1)' if level is None else level
    level = levels.get(level, level)
    seasonal = [None, seasonal] if seasonal is not None else False
    cycle = [False, True] if allow_cycle else False
    if max_autoregressive is not None:
        autoregressive = range(max_autoregressive)
    autoregressive = 0 if autoregressive is None else autoregressive
    stochastic_seasonal = ([False, True] if stochastic_seasonal is None else
                           stochastic_seasonal)
    stochastic_cycle = ([False, True] if stochastic_cycle is None else
                        stochastic_cycle)
    damped_cycle = [False, True] if damped_cycle is None else damped_cycle


    if options is not None:
        grid = [[np.array(options[j][i]) for j in range(len(options))]
                for i in range(len(options[0]))]
    else:
        grid = np.meshgrid(level, seasonal, cycle, autoregressive,
                           stochastic_seasonal, stochastic_cycle, damped_cycle)

    # Perform selection exercise
    evaluated = {}
    for key in np.nditer(grid):
        key = tuple(map(np.asscalar, key))
        (level, seasonal, cycle, autoregressive, stochastic_seasonal,
            stochastic_cycle, damped_cycle) = key

        if stochastic_seasonal and not seasonal:
            continue
        if (stochastic_cycle or damped_cycle) and not cycle:
            continue

        res, value, exception, warning = evaluate(
            endog, measure, level, seasonal, cycle, autoregressive,
            stochastic_seasonal, stochastic_cycle, damped_cycle,
            fit_kwargs=fit_kwargs, **spec)
        if not store_results:
            res = None
        if res is not None and not store_results_data:
            res.remove_data()
        specification = Specification(
            level, seasonal, cycle, autoregressive, stochastic_seasonal,
            stochastic_cycle, damped_cycle, fit_kwargs=fit_kwargs, **spec)
        evaluated[key] = EvaluatedSpecification(
            endog, specification, measure, value, exception, warning,
            selected=None, results=res)

    return EvaluationResults(endog, measure, evaluated, fit_kwargs=fit_kwargs,
                             **spec)


def auto(endog, measure='aic', level=None, seasonal=None, cycle=None,
         autoregressive=None, stochastic_seasonal=None, stochastic_cycle=None,
         damped_cycle=None, allow_cycle=None, max_autoregressive=None,
         options=None, fit_kwargs=None, **spec):
    r"""
    Retrieve optimal model via selection search of Unobserved Components models

    Parameters
    ----------
    endog : list
        The observed time-series process :math:`y`
    measure : {'aic', 'bic', 'hqic'}
        Which information criteria to use for model evaluation.
        Default is 'aic'.
    level : str or iterable of str
        Level specifications to consider. Either a single specification string,
        a list of specification strings, or a specification grouping.
        Specification strings are described in the `UnobservedComponents`
        documentation, and the groupings are ['stationary', 'trend_stationary',
        'I(1)', 'I(2)']. Default is 'I(1)' (i.e. models with one order of
        integration).
    seasonal : int or None, optional
        The period of the seasonal component, if any. Default is None.
    cycle : bool or iterable of bool, optional
        Cycle specifications to consider. Either a single specification
        boolean or a list of specification booleans.
    autoregressive : int or list of int, optional
        Autoregressive orders to consider. Either a single order or a list of
        orders to consider.
    stochastic_seasonal : bool or iterable of bool, optional
        Stochastic seasonal specifications to consider. Either a single
        specification boolean or a list of specification booleans.
    stochastic_cycle : bool or iterable of bool, optional
        Stochastic cycle specifications to consider. Either a single
        specification boolean or a list of specification booleans.
    damped_cycle : bool or iterable of bool, optional
        Damped cycle specifications to consider. Either a single
        specification boolean or a list of specification booleans.
    allow_cycle : bool, optional
        Flag to indicate that a cycle should be considered. Default is False.
        Cannot be used in combination with the `cycle` argument.
    max_autoregressive : iterable, optional
        Maximum order to consider for the autoregressive component. If this is
        set, then all orders up to `max_order` are evaluated. Default is 0.
    options : iterable of iterables, optional
        List of model specifications to consider. Each specification should be
        an iterable of the form `(level, seasonal, cycle, autoregressive,
        stochastic_seasonal, stochastic_cycle, damped_cycle)`.
    **spec
        Additional arguments to be passed to `UnobservedComponents` (for
        example `exog`). No model specification is made on these arguments -
        all evaluated models will be passed these arguments.

    Returns
    -------
    UnobservedComponentsResults

    Notes
    -----
    Status : Work In Progress.

    Example
    -------
    from statsmodels import datasets
    macrodata = datasets.macrodata.load_pandas().data
    macrodata.index = pd.PeriodIndex(start='1959Q1', end='2009Q3', freq='Q')
    mod = sarimax.auto(macrodata['infl'], 'I(1)')
    """
    res = specification_search(
        endog, measure=measure, level=level, seasonal=seasonal,
        cycle=cycle, autoregressive=autoregressive,
        stochastic_seasonal=stochastic_seasonal,
        stochastic_cycle=stochastic_cycle, damped_cycle=damped_cycle,
        allow_cycle=allow_cycle, max_autoregressive=max_autoregressive,
        options=options, fit_kwargs=fit_kwargs, store_results=False,
        store_results_data=False,
        **spec)

    fit_kwargs = {} if fit_kwargs is None else fit_kwargs
    return res[0].model().fit(**fit_kwargs)
