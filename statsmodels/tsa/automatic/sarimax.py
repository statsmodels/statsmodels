"""
Automatic model selection for SARIMAX models.

Author: Chad Fulton, Abhijeet Panda
License: BSD-3
"""
import warnings
from collections import OrderedDict

import numpy as np

from statsmodels.tsa.base.tsa_model import TimeSeriesModel
from statsmodels.tsa.statespace.sarimax import SARIMAX


def evaluate(endog, measure, order, trend, seasonal_order, fit_kwargs=None,
             **spec):
    r"""
    Evaluate a given SARIMAX model specification

    Parameters
    ----------
    endog : array_like
        The observed time-series process :math:`y`
    measure : {'aic', 'bic', 'hqic'}
        Which information criteria to use for model evaluation.
    order : iterable or iterable of iterables
        The (p,d,q) order of the model for the number of AR parameters,
        differences, and MA parameters.
    trend : str{'n','c','t','ct'} or iterable
        Parameter controlling the deterministic trend polynomial :math:`A(t)`.
        Can be specified as a string where 'c' indicates a constant (i.e. a
        degree zero component of the trend polynomial), 't' indicates a
        linear trend with time, and 'ct' is both. Can also be specified as an
        iterable defining the polynomial as in `numpy.poly1d`, where
        `[1,1,0,1]` would denote :math:`a + bt + ct^3`.
    seasonal_order : iterable or iterable of iterables
        The (P,D,Q,s) order of the seasonal component of the model for the
        AR parameters, differences, MA parameters, and periodicity.
    **spec
        Additional arguments to be passed to `SARIMAX` (for example `exog`).

    Returns
    -------
    results : SARIMAXResults or None
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
            mod = SARIMAX(endog, order=order, trend=trend,
                          seasonal_order=seasonal_order, **spec)
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
    """
    SARIMAX specification

    Parameters
    ----------
    order : iterable or iterable of iterables
        The (p,d,q) order of the model for the number of AR parameters,
        differences, and MA parameters.
    trend : str{'n','c','t','ct'} or iterable
        Parameter controlling the deterministic trend polynomial.
    seasonal_order : iterable
        The (P,D,Q,s) order of the seasonal component of the model for the
        AR parameters, differences, MA parameters, and periodicity.
    fit_kwargs : dict, optional
        Keyword arguments to be passed to a call to `fit`.
    **spec
        Additional keyword arguments to be passed to the SARIMAX constructor.
    """
    def __init__(self, order, trend, seasonal_order, fit_kwargs=None, **spec):
        self.order = order
        self.trend = trend
        self.seasonal_order = seasonal_order
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
        model : SARIMAX
            An SARIMAX model instance.
        """
        return SARIMAX(endog, order=self.order, trend=self.trend,
                       seasonal_order=self.seasonal_order, **self.spec)

    def __str__(self):
        components = ['order=(%s)' % (', '.join(map(str, self.order))),
                      'trend="%s"' % self.trend]
        if self.seasonal_order[:3] != (0, 0, 0):
            components.append('seasonal_order=(%s)' %
                              (', '.join(map(str, self.seasonal_order))))
        if len(self.spec) > 0:
            components.append('**%s' % str(self.spec))

        return ('SARIMAX(endog, %s)' % ', '.join(components))


class EvaluatedSpecification(object):
    """
    Evaluated SARIMAX specification

    Parameters
    ----------
    endog : array_like
        The observed time-series process :math:`y`
    specification : Specification
        SARIMAX specification object.
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
        """
        Status of `fit` call. Is 1 if there was an exception or warning, else 0
        """
        return 0 if self.exception is None and len(self.warnings) == 0 else 1

    def model(self, endog=None):
        """
        Construct a model using the specification information

        Parameters
        ----------
        endog : array_like, optional
            An observed time-series process :math:`y`. Default is to return a
            model object with the `endog` used in evaluation.

        Returns
        -------
        model : SARIMAX
            An SARIMAX model instance.
        """
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
    SARIMAX specification search results

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
        model_name = 'SARIMAX'

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
        order = []
        trend = []
        seasonal_order = []
        measure = []
        for res in self.values():
            spec = res.specification

            order.append(spec.order)
            trend.append(spec.trend)
            if spec.seasonal_order[:3] != (0, 0, 0):
                seasonal_order.append(spec.seasonal_order)
            else:
                seasonal_order.append(None)
            measure.append(res.value)

        # Add the model specification information
        param_header = ['order', 'trend']
        params_data = [order, trend]
        if np.any(seasonal_order):
            param_header.append('seasonal_order')
            params_data.append(seasonal_order)

        # Add the specified measure
        param_header.append(self.measure)
        params_data.append(['%#6.3f' % x for x in measure])

        # Add the other measures, if applicable
        for other_measure in ['aic', 'bic', 'hqic']:
            if other_measure == self.measure:
                continue
            param_header.append(other_measure)
            params_data.append(['%#6.2f' % getattr(x.results, other_measure)
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


def specification_search(endog, measure='aic', max_order=None,
                         max_seasonal_order=None, s=None, trend=None, p=None,
                         d=0, q=None, P=None, D=0, Q=None, stepwise=False,
                         options=None, fit_kwargs=None, store_results=True,
                         store_results_data=False, **spec):
    r"""
    Selection search over SARIMAX models.

    Parameters
    ----------
    endog : list
        The observed time-series process :math:`y`
    measure : {'aic', 'bic', 'hqic'}
        Which information criteria to use for model evaluation.
        Default is 'aic'.
    max_order : iterable of length 2, optional
        Maximum orders to consider for the autoregressive and moving average
        orders. If this is set, then all orders up to `max_order` are
        evaluated. Default is (3, 3)
    max_seasonal_order : iterable of length 2, optional
        Maximum orders to consider for the seasonal autoregressive and moving
        average orders. If this is set, then all orders up to
        `max_seasonal_order` are evaluated. Default is (3, 3)
    s : int, optional
        The period of the seasonal component, if any. Default is no seasonal
        component.
    trend : str, iterable, optional
        Trends to consider. Any of the trend types accepted by SARIMAX can be
        used (e.g. 'n', 'c', 't', etc. or a custom trend polynomial
        specification). Default is ['n', 'c'].
    p : int, iterable, optional
        Autoregressive orders to consider, if not using `max_order`,
        `stepwise`, or `options`. By default, this function uses `max_order`,
        so the default value of `p` is None.
    d : int, optional
        Order of integration. Default is zero.
    q : int, iterable or None, optional
        Moving average orders to consider, if not using `max_order`,
        `stepwise`, or `options`. By default, this function uses `max_order`,
        so the default value of `q` is None.
    P : int, iterable, optional
        Seasonal autoregressive orders to consider, if not using
        `max_seasonal_order`, `stepwise`, or `options`. Only used if `s` is
        greater than one. By default, this function uses `max_seasonal_order`,
        so the default value of `P` is None.
    D : int, optional
        Seasonal order of integration. Default is zero.
    Q : int, iterable, optional
        Seasonal moving average orders to consider, if not using
        `max_seasonal_order`, `stepwise`, or `options`. Only used if `s` is
        greater than one. By default, this function uses `max_seasonal_order`,
        so the default value of `Q` is None.
    stepwise : boolean, optional
        Specifies whether to use the stepwise algorithm. Default is False.
    options : iterable of iterables, optional
        List of model specifications to consider. Each specification should be
        an iterable of the form `(p, d, q, P, D, Q, s, trend)`.
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
        Additional arguments to be passed to `SARIMAX` (for example `exog`). No
        model specification is made on these arguments - all evaluated models
        will be passed these arguments.

    Returns
    -------
    EvaluationResults

    Notes
    -----
    Status : Work In Progress.

    References
    ----------
    .. [1] Hyndman, Rob J., and Yeasmin Khandakar.
       Automatic Time Series Forecasting: The forecast Package for R.

    Example
    -------
    from statsmodels import datasets
    macrodata = datasets.macrodata.load_pandas().data
    macrodata.index = pd.PeriodIndex(start='1959Q1', end='2009Q3', freq='Q')
    search_results = sarimax.specification_search(macrodata['infl'], d=0)
    print(search_results.summary())
    selected_results = search_results.selected.results
    """
    # Validate input
    if (p is not None or q is not None) and max_order is not None:
        raise ValueError('Cannot specify individual orders `p` and/or `q`'
                         ' in combination with `max_order`.')

    if (P is not None or Q is not None) and max_seasonal_order is not None:
        raise ValueError('Cannot specify individual orders `P` and/or `Q`'
                         ' in combination with `max_seasonal_order`.')

    some_order_given = (p is not None or q is not None or P is not None or
                        Q is not None or max_order is not None or
                        max_seasonal_order is not None)

    if stepwise and some_order_given:
        raise ValueError('Cannot specify individual orders or maximum orders'
                         ' in combination with `stepwise`.')

    if stepwise and options is not None:
        raise ValueError('Cannot specify specific options'
                         ' in combination with `stepwise`.')

    if options is not None and some_order_given:
        raise ValueError('Cannot specify individual orders or maximum orders'
                         ' in combination with `options`.')

    # Set default values
    if not some_order_given and not stepwise and options is None:
        max_order = (3, 3)
        max_seasonal_order = (1, 1)

    if trend is None:
        trend = ['n', 'c']

    p = 0 if p is None else p
    q = 0 if q is None else q
    P = 0 if P is None else P
    Q = 0 if Q is None else Q
    s = 1 if s is None else s

    if max_order is not None:
        p = np.arange(max_order[0] + 1)
        q = np.arange(max_order[1] + 1)

    if max_seasonal_order is not None and s > 1:
        P = np.arange(max_seasonal_order[0] + 1)
        Q = np.arange(max_seasonal_order[1] + 1)

    if stepwise:
        # TODO check that this is what is recommended
        options = [
            [2, d, 2, 1, D, 1, s, 'n'],
            [0, d, 0, 0, D, 0, s, 'n'],
            [1, d, 0, 1, D, 0, s, 'n'],
            [1, d, 0, 1, D, 0, s, 'n']]

        # Remove seasonal parameters if no seasonal
        if s == 1:
            for i in range(len(options)):
                options[i][3] = 0
                options[i][5] = 0

        # Trend
        # TODO check that this is what we want
        if d + D <= 1:
            new_options = []
            for option in options:
                new_option = option[:]
                new_option[-1] = 'c'
                new_options.append(new_option)
            options += new_options

    # Construct the grid
    if options is not None:
        grid = [[np.array(options[j][i]) for j in range(len(options))]
                for i in range(len(options[0]))]
    else:
        grid = np.meshgrid(p, d, q, P, D, Q, s, trend)

    # Perform first selection exercise
    evaluated = {}
    selection = None
    selection_value = np.inf
    for p, d, q, P, D, Q, s, trend in np.nditer(grid):
        key = tuple([int(x) for x in (p, d, q, P, D, Q, s)] + [str(trend)])
        res, value, exception, warning = evaluate(
            endog, measure, order=key[:3], trend=trend,
            seasonal_order=key[3:7], **spec)
        evaluated[key] = res, value, exception, warning

        if value < selection_value:
            selection = key
            selection_value = value

    # If stepwise, now perform second selection exercise
    if stepwise:
        p, d, q, P, D, Q, s, trend = selection
        # TODO check that this is right
        options = [
            [p + 1, d, q, P, D, Q, s, 'n'],
            [p - 1, d, q, P, D, Q, s, 'n'],
            [p, d, q + 1, P, D, Q, s, 'n'],
            [p, d, q - 1, P, D, Q, s, 'n'],
            [p + 1, d, q + 1, P, D, Q, s, 'n'],
            [p - 1, d, q - 1, P, D, Q, s, 'n']]
        if s > 1:
            options += [
                [p, d, q, P + 1, D, Q, s, 'n'],
                [p, d, q, P - 1, D, Q, s, 'n'],
                [p, d, q, P, D, Q + 1, s, 'n'],
                [p, d, q, P, D, Q - 1, s, 'n'],
                [p, d, q, P + 1, D, Q + 1, s, 'n'],
                [p, d, q, P - 1, D, Q - 1, s, 'n']]
        options = [option for option in options
                   if np.all(np.array(option[:-1]) >= 0)]

        # Trend
        # TODO check that this is what we want
        if d + D <= 1:
            new_options = []
            for option in options:
                new_option = option[:]
                new_option[-1] = 'c'
                new_options.append(new_option)
            options += new_options

        # Create the new grid
        grid = [[np.array(options[j][i]) for j in range(len(options))]
                for i in range(len(options[0]))]

        for p, d, q, P, D, Q, s, trend in np.nditer(grid):
            key = tuple([int(x) for x in (p, d, q, P, D, Q, s)] + [str(trend)])
            res, value, exception, warning = evaluate(
                endog, measure, order=key[:3], trend=trend,
                seasonal_order=key[3:7], fit_kwargs=fit_kwargs, **spec)
            evaluated[key] = res, value, exception, warning

            if value < selection_value:
                selection = key
                selection_value = value

    # Create a results object
    for key, output in evaluated.items():
        res, value, exception, warning = output
        if not store_results:
            res = None
        if res is not None and not store_results_data:
            res.remove_data()

        p, d, q, P, D, Q, s, trend = key
        specification = Specification(
            order=(p, d, q), trend=trend, seasonal_order=(P, D, Q, s),
            fit_kwargs=fit_kwargs, **spec)
        evaluated[key] = EvaluatedSpecification(
            endog, specification, measure, value, exception, warning,
            selected=None, results=res)

    return EvaluationResults(endog, measure, evaluated, fit_kwargs=fit_kwargs,
                             **spec)


def auto(endog, measure='aic', max_order=None, max_seasonal_order=None, s=None,
         trend=None, p=None, d=0, q=None, P=None, D=0, Q=None, stepwise=False,
         options=None, fit_kwargs=None, **spec):
    r"""
    Retrieve optimal model based on selection search of SARIMAX models.

    Parameters
    ----------
    endog : list
        The observed time-series process :math:`y`
    measure : {'aic', 'bic', 'hqic'}
        Which information criteria to use for model evaluation.
        Default is 'aic'.
    max_order : iterable of length 2, optional
        Maximum orders to consider for the autoregressive and moving average
        orders. If this is set, then all orders up to `max_order` are
        evaluated. Default is (3, 3)
    max_seasonal_order : iterable of length 2, optional
        Maximum orders to consider for the seasonal autoregressive and moving
        average orders. If this is set, then all orders up to
        `max_seasonal_order` are evaluated. Default is (3, 3)
    s : int, optional
        The period of the seasonal component, if any. Default is no seasonal
        component.
    trend : str, iterable, optional
        Trends to consider. Any of the trend types accepted by SARIMAX can be
        used (e.g. 'n', 'c', 't', etc. or a custom trend polynomial
        specification). Default is ['n', 'c'].
    p : int, iterable, optional
        Autoregressive orders to consider, if not using `max_order`,
        `stepwise`, or `options`. By default, this function uses `max_order`,
        so the default value of `p` is None.
    d : int, optional
        Order of integration. Default is zero.
    q : int, iterable or None, optional
        Moving average orders to consider, if not using `max_order`,
        `stepwise`, or `options`. By default, this function uses `max_order`,
        so the default value of `q` is None.
    P : int, iterable, optional
        Seasonal autoregressive orders to consider, if not using
        `max_seasonal_order`, `stepwise`, or `options`. Only used if `s` is
        greater than one. By default, this function uses `max_seasonal_order`,
        so the default value of `P` is None.
    D : int, optional
        Seasonal order of integration. Default is zero.
    Q : int, iterable, optional
        Seasonal moving average orders to consider, if not using
        `max_seasonal_order`, `stepwise`, or `options`. Only used if `s` is
        greater than one. By default, this function uses `max_seasonal_order`,
        so the default value of `Q` is None.
    stepwise : boolean, optional
        Specifies whether to use the stepwise algorithm. Default is False.
    options : iterable of iterables, optional
        List of model specifications to consider. Each specification should be
        an iterable of the form `(p, d, q, P, D, Q, s, trend)`.
    fit_kwargs : dict, optional
        Keyword arguments to be passed to each call to `fit`.
    **spec
        Additional arguments to be passed to `SARIMAX` (for example `exog`). No
        model specification is made on these arguments - all evaluated models
        will be passed these arguments.

    Returns
    -------
    SARIMAXResults

    Notes
    -----
    Status : Work In Progress.

    References
    ----------
    .. [1] Hyndman, Rob J., and Yeasmin Khandakar.
       Automatic Time Series Forecasting: The forecast Package for R.

    Example
    -------
    from statsmodels import datasets
    macrodata = datasets.macrodata.load_pandas().data
    macrodata.index = pd.PeriodIndex(start='1959Q1', end='2009Q3', freq='Q')
    mod = sarimax.auto(macrodata['infl'], d=0)
    """
    res = specification_search(
        endog, measure=measure, max_order=max_order,
        max_seasonal_order=max_seasonal_order, s=s, trend=trend, p=p, d=d, q=q,
        P=P, D=D, Q=Q, stepwise=stepwise, options=options,
        fit_kwargs=fit_kwargs, store_results=False, store_results_data=False,
        **spec)

    fit_kwargs = {} if fit_kwargs is None else fit_kwargs
    return res[0].model().fit(**fit_kwargs)
