"""
Run x12/x13-arima specs in a subprocess from Python and curry results back
into python.

Notes
-----
Many of the functions are called x12. However, they are also intended to work
for x13. If this is not the case, it's a bug.
"""

import os
import subprocess
import tempfile
import re
from warnings import warn

import pandas as pd

from statsmodels.tools.tools import Bunch
from statsmodels.tools.sm_exceptions import (X12NotFoundError, X12Error,
                                             IOWarning)

__all__ = ["select_arima_order", "x13arima_analysis"]

_binary_names = ('x13as.exe', 'x13as', 'x12a.exe', 'x12a')

class _freq_to_period:
    def __getitem__(self, key):
        if key.startswith('M'):
            return 12
        elif key.startswith('Q'):
            return 4


_freq_to_period = _freq_to_period()

_period_to_freq = {12 : 'M', 4 : 'Q'}
_log_to_x12 = {True : 'log', False : 'none', None : 'auto'}
_bool_to_yes_no = lambda x : 'yes' if x else 'no'


def _find_x12(x12path=None, prefer_x13=True):
    """
    If x12path is not given, then either x13as[.exe] or x12a[.exe] must
    be found on the PATH. Otherwise, the environmental variable X12PATH or
    X13PATH must be defined. If prefer_x13 is True, only X13PATH is searched
    for. If it is false, only X12PATH is searched for.
    """
    global _binary_names
    if x12path is not None and x12path.endswith(_binary_names):
        # remove binary from path if given
        x12path = os.path.dirname(x12path)

    if not prefer_x13:  # search for x12 first
        _binary_names = _binary_names[::-1]
        if x12path is None:
            x12path = os.getenv("X12PATH", "")
    elif x12path is None:
        x12path = os.getenv("X13PATH", "")

    for binary in _binary_names:
        x12 = os.path.join(x12path, binary)
        try:
            subprocess.check_call(x12, stdout=subprocess.PIPE,
                                  stderr=subprocess.PIPE)
            return x12
        except OSError:
            pass

    else:
        return False


def _check_x12(x12path=None):
    x12path = _find_x12(x12path)
    if not x12path:
        raise X12NotFoundError("x12a and x13a not found on path. Give the "
                               "path, put them on the path, or set the "
                               "X12PATH environmental variable.")
    return x12path


def _clean_order(order):
    """
    Takes something like (1 1 0)(0 1 1) and returns a arma order, sarma
    order tuple. Also accepts (1 1 0) and return arma order and (0, 0, 0)
    """
    order = re.findall("\([0-9 ]*?\)", order)
    clean = lambda x : tuple(map(int, re.sub("[()]", "", x).split(" ")))
    if len(order) > 1:
        order, sorder = map(clean, order)
    else:
        order = clean(order[0])
        sorder = (0, 0, 0)

    return order, sorder


def run_spec(x12path, specpath, outname=None, meta=False, datameta=False):

    if meta and datameta:
        raise ValueError("Cannot specify both meta and datameta.")
    if meta:
        args = [x12path, "-m " + specpath]
    elif datameta:
        args = [x12path, "-d " + specpath]
    else:
        args = [x12path, specpath]

    if outname:
        args += [outname]

    return subprocess.Popen(args, stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT)


def _make_automdl_options(maxorder, maxdiff, diff):
    options = "\n"
    options += "maxorder = ({} {})\n".format(maxorder[0], maxorder[1])
    if maxdiff is not None:  # maxdiff always takes precedence
        options += "maxdiff = ({} {})\n".format(maxdiff[0], maxdiff[1])
    else:
        options += "diff = ({} {})\n".format(diff[0], diff[1])
    return options


def _make_var_names(X):
    if hasattr(X, "name"):
        var_names = X.name
    elif hasattr(X, "columns"):
        var_names = X.columns
    else:
        raise ValueError("X is not a Series or DataFrame or is unnamed.")
    return " ".join(var_names)


def _make_regression_options(trading, X):
    if not trading and X is None:  # start regression spec
        return ""

    reg_spec = "regression{\n"
    if trading:
        reg_spec += "    variables = (td)\n"
    if X is not None:
        var_names = _make_var_names(X)
        reg_spec += "    user = ({})\n".format(var_names)
        reg_spec += "    data = ({})\n".format("\n".join(map(str,
                                               X.values.ravel().tolist())))

    reg_spec += "}\n"  # close out regression spec
    return reg_spec


def _check_errors(errors):
    errors = errors[errors.find("spc:")+4:].strip()
    if errors and 'ERROR' in errors:
        raise ValueError(errors)
    elif errors and 'WARNING' in errors:
        warn(errors, UserWarning)


def _convert_out_to_series(x, dates, name):
    """
    Convert x to a DataFrame where x is a string in the format given by
    x-13arima-seats output.
    """
    from StringIO import StringIO
    from pandas import read_table
    out = read_table(StringIO(x), skiprows=2, header=None)
    return out.set_index(dates).rename(columns={1 : name})[name]


def _open_and_read(fname):
    # opens a file, reads it, and make sure it's closed
    with open(fname, 'r') as fin:
        fout = fin.read()
    return fout


class Spec(object):
    @property
    def spec_name(self):
        return self.__class__.__name__.replace("Spec", "")

    def create_spec(self, **kwargs):
        spec = """{name} {{
        {options}
        }}
        """
        return spec.format(name=self.spec_name,
                           options=self.options)

    def set_options(self, **kwargs):
        options = ""
        for key, value in kwargs.iteritems():
            options += "{}={}\n".format(key, value)
            self.__dict__.update({key : value})
        self.options = options


class SeriesSpec(Spec):
    """
    Parameters
    ----------
    data
    appendbcst : bool
    appendfcst : bool
    comptype
    compwt
    decimals
    modelspan
    name
    period
    precision
    to_print
    to_save
    span
    start
    title
    type

    Notes
    -----
    Rarely used arguments

    divpower
    missingcode
    missingval
    saveprecision
    trimzero

    """
    def __init__(self, data, name='Unnamed Series', appendbcst=False,
                 appendfcst=False,
                 comptype=None, compwt=1, decimals=0, modelspan=(),
                 period=12, precision=0, to_print=[], to_save=[], span=(),
                 start=(1, 1), title='', series_type=None, divpower=None,
                 missingcode=-99999, missingval=1000000000):

        appendbcst, appendfcst = map(_bool_to_yes_no, [appendbcst,
                                                       appendfcst,
                                                       ])

        series_name = "\"{}\"".format(name[:64])  # trim to 64 characters
        title = "\"{}\"".format(title[:79])  # trim to 79 characters
        self.set_options(data=data, appendbcst=appendbcst,
                         appendfcst=appendfcst, period=period, start=start,
                         title=title, name=series_name,
                         )


def pandas_to_series_spec(x):
    #from statsmodels.tools.data import _check_period_index
    #_check_period_index(x)
    if hasattr(x, 'columns'):  # convert to series
        if len(x.columns) > 1:
            raise ValueError("Does not handle DataFrame with more than one "
                             "column")
        x = x[x.columns[0]]

    data = "({})".format("\n".join(map(str, x.values.tolist())))

    # get periodicity
    # get start / first data
    # give it a title
    try:
        period = _freq_to_period[x.index.freqstr]
    except (AttributeError, ValueError):
        from pandas.tseries.api import infer_freq
        period = _freq_to_period[infer_freq(x.index)]
    start_date = x.index[0]
    if period == 12:
        year, stperiod = start_date.year, start_date.month
    elif period == 4:
        year, stperiod = start_date.year, start_date.quarter
    else:  # pragma: no cover
        raise ValueError("Only monthly and quarterly periods are supported."
                         " Please report or send a pull request if you want "
                         "this extended.")

    if hasattr(x, 'name'):
        name = x.name or "Unnamed Series"
    else:
        name = 'Unnamed Series'
    series_spec = SeriesSpec(data=data, name=name, period=period,
                             title=name, start="{}.{}".format(year, stperiod))
    return series_spec


def x13arima_analysis(y, x12path=None, X=None, log=None, outlier=True,
                      maxorder=(2, 1), maxdiff=(2, 1), diff=None,
                      trading=False, retspec=False, speconly=False,
                      start=None, freq=None, print_stdout=False,
                      prefer_x13=True):
    """
    Perform x13-arima analysis for monthly or quarterly data.

    Parameters
    ----------
    y : array-like, pandas.Series
        The series to model. It is best to use a pandas object with a
        DatetimeIndex or PeriodIndex. However, you can pass an array-like
        object. If your object does not have a dates index then ``start`` and
        ``freq`` are not optional.
    x12path : str or None
        The path to x12 or x13 binary. If None, the program will attempt
        to find x13as or x12a on the PATH or by looking at X13PATH or X12PATH
        depending on the value of prefer_x13.
    X : array-like
        Exogenous variables.
    log : bool or None
        If None, it is automatically determined whether to log the series or not.
        If False, logs are not taken. If True, logs are taken.
    outlier : bool
        Whether or not outliers are tested for and corrected, if detected.
    maxorder : tuple
        The maximum order of the regular and seasonal ARMA polynomials to
        examine during the model identification. The order for the regular
        polynomial must be greater than zero and no larger than 4. The
        order for the seaonal polynomial may be 1 or 2.
    maxdiff : tuple
        The maximum orders for regular and seasonal differencing in the
        automatic differencing procedure. Acceptable inputs for regular
        differencing are 1 and 2. The maximum order for seasonal differencing
        is 1. If ``diff`` is specified then ``maxdiff`` should be None.
        Otherwise, ``diff`` will be ignored. See also ``diff``.
    diff : tuple
        Fixes the orders of differencing for the regular and seasonal
        differencing. Regular differencing may be 0, 1, or 2. Seasonal
        differencing may be 0 or 1. ``maxdiff`` must be None, otherwise
        ``diff`` is ignored.
    trading : bool
        Whether or not trading day effects are tested for.
    retspec : bool
        Whether to return the created specification file. Can be useful for
        debugging.
    speconly : bool
        Whether to create the specification file and then return it without
        performing the analysis. Can be useful for debugging.
    start : str, datetime
        Must be given if ``y`` does not have date information in its index.
        Anything accepted by pandas.DatetimeIndex for the start value.
    freq : str
        Must be givein if ``y`` does not have date information in its index.
        Anything accapted by pandas.DatetimeIndex for the freq value.
    print_stdout : bool
        The stdout from X12/X13 is suppressed. To print it out, set this
        to True. Default is False.
    prefer_x13 : bool
        If True, will look for x13as first and will fallback to the X13PATH
        environmental variable. If False, will look for x12a first and will
        fallback to the X12PATH environmental variable. If x12path points
        to the path for the X12/X13 binary, it does nothing.


    Returns
    -------
    results : str
        The full output from the X12/X13 run.
    seasadj : pandas.Series
        The final seasonally adjusted ``y``
    trend : pandas.Series
        The trend-cycle component of ``y``
    irregular : pandas.Series
        The final irregular component of ``y``
    stdout : str
        The captured stdout produced by x12/x13.
    spec : str, optional
        Returned if ``retspec`` is True. The only thing returned if ``speconly``
        is True.

    Notes
    -----
    This works by creating a specification file, writing it to a temporary
    directory, invoking X12/X13 in a subprocess, and reading the output back
    in.
    """
    x12path = _check_x12(x12path)

    if not isinstance(y, (pd.DataFrame, pd.Series)):
        if start is None or freq is None:
            raise ValueError("start and freq cannot be none if y is not "
                             "a pandas object")
        y = pd.Series(y, index=pd.DatetimeIndex(start=start, periods=len(y),
                                                freq=freq))
    spec_obj = pandas_to_series_spec(y)
    spec = spec_obj.create_spec()
    spec += "transform{{function={}}}\n".format(_log_to_x12[log])
    if outlier:
        spec += "outlier{}\n"
    options = _make_automdl_options(maxorder, maxdiff, diff)
    spec += "automdl{{{}}}\n".format(options)
    spec += _make_regression_options(trading, X)
    spec += "x11{ save=(d11 d12 d13) }"
    if speconly:
        return spec
    # write it to a tempfile
    #TODO: make this more robust - give the user some control?
    ftempin = tempfile.NamedTemporaryFile(delete=False, suffix='.spc')
    ftempout = tempfile.NamedTemporaryFile(delete=False)
    try:
        ftempin.write(spec)
        ftempin.close()
        ftempout.close()
        # call x12 arima
        p = run_spec(x12path, ftempin.name[:-4], ftempout.name)
        p.wait()
        stdout = p.stdout.read()
        if print_stdout:
            print p.stdout.read()
        # check for errors
        errors = _open_and_read(ftempout.name + '.err')
        _check_errors(errors)

        # read in results
        results = _open_and_read(ftempout.name + '.out')
        seasadj = _open_and_read(ftempout.name + '.d11')
        trend = _open_and_read(ftempout.name + '.d12')
        irregular = _open_and_read(ftempout.name + '.d13')
    except X12Error, err:
        raise err
    finally:
        try:  # sometimes this gives a permission denied error?
              # not sure why. no process should have these open
            os.remove(ftempin.name)
            os.remove(ftempout.name)
        except:
            if os.path.exists(ftempin.name):
                warn("Failed to delete resource {}".format(ftempin.name),
                     IOWarning)
            if os.path.exists(ftempout.name):
                warn("Failed to delete resource {}".format(ftempout.name),
                     IOWarning)

    seasadj = _convert_out_to_series(seasadj, y.index, 'seasadj')
    trend = _convert_out_to_series(trend, y.index, 'trend')
    irregular = _convert_out_to_series(irregular, y.index, 'irregular')

    #NOTE: there isn't likely anything in stdout that's not in results
    #      so may be safe to just suppress and remove it
    if not retspec:
        return results, seasadj, trend, irregular, stdout
    else:
        return results, seasadj, trend, irregular, stdout, spec


def select_arima_order(y, x12path=None, X=None, log=None, outlier=True,
                       trading=False, maxorder=(2, 1), maxdiff=(2, 1),
                       diff=None, print_stdout=False,
                       start=None, freq=None, prefer_x13=True):
    """
    Perform automatic seaonal ARIMA order identification using x12/x13 ARIMA.

    Parameters
    ----------
    y : array-like, pandas.Series
        The series to model. It is best to use a pandas object with a
        DatetimeIndex or PeriodIndex. However, you can pass an array-like
        object. If your object does not have a dates index then ``start`` and
        ``freq`` are not optional.
    x12path : str or None
        The path to x12 or x13 binary. If None, the program will attempt
        to find x13as or x12a on the PATH or by looking at X13PATH or X12PATH
        depending on the value of prefer_x13.
    X : array-like
        Exogenous variables.
    log : bool or None
        If None, it is automatically determined whether to log the series or not.
        If False, logs are not taken. If True, logs are taken.
    outlier : bool
        Whether or not outliers are tested for and corrected, if detected.
    trading : bool
        Whether or not trading day effects are tested for.
    maxorder : tuple
        The maximum order of the regular and seasonal ARMA polynomials to
        examine during the model identification. The order for the regular
        polynomial must be greater than zero and no larger than 4. The
        order for the seaonal polynomial may be 1 or 2.
    maxdiff : tuple
        The maximum orders for regular and seasonal differencing in the
        automatic differencing procedure. Acceptable inputs for regular
        differencing are 1 and 2. The maximum order for seasonal differencing
        is 1. If ``diff`` is specified then ``maxdiff`` should be None.
        Otherwise, ``diff`` will be ignored. See also ``diff``.
    diff : tuple
        Fixes the orders of differencing for the regular and seasonal
        differencing. Regular differencing may be 0, 1, or 2. Seasonal
        differencing may be 0 or 1. ``maxdiff`` must be None, otherwise
        ``diff`` is ignored.
    start : str, datetime
        Must be given if ``y`` does not have date information in its index.
        Anything accepted by pandas.DatetimeIndex for the start value.
    freq : str
        Must be givein if ``y`` does not have date information in its index.
        Anything accapted by pandas.DatetimeIndex for the freq value.
    prefer_x13 : bool
        If True, will look for x13as first and will fallback to the X13PATH
        environmental variable. If False, will look for x12a first and will
        fallback to the X12PATH environmental variable. If x12path points
        to the path for the X12/X13 binary, it does nothing.

    Returns
    -------
    results : Bunch
        A bunch object that has the following attributes:

        - order : tuple
          The regular order
        - sorder : tuple
          The seasonal order
        - include_mean : bool
          Whether to include a mean or not
        - results : str
          The full results from the X12/X13 analysis
        - stdout : str
          The captured stdout from the X12/X13 analysis


    Notes
    -----
    This works by creating a specification file, writing it to a temporary
    directory, invoking X12/X13 in a subprocess, and reading the output back
    in.
    """
    (results,
     seasadj,
     trend,
     irregular,
     stdout) = x13arima_analysis(y, x12path=x12path, X=X, log=log,
                                 outlier=outlier, trading=trading,
                                 maxorder=maxorder, maxdiff=maxdiff, diff=diff,
                                 start=start, freq=freq, prefer_x13=prefer_x13)
    model = re.search("(?<=Final automatic model choice : ).*", results)
    order = model.group()
    if re.search("Mean is not significant", results):
        include_mean = False
    elif re.search("Constant", results):
        include_mean = True
    else:
        include_mean = False
    order, sorder = _clean_order(order)
    res = Bunch(order=order, sorder=sorder, include_mean=include_mean,
                results=results, stdout=stdout)
    return res


if __name__ == "__main__":
    import pandas as pd
    import numpy as np
    from statsmodels.tsa.arima_process import ArmaProcess
    np.random.seed(123)
    ar = [1, .35, .8]
    ma = [1, .8]
    arma = ArmaProcess(ar, ma, nobs=100)
    assert arma.isstationary()
    assert arma.isinvertible()
    y = arma.generate_sample()
    dates = pd.date_range("1/1/1990", periods=len(y), freq='M')
    ts = pd.TimeSeries(y, index=dates)

    xpath = "/home/skipper/src/x12arima/x12a"

    try:
        results = x13arima_analysis(xpath, ts)
    except:
        print "Caught exception"

    results = x13arima_analysis(xpath, ts, log=False)

    #seas_y = pd.read_csv("usmelec.csv")
    #seas_y = pd.TimeSeries(seas_y["usmelec"].values,
    #                       index=pd.DatetimeIndex(seas_y["date"], freq="MS"))
    #results = x13arima_analysis(xpath, seas_y)
