from statsmodels.compat.pandas import MONTH_END

from pathlib import Path
import pickle

import numpy as np
from numpy.testing import assert_allclose
import pandas as pd
import pytest

from statsmodels.datasets import co2
from statsmodels.tsa.seasonal import STL, DecomposeResult
from statsmodels.tsa.tests._contract import SERIES_FORMS, series_forms

cur_dir = Path(__file__).resolve().parent
file_path = Path(cur_dir).joinpath("results", "stl_test_results.csv")
results = pd.read_csv(file_path)
results.columns = [c.strip() for c in results.columns]
results.scenario = results.scenario.apply(str.strip)
results = results.set_index(["scenario", "idx"])


@pytest.fixture(scope="module", params=[True, False])
def robust(request):
    return request.param


def default_kwargs_base():
    file_path = Path(cur_dir).joinpath("results", "stl_co2.csv")
    co2 = np.asarray(pd.read_csv(file_path, header=None).iloc[:, 0])
    y = co2
    nobs = y.shape[0]
    nperiod = 12
    work = np.zeros((nobs + 2 * nperiod, 7))
    rw = np.ones(nobs)
    trend = np.zeros(nobs)
    season = np.zeros(nobs)
    return dict(
        y=y,
        n=y.shape[0],
        np=nperiod,
        ns=35,
        nt=19,
        nl=13,
        no=2,
        ni=1,
        nsjump=4,
        ntjump=2,
        nljump=2,
        isdeg=1,
        itdeg=1,
        ildeg=1,
        rw=rw,
        trend=trend,
        season=season,
        work=work,
    )


@pytest.fixture
def default_kwargs():
    return default_kwargs_base()


@pytest.fixture
def default_kwargs_short():
    kwargs = default_kwargs_base()
    y = kwargs["y"][:-1]
    nobs = y.shape[0]
    work = np.zeros((nobs + 2 * kwargs["np"], 7))
    rw = np.ones(nobs)
    trend = np.zeros(nobs)
    season = np.zeros(nobs)
    kwargs.update(dict(y=y, n=nobs, rw=rw, trend=trend, season=season, work=work))
    return kwargs


def _to_class_kwargs(kwargs, robust=False):
    endog = kwargs["y"]
    np = kwargs["np"]
    ns = kwargs["ns"]
    nt = kwargs["nt"]
    nl = kwargs["nl"]
    isdeg = kwargs["isdeg"]
    itdeg = kwargs["itdeg"]
    ildeg = kwargs["ildeg"]
    nsjump = kwargs["nsjump"]
    ntjump = kwargs["ntjump"]
    nljump = kwargs["nljump"]
    outer_iter = kwargs["no"]
    inner_iter = kwargs["ni"]
    class_kwargs = dict(
        endog=endog,
        period=np,
        seasonal=ns,
        trend=nt,
        low_pass=nl,
        seasonal_deg=isdeg,
        trend_deg=itdeg,
        low_pass_deg=ildeg,
        robust=robust,
        seasonal_jump=nsjump,
        trend_jump=ntjump,
        low_pass_jump=nljump,
    )
    return class_kwargs, outer_iter, inner_iter


def test_baseline_class(default_kwargs):
    class_kwargs, outer, inner = _to_class_kwargs(default_kwargs)
    mod = STL(**class_kwargs)
    res = mod.fit(outer_iter=outer, inner_iter=inner)

    expected = results.loc["baseline"].sort_index()
    assert_allclose(res.trend, expected.trend)
    assert_allclose(res.seasonal, expected.season)
    assert_allclose(res.weights, expected.rw)
    resid = class_kwargs["endog"] - expected.trend - expected.season
    assert_allclose(res.resid, resid)


def test_short_class(default_kwargs_short):
    class_kwargs, outer, inner = _to_class_kwargs(default_kwargs_short)
    mod = STL(**class_kwargs)
    res = mod.fit(outer_iter=outer, inner_iter=inner)

    expected = results.loc["short"].sort_index()
    assert_allclose(res.seasonal, expected.season)
    assert_allclose(res.trend, expected.trend)
    assert_allclose(res.weights, expected.rw)


def test_nljump_1_class(default_kwargs):
    default_kwargs["nljump"] = 1
    class_kwargs, outer, inner = _to_class_kwargs(default_kwargs)
    mod = STL(**class_kwargs)
    res = mod.fit(outer_iter=outer, inner_iter=inner)

    expected = results.loc["nljump-1"].sort_index()
    assert_allclose(res.seasonal, expected.season)
    assert_allclose(res.trend, expected.trend)
    assert_allclose(res.weights, expected.rw)


def test_ntjump_1_class(default_kwargs):
    default_kwargs["ntjump"] = 1
    class_kwargs, outer, inner = _to_class_kwargs(default_kwargs)
    mod = STL(**class_kwargs)
    res = mod.fit(outer_iter=outer, inner_iter=inner)

    expected = results.loc["ntjump-1"].sort_index()
    assert_allclose(res.seasonal, expected.season)
    assert_allclose(res.trend, expected.trend)
    assert_allclose(res.weights, expected.rw)


def test_nljump_1_ntjump_1_class(default_kwargs):
    default_kwargs["nljump"] = 1
    default_kwargs["ntjump"] = 1
    class_kwargs, outer, inner = _to_class_kwargs(default_kwargs)
    mod = STL(**class_kwargs)
    res = mod.fit(outer_iter=outer, inner_iter=inner)

    expected = results.loc["nljump-1-ntjump-1"].sort_index()
    assert_allclose(res.seasonal, expected.season)
    assert_allclose(res.trend, expected.trend)
    assert_allclose(res.weights, expected.rw)


def test_parameter_checks_period(default_kwargs):
    class_kwargs, _, _ = _to_class_kwargs(default_kwargs)
    endog = class_kwargs["endog"]
    endog2 = np.hstack((endog[:, None], endog[:, None]))
    period = class_kwargs["period"]
    with pytest.raises(ValueError, match="endog is required to have ndim 1"):
        STL(endog=endog2, period=period)
    match = "period must be a positive integer >= 2"
    with pytest.raises(ValueError, match=match):
        STL(endog=endog, period=1)
    with pytest.raises(ValueError, match=match):
        STL(endog=endog, period=-12)
    with pytest.raises(ValueError, match=match):
        STL(endog=endog, period=4.5)


def test_parameter_checks_seasonal(default_kwargs):
    class_kwargs, _, _ = _to_class_kwargs(default_kwargs)
    endog = class_kwargs["endog"]
    period = class_kwargs["period"]
    match = "seasonal must be an odd positive integer >= 3"
    with pytest.raises(ValueError, match=match):
        STL(endog=endog, period=period, seasonal=2)
    with pytest.raises(ValueError, match=match):
        STL(endog=endog, period=period, seasonal=-7)
    with pytest.raises(ValueError, match=match):
        STL(endog=endog, period=period, seasonal=13.5)


def test_parameter_checks_trend(default_kwargs):
    class_kwargs, _, _ = _to_class_kwargs(default_kwargs)
    endog = class_kwargs["endog"]
    period = class_kwargs["period"]
    match = "trend must be an odd positive integer >= 3 where trend > period"
    with pytest.raises(ValueError, match=match):
        STL(endog=endog, period=period, trend=14)
    with pytest.raises(ValueError, match=match):
        STL(endog=endog, period=period, trend=11)
    with pytest.raises(ValueError, match=match):
        STL(endog=endog, period=period, trend=-19)
    with pytest.raises(ValueError, match=match):
        STL(endog=endog, period=period, trend=19.5)


def test_parameter_checks_low_pass(default_kwargs):
    class_kwargs, _, _ = _to_class_kwargs(default_kwargs)
    endog = class_kwargs["endog"]
    period = class_kwargs["period"]

    match = "low_pass must be an odd positive integer >= 3 where low_pass > period"
    with pytest.raises(ValueError, match=match):
        STL(endog=endog, period=period, low_pass=14)
    with pytest.raises(ValueError, match=match):
        STL(endog=endog, period=period, low_pass=7)
    with pytest.raises(ValueError, match=match):
        STL(endog=endog, period=period, low_pass=-19)
    with pytest.raises(ValueError, match=match):
        STL(endog=endog, period=period, low_pass=19.5)


def test_jump_errors(default_kwargs):
    class_kwargs, _, _ = _to_class_kwargs(default_kwargs)
    endog = class_kwargs["endog"]
    period = class_kwargs["period"]
    with pytest.raises(ValueError, match="low_pass_jump must be a positive"):
        STL(endog=endog, period=period, low_pass_jump=0)
    with pytest.raises(ValueError, match="low_pass_jump must be a positive"):
        STL(endog=endog, period=period, low_pass_jump=1.5)
    with pytest.raises(ValueError, match="seasonal_jump must be a positive"):
        STL(endog=endog, period=period, seasonal_jump=0)
    with pytest.raises(ValueError, match="seasonal_jump must be a positive"):
        STL(endog=endog, period=period, seasonal_jump=1.5)
    with pytest.raises(ValueError, match="trend_jump must be a positive"):
        STL(endog=endog, period=period, trend_jump=0)
    with pytest.raises(ValueError, match="trend_jump must be a positive"):
        STL(endog=endog, period=period, trend_jump=1.5)


def test_defaults_smoke(default_kwargs, robust):
    class_kwargs, _, _ = _to_class_kwargs(default_kwargs, robust)
    endog = class_kwargs["endog"]
    period = class_kwargs["period"]
    mod = STL(endog=endog, period=period)
    mod.fit()


def test_pandas(default_kwargs, robust):
    class_kwargs, _, _ = _to_class_kwargs(default_kwargs, robust)
    endog = pd.Series(class_kwargs["endog"], name="y")
    period = class_kwargs["period"]
    mod = STL(endog=endog, period=period)
    res = mod.fit()
    assert isinstance(res.trend, pd.Series)
    assert isinstance(res.seasonal, pd.Series)
    assert isinstance(res.resid, pd.Series)
    assert isinstance(res.weights, pd.Series)


def test_period_detection(default_kwargs):
    class_kwargs, _, _ = _to_class_kwargs(default_kwargs)
    mod = STL(**class_kwargs)
    res = mod.fit()

    del class_kwargs["period"]
    endog = class_kwargs["endog"]
    index = pd.date_range("1-1-1959", periods=348, freq=MONTH_END)
    class_kwargs["endog"] = pd.Series(endog, index=index)
    mod = STL(**class_kwargs)

    res_implicit_period = mod.fit()
    assert_allclose(res.seasonal, res_implicit_period.seasonal)


def test_no_period(default_kwargs):
    class_kwargs, _, _ = _to_class_kwargs(default_kwargs)
    del class_kwargs["period"]
    class_kwargs["endog"] = pd.Series(class_kwargs["endog"])
    with pytest.raises(ValueError, match="Unable to determine period from"):
        STL(**class_kwargs)


@pytest.mark.thread_unsafe(reason="Uses matplotlib")
@pytest.mark.matplotlib
def test_plot(default_kwargs, close_figures):
    class_kwargs, outer, inner = _to_class_kwargs(default_kwargs)
    res = STL(**class_kwargs).fit(outer_iter=outer, inner_iter=inner)
    res.plot()

    class_kwargs["endog"] = pd.Series(class_kwargs["endog"], name="CO2")
    res = STL(**class_kwargs).fit()
    res.plot()


def test_default_trend(default_kwargs):
    # GH 6686
    class_kwargs, _, _ = _to_class_kwargs(default_kwargs)
    class_kwargs["seasonal"] = 17
    class_kwargs["trend"] = None
    mod = STL(**class_kwargs)
    period = class_kwargs["period"]
    seasonal = class_kwargs["seasonal"]
    expected = int(np.ceil(1.5 * period / (1 - 1.5 / seasonal)))
    expected += 1 if expected % 2 == 0 else 0
    assert mod.config["trend"] == expected

    class_kwargs["seasonal"] = 7
    mod = STL(**class_kwargs)
    period = class_kwargs["period"]
    seasonal = class_kwargs["seasonal"]
    expected = int(np.ceil(1.5 * period / (1 - 1.5 / seasonal)))
    expected += 1 if expected % 2 == 0 else 0
    assert mod.config["trend"] == expected


def test_pickle(default_kwargs):
    class_kwargs, outer, inner = _to_class_kwargs(default_kwargs)
    mod = STL(**class_kwargs)
    res = mod.fit()
    pkl = pickle.dumps(mod)
    reloaded = pickle.loads(pkl)
    res2 = reloaded.fit()
    assert_allclose(res.trend, res2.trend)
    assert_allclose(res.seasonal, res2.seasonal)
    assert mod.config == reloaded.config


def test_squezable_to_1d():
    data = co2.load().data
    data = data.resample(MONTH_END).mean().ffill()
    res = STL(data).fit()
    assert isinstance(res, DecomposeResult)
    assert res.nobs == data.shape[0]


@pytest.mark.parametrize("form", SERIES_FORMS + ("int",))
def test_stl_input_forms(form):
    # endog is documented as array_like: every form of a 1-d series gives the
    # values of the ndarray, and pandas input gives Series with its index
    rng = np.random.default_rng(20261129)
    t = np.arange(72)
    y = 10 + 0.05 * t + np.sin(2 * np.pi * t / 12) + 0.2 * rng.standard_normal(72)
    forms = series_forms(y)
    forms["int"] = np.round(10 * y).astype(np.int64)
    y_form = forms[form]
    y_float = np.asarray(y_form, dtype=float).squeeze()
    res = STL(y_form, period=12).fit()
    expected = STL(y_float, period=12).fit()
    for attr in ("seasonal", "trend", "resid", "weights"):
        actual = getattr(res, attr)
        assert_allclose(np.asarray(actual), getattr(expected, attr), rtol=1e-12)
        if isinstance(y_form, (pd.Series, pd.DataFrame)):
            assert isinstance(actual, pd.Series)
            pd.testing.assert_index_equal(actual.index, y_form.index)
        else:
            assert isinstance(actual, np.ndarray)
    # the components make up the series
    combined = res.trend + res.seasonal + res.resid
    assert_allclose(np.asarray(combined), y_float, rtol=1e-12)
    observed = np.asarray(res.observed, dtype=float).squeeze()
    assert_allclose(observed, y_float, rtol=1e-15)


@pytest.mark.thread_unsafe(reason="Uses matplotlib")
@pytest.mark.matplotlib
@pytest.mark.parametrize("form", ["list", "tuple", "nested-list"])
def test_list_endog(default_kwargs, form, close_figures):
    # a list or tuple endog was stored as the observed data of the results,
    # so their nobs and plot raised an AttributeError. It is stored as an
    # ndarray, as ndarray input is, and the decomposition is the baseline.
    class_kwargs, outer, inner = _to_class_kwargs(default_kwargs)
    y = class_kwargs["endog"]
    if form == "list":
        class_kwargs["endog"] = y.tolist()
    elif form == "tuple":
        class_kwargs["endog"] = tuple(y)
    else:
        class_kwargs["endog"] = [[v] for v in y]
    res = STL(**class_kwargs).fit(outer_iter=outer, inner_iter=inner)
    assert isinstance(res.observed, np.ndarray)
    assert_allclose(np.squeeze(res.observed), y, rtol=1e-12)
    assert res.nobs == y.shape[0]
    expected = results.loc["baseline"].sort_index()
    assert_allclose(res.trend, expected.trend)
    assert_allclose(res.seasonal, expected.season)
    assert_allclose(res.weights, expected.rw)
    fig = res.plot()
    # observed, trend, seasonal and residual
    assert len(fig.axes) == 4


@pytest.mark.parametrize("container", ["series", "dataframe"])
@pytest.mark.parametrize("freq, period", [("M", 12), ("Q", 4), ("W", 52), ("D", 7)])
def test_period_from_period_index(freq, period, container):
    # a PeriodIndex has no inferred_freq, so without period STL raised
    # "Unable to determine period from endog". The period is that of the
    # frequency of the index, as in seasonal_decompose.
    y = np.random.default_rng(20261121).standard_normal(3 * period)
    index = pd.period_range("2000-01-01", periods=y.shape[0], freq=freq)
    if container == "series":
        endog = pd.Series(y, index=index)
    else:
        endog = pd.DataFrame({"y": y}, index=index)
    assert STL(endog).period == period


def test_period_index_baseline(default_kwargs):
    # with a monthly PeriodIndex and no period, the decomposition is the
    # baseline, which has period 12, and keeps the index
    class_kwargs, outer, inner = _to_class_kwargs(default_kwargs)
    del class_kwargs["period"]
    y = class_kwargs["endog"]
    index = pd.period_range("1959-01", periods=y.shape[0], freq="M")
    class_kwargs["endog"] = pd.Series(y, index=index)
    res = STL(**class_kwargs).fit(outer_iter=outer, inner_iter=inner)
    expected = results.loc["baseline"].sort_index()
    assert_allclose(res.trend, expected.trend)
    assert_allclose(res.seasonal, expected.season)
    assert_allclose(res.weights, expected.rw)
    assert res.trend.index.equals(index)


# The references below are from results/stl-fixed.f90, the Fortran STL that
# the results above come from, compiled with gfortran 13.2.


def test_cycle_subseries_with_one_point():
    # With fewer than two periods, some cycle-subseries have a single value
    y = np.arange(5.0) + np.array([0.0, 1.0, 0.0, -1.0, 0.0])
    res = STL(y, period=4).fit()
    seasonal = [-0.474609375, 0.7626953125, 0.0, -0.7626953125, 0.474609375]
    trend = [0.474609375, 1.2373046875, 2.0, 2.7626953125, 3.525390625]
    # atol: the middle seasonal value is zero up to rounding
    assert_allclose(res.seasonal, seasonal, rtol=1e-12, atol=1e-12)
    assert_allclose(res.trend, trend, rtol=1e-12)
    assert_allclose(res.weights, np.ones(5))


def test_robust_weights_exact_fit():
    # One spike late in a series of zeros: after the first pass the fit is
    # exact (zero) at the 168 values that are far enough before the spike, so
    # the median absolute residual is zero and every robustness weight is one.
    # The robust pass is then two more non-robust inner iterations.
    y = np.zeros(240)
    y[200] = 5.0
    res = STL(y, period=4, robust=True).fit(outer_iter=1)
    desired = STL(y, period=4).fit(inner_iter=4, outer_iter=0)
    assert_allclose(res.weights, np.ones(240))
    assert_allclose(res.seasonal, desired.seasonal, rtol=1e-12)
    assert_allclose(res.trend, desired.trend, rtol=1e-12)


OUTLIER_BLOCK = {
    "default": {
        "seasonal": [
            -1265.18853107, 474.504518813, 1087.15601271, -368.909405747,
            -1018.69588225, 363.210322759, 860.016079546, -276.967925347,
            -772.203233438, 251.916126705, 632.876146386, -184.994090781,
            -524.976892129, 144.163896401, 415.716773613, 44.2987649471,
            272.88484211, 1311.87694607, 2629.02314777, 3441.76395254,
            -16252.7952421, 1947.88817803, -455.779085302, -11808.552457,
            34345.4093875, -3481.11011036, -2452.81077638, -42669.4182331,
            45645.1456387, -289.7773448, 161.785390331, -45645.5743856,
            42982.981839, 1917.94604106, 2688.55654342, -38297.005282,
            25218.0351642, 3753.74605783, 3737.30653274, -17803.0495446,
            5311.95806376, 3580.44209922, 3470.05433796, 1693.53689003,
            -12767.7719194, 3129.28297745, 2995.64131491, 18813.4947532,
        ],
        "trend": [
            1266.18823688, -475.498954489, -1085.15163952, 366.89319275,
            1019.69343629, -364.221236826, -858.027892385, 274.968498203,
            1000773.2137, 999747.090082, 999369.128149, 1000183.00823,
            1000525.97631, 999854.829211, 998687.135136, 1001315.45519,
            999746.16827, 995461.433523, 998913.358053, 1001762.55821,
            1002869.98348, 1003019.02609, 999591.193235, 996527.02497,
            997835.397623, 1000297.96982, 999544.548877, 996915.412761,
            833406.641945, 597629.230511, 401934.380364, 166422.036253,
            3087.45289178, 881.248627385, -1326.08745162, -157.033891497,
            1642.65300428, 1049.33292197, -888.870866343, -291.797754385,
            510.133414668, 114.72284339, -151.035087161, 231.901921303,
            -1615.49676325, -4053.14660425, -6768.32924712, -9946.14345081,
        ],
        "weights": [
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0, 0.469165287298, 0.716025978415, 0.825520057534,
            0.99570291285, 0.999683322823, 0.99932184934, 0.979564406143,
            0.992145277733, 0.995888140068, 0.976819592351, 1.0, 0.950422622231,
            0.0, 0.0, 0.947855780443, 1.0, 0.942514054288, 0.853717662313,
            0.99560959229, 0.995229492276, 0.99358932711, 0.983249889964,
            0.992663011117, 0.999979666901, 0.999988402994, 0.999011068068,
            0.999122147007, 0.999375044102, 0.999797473977, 0.998983204596,
            0.999520070666,
        ],
    },
    "jumps": {
        "seasonal": [
            298.515718699, -166.318926543, -141.020460026, 8.60776076997,
            278.210019467, -143.776986673, -123.598062385, -9.65746025234,
            261.795753114, -110.480139827, -88.5572836726, -4.70718542369,
            139.372289737, -60.4968673995, -40.7246852285, 9.50857107003,
            18.7470757439, -16.9433376183, -7.54982146033, 5.96840940171,
            12.24151434, -9.2752808697, -3.60130405571, 1.13441593599,
            5.22997886839, -1.08070307041, 1.90622951798, -1.01821338846,
            1.97604317363, -1.83859463156, 1.63243093119, -1.78117404442,
            1.39116456567, -0.911623468174, 2.05930070665, -1.94157002125,
            1.03170847554, -1.05144464967, 2.00533784561, -1.96447634335,
            0.97525318409, -1.01227305102, 2.00635974617, -1.96701213402,
            0.911023957672, -0.979646867356, 2.03391907475, -1.93056605837,
        ],
        "trend": [
            -277.21246543, -277.21246543, -277.21246543, -277.21246543,
            -277.21246543, 199800.127297, 399877.46706, 700002.939089,
            1000128.41112, 1000111.283, 1000094.15488, 1000081.67529,
            1000069.19569, 1000057.54218, 1000045.88866, 1000036.08071,
            1000026.27277, 1000019.06312, 1000011.85347, 1000010.10589,
            1000008.35831, 1000006.31619, 1000004.27408, 1000002.19398,
            1000000.11389, 999999.591753, 999999.06962, 999998.985208,
            999998.900797, 749995.010112, 499991.119428, 249995.399387,
            -0.320653703466, -0.217641656241, -0.114629609015, -0.0665229577625,
            -0.0184163065096, -0.00649265251758, 0.00543100147444,
            0.00484695443179, 0.00426290738915, 0.00755589938652,
            0.0108488913839, 0.00350579907736, -0.00383729322917,
            -0.0169342182317, -0.0300311432343, -0.045126270648,
        ],
        "weights": [
            0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.953486628619,
            0.82059247907, 0.0, 0.0, 0.945533471482, 0.860061442205, 0.0, 0.0,
            0.865881062355, 0.924851308327, 0.0, 0.0, 0.944897745416,
            0.97449973013, 0.635556930512, 0.746138839845, 0.996232076878,
            0.984740622068, 0.999960754703, 0.999738508839, 0.0, 0.0, 0.0,
            0.999955740754, 0.999801032867, 0.999949647217, 1.0, 0.999997983394,
            0.999913141656, 1.0, 0.99998026799, 0.999994155123, 1.0, 1.0,
            0.999986262902, 0.999896631272, 1.0, 1.0, 0.999995969221,
        ],
    },
}


@pytest.mark.parametrize("case", ["default", "jumps"])
def test_robust_outlier_block(case):
    # 22 consecutive values are shifted by 1e6, more than the trend window, so
    # in some windows all robustness weights are zero and the loess fits fall
    # back to the data. The noise is the series in which these paths were
    # found: draws 240 to 287 of the seeded generator.
    noise = 0.01 * np.random.default_rng(1234).standard_normal(288)[240:]
    y = np.tile([1.0, -1.0, 2.0, -2.0], 12) + noise
    y[8:30] += 1e6
    kwargs = {}
    if case == "jumps":
        kwargs = {"seasonal_jump": 2, "trend_jump": 2, "low_pass_jump": 2}
    res = STL(y, period=4, robust=True, **kwargs).fit()
    for name, desired in OUTLIER_BLOCK[case].items():
        desired = np.array(desired)
        # atol: the 15 robustness iterations amplify rounding differences
        # between compilers (3e-11 of the largest value between gfortran and
        # the compiled extension on x86-64); a wrong fallback moves the values
        # by orders of magnitude more
        atol = 1e-5 * np.max(np.abs(desired))
        assert_allclose(getattr(res, name), desired, rtol=0, atol=atol)


INT_LIKE_PARAMETERS = [
    "period",
    "seasonal",
    "trend",
    "low_pass",
    "seasonal_jump",
    "trend_jump",
    "low_pass_jump",
]


@pytest.mark.parametrize("name", INT_LIKE_PARAMETERS)
@pytest.mark.parametrize(
    "convert", [np.int64, np.int32, np.array, float], ids=["int64", "int32", "0d", "float"]
)
def test_int_like_parameters(default_kwargs, name, convert):
    # A 0-d integer array or an integral float is accepted as an int is. They
    # raised a ValueError. The baseline results are the reference.
    class_kwargs, outer, inner = _to_class_kwargs(default_kwargs)
    class_kwargs[name] = convert(class_kwargs[name])
    mod = STL(**class_kwargs)
    res = mod.fit(outer_iter=outer, inner_iter=inner)
    expected = results.loc["baseline"].sort_index()
    assert_allclose(res.trend, expected.trend)
    assert_allclose(res.seasonal, expected.season)
    assert_allclose(res.weights, expected.rw)


@pytest.mark.parametrize("name", INT_LIKE_PARAMETERS)
@pytest.mark.parametrize("value", [True, np.True_], ids=["True", "np.True_"])
def test_bool_parameters(default_kwargs, name, value):
    # A bool is not an integer: True was a jump of 1
    class_kwargs, _, _ = _to_class_kwargs(default_kwargs)
    class_kwargs[name] = value
    with pytest.raises(ValueError, match=f"{name} must be"):
        STL(**class_kwargs)
