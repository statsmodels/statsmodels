import re

import numpy as np
from numpy.testing import assert_allclose
import pytest

from statsmodels.robust import norms
from statsmodels.tools.numdiff import (  # _approx_fprime_cs_scalar,  # not yet
    _approx_fprime_scalar,
)

from .results import results_norms as res_r

cases = [
    (norms.Hampel, (1.5, 3.5, 8.), res_r.res_hampel),
    (norms.TukeyBiweight, (4,), res_r.res_biweight),
    (norms.HuberT, (1.345,), res_r.res_huber),
    ]

norms_other = [
    (norms.LeastSquares, ()),
    (norms.TrimmedMean, (1.9,)),  # avoid arg at integer used in example
    (norms.HuberT, ()),
    (norms.AndrewWave, ()),
    (norms.RamsayE, ()),
    (norms.Hampel, ()),
    (norms.TukeyBiweight, ()),
    (norms.TukeyQuartic, ()),
    (norms.StudentT, ()),
    # norms.MQuantileNorm,  # requires keywords in init
    ]

dtypes = ["int", np.float64, np.complex128]


@pytest.mark.parametrize("dtype", dtypes)
@pytest.mark.parametrize("case", cases)
def test_norm(case, dtype):
    ncls, args, res = case
    if ncls == norms.HuberT and dtype == np.complex128:
        # skip for now
        return

    norm = ncls(*args)
    x = np.array([-9, -6, -2, -1, 0, 1, 2, 6, 9], dtype=dtype)

    weights = norm.weights(x)
    rho = norm.rho(x)
    psi = norm.psi(x)
    psi_deriv = norm.psi_deriv(x)
    assert_allclose(weights, res.weights, rtol=1e-12, atol=1e-20)
    assert_allclose(rho, res.rho, rtol=1e-12, atol=1e-20)
    assert_allclose(psi, res.psi, rtol=1e-12, atol=1e-20)
    assert_allclose(psi_deriv, res.psi_deriv, rtol=1e-12, atol=1e-20)

    dtype2 = np.promote_types(dtype, "float")
    assert weights.dtype == dtype2
    assert rho.dtype == dtype2
    assert psi.dtype == dtype2
    assert psi_deriv.dtype == dtype2

    psid = _approx_fprime_scalar(x, norm.rho)
    assert_allclose(psid, res.psi, rtol=1e-6, atol=1e-8)
    psidd = _approx_fprime_scalar(x, norm.psi)
    assert_allclose(psidd, res.psi_deriv, rtol=1e-6, atol=1e-8)

    # complex step derivatives are not yet supported if method uses np.abs
    # psid = _approx_fprime_cs_scalar(x, norm.rho)
    # assert_allclose(psid, res.psi, rtol=1e-12, atol=1e-20)
    # psidd = _approx_fprime_cs_scalar(x, norm.psi)
    # assert_allclose(psidd, res.psi_deriv, rtol=1e-12, atol=1e-20)

    # check scalar value
    methods = ["weights", "rho", "psi", "psi_deriv"]
    for meth in methods:
        resm = [getattr(norm, meth)(xi) for xi in x]
        assert_allclose(resm, getattr(res, meth))


@pytest.mark.parametrize("case", norms_other)
def test_norms_consistent(case):
    # test that norm methods are consistent with each other
    ncls, args = case
    norm = ncls(*args)
    x = np.array([-9, -6, -2, -1, 0, 1, 2 - 1e-4, 6, 9], dtype=float)
    # 2 - 1e-4 because Hample psi has discontinuity at 2, numdiff problem

    weights = norm.weights(x)
    rho = norm.rho(x)  # not used
    psi = norm.psi(x)
    psi_deriv = norm.psi_deriv(x)

    # check location and u-shape of rho
    assert_allclose(rho[4], 0, atol=1e-12)
    assert np.all(np.diff(rho[4:]) >= 0)
    assert np.all(np.diff(rho[:4]) <= 0)

    # check weights at and around zero
    assert_allclose(weights[4], 1, atol=1e-12)
    assert np.all(norm.weights([-1e-6, 1e-6]) >= 1 - 1e-5)

    # avoid zero division nan:
    assert_allclose(weights, (psi + 1e-50) / (x + 1e-50), rtol=1e-6, atol=1e-8)
    psid = _approx_fprime_scalar(x, norm.rho)
    assert_allclose(psi, psid, rtol=1e-6, atol=1e-6)
    psidd = _approx_fprime_scalar(x, norm.psi)
    assert_allclose(psi_deriv, psidd, rtol=1e-6, atol=1e-8)

    # attributes
    if norm.redescending == "hard":
        assert_allclose(norm.max_rho(), norm.rho(100), rtol=1e-12)
    else:
        assert np.isposinf(norm.max_rho())

    if norm.redescending == "soft":
        # we don't have info where argmax psi is, use simple values for x
        assert norm.psi(100) < norm.psi(2)


# Validation of the tuning constants
# ----------------------------------
# The norms with one tuning constant that accepts an array or a scalar.
# The tuple is (class, name of the attribute with the constant).
norms_one_constant = [
    (norms.HuberT, "t"),
    (norms.RamsayE, "a"),
    (norms.AndrewWave, "a"),
    (norms.TrimmedMean, "c"),
    (norms.TukeyBiweight, "c"),
    (norms.TukeyQuartic, "c"),
    (norms.StudentT, "c"),
]

# the subset where the array valued constant is documented
norms_array_constant = [
    (norms.HuberT, "t"),
    (norms.AndrewWave, "a"),
    (norms.TukeyBiweight, "c"),
]

# ways to set the constant: the constructor, `_set_tuning_param` that returns
# a copy, and `_set_tuning_param` that modifies the instance in place
set_ways = ["init", "copy", "inplace"]


def set_constant(cls, name, value, way):
    """Create a norm with the constant `value` in one of the ways"""
    if way == "init":
        return cls(**{name: value})
    norm = cls()
    return norm._set_tuning_param(value, inplace=(way == "inplace"))


@pytest.mark.parametrize("way", set_ways)
@pytest.mark.parametrize("value", [0, 0.0, -1.0, -np.inf, np.nan, np.float32(-1)])
@pytest.mark.parametrize("cls, name", norms_one_constant)
def test_invalid_constant_raises(cls, name, value, way):
    # zero, negative or nan constants previously constructed degenerate norms
    # whose psi is identically zero or nan
    with pytest.raises(ValueError, match=f"{name} must be positive, got "):
        set_constant(cls, name, value, way)


@pytest.mark.parametrize("cls, name", norms_one_constant)
def test_invalid_constant_inplace_keeps_state(cls, name):
    norm = cls()
    old = getattr(norm, name)
    with pytest.raises(ValueError, match=f"{name} must be positive"):
        norm._set_tuning_param(-1.0, inplace=True)
    assert getattr(norm, name) == old


@pytest.mark.parametrize("way", set_ways)
@pytest.mark.parametrize(
    "value, exc, msg",
    [
        (None, TypeError, "must be float_like"),
        ("2", TypeError, "must be float_like"),
        (True, TypeError, "must be float_like"),
        (1 + 1j, TypeError, "must be float_like"),
        (1 + 0j, TypeError, "must be float_like"),
        (np.complex128(2), TypeError, "must be float_like"),
        ([], ValueError, "must not be empty"),
        (np.empty((0, 2)), ValueError, "must not be empty"),
        ([True, True], TypeError, "must be array_like of real numbers"),
        ([1 + 1j, 2.0], TypeError, "must be array_like of real numbers"),
        (["a", "b"], TypeError, "must be array_like of real numbers"),
        ([None, 1.0], TypeError, "must be array_like of real numbers"),
    ],
)
@pytest.mark.parametrize("cls, name", norms_one_constant)
def test_wrong_type_constant_raises(cls, name, value, exc, msg, way):
    with pytest.raises(exc, match=f"{name} {msg}"):
        set_constant(cls, name, value, way)


@pytest.mark.parametrize("way", set_ways)
def test_invalid_array_names_elements(way):
    with pytest.raises(ValueError) as excinfo:
        set_constant(norms.HuberT, "t", [1.5, -2.0, np.nan, 3.0], way)
    msg = str(excinfo.value)
    assert "t must be positive" in msg
    assert "t[1]=-2.0" in msg
    assert "t[2]=nan" in msg
    # the valid elements are not reported
    assert "t[0]" not in msg
    assert "t[3]" not in msg

    with pytest.raises(ValueError, match=re.escape("t[1, 0]=-1")):
        set_constant(norms.HuberT, "t", np.array([[1, 2], [-1, 3]]), way)

    # only the first few offending elements are listed
    with pytest.raises(ValueError, match=re.escape("t[4]=0.0, ...")):
        set_constant(norms.HuberT, "t", np.zeros(8), way)


@pytest.mark.parametrize("way", set_ways)
@pytest.mark.parametrize(
    "value",
    [1e-3, 2, 2.0, np.float32(1.5), np.int64(3), np.float64(2.5), np.inf],
)
@pytest.mark.parametrize("cls, name", norms_one_constant)
def test_valid_constant_stored_as_given(cls, name, value, way):
    # the original object is stored, the validation does not convert it
    norm = set_constant(cls, name, value, way)
    stored = getattr(norm, name)
    assert stored is value
    assert type(stored) is type(value)


@pytest.mark.parametrize("way", set_ways)
@pytest.mark.parametrize("cls, name", norms_array_constant)
def test_array_constant_validated(cls, name, way):
    # an array valued constant is checked element-wise and stored as given
    x = np.array([-2.0, -0.5, 0.5, 2.0])
    value = np.array([1.5, 2.0, 1.5, 2.0])
    norm = set_constant(cls, name, value, way)
    assert getattr(norm, name) is value
    assert np.isfinite(norm.psi(x)).all()
    # one invalid element is rejected, the others do not mask it
    for bad in ([1.5, -2.0], [np.nan, 2.0], [1.5, np.nan], [1.5, 0.0]):
        with pytest.raises(ValueError, match=f"{name} must be positive"):
            set_constant(cls, name, np.array(bad), way)


@pytest.mark.parametrize("cls, name", norms_array_constant)
def test_array_constant_matches_elementwise(cls, name):
    # a vector valued tuning constant gives the element-wise results of the
    # same calls with scalars, for every method of the norm
    values = np.array([1.5, 2.5, 1.5, 2.5])
    norm = cls(**{name: values})
    x = np.array([-2.0, -0.5, 0.5, 2.0])
    for method in ["rho", "psi", "psi_deriv", "weights"]:
        combined = getattr(norm, method)(x)
        scalar = np.array(
            [
                getattr(cls(**{name: value}), method)(z)
                for value, z in zip(values, x, strict=True)
            ]
        )
        assert_allclose(combined, scalar, rtol=1e-13)


@pytest.mark.parametrize("cls, name", norms_one_constant)
def test_valid_constants_defaults(cls, name):
    # defaults and custom values remain constructible and finite
    x = np.array([-2.0, -0.5, 0.5, 2.0])
    for norm in [cls(), cls(**{name: 1.5}), cls(**{name: 5.0})]:
        assert np.isfinite(norm.psi(x)).all()
        assert np.isfinite(norm.weights(x)).all()


@pytest.mark.parametrize("way", set_ways)
@pytest.mark.parametrize(
    "cls, name",
    [
        (norms.TrimmedMean, "c"),
        (norms.TukeyBiweight, "c"),
        (norms.TukeyQuartic, "c"),
        (norms.StudentT, "c"),
    ],
)
def test_inf_constant_limit(cls, name, way):
    # c = inf is the limit where nothing is down-weighted, psi(z) = z
    x = np.array([-9.0, -2.0, 0.5, 6.0])
    norm = set_constant(cls, name, np.inf, way)
    assert_allclose(norm.psi(x), x, rtol=1e-13)
    assert_allclose(norm.weights(x), np.ones(4), rtol=1e-13)


@pytest.mark.parametrize("k", [2, 4, 6, np.int64(4), 4.0, np.float32(2)])
def test_tukeyquartic_valid_k(k):
    norm = norms.TukeyQuartic(c=4.0, k=k)
    assert norm.k is k
    # psi is an odd function for an even power
    x = np.array([0.5, 1.5, 3.0])
    assert_allclose(norm.psi(-x), -norm.psi(x), rtol=1e-13)
    assert norm._set_tuning_param(3.0).k is k
    assert norm._set_tuning_param(3.0, inplace=True).k is k


def test_tukeyquartic_k2_is_biweight():
    x = np.array([-9.0, -2.0, 0.5, 6.0])
    quartic = norms.TukeyQuartic(c=4.685, k=2)
    biweight = norms.TukeyBiweight(c=4.685)
    assert_allclose(quartic.psi(x), biweight.psi(x), rtol=1e-13)
    assert_allclose(quartic.rho(x), biweight.rho(x), rtol=1e-13)


@pytest.mark.parametrize("k", [0, -2, 1, 3, 5, -4.0])
def test_tukeyquartic_invalid_k_raises(k):
    with pytest.raises(ValueError, match="k must be a positive even integer"):
        norms.TukeyQuartic(k=k)


@pytest.mark.parametrize("k", [None, "4", True, 2.5, np.nan, np.inf, 4 + 0j])
def test_tukeyquartic_wrong_type_k_raises(k):
    with pytest.raises(TypeError, match="k must be integer_like"):
        norms.TukeyQuartic(k=k)


@pytest.mark.parametrize("df", [0.5, 1, 4, 30.0, np.int64(3), np.float32(2), np.inf])
def test_studentt_valid_df(df):
    norm = norms.StudentT(df=df)
    assert norm.df is df
    # df is kept by _set_tuning_param
    assert norm._set_tuning_param(3.0).df is df
    assert norm._set_tuning_param(3.0, inplace=True).df is df


@pytest.mark.parametrize("df", [0, 0.0, -1, -np.inf, np.nan])
def test_studentt_invalid_df_raises(df):
    with pytest.raises(ValueError, match="df must be positive, got "):
        norms.StudentT(df=df)


@pytest.mark.parametrize(
    "df, exc, msg",
    [
        (None, TypeError, "df must be float_like"),
        ("4", TypeError, "df must be float_like"),
        (True, TypeError, "df must be float_like"),
        (4 + 1j, TypeError, "df must be float_like"),
        # df is a scalar
        ([4.0, 5.0], TypeError, "df must be float_like"),
        (np.array([4.0]), TypeError, "df must be float_like"),
    ],
)
def test_studentt_wrong_type_df_raises(df, exc, msg):
    with pytest.raises(exc, match=msg):
        norms.StudentT(df=df)


@pytest.mark.parametrize("q", [1e-3, 0.25, 0.5, 0.999, np.float32(0.25)])
def test_mquantile_valid_q(q):
    base = norms.HuberT()
    norm = norms.MQuantileNorm(q, base)
    assert norm.q is q
    assert norm.base_norm is base


@pytest.mark.parametrize("q", [0, 1, 0.0, 1.0, -0.1, 1.5, 2, np.nan, np.inf, -np.inf])
def test_mquantile_invalid_q_raises(q):
    with pytest.raises(ValueError, match="q must be between 0 and 1, got "):
        norms.MQuantileNorm(q, norms.HuberT())


@pytest.mark.parametrize(
    "q", [None, "0.5", True, 0.5 + 0j, [0.5], np.array([0.5, 0.5])]
)
def test_mquantile_wrong_type_q_raises(q):
    with pytest.raises(TypeError, match="q must be float_like"):
        norms.MQuantileNorm(q, norms.HuberT())


@pytest.mark.parametrize(
    "a, b, c",
    [
        (0.0, 4.0, 8.0),
        (-2.0, 4.0, 8.0),
        (2.0, 1.0, 8.0),
        (2.0, 4.0, 3.0),
        (2.0, 4.0, -8.0),
        (np.nan, 4.0, 8.0),
        (2.0, np.nan, 8.0),
        (2.0, 4.0, np.nan),
        (np.inf, 4.0, 8.0),
        (2.0, np.inf, 8.0),
    ],
)
def test_hampel_invalid_constants_raise(a, b, c):
    with pytest.raises(ValueError, match="must satisfy 0 < a <= b <= c, got a="):
        norms.Hampel(a=a, b=b, c=c)


@pytest.mark.parametrize(
    "a, b, c",
    [
        (1.5, 3.5, 7.0),
        (1, 2, 3),
        (2.0, 2.0, 8.0),
        (2.0, 8.0, 8.0),
        (2.0, 2.0, 2.0),
        (np.float32(1), np.float32(2), np.float32(3)),
    ],
)
def test_hampel_valid_constants(a, b, c):
    # ties and a <= b <= c are valid, the original objects are stored
    norm = norms.Hampel(a=a, b=b, c=c)
    assert norm.a is a
    assert norm.b is b
    assert norm.c is c
    x = np.array([-9.0, -2.0, 0.5, 6.0])
    assert np.isfinite(norm.psi(x)).all()
    assert np.isfinite(norm.weights(x)).all()


@pytest.mark.parametrize(
    "a, b, c", [(1.0, 2.0, np.inf), (1.0, np.inf, np.inf), (np.inf,) * 3]
)
def test_hampel_inf_constants(a, b, c):
    # inf is the limit with no down-weighting, psi(z) = z for z <= a
    norm = norms.Hampel(a=a, b=b, c=c)
    x = np.array([-0.9, 0.5])
    assert_allclose(norm.psi(x), x, rtol=1e-13)
    assert_allclose(norm.weights(x), np.ones(2), rtol=1e-13)


@pytest.mark.parametrize("pos", ["a", "b", "c"])
@pytest.mark.parametrize(
    "value",
    [
        None,
        "2",
        True,
        1 + 1j,
        [],
        # arrays do not work in the methods of the norm, a scalar is required
        np.array([2.0]),
        np.array(2.0),
        [2.0, 2.0, 2.0, 2.0],
        np.ones(4),
    ],
)
def test_hampel_scalar_required(pos, value):
    constants = dict(a=2.0, b=4.0, c=8.0)
    constants[pos] = value
    with pytest.raises(TypeError, match=f"{pos} must be float_like"):
        norms.Hampel(**constants)


@pytest.mark.parametrize("inplace", [False, True])
def test_hampel_set_tuning_param(inplace):
    norm = norms.Hampel()
    res = norm._set_tuning_param(6, inplace=inplace)
    assert (res is norm) == inplace
    assert (res.a, res.b, res.c) == (1.5, 3.0, 6)
    # a copy leaves the original unchanged, in place modifies it
    assert (norm.a, norm.b, norm.c) == ((1.5, 3.0, 6) if inplace else (2, 4, 8))
    assert np.isfinite(res.psi(np.array([-9.0, 1.0, 5.0]))).all()


@pytest.mark.parametrize("inplace", [False, True])
@pytest.mark.parametrize("c", [-1.0, 0, np.nan, -np.inf])
def test_hampel_set_tuning_param_invalid(c, inplace):
    # the message refers to the value that was passed, not to the derived a, b
    norm = norms.Hampel()
    with pytest.raises(ValueError, match="c must be positive, got ") as excinfo:
        norm._set_tuning_param(c, inplace=inplace)
    assert "a=" not in str(excinfo.value)
    assert (norm.a, norm.b, norm.c) == (2, 4, 8)


@pytest.mark.parametrize("inplace", [False, True])
@pytest.mark.parametrize("c", [None, "8", True, 1 + 1j, [], np.ones(3)])
def test_hampel_set_tuning_param_wrong_type(c, inplace):
    norm = norms.Hampel()
    with pytest.raises(TypeError, match="c must be "):
        norm._set_tuning_param(c, inplace=inplace)
    assert (norm.a, norm.b, norm.c) == (2, 4, 8)


def test_hampel_set_tuning_param_inf():
    norm = norms.Hampel()._set_tuning_param(np.inf, inplace=True)
    assert (norm.a, norm.b, norm.c) == (np.inf, np.inf, np.inf)
