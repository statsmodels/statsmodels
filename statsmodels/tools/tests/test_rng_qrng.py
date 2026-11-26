import contextlib
import warnings

import numpy as np
from numpy.testing import assert_equal
import pytest
from scipy import stats

from statsmodels.tools.rng_qrng import check_random_state


@contextlib.contextmanager
def warnings_as_errors():
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        yield


def test_none_returns_fresh_generator():
    rng1 = check_random_state(None)
    rng2 = check_random_state(None)
    assert isinstance(rng1, np.random.Generator)
    assert isinstance(rng2, np.random.Generator)
    # Two fresh, OS-entropy-seeded generators should not produce the
    # same stream.
    assert not np.allclose(rng1.standard_normal(10), rng2.standard_normal(10))


def test_int_returns_generator_by_default():
    rng = check_random_state(0)
    assert isinstance(rng, np.random.Generator)


def test_int_reproducible():
    rng1 = check_random_state(123)
    rng2 = check_random_state(123)
    assert_equal(rng1.standard_normal(10), rng2.standard_normal(10))


def test_array_like_int_seed():
    rng1 = check_random_state([1, 2, 3])
    rng2 = check_random_state([1, 2, 3])
    assert isinstance(rng1, np.random.Generator)
    assert_equal(rng1.standard_normal(10), rng2.standard_normal(10))


def test_non_integer_array_raises():
    with pytest.raises(TypeError):
        check_random_state([1.5, 2.5])


def test_randomstate_instance_passthrough():
    rs = np.random.RandomState(0)
    out = check_random_state(rs)
    assert out is rs


def test_generator_instance_passthrough():
    gen = np.random.default_rng(0)
    out = check_random_state(gen)
    assert out is gen


def test_qmc_engine_passthrough():
    if not hasattr(stats, "qmc"):
        pytest.skip("scipy.stats.qmc not available")
    engine = stats.qmc.Sobol(d=2, seed=0)
    out = check_random_state(engine)
    assert out is engine


def test_deprecated_int_warns_and_returns_randomstate():
    with pytest.warns(FutureWarning, match="After statsmodels 0.15"):
        rng = check_random_state(0, deprecated=True)
    assert isinstance(rng, np.random.RandomState)


def test_deprecated_int_warn_false_suppresses_warning():
    with warnings_as_errors():
        rng = check_random_state(0, deprecated=True, warn=False)
    assert isinstance(rng, np.random.RandomState)


def test_deprecated_none_does_not_warn_and_returns_generator():
    # None should never trigger the legacy-seed FutureWarning: there is
    # no "value" being reinterpreted, so the deprecated flag should have
    # no effect and callers should always get a fresh Generator.
    with warnings_as_errors():
        rng = check_random_state(None, deprecated=True)
    assert isinstance(rng, np.random.Generator)


def test_deprecated_instance_passthrough_does_not_warn():
    rs = np.random.RandomState(0)
    gen = np.random.default_rng(0)
    with warnings_as_errors():
        assert check_random_state(rs, deprecated=True) is rs
        assert check_random_state(gen, deprecated=True) is gen


def _draws(rng):
    return rng.random(5) if isinstance(rng, np.random.Generator) else rng.random_sample(5)


@pytest.mark.parametrize(
    "seed", [np.int64(3), np.uint8(3), np.array(3)], ids=["int64", "uint8", "0-d"]
)
@pytest.mark.parametrize("deprecated", [False, True])
def test_integer_like_seed(seed, deprecated):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", FutureWarning)  # deprecated=True warns
        rng = check_random_state(seed, deprecated=deprecated)
        expected = check_random_state(3, deprecated=deprecated)
    assert_equal(_draws(rng), _draws(expected))


@pytest.mark.parametrize(
    "seed, value",
    [(3.0, 3), (2.5, 2), ("3", 3), (True, 1), (np.True_, 1), (np.float32(4.0), 4)],
    ids=["float", "fraction", "str", "bool", "np.bool_", "float32"],
)
def test_non_integer_scalar_seed_deprecated(seed, value):
    # int(seed) converted these silently; they now warn with the same result
    match = (
        rf"Passing seed of type {type(seed).__name__} is deprecated and will "
        rf"raise a TypeError in statsmodels 0\.17\. Pass seed={value} as an int"
    )
    with pytest.warns(FutureWarning, match=match):
        rng = check_random_state(seed)
    assert_equal(_draws(rng), _draws(np.random.default_rng(value)))


@pytest.mark.parametrize("seed", ["abc", 1 + 2j, np.array([1.5])])
def test_seed_not_convertible_raises(seed):
    with pytest.raises(TypeError, match="seed must either be an integer"):
        check_random_state(seed)


def test_one_element_array_seed_is_array_of_ints():
    # a 1-element array is an array of ints, which seeds the same stream
    assert_equal(
        _draws(check_random_state(np.array([3]))), _draws(check_random_state(3))
    )
