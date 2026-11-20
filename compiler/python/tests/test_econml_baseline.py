import numpy as np
import pytest

pytest.importorskip("econml")

from causal_compiler import econml_baseline as eb


@pytest.fixture(scope="module")
def fitted():
    try:
        from causal_compiler.datasets import load_ihdp
        data = load_ihdp(1)
    except OSError:
        pytest.skip("IHDP download unavailable")
    return data, eb.fit_linear_dml(data)


def test_linear_dml_effect_is_linear_in_x(fitted):
    data, est = fitted
    x = data[eb.COVARIATES].to_numpy()
    coeffs = eb.coefficient_vector(est)
    np.testing.assert_allclose(est.effect(x), coeffs[0] + x @ coeffs[1:],
                               rtol=0, atol=1e-10)


def test_emitted_ir_shapes():
    text = eb.emit_mlir(10, 25)
    assert '"ci.treatment_effect"' in text
    assert "tensor<10x25xf64>" in text and "tensor<26xf64>" in text
    assert "main_effect_index = 0" in text
