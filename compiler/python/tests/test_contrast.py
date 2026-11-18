import numpy as np
import pytest

from causal_compiler import treatment_effect


def test_treatment_effect_matches_hand_calculation():
    interaction_data = np.array([[2.0, 3.0], [-1.0, 4.0]])
    interaction_coefficients = np.array([0.5, -0.25])

    result = treatment_effect(
        interaction_data,
        main_effect=1.5,
        interaction_coefficients=interaction_coefficients,
        treatment_delta=2.0,
    )

    np.testing.assert_allclose(result, [3.5, 0.0])


def test_treatment_effect_rejects_mismatched_feature_count():
    with pytest.raises(ValueError, match="columns must match"):
        treatment_effect(
            np.ones((3, 2)),
            main_effect=1.0,
            interaction_coefficients=np.ones(1),
            treatment_delta=1.0,
        )


@pytest.mark.parametrize(
    "interaction_data,coefficients",
    [
        (np.ones(3), np.ones(3)),
        (np.ones((2, 3)), np.ones((3, 1))),
    ],
)
def test_treatment_effect_requires_expected_array_ranks(
    interaction_data, coefficients
):
    with pytest.raises(ValueError):
        treatment_effect(
            interaction_data,
            main_effect=1.0,
            interaction_coefficients=coefficients,
            treatment_delta=1.0,
        )
