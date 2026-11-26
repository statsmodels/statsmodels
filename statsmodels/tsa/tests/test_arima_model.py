"""Tests of the stub left in statsmodels.tsa.arima_model after its removal"""

import numpy as np
import pytest

from statsmodels.tsa import arima_model


@pytest.mark.parametrize(
    "cls",
    [
        arima_model.ARMA,
        arima_model.ARIMA,
        arima_model.ARMAResults,
        arima_model.ARIMAResults,
    ],
)
def test_removed_class_raises(cls):
    y = np.arange(20.0)
    with pytest.raises(NotImplementedError) as exc_info:
        cls(y, order=(1, 0, 0))
    message = " ".join(str(exc_info.value).split())
    # The message names the replacement and says when the module goes
    assert "in favor of statsmodels.tsa.arima.model.ARIMA" in message
    assert "statsmodels.tsa.statespace.sarimax.SARIMAX" in message
    assert (
        "statsmodels.tsa.arima_model module will be removed after "
        "statsmodels 0.16 is released" in message
    )


def test_replacements_named_in_message_import():
    # The migration advice names import paths that must exist
    from statsmodels.tsa.arima.model import ARIMA
    from statsmodels.tsa.statespace.sarimax import SARIMAX

    assert "statsmodels.tsa.arima.model.ARIMA" in arima_model.ARIMA_DEPRECATION_ERROR
    assert ARIMA.__module__ == "statsmodels.tsa.arima.model"
    assert SARIMAX.__module__ == "statsmodels.tsa.statespace.sarimax"
