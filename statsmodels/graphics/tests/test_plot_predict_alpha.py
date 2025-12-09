import numpy as np
import statsmodels.api as sm
from statsmodels.graphics.tsaplots import plot_predict


def test_plot_predict_passes_alpha_to_conf_int():
    # Create a small, reproducible time series
    np.random.seed(0)
    data = np.random.normal(size=100)

    model = sm.tsa.ARIMA(data, order=(1, 0, 0))
    res = model.fit()

    # Get an actual prediction object
    pred = res.get_prediction()
    original_conf_int = pred.conf_int

    recorded_alpha = []

    # Wrap conf_int so we can see what alpha it receives
    def recording_conf_int(obs=False, alpha=0.05, *args, **kwargs):
        recorded_alpha.append(alpha)
        # Call the original implementation so behaviour is unchanged
        return original_conf_int(obs=obs, alpha=alpha, *args, **kwargs)

    # Use our wrapped conf_int on the prediction object
    pred.conf_int = recording_conf_int

    # Make get_prediction() always return this patched prediction object
    def fake_get_prediction(*args, **kwargs):
        return pred

    res.get_prediction = fake_get_prediction

    # Call plot_predict with a non-default alpha
    plot_predict(res, alpha=0.32)

    # Before the bug fix, recorded_alpha would contain [0.05]
    # After the fix, it must contain [0.32]
    assert recorded_alpha == [0.32]
