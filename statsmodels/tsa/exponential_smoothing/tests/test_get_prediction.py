import numpy as np
import pandas as pd
import pytest

from statsmodels.tsa.api import ETSModel


def test_prediction_start_out_of_sample():
    values = np.arange(100)* (1+np.random.randn(100)/10)
    dates = [np.datetime64('today') + np.timedelta64(-101+i) for i in range(100)]
    train_x = pd.Series(values, index=dates).asfreq(freq='D')
    model = ETSModel(train_x, trend='add')
    fit = model.fit()

    with pytest.raises(ValueError, match="Prediction start cannot lie outside of the sample."):
        prediction = fit.get_prediction(start=np.datetime64('today'),
                     end=np.datetime64('today') + np.timedelta64(5))
