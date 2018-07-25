"""Time-Series Cross Validation."""


def _fit_model(train, model, **spec):
    """Private method to fit individual models with the data and parameters."""
    mod = model(train, **spec)
    res = mod.fit()
    return res.forecast(1)


def evaluate(data, model, roll_window=10, **spec):
    error = []
    for i in range(len(data)):
        if i < roll_window:
            error.append('NaN')
        else:
            train = data[i-roll_window:i]
            try:
                err = data.iloc[i] - _fit_model(train, model, **spec)[0]
            except Exception as e:
                err = 'NaN'
            error.append(err)
    return error
