from statsmodels.datasets import co2

def test_co2_python3():
    # this failed in pd.to_datetime on Python 3 with pandas <= 0.12.0
    dta = co2.load_pandas()
