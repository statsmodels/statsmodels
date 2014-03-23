from statsmodels.tools.tools import Bunch
from pandas import Series, DatetimeIndex


class ExpSmoothingResults:
    def ses(self):
        import co2_ses_results as res
        return Bunch(forecasts=res.forecasts, fitted=res.fitted,
                     level=res.level.squeeze(), resid=res.resid)

    def holt_des(self):
        import co2_holt_des_results as res
        return Bunch(forecasts=res.forecasts, fitted=res.xhat,
                     level=res.level,
                     resid=res.resid, trend=res.trend)

    def holt_des_mult(self):
        import co2_holt_des_mult_results as res
        return Bunch(forecasts=res.forecasts, fitted=res.xhat,
                     level=res.level,
                     resid=res.resid, trend=res.trend)

    def hw_seas(self):
        import co2_holt_seas_results as res
        return Bunch(forecasts=res.forecasts, fitted=res.xhat,
                     level=res.level,
                     resid=res.resid)

    def hw_seas_mult(self):
        import co2_holt_seas_mult_results as res
        return Bunch(forecasts=res.forecasts, fitted=res.xhat,
                     level=res.level,
                     resid=res.resid)

    def damped_trend(self):
        import co2_damped_results as res
        return Bunch(forecasts=res.forecasts, fitted=res.xhat,
                     level=res.level,
                     resid=res.resid, trend=res.trend)

    def damped_mult_trend(self):
        import co2_damped_mult_results as res
        return Bunch(forecasts=res.forecasts, fitted=res.xhat,
                     level=res.level,
                     resid=res.resid, trend=res.trend)

    def multmult(self):
        import co2_mult_mult_results as res
        return Bunch(forecasts=res.forecasts, fitted=res.xhat,
                     level=res.level,
                     resid=res.resid, trend=res.trend)

    def multmult_pandas(self):
        import co2_mult_mult_results as res
        index = DatetimeIndex(start='3/1/1958', periods=526,
                              freq='MS')
        index_with_initial = DatetimeIndex(start='2/1/1958', periods=527,
                              freq='MS')
        season_index = DatetimeIndex(start='3/1/1957', periods=538,
                                     freq='MS')
        fc_index = DatetimeIndex(start='1/1/2002', periods=48, freq='MS')
        make_series = lambda x, name, index : Series(x, name=name,
                                                     index=index)

        return Bunch(forecasts=make_series(res.forecasts, 'forecast',
                                           fc_index),
                     fitted=make_series(res.xhat, 'fittedvalues', index),
                     level=make_series(res.level, 'level',
                                       index_with_initial),
                     resid=make_series(res.resid, 'resid', index),
                     trend=make_series(res.trend, 'trend',
                                       index_with_initial),
                     seasonal=make_series(res.seasonal.squeeze(), 'seasonal',
                                          season_index))
