from statsmodels.tools.tools import Bunch

class ExpSmoothingResults:
    def ses(self):
        import co2_ses_results as res
        return Bunch(forecasts=res.forecasts, fitted=res.fitted,
                     resid=res.resid)

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
