from statsmodels.tools.tools import Bunch

class ExpSmoothingResults:
    def ses(self):
        import co2_ses_results as res
        return Bunch(forecasts=res.forecasts, fitted=res.fitted,
                     resid=res.resid)
