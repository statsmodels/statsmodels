import scikits.statsmodels.base.model as base

class TimeSeriesModel(base.LikelihoodModel):

    def __init__(self, endog, exog=None, dates=None):
        #        self._data = _data = smdata.handle_data(endog, exog)
        #self.exog = _data.exog
        #self.endog = _data.endog
        super(TimeSeriesModel, self).__init__(endog, exog)
        self.dates = dates or self._data.row_labels

if __name__ == "__main__":
    import scikits.statsmodels.api as sm
    import datetime
    import pandas

    data = sm.datasets.macrodata.load()

    #make a DataFrame
    #TODO: attach a DataFrame to some of the datasets, for quicker use
    dates = [str(int(x[0])) +':'+ str(int(x[1])) \
             for x in data.data[['year','quarter']]]
    try:
        import scikits.timeseries as ts
        ts_dates = ts.date_array(start_date = ts.Date(year=1959,quarter=1,freq='Q'),
                             length=len(data.data))
    except:
        pass


    df = pandas.DataFrame(data.data[['realgdp','realinv','realcons']], index=dates)
    ex_mod = TimeSeriesModel(df)
    #ts_series = pandas.TimeSeries()


