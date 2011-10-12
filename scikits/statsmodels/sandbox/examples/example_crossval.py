if __name__ == '__main__':
    #A: josef-pktd

    import scikits.statsmodels.api as sm
    from scikits.statsmodels.api import OLS
    from scikits.statsmodels.datasets.longley import load
    from scikits.statsmodels.iolib.table import (SimpleTable, default_txt_fmt,
                            default_latex_fmt, default_html_fmt)
    import numpy as np

    data = load()
    data.exog = sm.tools.add_constant(data.exog)

    for inidx, outidx in LeaveOneOut(len(data.endog)):
        res = sm.OLS(data.endog[inidx], data.exog[inidx,:]).fit()
        print data.endog[outidx], res.model.predict(data.exog[outidx,:]),
        print data.endog[outidx] - res.model.predict(data.exog[outidx,:])

    resparams = []
    for inidx, outidx in LeavePOut(len(data.endog), 2):
        res = sm.OLS(data.endog[inidx], data.exog[inidx,:]).fit()
        #print data.endog[outidx], res.model.predict(data.exog[outidx,:]),
        #print ((data.endog[outidx] - res.model.predict(data.exog[outidx,:]))**2).sum()
        resparams.append(res.params)

    resparams = np.array(resparams)
    doplots = 1
    if doplots:
        import matplotlib.pyplot as plt
        from matplotlib.font_manager import FontProperties

        plt.figure()
        figtitle = 'Leave2out parameter estimates'

        t = plt.gcf().text(0.5,
        0.95, figtitle,
        horizontalalignment='center',
        fontproperties=FontProperties(size=16))

        for i in range(resparams.shape[1]):
            plt.subplot(4, 2, i+1)
            plt.hist(resparams[:,i], bins = 10)
            #plt.title("Leave2out parameter estimates")




    for inidx, outidx in KStepAhead(20,2):
        #note the following were broken because KStepAhead returns now a slice by default
        print inidx
        print np.ones(20)[inidx].sum(), np.arange(20)[inidx][-4:]
        print outidx
        print np.nonzero(np.ones(20)[outidx])[0][()]