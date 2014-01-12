import numpy as np


for (n,p) in (20,1),(50,1),(50,2),(100,5),(1000,10):

    exog = np.random.normal(size=(5*n,p))
    coef = np.linspace(-0.5, 0.5, p)
    lpred = np.dot(exog, coef)
    expected_survival_time = np.exp(-lpred)
    survival_time = -expected_survival_time*\
        np.log(np.random.uniform(size=5*n))
    expected_censoring_time = np.mean(expected_survival_time)
    censoring_time = -expected_censoring_time*\
        np.log(np.random.uniform(size=5*n))
    entry_time = -0.5*expected_censoring_time*\
        np.log(np.random.uniform(size=5*n))

    ##DEBUG
#    censoring_time *= 1000

    status = 1*(survival_time <= censoring_time)
    time = np.where(status==1, survival_time, censoring_time)
    time = np.around(time, decimals=1)

    ii = np.flatnonzero(entry_time < time)
    ii = ii[np.random.permutation(len(ii))[0:n]]
    status = status[ii]
    time = time[ii]
    exog = exog[ii,:]
    entry_time = entry_time[ii]


    ##DEBUG
#    strata = np.kron(range(10), np.ones(n/10))
#    for j in set(strata):
#        ii = np.flatnonzero(strata==j)
#        jj = np.argsort(time[ii])
#        ii = ii[jj]
#        status[ii[-1]] = 0


    data = np.concatenate((time[:,None], status[:,None],
                           entry_time[:,None], exog),
                          axis=1)

    fname = "results/survival_%d_%d.csv" % (n, p)
    np.savetxt(fname, data, fmt="%.5f")
    print fname
