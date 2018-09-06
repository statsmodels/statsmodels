# TODO: This file was moved from emplike.koul_and_mc, where it would
#   have raised an OSError if it were ever imported.  See if it can be made
#   into a useful test or example

from statsmodels.compat.python import range
import statsmodels.api as sm
import numpy as np

##################
#Monte Carlo test#
##################
modrand1 = np.random.RandomState(5676576)
modrand2 = np.random.RandomState(1543543)
modrand3 = np.random.RandomState(5738276)
X = modrand1.uniform(0, 5, (1000, 4))
X = sm.add_constant(X)
beta = np.array([[10], [2], [3], [4], [5]])
y = np.dot(X, beta)
params = []
for i in range(10000):
    yhat = y + modrand2.standard_normal((1000, 1))
    cens_times = 50 + (modrand3.standard_normal((1000, 1)) * 5)
    yhat_observed = np.minimum(yhat, cens_times)
    censors = np.int_(yhat < cens_times)
    model = sm.emplike.emplikeAFT(yhat_observed, X, censors)
    new_params = model.fit().params
    params.append(new_params)

mc_est = np.mean(params, axis=0)  # Gives MC parameter estimate

##################
#Koul replication#
##################

koul_data = np.genfromtxt('/home/justin/rverify.csv', delimiter=';')
# ^ Change path to where file is located.
koul_y = np.log10(koul_data[:, 0])
koul_x = sm.add_constant(koul_data[:, 2])
koul_censors = koul_data[:, 1]
koul_params = sm.emplike.emplikeAFT(koul_y, koul_x, koul_censors).fit().params
