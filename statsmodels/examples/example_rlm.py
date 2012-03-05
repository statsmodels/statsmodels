"""
Examples: statsmodels.models.RLM

Notes
-----
The syntax for the arguments will be shortened to accept string arguments
in the future.
"""

import statsmodels.api as sm

### Example for using Huber's T norm with the default
### median absolute deviation scaling

data = sm.datasets.stackloss.load()
data.exog = sm.add_constant(data.exog)
huber_t = sm.RLM(data.endog, data.exog, M=sm.robust.norms.HuberT())
hub_results = huber_t.fit()
print hub_results.params
print hub_results.bse

### or with the 'H2' covariance matrix
hub_results2 = huber_t.fit(cov="H2")
print hub_results2.params
print hub_results2.bse

### Example for using Andrew's Wave norm with
### Huber's Proposal 2 scaling and 'H3' covariance matrix
andrew_mod = sm.RLM(data.endog, data.exog, M=sm.robust.norms.AndrewWave())
andrew_results = andrew_mod.fit(scale_est=sm.robust.scale.HuberScale(), cov="H3")
print andrew_results.params

print hub_results.summary(yname='y',
                          xname=['var_%d' % i for i in range(len(hub_results.params))])
