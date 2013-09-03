from vbench.api import Benchmark
from datetime import datetime

common_setup = """from sm_vb_common import *
"""

#-----------------------------------------------------------------------------
# Straight from exapmles/example_distcrete.py
setup = common_setup + """
spector_data = sm.datasets.spector.load()
spector_data.exog = sm.add_constant(spector_data.exog, prepend=False)
logit_mod = sm.Logit(spector_data.endog, spector_data.exog)
"""

logit_fit = Benchmark("logit_mod.fit()", setup,
                      start_date=datetime(2013, 6, 1))

#-----------------------------------------------------------------------------
# Probit
setup = common_setup + """
spector_data = sm.datasets.spector.load()
spector_data.exog = sm.add_constant(spector_data.exog, prepend=False)
probit_mod = sm.Probit(spector_data.endog, spector_data.exog)
"""
stmt1 = "probit_mod.fit()"
probit_fit = Benchmark(stmt1, setup, start_date=datetime(2013, 6, 1))

setup = setup + """
probit_res = probit_mod.fit()
"""
stmt2 = "probit_res.get_margeff()"
probit_margeff = Benchmark(stmt2, setup, start_date=datetime(2013, 6, 1))

#-----------------------------------------------------------------------------
# Multinomial Logit

setup = common_setup + """
anes_data = sm.datasets.anes96.load()
anes_exog = anes_data.exog
anes_exog = sm.add_constant(anes_exog, prepend=False)
mlogit_mod = sm.MNLogit(anes_data.endog, anes_exog)
"""

stmt1 = "mlogit_mod.fit()"
multi_logit_fit = Benchmark(stmt1, setup, start_date=datetime(2013, 6, 1))

# Marginal Effects
setup = setup + """
mlogit_res = mlogit_mod.fit()
"""
stmt2 = "mlogit_res.get_margeff()"
mutli_logit_margeff = Benchmark(stmt2, setup, start_date=datetime(2013, 6, 1))

# Alternative solver
stmt3 = "mlogit_mod.fit(method='bfgs', maxiter=500)"
mutli_logit_alt = Benchmark(stmt3, setup, start_date=datetime(2013, 6, 1))

#-----------------------------------------------------------------------------
# Poisson
setup = common_setup + """
rand_data = sm.datasets.randhie.load()
rand_exog = rand_data.exog.view(float).reshape(len(rand_data.exog), -1)
rand_exog = sm.add_constant(rand_exog, prepend=False)

poisson_mod = sm.Poisson(rand_data.endog, rand_exog)
"""

stmt1 = 'poisson_res = poisson_mod.fit(method="newton")'
poisson_newton_fit = Benchmark(stmt1, setup, start_date=datetime(2013, 6, 1))

setup = setup + """
poisson_res = poisson_mod.fit(method="newton")
"""
stmt2 = "poisson_res.get_margeff()"
poisson_marg_eff = Benchmark(stmt2, setup, start_date=datetime(2013, 6, 1))
