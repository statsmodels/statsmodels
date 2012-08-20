# To be finished

# Munnell (1990) proposed a model of productivity of public capital at the state
# level. See Greene Example 10.1 (7th edition)

# Warning : 'R' and 'AK' states does not exist in the dataset.
# So 'R' was replaced by 'AR' and 'AK' was deleted.

import numpy as np
import statsmodels.api as sm
from statsmodels.sysreg.sysmodel import *

munnell_data = sm.datasets.munnell.load()

# Aggregate states into regions
regions = {'GF' : ['AL','FL','LA','MS'],
          'MW' : ['IL','IN','KY','MI','MN','OH','WI'],
          'MA' : ['DE','MD','NJ','NY','PA','VA'],
          'MT' : ['CO','ID','MT','ND','SD','WY'],
          'NE' : ['CT','ME','MA','NH','RI','VT'],
          'SO' : ['GA','NC','SC','TN','WV','AR'],
          'SW' : ['AZ','NV','NM','TX','UT'],
          'CN' : ['IA','KS','MO','NE','OK'],
          'WC' : ['CA','OR','WA']}

munnell_sys = []

for m in regions:
    eq_m = {}
    eq_m['endog'] = np.zeros(17)
    pc, hwy, water, util, emp, unemp = (np.zeros(17) for i in range(6))

    for state in regions[m]:
        state_index = munnell_data.data['state'] == state
        eq_m['endog'] += munnell_data.endog[state_index]
        pc += munnell_data.data['pc'][state_index]
        hwy += munnell_data.data['hwy'][state_index]
        water +=munnell_data.data['water'][state_index]
        util += munnell_data.data['util'][state_index]
        emp += munnell_data.data['emp'][state_index]
    for state in regions[m]:
        state_index = munnell_data.data['state'] == state
        weights = munnell_data.data['emp'][state_index] / emp
        unemp += weights*munnell_data.data['unemp'][state_index]
    
    eq_m['endog'] = np.log(eq_m['endog'])
    eq_m['exog'] = np.column_stack((np.ones(17),np.log(pc), np.log(hwy), 
        np.log(water), np.log(util), np.log(emp), unemp))
    #eq_m['exog'] = sm.add_constant(eq_m['exog'], prepend=True) # Doesn't work (without np.ones)
    munnell_sys.append(eq_m)

munnell_mod = SysSUR(munnell_sys)
munnell_res = munnell_mod.fit()
print munnell_res.summary()

