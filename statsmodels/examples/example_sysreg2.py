# To be finished

# Munnell (1990) proposed a model of productivity of public capital at the state
# level. See Greene Example 10.1 (7th edition)

# Warning : 'R' and 'AK' states does not exist in the dataset.
# So 'R' was replaced by 'AR' and 'AK' was deleted.

import numpy as np
import statsmodels.api as sm
from statsmodels.sysreg.sysreg import *

munnell_data = sm.datasets.munnell.load()

groups = {'GF' : ['AL','FL','LA','MS'],
          'MW' : ['IL','IN','KY','MI','MN','OH','WI'],
          'MA' : ['DE','MD','NJ','NY','PA','VA'],
          'MT' : ['CO','ID','MT','ND','SD','WY'],
          'NE' : ['CT','ME','MA','NH','RI','VT'],
          'SO' : ['GA','NC','SC','TN','WV','AR'],
          'SW' : ['AZ','NV','NM','TX','UT'],
          'CN' : ['IA','KS','MO','NE','OK'],
          'WC' : ['CA','OR','WA']}

munnell_sys = []

for m in groups:
    endog = np.zeros(17) # 17 obs for each state
    for state in groups[m]:
        index = munnell_data.exog['state'] == state
        
        # endog
        endog += munnell_data.endog[index]
        # exog
        regs = ['pc','hwy','water','util','emp']

        

    # add eq to system
    munnell_sys.append(endog)
    exog = sm.add_constant(exog)
    munnell_sys.append(exog)

