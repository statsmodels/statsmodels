# -*- coding: utf-8 -*-
"""
Created on Thu Oct 27 10:01:25 2011

Author: Josef Perktold
License: BSD-3
"""

import numpy as np
from scikits.statsmodels.iolib import SimpleTable
from scikits.statsmodels.iolib.summary import summary_params

def summary_params_2d(res, below=None, endog_names=None, exog_names=None):
    '''create summary table of regression parameters with several equations

    This allows interleafing of parameters with bse or tvalues

    '''
    if endog_names is None:
        endog_names = ['endog_%d' % i for i in
                            np.unique(res.model.endog)[1:]]
    if exog_names is None:
        exog_names = ['var%d' %i for i in range(len(res.params))]

    res_params = [['%10.4f'%item for item in row] for row in res.params]
    if below: #not None or non-empty
        below_list = [[['%10s' % ('('+('%10.3f'%v).strip()+')')
                                for v in col]
                                for col in getattr(res, what)]
                                for what in below
                                ]
        data = zip(res_params, *below_list)
        data = [i for j in data for i in j]
        stubs = zip(endog_names, *[['']*len(endog_names)]*len(below))
        stubs = [i for j in stubs for i in j]
        #return SimpleTable(data, headers=exog_names, stubs=stubs)
    else:
        data = res_params
        stubs = endog_names
#        return SimpleTable(data, headers=exog_names, stubs=stubs,
#                       data_fmts=['%10.4f'])

    return SimpleTable(data, headers=exog_names, stubs=stubs)

    return data, stubs, endog_names, res_params, below_list

    return SimpleTable(data, headers=exog_names, stubs=stubs,
                       data_fmts=['%10.4f'])



def summary_params_2dflat(result, endog_names=None, exog_names=None, alpha=0.95):

    res = result
    if endog_names is None:
        endog_names = ['endog_%d' % i for i in
                            np.unique(res.model.endog)[1:]]

    res = result
    n_equ = res.params.shape[0]
    tables = []
    for row in range(n_equ):
        restup = res, res.params[row], res.bse[row], res.tvalues[row], res.pvalues[row], res.conf_int(alpha)[row]
        tble = summary_params(restup, yname=endog_names[row], 
                              xname=exog_names, alpha=.05, use_t=True)
        tables.append(tble)
    from copy import deepcopy
    table_all = deepcopy(tables[0])
    for t in tables[1:]:
        table_all.extend(t)
    
    #TODO: make function argument?
    for i in range(len(endog_names)):
        tables[i].title = endog_names[i]

    return tables, table_all



#from scikits.statsmodels.iolib.summary import Summary
#smry = Summary()
#smry.add_table_2cols(self, gleft=top_left, gright=top_right,
#                  yname=yname, xname=xname, title=title)
#smry.add_table_params(self, yname=yname, xname=xname, alpha=.05,
#                     use_t=True)