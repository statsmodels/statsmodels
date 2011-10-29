# -*- coding: utf-8 -*-
"""
Created on Thu Oct 27 10:01:25 2011

Author: Josef Perktold
License: BSD-3
"""

import numpy as np
from scikits.statsmodels.iolib import SimpleTable
from scikits.statsmodels.iolib.summary import summary_params, forg
from scikits.statsmodels.iolib.tableformatting import fmt_params

def summary_params_2d(result, extras=None, endog_names=None, exog_names=None,
                      title=None):
    '''create summary table of regression parameters with several equations

    This allows interleaving of parameters with bse and/or tvalues

    Parameter
    ---------
    result : result instance
        the result instance with params and attributes in extras
    extras : list of strings
        additional attributes to add below a parameter row, e.g. bse or tvalues
    endog_names : None or list of strings
        names for rows of the parameter array (multivariate endog)
    exog_names : None or list of strings
        names for columns of the parameter array (exog)
    alpha : float
        level for confidence intervals, default 0.95
    title : None or string

    Returns
    -------
    tables : list of SimpleTable
        this contains a list of all seperate Subtables
    table_all : SimpleTable
        the merged table with results concatenated for each row of the parameter
        array

    '''
    if endog_names is None:
        endog_names = ['endog_%d' % i for i in
                            np.unique(result.model.endog)[1:]]
    if exog_names is None:
        exog_names = ['var%d' %i for i in range(len(result.params))]

    #TODO: check formatting options with different values
    #res_params = [['%10.4f'%item for item in row] for row in result.params]
    res_params = [[forg(item, prec=4) for item in row] for row in result.params]
    if extras: #not None or non-empty
        #maybe this should be a simple triple loop instead of list comprehension?
        #below_list = [[['%10s' % ('('+('%10.3f'%v).strip()+')')
        extras_list = [[['%10s' % ('(' + forg(v, prec=3).strip() + ')')
                                for v in col]
                                for col in getattr(result, what)]
                                for what in extras
                                ]
        data = zip(res_params, *extras_list)
        data = [i for j in data for i in j]  #flatten
        stubs = zip(endog_names, *[['']*len(endog_names)]*len(extras))
        stubs = [i for j in stubs for i in j] #flatten
        #return SimpleTable(data, headers=exog_names, stubs=stubs)
    else:
        data = res_params
        stubs = endog_names
#        return SimpleTable(data, headers=exog_names, stubs=stubs,
#                       data_fmts=['%10.4f'])

    import copy
    txt_fmt = copy.deepcopy(fmt_params)
    txt_fmt.update(dict(data_fmts = ["%s"]*result.params.shape[1]))
    return SimpleTable(data, headers=exog_names,
                             stubs=stubs,
                             title=title,
#                             data_fmts = ["%s"]),
                             txt_fmt = txt_fmt)




def summary_params_2dflat(result, endog_names=None, exog_names=None, alpha=0.95,
                          keep_headers=True):
                          #skip_headers2=True):
    '''summary table for parameters that are 2d, e.g. multi-equation models

    Parameter
    ---------
    result : result instance
        the result instance with params, bse, tvalues and conf_int
    endog_names : None or list of strings
        names for rows of the parameter array (multivariate endog)
    exog_names : None or list of strings
        names for columns of the parameter array (exog)
    alpha : float
        level for confidence intervals, default 0.95
    keep_headers : bool
        If true (default), then sub-tables keep their headers. If false, then
        only the first headers are kept, the other headerse are blanked out

    Returns
    -------
    tables : list of SimpleTable
        this contains a list of all seperate Subtables
    table_all : SimpleTable
        the merged table with results concatenated for each row of the parameter
        array

    '''

    res = result
    if endog_names is None:
        endog_names = ['endog_%d' % i for i in
                            np.unique(res.model.endog)[1:]]

    res = result
    n_equ = res.params.shape[0]
    tables = []
    for row in range(n_equ):
        restup = (res, res.params[row], res.bse[row], res.tvalues[row],
                  res.pvalues[row], res.conf_int(alpha)[row])

        #not used anymore in current version
#        if skip_headers2:
#            skiph = (row != 0)
#        else:
#            skiph = False
        skiph = False
        tble = summary_params(restup, yname=endog_names[row],
                              xname=exog_names, alpha=.05, use_t=True,
                              skip_header=skiph)

        tables.append(tble)

    #add titles, they will be moved to header lines in table_extend
    for i in range(len(endog_names)):
        tables[i].title = endog_names[i]

    table_all = table_extend(tables, keep_headers=keep_headers)

    return tables, table_all


def table_extend(tables, keep_headers=True):
    '''extend a list of SimpleTables, adding titles to header of subtables

    This function returns the merged table as a deepcopy, in contrast to the
    SimpleTable extend method.

    Parameter
    ---------
    tables : list of SimpleTable instances
    keep_headers : bool
        If true, then all headers are kept. If falls, then the headers of
        subtables are blanked out.

    Returns
    -------
    table_all : SimpleTable
        merged tables as a single SimpleTable instance

    '''
    from copy import deepcopy
    for ii, t in enumerate(tables[:]): #[1:]:
        t = deepcopy(t)

        #move title to first cell of header
        #TODO: check if we have multiline headers
        if t[0].datatype == 'header':
            t[0][0].data = t.title
            t[0][0]._datatype = None
            t[0][0].row = t[0][1].row
            if not keep_headers and (ii > 0):
                for c in t[0][1:]:
                    c.data = ''

        #add separating line and extend tables
        if ii == 0:
            table_all = t
        else:
            r1 = table_all[-1]
            r1.add_format('txt', row_dec_below='-')
            table_all.extend(t)

    table_all.title = None
    return table_all

#from scikits.statsmodels.iolib.summary import Summary
#smry = Summary()
#smry.add_table_2cols(self, gleft=top_left, gright=top_right,
#                  yname=yname, xname=xname, title=title)
#smry.add_table_params(self, yname=yname, xname=xname, alpha=.05,
#                     use_t=True)