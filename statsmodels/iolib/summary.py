from statsmodels.compat.python import range, lrange, lmap, lzip, zip_longest
import numpy as np
from statsmodels.iolib.table import SimpleTable
from statsmodels.iolib.tableformatting import (gen_fmt, fmt_2,
                                                fmt_params, fmt_base, fmt_2cols)
#from statsmodels.iolib.summary2d import summary_params_2dflat
#from summary2d import summary_params_2dflat

def forg(x, prec=3):
    if prec == 3:
    #for 3 decimals
        if (abs(x) >= 1e4) or (abs(x) < 1e-4):
            return '%9.3g' % x
        else:
            return '%9.3f' % x
    elif prec == 4:
        if (abs(x) >= 1e4) or (abs(x) < 1e-4):
            return '%10.4g' % x
        else:
            return '%10.4f' % x
    else:
        raise NotImplementedError


def summary(self, yname=None, xname=None, title=0, alpha=.05,
            returns='text', model_info=None):
    """
    Parameters
    -----------
    yname : string
            optional, Default is `Y`
    xname : list of strings
            optional, Default is `X.#` for # in p the number of regressors
    Confidance interval : (0,1) not implimented
    title : string
            optional, Defualt is 'Generalized linear model'
    returns : string
              'text', 'table', 'csv', 'latex', 'html'

    Returns
    -------
    Default :
    returns='print'
            Prints the summarirized results

    Option :
    returns='text'
            Prints the summarirized results

    Option :
    returns='table'
             SimpleTable instance : summarizing the fit of a linear model.

    Option :
    returns='csv'
            returns a string of csv of the results, to import into a spreadsheet

    Option :
    returns='latex'
    Not implimented yet

    Option :
    returns='HTML'
    Not implimented yet


    Examples (needs updating)
    --------
    >>> import statsmodels as sm
    >>> data = sm.datasets.longley.load()
    >>> data.exog = sm.add_constant(data.exog)
    >>> ols_results = sm.OLS(data.endog, data.exog).results
    >>> print ols_results.summary()
    ...

    Notes
    -----
    conf_int calculated from normal dist.
    """
    import time as time



    #TODO Make sure all self.model.__class__.__name__ are listed
    model_types = {'OLS' : 'Ordinary least squares',
                   'GLS' : 'Generalized least squares',
                   'GLSAR' : 'Generalized least squares with AR(p)',
                   'WLS' : 'Weighted least squares',
                   'RLM' : 'Robust linear model',
                   'GLM' : 'Generalized linear model'
                   }
    model_methods = {'OLS' : 'Least Squares',
                   'GLS' : 'Least Squares',
                   'GLSAR' : 'Least Squares',
                   'WLS' : 'Least Squares',
                   'RLM' : '?',
                   'GLM' : '?'
                   }
    if title==0:
        title = model_types[self.model.__class__.__name__]
    if yname is None:
        try:
            yname = self.model.endog_names
        except AttributeError:
            yname = 'y'
    if xname is None:
        try:
            xname = self.model.exog_names
        except AttributeError:
            xname = ['var_%d' % i for i in range(len(self.params))]
    time_now = time.localtime()
    time_of_day = [time.strftime("%H:%M:%S", time_now)]
    date = time.strftime("%a, %d %b %Y", time_now)
    modeltype = self.model.__class__.__name__
    #dist_family = self.model.family.__class__.__name__
    nobs = self.nobs
    df_model = self.df_model
    df_resid = self.df_resid

    #General part of the summary table, Applicable to all? models
    #------------------------------------------------------------
    #TODO: define this generically, overwrite in model classes
    #replace definition of stubs data by single list
    #e.g.
    gen_left =   [('Model type:', [modeltype]),
                  ('Date:', [date]),
                  ('Dependent Variable:', yname), #What happens with multiple names?
                  ('df model', [df_model])
                  ]
    gen_stubs_left, gen_data_left = zip_longest(*gen_left) #transpose row col

    gen_title = title
    gen_header = None
##    gen_stubs_left = ('Model type:',
##                      'Date:',
##                      'Dependent Variable:',
##                      'df model'
##                  )
##    gen_data_left = [[modeltype],
##                     [date],
##                     yname, #What happens with multiple names?
##                     [df_model]
##                     ]
    gen_table_left = SimpleTable(gen_data_left,
                                 gen_header,
                                 gen_stubs_left,
                                 title = gen_title,
                                 txt_fmt = gen_fmt
                                 )

    gen_stubs_right = ('Method:',
                      'Time:',
                      'Number of Obs:',
                      'df resid'
                      )
    gen_data_right = ([modeltype], #was dist family need to look at more
                      time_of_day,
                      [nobs],
                      [df_resid]
                      )
    gen_table_right = SimpleTable(gen_data_right,
                                 gen_header,
                                 gen_stubs_right,
                                 title = gen_title,
                                 txt_fmt = gen_fmt
                                 )
    gen_table_left.extend_right(gen_table_right)
    general_table = gen_table_left

    #Parameters part of the summary table
    #------------------------------------
    #Note: this is not necessary since we standardized names, only t versus normal
    tstats = {'OLS' : self.t(),
            'GLS' : self.t(),
            'GLSAR' : self.t(),
            'WLS' : self.t(),
            'RLM' : self.t(),
            'GLM' : self.t()
            }
    prob_stats = {'OLS' : self.pvalues,
                 'GLS' : self.pvalues,
                 'GLSAR' : self.pvalues,
                 'WLS' : self.pvalues,
                 'RLM' : self.pvalues,
                 'GLM' : self.pvalues
                }
    #Dictionary to store the header names for the parameter part of the
    #summary table. look up by modeltype
    alp = str((1-alpha)*100)+'%'
    param_header = {
         'OLS'   : ['coef', 'std err', 't', 'P>|t|', alp + ' Conf. Interval'],
         'GLS'   : ['coef', 'std err', 't', 'P>|t|', alp + ' Conf. Interval'],
         'GLSAR' : ['coef', 'std err', 't', 'P>|t|', alp + ' Conf. Interval'],
         'WLS'   : ['coef', 'std err', 't', 'P>|t|', alp + ' Conf. Interval'],
         'GLM'   : ['coef', 'std err', 't', 'P>|t|', alp + ' Conf. Interval'], #glm uses t-distribution
         'RLM'   : ['coef', 'std err', 'z', 'P>|z|', alp + ' Conf. Interval']  #checke z
                   }
    params_stubs = xname
    params = self.params
    conf_int = self.conf_int(alpha)
    std_err = self.bse
    exog_len = lrange(len(xname))
    tstat = tstats[modeltype]
    prob_stat = prob_stats[modeltype]

    # Simpletable should be able to handle the formating
    params_data = lzip(["%#6.4g" % (params[i]) for i in exog_len],
                       ["%#6.4f" % (std_err[i]) for i in exog_len],
                       ["%#6.4f" % (tstat[i]) for i in exog_len],
                       ["%#6.4f" % (prob_stat[i]) for i in exog_len],
                       ["(%#5g, %#5g)" % tuple(conf_int[i]) for i in \
                                                             exog_len]
                      )
    parameter_table = SimpleTable(params_data,
                                  param_header[modeltype],
                                  params_stubs,
                                  title = None,
                                  txt_fmt = fmt_2, #gen_fmt,
                                  )

    #special table
    #-------------
    #TODO: exists in linear_model, what about other models
    #residual diagnostics


    #output options
    #--------------
    #TODO: JP the rest needs to be fixed, similar to summary in linear_model

    def ols_printer():
        """
        print summary table for ols models
        """
        table = str(general_table)+'\n'+str(parameter_table)
        return table

    def ols_to_csv():
        """
        exports ols summary data to csv
        """
        pass
    def glm_printer():
        table = str(general_table)+'\n'+str(parameter_table)
        return table
        pass

    printers  = {'OLS': ols_printer,
                'GLM' : glm_printer
                }

    if returns=='print':
        try:
            return printers[modeltype]()
        except KeyError:
            return printers['OLS']()

def _getnames(self, yname=None, xname=None):
    '''extract names from model or construct names
    '''
    if yname is None:
        if hasattr(self.model, 'endog_names') and (
               not self.model.endog_names is None):
            yname = self.model.endog_names
        else:
            yname = 'y'

    if xname is None:
        if hasattr(self.model, 'exog_names') and (
               not self.model.exog_names is None):
            xname = self.model.exog_names
        else:
            xname = ['var_%d' % i for i in range(len(self.params))]

    return yname, xname



def summary_top(results, title=None, gleft=None, gright=None, yname=None, xname=None):
    '''generate top table(s)


    TODO: this still uses predefined model_methods
    ? allow gleft, gright to be 1 element tuples instead of filling with None?

    '''
    #change of names ?
    gen_left, gen_right = gleft, gright

    #time and names are always included
    import time
    time_now = time.localtime()
    time_of_day = [time.strftime("%H:%M:%S", time_now)]
    date = time.strftime("%a, %d %b %Y", time_now)

    yname, xname = _getnames(results, yname=yname, xname=xname)

    #create dictionary with default
    #use lambdas because some values raise exception if they are not available
    #alternate spellings are commented out to force unique labels
    default_items = dict([
          ('Dependent Variable:', lambda: [yname]),
          ('Dep. Variable:', lambda: [yname]),
          ('Model:', lambda: [results.model.__class__.__name__]),
          #('Model type:', lambda: [results.model.__class__.__name__]),
          ('Date:', lambda: [date]),
          ('Time:', lambda: time_of_day),
          ('Number of Obs:', lambda: [results.nobs]),
          #('No. of Observations:', lambda: ["%#6d" % results.nobs]),
          ('No. Observations:', lambda: ["%#6d" % results.nobs]),
          #('Df model:', lambda: [results.df_model]),
          ('Df Model:', lambda: ["%#6d" % results.df_model]),
          #TODO: check when we have non-integer df
          ('Df Residuals:', lambda: ["%#6d" % results.df_resid]),
          #('Df resid:', lambda: [results.df_resid]),
          #('df resid:', lambda: [results.df_resid]), #check capitalization
          ('Log-Likelihood:', lambda: ["%#8.5g" % results.llf]) #doesn't exist for RLM - exception
          #('Method:', lambda: [???]), #no default for this
          ])

    if title is None:
        title = results.model.__class__.__name__ + 'Regression Results'

    if gen_left is None:
        #default: General part of the summary table, Applicable to all? models
        gen_left = [('Dep. Variable:', None),
                    ('Model type:', None),
                    ('Date:', None),
                    ('No. Observations:', None),
                    ('Df model:', None),
                    ('Df resid:', None)]

        try:
            llf = results.llf
            gen_left.append(('Log-Likelihood', None))
        except: #AttributeError, NotImplementedError
            pass

        gen_right = []


    gen_title = title
    gen_header = None

    #needed_values = [k for k,v in gleft + gright if v is None] #not used anymore
    #replace missing (None) values with default values
    gen_left_ = []
    for item, value in gen_left:
        if value is None:
            value = default_items[item]()  #let KeyErrors raise exception
        gen_left_.append((item, value))
    gen_left = gen_left_

    if gen_right:
        gen_right_ = []
        for item, value in gen_right:
            if value is None:
                value = default_items[item]()  #let KeyErrors raise exception
            gen_right_.append((item, value))
        gen_right = gen_right_

    #check
    missing_values = [k for k,v in gen_left + gen_right if v is None]
    assert missing_values == [], missing_values

    #pad both tables to equal number of rows
    if gen_right:
        if len(gen_right) < len(gen_left):
            #fill up with blank lines to same length
            gen_right += [(' ', ' ')] * (len(gen_left) - len(gen_right))
        elif len(gen_right) > len(gen_left):
            #fill up with blank lines to same length, just to keep it symmetric
            gen_left += [(' ', ' ')] * (len(gen_right) - len(gen_left))

        #padding in SimpleTable doesn't work like I want
        #force extra spacing and exact string length in right table
        gen_right = [('%-21s' % ('  '+k), v) for k,v in gen_right]
        gen_stubs_right, gen_data_right = zip_longest(*gen_right) #transpose row col
        gen_table_right = SimpleTable(gen_data_right,
                                      gen_header,
                                      gen_stubs_right,
                                      title = gen_title,
                                      txt_fmt = fmt_2cols #gen_fmt
                                      )
    else:
        gen_table_right = []  #because .extend_right seems works with []


    #moved below so that we can pad if needed to match length of gen_right
    #transpose rows and columns, `unzip`
    gen_stubs_left, gen_data_left = zip_longest(*gen_left) #transpose row col

    gen_table_left = SimpleTable(gen_data_left,
                                 gen_header,
                                 gen_stubs_left,
                                 title = gen_title,
                                 txt_fmt = fmt_2cols
                                 )

    gen_table_left.extend_right(gen_table_right)
    general_table = gen_table_left

    return general_table #, gen_table_left, gen_table_right



def summary_params(results, yname=None, xname=None, alpha=.05, use_t=True,
                   skip_header=False, title=None):
    '''create a summary table for the parameters

    Parameters
    ----------
    res : results instance
        some required information is directly taken from the result
        instance
    yname : string or None
        optional name for the endogenous variable, default is "y"
    xname : list of strings or None
        optional names for the exogenous variables, default is "var_xx"
    alpha : float
        significance level for the confidence intervals
    use_t : bool
        indicator whether the p-values are based on the Student-t
        distribution (if True) or on the normal distribution (if False)
    skip_headers : bool
        If false (default), then the header row is added. If true, then no
        header row is added.

    Returns
    -------
    params_table : SimpleTable instance
    '''

    #Parameters part of the summary table
    #------------------------------------
    #Note: this is not necessary since we standardized names, only t versus normal

    if isinstance(results, tuple):
        #for multivariate endog
        #TODO: check whether I don't want to refactor this
        #we need to give parameter alpha to conf_int
        results, params, std_err, tvalues, pvalues, conf_int = results
    else:
        params = results.params
        std_err = results.bse
        tvalues = results.tvalues  #is this sometimes called zvalues
        pvalues = results.pvalues
        conf_int = results.conf_int(alpha)


    #Dictionary to store the header names for the parameter part of the
    #summary table. look up by modeltype
    if use_t:
        param_header = ['coef', 'std err', 't', 'P>|t|',
                        '[' + str(alpha/2), str(1-alpha/2) + ']']
    else:
        param_header = ['coef', 'std err', 'z', 'P>|z|',
                        '[' + str(alpha/2), str(1-alpha/2) + ']']

    if skip_header:
        param_header = None


    _, xname = _getnames(results, yname=yname, xname=xname)

    if len(xname) != len(params):
        raise ValueError('xnames and params do not have the same length')

    params_stubs = xname

    exog_idx = lrange(len(xname))

    params_data = lzip([forg(params[i], prec=4) for i in exog_idx],
                       [forg(std_err[i]) for i in exog_idx],
                       [forg(tvalues[i]) for i in exog_idx],
                       ["%#6.3f" % (pvalues[i]) for i in exog_idx],
                       [forg(conf_int[i,0]) for i in exog_idx],
                       [forg(conf_int[i,1]) for i in exog_idx]
                      )
    parameter_table = SimpleTable(params_data,
                                  param_header,
                                  params_stubs,
                                  title = title,
                                  txt_fmt = fmt_params #gen_fmt #fmt_2, #gen_fmt,
                                  )

    return parameter_table


def summary_params_frame(results, yname=None, xname=None, alpha=.05,
                         use_t=True):
    '''create a summary table for the parameters

    Parameters
    ----------
    res : results instance
        some required information is directly taken from the result
        instance
    yname : string or None
        optional name for the endogenous variable, default is "y"
    xname : list of strings or None
        optional names for the exogenous variables, default is "var_xx"
    alpha : float
        significance level for the confidence intervals
    use_t : bool
        indicator whether the p-values are based on the Student-t
        distribution (if True) or on the normal distribution (if False)
    skip_headers : bool
        If false (default), then the header row is added. If true, then no
        header row is added.

    Returns
    -------
    params_table : SimpleTable instance
    '''

    #Parameters part of the summary table
    #------------------------------------
    #Note: this is not necessary since we standardized names, only t versus normal

    if isinstance(results, tuple):
        #for multivariate endog
        #TODO: check whether I don't want to refactor this
        #we need to give parameter alpha to conf_int
        results, params, std_err, tvalues, pvalues, conf_int = results
    else:
        params = results.params
        std_err = results.bse
        tvalues = results.tvalues  #is this sometimes called zvalues
        pvalues = results.pvalues
        conf_int = results.conf_int(alpha)


    #Dictionary to store the header names for the parameter part of the
    #summary table. look up by modeltype
    alp = str((1-alpha)*100)+'%'
    if use_t:
        param_header = ['coef', 'std err', 't', 'P>|t|',
                        'Conf. Int. Low', 'Conf. Int. Upp.']
    else:
        param_header = ['coef', 'std err', 'z', 'P>|z|',
                        'Conf. Int. Low', 'Conf. Int. Upp.']

    _, xname = _getnames(results, yname=yname, xname=xname)


    #------------------

    from pandas import DataFrame
    table = np.column_stack((params, std_err, tvalues, pvalues, conf_int))
    return DataFrame(table, columns=param_header, index=xname)


def summary_params_2d(result, extras=None, endog_names=None, exog_names=None,
                      title=None):
    '''create summary table of regression parameters with several equations

    This allows interleaving of parameters with bse and/or tvalues

    Parameters
    ----------
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
        #TODO: note the [1:] is specific to current MNLogit
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
        data = lzip(res_params, *extras_list)
        data = [i for j in data for i in j]  #flatten
        stubs = lzip(endog_names, *[['']*len(endog_names)]*len(extras))
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


def summary_params_2dflat(result, endog_names=None, exog_names=None, alpha=0.05,
                          use_t=True, keep_headers=True, endog_cols=False):
                          #skip_headers2=True):
    '''summary table for parameters that are 2d, e.g. multi-equation models

    Parameters
    ----------
    result : result instance
        the result instance with params, bse, tvalues and conf_int
    endog_names : None or list of strings
        names for rows of the parameter array (multivariate endog)
    exog_names : None or list of strings
        names for columns of the parameter array (exog)
    alpha : float
        level for confidence intervals, default 0.95
    use_t : bool
        indicator whether the p-values are based on the Student-t
        distribution (if True) or on the normal distribution (if False)
    keep_headers : bool
        If true (default), then sub-tables keep their headers. If false, then
        only the first headers are kept, the other headerse are blanked out
    endog_cols : bool
        If false (default) then params and other result statistics have
        equations by rows. If true, then equations are assumed to be in columns.
        Not implemented yet.

    Returns
    -------
    tables : list of SimpleTable
        this contains a list of all seperate Subtables
    table_all : SimpleTable
        the merged table with results concatenated for each row of the parameter
        array

    '''

    res = result
    params = res.params
    if params.ndim == 2: # we've got multiple equations
        n_equ = params.shape[1]
        if not len(endog_names) == params.shape[1]:
            raise ValueError('endog_names has wrong length')
    else:
        if not len(endog_names) == len(params):
            raise ValueError('endog_names has wrong length')
        n_equ = 1

    #VAR doesn't have conf_int
    #params = res.params.T # this is a convention for multi-eq models

    if not isinstance(endog_names, list):
        #this might be specific to multinomial logit type, move?
        if endog_names is None:
            endog_basename = 'endog'
        else:
            endog_basename = endog_names
        #TODO: note, the [1:] is specific to current MNLogit
        endog_names = res.model.endog_names[1:]

    #check if we have the right length of names

    tables = []
    for eq in range(n_equ):
        restup = (res, res.params[:,eq], res.bse[:,eq], res.tvalues[:,eq],
                  res.pvalues[:,eq], res.conf_int(alpha)[eq])

        #not used anymore in current version
#        if skip_headers2:
#            skiph = (row != 0)
#        else:
#            skiph = False
        skiph = False
        tble = summary_params(restup, yname=endog_names[eq],
                              xname=exog_names, alpha=alpha, use_t=use_t,
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

    Parameters
    ----------
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


def summary_return(tables, return_fmt='text'):
    ########  Return Summary Tables ########
        # join table parts then print
    if return_fmt == 'text':
        strdrop = lambda x: str(x).rsplit('\n',1)[0]
        #convert to string drop last line
        return '\n'.join(lmap(strdrop, tables[:-1]) + [str(tables[-1])])
    elif return_fmt == 'tables':
        return tables
    elif return_fmt == 'csv':
        return '\n'.join(map(lambda x: x.as_csv(), tables))
    elif return_fmt == 'latex':
        #TODO: insert \hline after updating SimpleTable
        import copy
        table = copy.deepcopy(tables[0])
        del table[-1]
        for part in tables[1:]:
            table.extend(part)
        return table.as_latex_tabular()
    elif return_fmt == 'html':
        return "\n".join(table.as_html() for table in tables)
    else:
        raise ValueError('available output formats are text, csv, latex, html')


class Summary(object):
    '''class to hold tables for result summary presentation

    Construction does not take any parameters. Tables and text can be added
    with the `add_` methods.

    Attributes
    ----------
    tables : list of tables
        Contains the list of SimpleTable instances, horizontally concatenated tables are not saved separately.
    extra_txt : string
        extra lines that are added to the text output, used for warnings and explanations.
    '''
    def __init__(self):
        self.tables = []
        self.extra_txt = None

    def __str__(self):
        return self.as_text()

    def __repr__(self):
        #return '<' + str(type(self)) + '>\n"""\n' + self.__str__() + '\n"""'
        return str(type(self)) + '\n"""\n' + self.__str__() + '\n"""'

    def _repr_html_(self):
        '''Display as HTML in IPython notebook.'''
        return self.as_html()

    def add_table_2cols(self, res,  title=None, gleft=None, gright=None,
                            yname=None, xname=None):
        '''add a double table, 2 tables with one column merged horizontally

        Parameters
        ----------
        res : results instance
            some required information is directly taken from the result
            instance
        title : string or None
            if None, then a default title is used.
        gleft : list of tuples
            elements for the left table, tuples are (name, value) pairs
            If gleft is None, then a default table is created
        gright : list of tuples or None
            elements for the right table, tuples are (name, value) pairs
        yname : string or None
            optional name for the endogenous variable, default is "y"
        xname : list of strings or None
            optional names for the exogenous variables, default is "var_xx"

        Returns
        -------
        None : tables are attached

        '''

        table = summary_top(res, title=title, gleft=gleft, gright=gright,
                            yname=yname, xname=xname)
        self.tables.append(table)

    def add_table_params(self, res, yname=None, xname=None, alpha=.05,
                         use_t=True):
        '''create and add a table for the parameter estimates

        Parameters
        ----------
        res : results instance
            some required information is directly taken from the result
            instance
        yname : string or None
            optional name for the endogenous variable, default is "y"
        xname : list of strings or None
            optional names for the exogenous variables, default is "var_xx"
        alpha : float
            significance level for the confidence intervals
        use_t : bool
            indicator whether the p-values are based on the Student-t
            distribution (if True) or on the normal distribution (if False)

        Returns
        -------
        None : table is attached

        '''
        if res.params.ndim == 1:
            table = summary_params(res, yname=yname, xname=xname, alpha=alpha,
                                   use_t=use_t)
        elif res.params.ndim == 2:
#            _, table = summary_params_2dflat(res, yname=yname, xname=xname,
#                                             alpha=alpha, use_t=use_t)
            _, table = summary_params_2dflat(res, endog_names=yname,
                                             exog_names=xname,
                                             alpha=alpha, use_t=use_t)
        else:
            raise ValueError('params has to be 1d or 2d')
        self.tables.append(table)

    def add_extra_txt(self, etext):
        '''add additional text that will be added at the end in text format

        Parameters
        ----------
        etext : list[str]
            string with lines that are added to the text output.

        '''
        self.extra_txt = '\n'.join(etext)

    def as_text(self):
        '''return tables as string

        Returns
        -------
        txt : string
            summary tables and extra text as one string

        '''
        txt = summary_return(self.tables, return_fmt='text')
        if not self.extra_txt is None:
            txt = txt + '\n\n' + self.extra_txt
        return txt

    def as_latex(self):
        '''return tables as string

        Returns
        -------
        latex : string
            summary tables and extra text as string of Latex

        Notes
        -----
        This currently merges tables with different number of columns.
        It is recommended to use `as_latex_tabular` directly on the individual
        tables.

        '''
        latex = summary_return(self.tables, return_fmt='latex')
        if not self.extra_txt is None:
            latex = latex + '\n\n' + self.extra_txt.replace('\n', ' \\newline\n ')
        return latex

    def as_csv(self):
        '''return tables as string

        Returns
        -------
        csv : string
            concatenated summary tables in comma delimited format

        '''
        csv = summary_return(self.tables, return_fmt='csv')
        if not self.extra_txt is None:
            csv = csv + '\n\n' + self.extra_txt
        return csv

    def as_html(self):
        '''return tables as string

        Returns
        -------
        html : string
            concatenated summary tables in HTML format

        '''
        html = summary_return(self.tables, return_fmt='html')
        if not self.extra_txt is None:
            html = html + '<br/><br/>' + self.extra_txt.replace('\n', '<br/>')
        return html


if __name__ == "__main__":
    import statsmodels.api as sm
    data = sm.datasets.longley.load()
    data.exog = sm.add_constant(data.exog)
    res = sm.OLS(data.endog, data.exog).fit()
    #summary(

