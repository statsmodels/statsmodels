

from scikits.statsmodels.iolib.table import SimpleTable
from scikits.statsmodels.iolib.tableformatting import gen_fmt, fmt_2


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
    Defualt :
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
    >>> import scikits.statsmodels as sm
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
                   'WLS' : 'Weigthed least squares',
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
    gen_stubs_left, gen_data_left = map(None, *gen_left) #transpose row col
    
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
    exog_len = xrange(len(xname))
    tstat = tstats[modeltype]
    prob_stat = prob_stats[modeltype]
    
    # Simpletable should be able to handle the formating
    params_data = zip(["%#6.4g" % (params[i]) for i in exog_len],
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
        try:
            yname = self.model.endog_names
        except AttributeError:
            yname = 'y'
    if xname is None:
        try:
            xname = self.model.exog_names
        except AttributeError:
            xname = ['var_%d' % i for i in range(len(self.params))]

    return yname, xname



def summary_top(results, title=None, gleft=None, gright=None, yname=None, xname=None):
    '''generate top table(s)


    TODO: this still uses predefined model_methods
    ? allow gleft, gright to be 1 element tuples instead of filling with None?

    '''

    import time as time

    #TODO Make sure all results.model.__class__.__name__ are listed    
    model_types = {'OLS' : 'Ordinary least squares',
                   'GLS' : 'Generalized least squares',
                   'GLSAR' : 'Generalized least squares with AR(p)',
                   'WLS' : 'Weigthed least squares',
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
    if title is None:
        title = model_types[results.model.__class__.__name__]

    yname, xname = _getnames(results, yname=yname, xname=xname)
    
    time_now = time.localtime()
    time_of_day = [time.strftime("%H:%M:%S", time_now)]
    date = time.strftime("%a, %d %b %Y", time_now)
    
    modeltype = results.model.__class__.__name__
    #dist_family = results.model.family.__class__.__name__
    nobs = results.nobs
    df_model = results.df_model
    df_resid = results.df_resid

    
    
    #General part of the summary table, Applicable to all? models
    #------------------------------------------------------------
    #TODO: define this generically, overwrite in model classes
    #replace definition of stubs data by single list
    #e.g.
    gen_left_ =   [('Model type:', [modeltype]),
                  ('Date:', [date]),
                  ('Dependent Variable:', yname), #What happens with multiple names?
                  ('df model', [df_model])
                  ]

    gen_right_ = [('Method:', [model_methods.get(modeltype)]), #[modeltype]),
                  ('Time:', time_of_day),
                  ('Number of Obs:', [nobs]),
                  ('df resid', [df_resid])
                  ]

    gen_title = title
    gen_header = None

    if gleft is None :
        gen_left = gen_left_
    else:
        gl = dict(gen_left_ + gen_right_)
        gen_left = []
        for item, value in gleft:
            if value is None:
                value = gl.get(item)  #adds None if not a key
            gen_left.append((item, value))
    

    if gright is None :
        gen_right = gen_right_
    else:
        gr = dict(gen_left_ + gen_right_)
        gen_right = []
        for item, value in gright:
            if value is None:
                value = gr.get(item)  #adds None if not a key
            gen_right.append((item, value))

    if gen_right:
        if len(gen_right) < len(gen_left):
            #fill up with blank lines to same length
            gen_right += [(' ', ' ')] * (len(gen_left) - len(gen_right))
        elif len(gen_right) > len(gen_left):
            #fill up with blank lines to same length, just to keep it symmetric
            gen_left += [(' ', ' ')] * (len(gen_right) - len(gen_left))
        gen_stubs_right, gen_data_right = map(None, *gen_right) #transpose row col            
        gen_table_right = SimpleTable(gen_data_right,
                                      gen_header,
                                      gen_stubs_right,
                                      title = gen_title,
                                      txt_fmt = gen_fmt
                                      )
    else:
        gen_table_right = []  #because .extend_right seems to work with []


    #moved below so that we can pad if needed to match length of gen_right
    gen_stubs_left, gen_data_left = map(None, *gen_left) #transpose row col
    gen_table_left = SimpleTable(gen_data_left,
                                 gen_header,
                                 gen_stubs_left,
                                 title = gen_title,
                                 txt_fmt = gen_fmt
                                 )
        

    gen_table_left.extend_right(gen_table_right)
    general_table = gen_table_left

    return general_table #, gen_table_left, gen_table_right


def summary_params(results, yname=None, xname=None, alpha=.05, use_t=True):
    
    #Parameters part of the summary table
    #------------------------------------
    #Note: this is not necessary since we standardized names, only t versus normal

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
                        alp + ' Conf. Int.']
    else:
        param_header = ['coef', 'std err', 'z', 'P>|z|',
                        alp + ' Conf. Int.']

    
    _, xname = _getnames(results, yname=yname, xname=xname)
    
    params_stubs = xname

    exog_idx = xrange(len(xname))
    
    # Simpletable should be able to handle the formating
    # alternative would be to use format in fmt_2
    params_data = zip(["%#6.4g" % (params[i]) for i in exog_idx],
                       ["%#6.4f" % (std_err[i]) for i in exog_idx],
                       ["%#6.3f" % (tvalues[i]) for i in exog_idx],
                       ["%#5.3f" % (pvalues[i]) for i in exog_idx],
                       ["(%#6.3g, %#6.3g)" % tuple(conf_int[i]) for i in \
                                                             exog_idx]
                      )
    parameter_table = SimpleTable(params_data,
                                  param_header,
                                  params_stubs,
                                  title = None,
                                  txt_fmt = fmt_2, #gen_fmt,
                                  )

    return parameter_table


def summary_return(tables, return_fmt='text'):
    ########  Return Summary Tables ########
        # join table parts then print
    if return_fmt == 'text':
        return '\n'.join(map(str, tables))
    elif return_fmt == 'tables':
        return tables
    elif return_fmt == 'csv':
        return '\n'.join(map(lambda x: x.as_csv(), tables))
    elif return_fmt == 'latex':
        #TODO: insert \hline after updating SimpleTable
        import copy
        table = copy.deepcopy(tables[0])
        for part in tables[1:]:
            table.extend(part)
        return table.as_latex_tabular()                
    elif return_fmt == 'html':
        import copy
        table = copy.deepcopy(tables[0])
        for part in tables[1:]:
            table.extend(part)
        return table.as_html
    else:
        raise ValueError('available output formats are text, csv, latex, html')


class Summary(object):
    def __init__(self):
        self.tables = []
        
    def __str__(self):
        return self.as_text()
    
    def __repr__(self):
        #return '<' + str(type(self)) + '>\n"""\n' + self.__str__() + '\n"""'
        return str(type(self)) + '\n"""\n' + self.__str__() + '\n"""'
    
    def add_table_2cols(self, res,  title=None, gleft=None, gright=None, 
                            yname=None, xname=None):
        table = summary_top(res, title=title, gleft=gleft, gright=gright, 
                            yname=yname, xname=xname)
        self.tables.append(table)
        
    def add_table_params(self, res, yname=None, xname=None, alpha=.05, use_t=True):
        table = summary_params(res, yname=yname, xname=xname, alpha=alpha, 
                               use_t=use_t)
        self.tables.append(table)
        
    def as_text(self):
        return summary_return(self.tables, return_fmt='text')
    
    def as_latex(self):
        return summary_return(self.tables, return_fmt='latex')
    
    def as_csv(self):
        return summary_return(self.tables, return_fmt='csv')
        

if __name__ == "__main__":
    import scikits.statsmodels.api as sm
    data = sm.datasets.longley.load()
    data.exog = sm.add_constant(data.exog)
    res = sm.OLS(data.endog, data.exog).fit()
    #summary(
  
