def summary(self, yname=None, xname=None, title=0, alpha=.05,
            returns='print'):
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
    from scikits.statsmodels.iolib.table import SimpleTable
    from scikits.statsmodels.iolib.tableformatting import gen_fmt, fmt_2 #, summaries
    
    
    
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
    stat = stats[modeltype]
    prob_stat = prob_stats[modeltype]
    
    # Simpletable should be able to handle the formating
    params_data = zip(["%#6.4g" % (params[i]) for i in exog_len],
                       ["%#6.4f" % (std_err[i]) for i in exog_len],
                       ["%#6.4f" % (stat[i]) for i in exog_len],
                       ["%#6.4f" % (prob_stat[i]) for i in exog_len],
                       ["""(%#6.3f, %#6.3f)""" % tuple(conf_int[i]) for i in \
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
            
        


#if __name__ == "__main__":
    #import scikits.statsmodels as sm
    #data = sm.datasets.longley.load()
    #data.exog = add_constant(data.exog)
    #ols_results = OLS(data.endog, data.exog).results
    #summary(
  
