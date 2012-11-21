import numpy as np
import pandas as pd
import datetime
import copy
import collections
from statsmodels.iolib.table import SimpleTable
import StringIO
import textwrap

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

def _run_dict(res, d):
    '''
    If the Dict contains lambda functions, apply the, to the results instance. 
    Otherwise, return the Dict value. The value must be string, or the lambda 
    function must return a string.  

    Parameters
    ----------
    res : results instance
    d : dict that contains either strings or lambda functions which generate 
        strings when they are applied to the results instance.

    Returns
    -------
    Dict of strings
    '''
    out = collections.OrderedDict() # Preserve order in summary table
    for key in d.keys():
        if type(d[key]) == str:
            out[key] = d[key]
        else:
            try:
                out[key] = str(d[key](res))
            except:
                pass
    return out

def _dict_to_df(d, ncols=2):
    '''Convert a dict to DataFrame

    Parameters
    ----------
    d : dict of strings or lambda functions which generate 
        strings when they are applied to a results instance.
    ncols : Break values in ncols columns 

    Returns
    -------
    DataFrame with ncols * 2 columns (Values/Keys in separate columns)
    '''
    # Dict -> Numpy array
    data = np.array([d.keys(), d.values()]).T
    n = data.shape[0]
    # Find next multiple of ncols and pad array
    if n % ncols != 0:
        multiple = np.arange(n, 1000) % ncols
        multiple = np.arange(n, 1000)[multiple == 0][0]
        pad = np.array((multiple-n) * [['','']])
        data = np.vstack([data, pad])
    n = data.shape[0]
    # Split array in ncols chunks and stack horizontally
    def chunks(l, n):
        return [l[i:i+n] for i in range(0, len(l), n)]
    idx = chunks(range(data.shape[0]), data.shape[0]/ncols)
    out = data[idx[0]]
    for i in range(1,len(idx)):
        out = np.hstack([out, data[idx[i]]])
    # Numpy -> DataFrame
    out = pd.DataFrame(out)
    return out

def _df_to_ascii(df, pad_sep=2, pad_stub=0, header=False, index=False, 
                 float_format='%.4f', align='l', **kwargs):
    '''Convert a DataFrame

    Parameters
    ----------
    df : DataFrame to print as ASCII table
    pad_sep : Number of spaces in between columns
    pad_stub : Number of spaces after the first column 
               (for internal use, leave at 0)
    header : Reproduce the DataFrame header in ASCII Table?
    index : Reproduce the DataFrame row index in ASCII Table?
    float_format : Float formatter
    align : data alignment (l/c/r)

    Returns
    -------
    ASCII table as string 
    '''
    # Format numbers where possible and convert to Numpy array (type str)
    for col in df.columns:
        try:
            df[col] = map(lambda x: float_format % x, df[col])
            df[col] = map(lambda x: float_format % x, df[col])
        except:
            pass
    data = np.array(df)
    # Pandas inserts undesirable white space
    for i in range(data.shape[1]):
        try:
            data[:,i] = map(lambda x: x.lstrip(), data[:,i])
            data[:,i] = map(lambda x: x.rstrip(), data[:,i])
        except:
            pass
    # Headers and index
    if header:
        headers = df.columns.tolist()
    else:
        headers = None
    if index:
        index = df.index.tolist()
        # Pad right-side of first column if necessary
        try:
            index = map(lambda x: x + ' ' * pad_stub, index)
        except:
            pass
    else:
        index=None
        try:
            # Pad right-side of first column if necessary
            data[:,0] = data[:,0] + ' ' * pad_stub
        except:
            pass
    # Numpy -> SimpleTable -> ASCII
    st_fmt = {'fmt':'txt', 'title_align':'c', 'data_aligns':align, 
              'table_dec_above':None, 'table_dec_below':None}
    st_fmt['colsep'] = ' ' * pad_sep
    ascii = SimpleTable(data, headers=headers, stubs=index, txt_fmt=st_fmt).as_text()
    return ascii

def _pad_target(tables, settings):
    '''Compare width of tables in a list and calculate padding values.
    We add space to each col_sep to get us as close as possible to the
    width of the largest table. Then, we add a few spaces to the first
    column to pad the rest.
    '''
    tab = []
    for i in range(len(tables)):
        # Convert DataFrames to string with default padding
        tab.append(_df_to_ascii(tables[i], **settings[i]))
    # Measure tables width
    length = map(lambda x: len(x.splitlines()[0]), tab)
    # Calculate padding
    pad_sep = []
    target = []
    for i in range(len(length)):
        # Space between each column remains equal (i.e. nsep col separators)
        nsep = settings[i]['ncols'] - 1
        # Padding domain
        temp = np.arange(1000) * nsep + length[i] <= max(length)
        # Max pad which keeps us under the width of largest table
        pad_sep.append(max(np.arange(1000)[temp]))
        target.append(pad_sep[i] * nsep + length[i])
    pad_stub = map(lambda x: max(length) - x, target)
    return pad_sep, pad_stub, length

class Summary(object):
    def __init__(self):
        self.tables = []
        self.settings = []
        self.extra_txt = []

    def __str__(self):
        return self.as_text()

    def __repr__(self):
        return str(type(self)) + '\n"""\n' + self.__str__() + '\n"""'

    def _repr_html_(self):
        '''Display as HTML in IPython notebook.'''
        return self.as_html()

    def add_dict(self, d, ncols=2):
        table = _run_dict(self, d)
        table = _dict_to_df(d, ncols=ncols) 
        settings = {'ncols':table.shape[1], 
                    'index':False, 'header':False, 'float_format':None, 
                    'align':'l'}
        self.tables.append(table)
        self.settings.append(settings)

    def add_df(self, df, index=True, header=True, float_format='%.4f', align='r'):
        # Needs a deep copy somewhere, otherwise float_formatting gets hardcoded
        settings = {'ncols':df.shape[1], 
                    'index':index, 'header':header, 'float_format':float_format, 
                    'align':align} 
        if header:
            settings['ncols'] += 1
        self.tables.append(copy.deepcopy(df))
        self.settings.append(settings)
        
    def add_array(self, array):
        table = pd.DataFrame(array)
        settings = {'ncols':table.shape[1], 
                    'index':False, 'header':False, 'float_format':None, 
                    'align':'l'}
        self.tables.append(table)
        self.settings.append(settings)

    def add_text(self, string, max_len=79):
        self.extra_txt.append(string)

    def as_text(self):
        pad_sep, pad_stub, length = _pad_target(self.tables, self.settings)
        tab = []
        for i in range(len(self.tables)):
            tab.append(_df_to_ascii(self.tables[i], pad_sep[i]+2, pad_stub[i], **self.settings[i]))
        rule_equal = '\n' + max(length) * '=' + '\n' 
        rule_dash = '\n' + max(length) * '-' + '\n'
        ntxt = len(self.extra_txt)
        if ntxt > 0:
            txt = copy.deepcopy(self.extra_txt)
            for i in range(ntxt):
                txt[i] = '\n'.join(textwrap.wrap(txt[i], max(length)))
            txt = '\n'.join(txt)
            out = rule_equal + rule_dash.join(tab) + rule_equal + txt
        else: 
            out = rule_equal + rule_dash.join(tab) + rule_equal 
        return out

    def as_html(self):
        tables = copy.deepcopy(self.tables)
        for i in range(len(tables)):
            tables[i] = tables[i].to_html(header=self.settings[i]['header'], 
                                          index=self.settings[i]['index'], 
                                          float_format=self.settings[i]['float_format']) 
        out = '\n'.join(tables)
        return out

    def as_latex(self):
        tables = copy.deepcopy(self.tables)
        for i in range(len(tables)):
            tables[i] = tables[i].to_latex(header=self.settings[i]['index'], 
                                           index=self.settings[i]['index']) 
            tables[i] = tables[i].replace('\\hline\n', '')
        out = '\\begin{table}\n' + '\n'.join(tables) + '\\end{table}\n'
        return out

# Useful stuff
model_types = {'OLS' : 'Ordinary least squares',
               'GLS' : 'Generalized least squares',
               'GLSAR' : 'Generalized least squares with AR(p)',
               'WLS' : 'Weigthed least squares',
               'RLM' : 'Robust linear model',
               'NBin': 'Negative binomial model', 
               'GLM' : 'Generalized linear model'
               }

def summary_model(results):
    '''Create a dict with information about the model
    '''
    def time_now(**kwrds):
        now = datetime.datetime.now()
        return now.strftime('%Y-%m-%d %H:%M')
    info = collections.OrderedDict()
    info['Model:'] = lambda x: x.model.__class__.__name__
    info['Model Family:'] = lambda x: x.family.__class.__name__
    info['Link Function:'] = lambda x: x.family.link.__class__.__name__
    info['Dependent Variable:'] = lambda x: _getnames(x)[0]
    info['Date:'] = time_now()
    info['No. Observations:'] = lambda x: "%#6d" % x.nobs
    info['Df Model:'] = lambda x: "%#6d" % x.df_model
    info['Df Residuals:'] = lambda x: "%#6d" % x.df_resid
    info['Converged:'] = lambda x: x.mle_retvals['converged']
    info['No. Iterations:'] = lambda x: x.mle_retvals['iterations']
    info['Method:'] = lambda x: x.method
    info['Norm:'] = lambda x: x.fit_options['norm']
    info['Scale Est.:'] = lambda x: x.fit_options['scale_est']
    info['Cov. Type:'] = lambda x: x.fit_options['cov']
    info['R-squared:'] = lambda x: "%#8.3f" % x.rsquared
    info['Adj. R-squared:'] = lambda x: "%#8.3f" % x.rsquared_adj
    info['Pseudo R-squared:'] = lambda x: "%#8.3f" % x.prsquared
    info['AIC:'] = lambda x: "%8.4f" % x.aic
    info['BIC:'] = lambda x: "%8.4f" % x.bic
    info['Log-Likelihood:'] = lambda x: "%#8.5g" % x.llf
    info['Deviance:'] = lambda x: "%#8.5g" % x.deviance 
    info['Pearson chi2:'] = lambda x: "%#6.3g" % x.pearson_chi2
    info ['F-statistic:'] = lambda x: "%#8.4g" % self.fvalue
    info ['Prob (F-statistic):'] = lambda x: "%#6.3g" % self.f_pvalue
    info['Scale:'] = lambda x: "%#8.5g" % x.scale
    out = _run_dict(results, info)
    return out 

def summary_params(results, xname=None, alpha=.05):
    '''create a summary table of parameters from results instance

    Parameters
    ----------
    res : results instance
        some required information is directly taken from the result
        instance
    xname : list of strings or None
        optional names for the exogenous variables, default is "var_xx"
    alpha : float
        significance level for the confidence intervals

    Returns
    -------
    params_table : DataFrame instance
    '''
    #Parameters part of the summary table
    data = np.array([results.params, results.bse, results.tvalues, results.pvalues]).T
    data = np.hstack([data, results.conf_int(alpha)])
    data = pd.DataFrame(data)
    data.columns = ['Coef.', 'Std.Err.', 't', 'P>|t|', 
                    '[' + str(alpha/2), str(1-alpha/2) + ']']
    yname, xname = _getnames(results)
    data.index = xname
    return data


