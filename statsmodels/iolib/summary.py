import numpy as np
import pandas as pd
import datetime
import copy
import collections
from statsmodels.iolib.table import SimpleTable
import StringIO
import textwrap

class Summary(object):
    def __init__(self):
        self.tables = []
        self.settings = []
        self.extra_txt = []
        self.title = None

    def __str__(self):
        return self.as_text()

    def __repr__(self):
        return str(type(self)) + '\n"""\n' + self.__str__() + '\n"""'

    def _repr_html_(self):
        '''Display as HTML in IPython notebook.'''
        return self.as_html()

    def add_df(self, df, index=True, header=True, float_format='%.4f', 
               align='r'):
        '''Add the contents of a DataFrame to summary table

        Parameters
        ----------
        df : DataFrame
        header: bool 
            Reproduce the DataFrame column labels in summary table
        index: bool 
            Reproduce the DataFrame row labels in summary table
        float_format: string
            Formatting to float data columns
        align : string 
            Data alignment (l/c/r)
        '''

        settings = {'ncols':df.shape[1], 
                    'index':index, 'header':header, 
                    'float_format':float_format, 'align':align}
        if index:
            settings['ncols'] += 1
        self.tables.append(df)
        self.settings.append(settings)

    def add_array(self, array, align='l', float_format="%.4f"):
        '''Add the contents of a Numpy array to summary table

        Parameters
        ----------
        array : numpy array (2D)
        float_format: string
            Formatting to array if type is float
        align : string 
            Data alignment (l/c/r)
        '''

        table = pd.DataFrame(array)
        self.add_df(table, index=False, header=False,
                float_format=float_format, align=align)

    def add_dict(self, d, ncols=2, align='l'):
        '''Add the contents of a Dict to summary table

        Parameters
        ----------
        d : dict
            Keys and values are automatically coerced to strings with str().
            Users are encouraged to format them before using add_dict.
        ncols: int
            Number of columns of the output table
        align : string
            Data alignment (l/c/r)
        '''

        key = map(str, d.keys())
        val = map(str, d.values())
        data = np.array([key, val]).T
        # Pad if necessary to fill columns
        if data.shape[0] % ncols != 0:
            pad = ncols - (data.shape[0] % ncols)
            data = np.vstack([data, np.array(pad * [['','']])])
        # Split and join as multi-columns
        data = np.split(data, ncols)
        data = reduce(lambda x,y: np.hstack([x,y]), data)
        self.add_array(data, align=align)

    def add_text(self, string):
        '''Append a note to the bottom of the summary table. In ASCII tables,
        the note will be wrapped to table width. Notes are not indendented. 
        '''
        self.extra_txt.append(string)

    def add_title(self, title=None, results=None):
        '''Insert a title on top of the summary table. If a string is provided
        in the title argument, that string is printed. If no title string is
        provided but a results instance is provided, statsmodels attempts
        to construct a useful title automatically.
        '''
        if type(title) == str:
            self.title = title 
        else:
            try:
                model = results.model.__class__.__name__
                if model in _model_types:
                    model = _model_types[model]
                self.title = 'Results: ' + model
            except:
                self.title = '' 

    def add_base(self, results, alpha=0.05, float_format="%.4f", title=None, 
            xname=None, yname=None):
        '''Try to construct a basic summary instance. 

        Parameters
        ----------
        results : Model results instance
        alpha : float
            significance level for the confidence intervals (optional)
        float_formatting: string
            Float formatting for summary of parameters (optional)
        title : string
            Title of the summary table (optional)
        xname : List of strings of length equal to the number of parameters
            Names of the independent variables (optional)
        yname : string
            Name of the dependent variable (optional)
        '''

        param = summary_params(results, alpha=alpha)
        info = summary_model(results)
        if xname != None:
            param.index = xname
        if yname != None: 
            info['Dependent Variable:'] = yname
        self.add_dict(info)
        self.add_df(param, float_format=float_format)
        self.add_title(title=title, results=results)

    def as_text(self):
        '''Generate ASCII Summary Table
        '''
        pad_sep, pad_index, widest = _pad(self.tables, self.settings)
        tables = []
        for i in range(len(self.tables)):
            tab = _df_to_ascii(self.tables[i], pad_sep[i]+2, pad_index[i], **self.settings[i])
            tables.append(tab)
        rule_equal = '\n' + widest * '=' + '\n' 
        rule_dash = '\n' + widest * '-' + '\n'
        ntxt = len(self.extra_txt)
        if ntxt > 0:
            txt = copy.deepcopy(self.extra_txt)
            for i in range(ntxt):
                txt[i] = '\n'.join(textwrap.wrap(txt[i], widest))
            txt = '\n'.join(txt)
            out = rule_equal + rule_dash.join(tables) + rule_equal + txt
        else: 
            out = rule_equal + rule_dash.join(tables) + rule_equal 
        if type(self.title) == str:
            if len(self.title) < widest:
                title = ' ' * int(widest/2 - len(self.title)/2) + self.title
                out = title + out
        return out

    def as_html(self):
        '''Generate HTML Summary Table
        '''
        tables = copy.deepcopy(self.tables)
        for i in range(len(tables)):
            tables[i] = tables[i].to_html(header=self.settings[i]['header'], 
                                          index=self.settings[i]['index'], 
                                          float_format=self.settings[i]['float_format']) 
        out = '\n'.join(tables)
        return out

    def as_latex(self):
        '''Generate LaTeX Summary Table
        '''
        tables = copy.deepcopy(self.tables)
        for i in range(len(tables)):
            tables[i] = tables[i].to_latex(header=self.settings[i]['index'], 
                                           index=self.settings[i]['index']) 
            tables[i] = tables[i].replace('\\hline\n', '')
        out = '\\begin{table}\n' + '\n'.join(tables) + '\\end{table}\n'
        return out


# ASCII table formatting
def _df_to_ascii(df, pad_sep=2, pad_index=0, header=False, index=False, 
                 float_format='%.4f', align='l', **kwargs):
    '''Convert a DataFrame to ASCII table

    Parameters
    ----------
    df : DataFrame
        Print df as ASCII table
    pad_sep : int
        Number of spaces in between columns
    pad_index : int
        Number of spaces after the first column 
    header : bool
        Reproduce the DataFrame header in ASCII Table?
    index : bool
        Reproduce the DataFrame row index in ASCII Table?
    float_format : string
        Float format
    align : string: 
        data alignment (l/c/r)

    Returns
    -------
    ASCII table as string 
    '''
    # Format numbers where possible and convert to Numpy array (type str)
    for col in df.columns:
        try:
            df[col] = [float_format % x for x in df[col]]
        except:
            pass
        try:
            df[col] = [x.lstrip().rstrip() for x in df[col]]
        except:
            pass
    data = np.array(df)
    # Headers and index
    if header:
        headers = map(str, df.columns)
    else:
        headers = None
    if index:
        # Pad right-side of index if necessary
        try:
            index = [str(x) + ' ' * pad_index for x in df.index]
        except:
            pass
    else:
        index=None
        try:
            # Pad right-side of first column if necessary
            data[:,0] = [str(x) + ' ' * pad_index for x in data[:,0]]
        except:
            pass
    # Numpy -> SimpleTable -> ASCII
    st_fmt = {'fmt':'txt', 'title_align':'c', 'data_aligns':align, 
              'table_dec_above':None, 'table_dec_below':None}
    st_fmt['colsep'] = ' ' * int(pad_sep)
    ascii = SimpleTable(data, headers=headers, stubs=index, txt_fmt=st_fmt).as_text()
    return ascii

def _pad(tables, settings):
    '''Compare width of ascii tables in a list and calculate padding values.
    We add space to each col_sep to get us as close as possible to the
    width of the largest table. Then, we add a few spaces to the first
    column to pad the rest.
    '''
    tab = []
    pad_index = []
    pad_sep = []
    for i in range(len(tables)):
        tab.append(_df_to_ascii(tables[i], **settings[i]))
    length = [len(x.splitlines()[0]) for x in tab]
    len_max = max(length)
    pad_sep = []
    for i in range(len(tab)):
        nsep = settings[i]['ncols'] - 1
        pad = (len_max - length[i]) / nsep 
        pad_sep.append(pad)
        len_new = length[i] + nsep * pad
        pad_index.append(len_max - len_new) 
    return pad_sep, pad_index, max(length)


# Useful stuff
_model_types = {'OLS' : 'Ordinary least squares',
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
    info['Dependent Variable:'] = lambda x: x.model.endog_names
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
    info['LL-Null:'] = lambda x: "%#8.5g" % x.llnull
    info['LLR p-value:'] = lambda x: "%#8.5g" % x.llr_pvalue
    info['Deviance:'] = lambda x: "%#8.5g" % x.deviance 
    info['Pearson chi2:'] = lambda x: "%#6.3g" % x.pearson_chi2
    info ['F-statistic:'] = lambda x: "%#8.4g" % self.fvalue
    info ['Prob (F-statistic):'] = lambda x: "%#6.3g" % self.f_pvalue
    info['Scale:'] = lambda x: "%#8.5g" % x.scale
    out = collections.OrderedDict()
    for key in info.keys():
        try: 
            out[key] = info[key](results)
        except:
            pass 
    return out 

def summary_params(results, alpha=.05, params=None, bse=None, tvalues=None,
        pvalues=None, confint=None):
    '''create a summary table of parameters from results instance

    Parameters
    ----------
    res : results instance
        some required information is directly taken from the result
        instance
    alpha : float
        significance level for the confidence intervals

    Returns
    -------
    params_table : DataFrame instance
    '''
    #Parameters part of the summary table
    vals = [params, bse, tvalues, pvalues, confint]
    if any([x is None for x in vals]):
        params = results.params
        bse = results.bse
        tvalues = results.tvalues
        pvalues = results.pvalues
        confint = results.conf_int(alpha)
    data = np.array([params, bse, tvalues, pvalues]).T
    data = np.hstack([data, confint])
    data = pd.DataFrame(data)
    data.columns = ['Coef.', 'Std.Err.', 't', 'P>|t|', 
                    '[' + str(alpha/2), str(1-alpha/2) + ']']
    data.index = results.model.exog_names
    return data


# Vertical summary instance for multiple models
def _col_params(result, float_format='%.4f', stars=True):
    '''Stack coefficients and standard errors in single column
    '''

    # Extract parameters
    res = summary_params(result)
    # Format float
    for col in res.columns[:2]:
        res[col] = res[col].apply(lambda x: float_format % x)
    # Std.Errors in parentheses
    res.ix[:,1] = '(' + res.ix[:,1] + ')'
    # Significance stars
    if stars:
        idx = res.ix[:,3] < .1
        res.ix[:,0][idx] = res.ix[:,0][idx] + '*'
        idx = res.ix[:,3] < .05
        res.ix[:,0][idx] = res.ix[:,0][idx] + '*'
        idx = res.ix[:,3] < .01
        res.ix[:,0][idx] = res.ix[:,0][idx] + '*'
    # Stack Coefs and Std.Errors
    res = res.ix[:,:2]
    res = res.stack()
    res = pd.DataFrame(res)
    return res

def _col_info(result, info_dict=None):
    '''Stack model info in a column
    '''

    if info_dict == None:
        info_dict = {'N': lambda x: str(int(x.nobs)), 
                    'AIC': lambda x: '%.3f' % x.aic, 
                    'R2': lambda x: '%.3f' % x.rsquared}
    out = []
    for i in info_dict:
        try:
            out.append(info_dict[i](result))
        except:
            out.append('')
    out = pd.DataFrame(out)
    out.index = pd.Index(info_dict.keys())
    return out

def summary_col(results, float_format='%.4f', model_names=None, stars=True,
        info_dict=None):
    '''Add the contents of a Dict to summary table

    Parameters
    ----------
    results : statsmodels results instance or list of result instances
    float_format : string 
        float format for coefficients and standard errors
    model_names : list of strings of length len(results)
    stars : bool
        print significance stars 
    info_dict : dict
        dict of lambda functions to be applied to results instances to retrieve
        model info 
    '''

    # Coerce to list if user feeds a results instance
    if type(results) != list:
        results = [results]
    # Params as dataframe columns
    cols = [_col_params(x, stars=stars, float_format=float_format) for x in results]
    merg = lambda x,y: x.merge(y, how='outer', right_index=True, left_index=True)
    summ = reduce(merg, cols)
    # Index
    idx1 = summ.index.get_level_values(0).tolist()
    idx2 = range(1,len(idx1),2)
    for i in idx2:
        idx1[i] = ''
    summ.index = pd.Index(idx1)
    # Header
    if model_names == None:
        header = []
        try:
            for r in results:
                header.append(r.model.endog_names)
        except:
            i = 0
            for r in results:
                header.append('Model ' + i)
                i += 1
    else:
        header = model_names
    summ.columns = pd.Index(header)
    summ = summ.fillna('')
    # Info as dataframe columns
    cols = [_col_info(x, info_dict) for x in results]
    merg = lambda x,y: x.merge(y, how='outer', right_index=True, left_index=True)
    info = reduce(merg, cols)
    dat = pd.DataFrame(np.vstack([summ,info])) # pd.concat better, but error
    dat.columns = summ.columns
    dat.index = pd.Index(summ.index.tolist() + info.index.tolist())
    # Summary
    smry = Summary()
    smry.add_df(dat, header=True, align='l')
    smry.add_text('Standard errors in parentheses.')
    if stars:
        smry.add_text('* p<.1, ** p<.05, ***p<.01')
    return smry
