import numpy as np
import pandas as pd
import datetime
import copy
import collections
import StringIO
import textwrap
from copy import copy
from table import SimpleTable
from tableformatting import fmt_latex, fmt_txt
from collections import OrderedDict

class Summary(object):
    def __init__(self):
        self.tables = []
        self.settings = []
        self.extra_txt = []
        self.rule = []
        self.title = None

    def __str__(self):
        return self.as_text()

    def __repr__(self):
        return str(type(self)) + '\n"""\n' + self.__str__() + '\n"""'

    def _repr_html_(self):
        '''Display as HTML in IPython notebook.'''
        return self.as_html()

    def add_df(self, df, index=True, header=True, float_format=None, 
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

        cols = df.columns.tolist()
        if len(cols) != len(set(cols)):
            raise Exception('DataFrame must not include duplicated column names')
        settings = {'index':index, 'header':header, 
                    'float_format':float_format, 'align':align}
        self.tables.append(df)
        self.settings.append(settings)

    def add_array(self, array, align='r', float_format=None):
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

    def add_dict(self, d, ncols=2, align='l', float_format=None):
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

        keys = [_formatter(x, float_format) for x in d.keys()]
        vals = [_formatter(x, float_format) for x in d.values()]
        data = np.array(zip(keys, vals))

        if data.shape[0] % ncols != 0:
            pad = ncols - (data.shape[0] % ncols)
            data = np.vstack([data, np.array(pad * [['','']])])

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

    def add_base(self, results, alpha=0.05, float_format=None, title=None, 
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

        param = summary_params(results, alpha=alpha, float_format=float_format)
        info = summary_model(results)
        if xname != None:
            param.index = xname
        if yname != None: 
            info['Dependent Variable:'] = yname
        self.add_dict(info, align='l')
        self.add_df(param, float_format=float_format)
        self.add_title(title=title, results=results)

    def add_rule(self, row):
        '''In as_text() ascii output, add a horizontal rule below row. Row
        index starts at 0 (top rule) and includes any existing separators.'''

        self.rule.append(row)

    def as_text(self):
        '''Generate ASCII Summary Table
        '''

        tables = self.tables
        settings = self.settings
        title = self.title
        extra_txt = self.extra_txt

        pad_col, pad_index, widest = _measure_tables(tables, settings)

        rule_equal = widest * '='
        rule_dash = widest * '-'

        simple_tables = _simple_tables(tables, settings, pad_col, pad_index)
        tab = [x.as_text() for x in simple_tables]

        tab = '\n'.join(tab)
        tab = tab.split('\n')
        tab[0] = rule_equal
        tab.append(rule_equal)
        tab = '\n'.join(tab)

        if title != None:
            title = title
            if len(title) < widest:
                title = ' ' * int(widest/2 - len(title)/2) + title
        else:
            title = ''

        txt = [textwrap.wrap(x, widest) for x in extra_txt]
        txt = ['\n'.join(x) for x in txt]
        txt = '\n'.join(txt)

        out = '\n'.join([title, tab, txt])

        if len(self.rule) > 0:
            self.rule.sort(reverse=True)
            out = out.splitlines()
            for row in self.rule:
                out[row] = out[row] + '\n' + rule_dash
            out = '\n'.join(out)

        return out

    def as_html(self):
        '''Generate HTML Summary Table
        '''

        tables = self.tables
        settings = self.settings
        title = self.title

        simple_tables = _simple_tables(tables, settings)
        tab = [x.as_html() for x in simple_tables]
        tab = '\n'.join(tab)

        return tab

    def as_latex(self):
        '''Generate LaTeX Summary Table
        '''
        tables = self.tables
        settings = self.settings
        title = self.title
        if title != None:
            title = '\\caption{' + title + '} \\\\\n%\\label{}'
        else:
            title = '%\\caption{}\n%\\label{}'

        simple_tables = _simple_tables(tables, settings)
        tab = [x.as_latex_tabular() for x in simple_tables]
        tab = '\n\\hline\n'.join(tab)

        out = '\\begin{table}', title, tab, '\\end{table}'
        out = '\n'.join(out)
        return out

def _measure_tables(tables, settings):
    '''Compare width of ascii tables in a list and calculate padding values.
    We add space to each col_sep to get us as close as possible to the
    width of the largest table. Then, we add a few spaces to the first
    column to pad the rest.
    '''

    simple_tables = _simple_tables(tables, settings)
    tab = [x.as_text() for x in simple_tables] 

    length = [len(x.splitlines()[0]) for x in tab]
    len_max = max(length)
    pad_sep = []
    pad_index = []

    for i in range(len(tab)):
        nsep = tables[i].shape[1] - 1
        if settings[i]['index']:
            nsep += 1
        pad = int((len_max - length[i]) / nsep)
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

def summary_model(results, info_dict=None):
    '''Create a dict with information about the model
    '''
    if info_dict == None:
        info_dict = collections.OrderedDict()
        info_dict['Model:'] = lambda x: x.model.__class__.__name__
        info_dict['Model Family:'] = lambda x: x.family.__class.__name__
        info_dict['Link Function:'] = lambda x: x.family.link.__class__.__name__
        info_dict['Dependent Variable:'] = lambda x: x.model.endog_names
        now = datetime.datetime.now()
        info_dict['Date:'] = lambda x: now.strftime('%Y-%m-%d %H:%M')
        info_dict['No. Observations:'] = lambda x: "%#6d" % x.nobs
        info_dict['Df Model:'] = lambda x: "%#6d" % x.df_model
        info_dict['Df Residuals:'] = lambda x: "%#6d" % x.df_resid
        info_dict['Converged:'] = lambda x: x.mle_retvals['converged']
        info_dict['No. Iterations:'] = lambda x: x.mle_retvals['iterations']
        info_dict['Method:'] = lambda x: x.method
        info_dict['Norm:'] = lambda x: x.fit_options['norm']
        info_dict['Scale:'] = lambda x: _formatter(x.scale)
        info_dict['Scale Est.:'] = lambda x: x.fit_options['scale_est']
        info_dict['Cov. Type:'] = lambda x: x.fit_options['cov']
        info_dict['R-squared:'] = lambda x: _formatter(x.rsquared)
        info_dict['Adj. R-squared:'] = lambda x: _formatter(x.rsquared_adj)
        info_dict['Pseudo R-squared:'] = lambda x: _formatter(x.prsquared)
        info_dict['AIC:'] = lambda x: _formatter(x.aic)
        info_dict['BIC:'] = lambda x: _formatter(x.bic)
        info_dict['Log-Likelihood:'] = lambda x: _formatter(x.llf)
        info_dict['LL-Null:'] = lambda x: _formatter(x.llnull)
        info_dict['LLR p-value:'] = lambda x: _formatter(x.llr_pvalue)
        info_dict['Deviance:'] = lambda x: _formatter(x.deviance)
        info_dict['Pearson chi2:'] = lambda x: _formatter(x.pearson_chi2)
        info_dict ['F-statistic:'] = lambda x: _formatter(self.fvalue)
        info_dict ['Prob (F-statistic):'] = lambda x: _formatter(self.f_pvalue)
    out = collections.OrderedDict()
    for key in info_dict.keys():
        try: 
            out[key] = info_dict[key](results)
        except:
            pass 
    return out 

def summary_params(results, alpha=.05, float_format=None, vertical=False,
        stars=False):
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

    params = results.params
    bse = results.bse.tolist()
    tvalues = results.tvalues.tolist()
    pvalues = results.pvalues.tolist()
    confint = np.array(results.conf_int(alpha))
    confint_lb = confint[:,0].tolist()
    confint_ub = confint[:,1].tolist()

    if stars:
        r = range(len(params))
        params = [_formatter(x, float_format) for x in params]
        params = [params[i] + '*' if pvalues[i] < .1 else params[i] for i in r]
        params = [params[i] + '*' if pvalues[i] < .05 else params[i] for i in r]
        params = [params[i] + '*' if pvalues[i] < .01 else params[i] for i in r]

    pvalues = [x if x >= 2e-16 else '<2e-16' for x in pvalues]

    values = [params, bse, tvalues, pvalues, confint_lb, confint_ub]
    f = lambda x: _formatter(x, float_format)
    values = [[f(y) for y in x] for x in values]
    data = pd.DataFrame(values).T
    data.columns = ['Coef.', 'Std.Err.', 't', 'P>|t|', 
                    '[' + str(alpha/2), str(1-alpha/2) + ']'] 

    data.index = results.model.exog_names

    if vertical:
        data = data.ix[:,:2].stack()
        idx = data.index.get_level_values(1) == 'Std.Err.'
        data[idx] = '(' + data[idx] + ')'

    return data

def summary_col(results, float_format=None, model_names=None, stars=True,
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

    if type(results) != list:
        results = [results]

    f = lambda x: summary_params(x, stars=stars, float_format=float_format,
            vertical=True)
    cols = [f(x) for x in results]
    summ = pd.DataFrame(cols).T
    
    if info_dict == None:
        info_dict = {'N':lambda x: str(int(x.nobs)), 
                     'R2':lambda x: '%.3f' % x.rsquared}
    info = [summary_model(x, info_dict) for x in results]
    info = [pd.Series(x) for x in info]
    info = pd.DataFrame(info).T

    out = pd.concat([summ, info])

    if model_names == None:
        header = ['Model ' + str(x) for x in range(len(results))]
    else:
        header = model_names
    out.columns = header

    out = out.fillna('')

    idx = pd.Series(summ.index.get_level_values(0).tolist() + info.index.tolist())
    idx[1:summ.shape[0]:2] = ''
    out.index = idx

    smry = Summary()
    smry.add_df(out)
    smry

    smry.add_text('Standard errors in parentheses.')
    if stars:
        smry.add_text('* p<.1, ** p<.05, ***p<.01')

    smry.add_rule(summ.shape[0] + 3)

    return smry

def _formatter(element, float_format=None):
    try:
        # statsmodels-wide default values for float formatting
        if float_format is None:
            if (abs(element) >= 1e4) or (abs(element) < 1e-4):
                out = "%4.5g" % element
            else:
                out = '%.4f' % element
        else:
            out = float_format % element
    except:
        out = str(element)
    return out.strip()

def _df_to_simpletable(dat, align='r', float_format=None, header=True, index=True,
        table_dec_above='-', table_dec_below=None, header_dec_below='-', 
        pad_col=0, pad_index=0):
    dat = dat.applymap(lambda x: _formatter(x, float_format))
    if header:
        headers = [str(x) for x in dat.columns.tolist()]
    else: 
        headers = None
    if index:
        stubs = [str(x) + int(pad_index) * ' ' for x in dat.index.tolist()]
    else: 
        dat.ix[:,0] = [str(x) + int(pad_index) * ' ' for x in dat.ix[:,0]]
        stubs = None
    st = SimpleTable(np.array(dat), headers=headers, stubs=stubs, 
            ltx_fmt=fmt_latex, txt_fmt=fmt_txt)
    st.output_formats['latex']['data_aligns'] = align
    st.output_formats['txt']['data_aligns'] = align
    st.output_formats['txt']['table_dec_above'] = table_dec_above
    st.output_formats['txt']['table_dec_below'] = table_dec_below
    st.output_formats['txt']['header_dec_below'] = header_dec_below
    st.output_formats['txt']['colsep'] = ' ' * int(pad_col + 1)
    return st

def _simple_tables(tables, settings, pad_col=None, pad_index=None):
    simple_tables = []
    if pad_col == None:
        pad_col = [0] * len(tables) 
    if pad_index == None:
        pad_index = [0] * len(tables) 
    for i,v in enumerate(tables):
        index = settings[i]['index']
        header = settings[i]['header']
        align = settings[i]['align']
        float_format = settings[i]['float_format']
        simple_tables.append(_df_to_simpletable(v, align=align,
            float_format=float_format, header=header, index=index,
            pad_col=pad_col[i], pad_index=pad_index[i]))
    return simple_tables

