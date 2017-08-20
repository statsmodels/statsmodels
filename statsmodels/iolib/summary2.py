from statsmodels.compat.python import (lrange, iterkeys, iteritems, lzip,
                                       reduce, itervalues, zip, string_types,
                                       range)
from statsmodels.compat.collections import OrderedDict
import numpy as np
import pandas as pd
import datetime
import textwrap
from .table import SimpleTable
from .tableformatting import fmt_latex, fmt_txt


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

        settings = {'index': index, 'header': header,
                    'float_format': float_format, 'align': align}
        self.tables.append(df)
        self.settings.append(settings)

    def add_array(self, array, align='r', float_format="%.4f"):
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

    def add_dict(self, d, ncols=2, align='l', float_format="%.4f"):
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

        keys = [_formatter(x, float_format) for x in iterkeys(d)]
        vals = [_formatter(x, float_format) for x in itervalues(d)]
        data = np.array(lzip(keys, vals))

        if data.shape[0] % ncols != 0:
            pad = ncols - (data.shape[0] % ncols)
            data = np.vstack([data, np.array(pad * [['', '']])])

        data = np.split(data, ncols)
        data = reduce(lambda x, y: np.hstack([x, y]), data)
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
        if isinstance(title, string_types):
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

        param = summary_params(results, alpha=alpha, use_t=results.use_t)
        info = summary_model(results)
        if xname is not None:
            param.index = xname
        if yname is not None:
            info['Dependent Variable:'] = yname
        self.add_dict(info, align='l')
        self.add_df(param, float_format=float_format)
        self.add_title(title=title, results=results)

    def as_text(self):
        '''Generate ASCII Summary Table
        '''

        tables = self.tables
        settings = self.settings
        title = self.title
        extra_txt = self.extra_txt

        pad_col, pad_index, widest = _measure_tables(tables, settings)

        rule_equal = widest * '='
        #TODO: this isn't used anywhere?
        rule_dash = widest * '-'

        simple_tables = _simple_tables(tables, settings, pad_col, pad_index)
        tab = [x.as_text() for x in simple_tables]

        tab = '\n'.join(tab)
        tab = tab.split('\n')
        tab[0] = rule_equal
        tab.append(rule_equal)
        tab = '\n'.join(tab)

        if title is not None:
            title = title
            if len(title) < widest:
                title = ' ' * int(widest/2 - len(title)/2) + title
        else:
            title = ''

        txt = [textwrap.wrap(x, widest) for x in extra_txt]
        txt = ['\n'.join(x) for x in txt]
        txt = '\n'.join(txt)

        out = '\n'.join([title, tab, txt])

        return out

    def as_html(self):
        '''Generate HTML Summary Table
        '''

        tables = self.tables
        settings = self.settings
        #TODO: this isn't used anywhere
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
        if title is not None:
            title = '\\caption{' + title + '} \\\\'
        else:
            title = '\\caption{}'

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
        pad = int((len_max - length[i]) / nsep)
        pad_sep.append(pad)
        len_new = length[i] + nsep * pad
        pad_index.append(len_max - len_new)

    return pad_sep, pad_index, max(length)


# Useful stuff
_model_types = {'OLS' : 'Ordinary least squares',
                'GLS' : 'Generalized least squares',
                'GLSAR' : 'Generalized least squares with AR(p)',
                'WLS' : 'Weighted least squares',
                'RLM' : 'Robust linear model',
                'NBin': 'Negative binomial model',
                'GLM' : 'Generalized linear model'
                }


def summary_model(results):
    '''Create a dict with information about the model
    '''
    def time_now(*args, **kwds):
        now = datetime.datetime.now()
        return now.strftime('%Y-%m-%d %H:%M')
    info = OrderedDict()
    info['Model:'] = lambda x: x.model.__class__.__name__
    info['Model Family:'] = lambda x: x.family.__class.__name__
    info['Link Function:'] = lambda x: x.family.link.__class__.__name__
    info['Dependent Variable:'] = lambda x: x.model.endog_names
    info['Date:'] = time_now
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
    info['F-statistic:'] = lambda x: "%#8.4g" % x.fvalue
    info['Prob (F-statistic):'] = lambda x: "%#6.3g" % x.f_pvalue
    info['Scale:'] = lambda x: "%#8.5g" % x.scale
    out = OrderedDict()
    for key, func in iteritems(info):
        try:
            out[key] = func(results)
        # NOTE: some models don't have loglike defined (RLM), so that's NIE
        except (AttributeError, KeyError, NotImplementedError):
            pass
    return out


def summary_params(results, yname=None, xname=None, alpha=.05, use_t=True,
                   skip_header=False, float_format="%.4f"):
    '''create a summary table of parameters from results instance

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
    float_format : string
        float formatting options (e.g. ".3g")

    Returns
    -------
    params_table : SimpleTable instance
    '''

    if isinstance(results, tuple):
        results, params, bse, tvalues, pvalues, conf_int = results
    else:
        params = results.params
        bse = results.bse
        tvalues = results.tvalues
        pvalues = results.pvalues
        conf_int = results.conf_int(alpha)

    data = np.array([params, bse, tvalues, pvalues]).T
    data = np.hstack([data, conf_int])
    data = pd.DataFrame(data)

    if use_t:
        data.columns = ['Coef.', 'Std.Err.', 't', 'P>|t|',
                        '[' + str(alpha/2), str(1-alpha/2) + ']']
    else:
        data.columns = ['Coef.', 'Std.Err.', 'z', 'P>|z|',
                        '[' + str(alpha/2), str(1-alpha/2) + ']']

    if not xname:
        data.index = results.model.exog_names
    else:
        data.index = xname

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
    res.iloc[:, 1] = '(' + res.iloc[:, 1] + ')'
    # Significance stars
    if stars:
        idx = res.iloc[:, 3] < .1
        res.loc[idx, res.columns[0]] = res.loc[idx, res.columns[0]] + '*'
        idx = res.iloc[:, 3] < .05
        res.loc[idx, res.columns[0]] = res.loc[idx, res.columns[0]] + '*'
        idx = res.iloc[:, 3] < .01
        res.loc[idx, res.columns[0]] = res.loc[idx, res.columns[0]] + '*'
    # Stack Coefs and Std.Errors
    res = res.iloc[:, :2]
    res = res.stack()
    res = pd.DataFrame(res)
    res.columns = [str(result.model.endog_names)]
    return res


def _col_info(result, info_dict=None):
    '''Stack model info in a column
    '''

    if info_dict is None:
        info_dict = {}
    out = []
    index = []
    for i in info_dict:
        if isinstance(info_dict[i], dict):
            # this is a specific model info_dict, but not for this result...
            continue
        try:
            out.append(info_dict[i](result))
        except:
            out.append('')
        index.append(i)
    out = pd.DataFrame({str(result.model.endog_names): out}, index=index)
    return out


def _make_unique(list_of_names):
    if len(set(list_of_names)) == len(list_of_names):
        return list_of_names
    # pandas does not like it if multiple columns have the same names
    from collections import defaultdict
    name_counter = defaultdict(str)
    header = []
    for _name in list_of_names:
        name_counter[_name] += "I"
        header.append(_name+" " + name_counter[_name])
    return header


def summary_col(results, float_format='%.4f', model_names=[], stars=False,
                info_dict=None, regressor_order=[]):
    """
    Summarize multiple results instances side-by-side (coefs and SEs)

    Parameters
    ----------
    results : statsmodels results instance or list of result instances
    float_format : string
        float format for coefficients and standard errors
        Default : '%.4f'
    model_names : list of strings of length len(results) if the names are not
        unique, a roman number will be appended to all model names
    stars : bool
        print significance stars
    info_dict : dict
        dict of lambda functions to be applied to results instances to retrieve
        model info. To use specific information for different models, add a
        (nested) info_dict with model name as the key.
        Example: `info_dict = {"N":..., "R2": ..., "OLS":{"R2":...}}` would
        only show `R2` for OLS regression models, but additionally `N` for
        all other results.
        Default : None (use the info_dict specified in
        result.default_model_infos, if this property exists)
    regressor_order : list of strings
        list of names of the regressors in the desired order. All regressors
        not specified will be appended to the end of the list.
    """

    if not isinstance(results, list):
        results = [results]

    cols = [_col_params(x, stars=stars, float_format=float_format) for x in
            results]

    # Unique column names (pandas has problems merging otherwise)
    if model_names:
        colnames = _make_unique(model_names)
    else:
        colnames = _make_unique([x.columns[0] for x in cols])
    for i in range(len(cols)):
        cols[i].columns = [colnames[i]]

    merg = lambda x, y: x.merge(y, how='outer', right_index=True,
                                left_index=True)
    summ = reduce(merg, cols)

    if regressor_order:
        varnames = summ.index.get_level_values(0).tolist()
        ordered = [x for x in regressor_order if x in varnames]
        unordered = [x for x in varnames if x not in regressor_order + ['']]
        order = ordered + list(np.unique(unordered))

        f = lambda idx: sum([[x + 'coef', x + 'stde'] for x in idx], [])
        summ.index = f(np.unique(varnames))
        summ = summ.reindex(f(order))
        summ.index = [x[:-4] for x in summ.index]

    idx = pd.Series(lrange(summ.shape[0])) % 2 == 1
    summ.index = np.where(idx, '', summ.index.get_level_values(0))

    # add infos about the models.
    if info_dict:
        cols = [_col_info(x, info_dict.get(x.model.__class__.__name__,
                                           info_dict)) for x in results]
    else:
        cols = [_col_info(x, getattr(x, "default_model_infos", None)) for x in
                results]
    # use unique column names, otherwise the merge will not succeed
    for df , name in zip(cols, _make_unique([df.columns[0] for df in cols])):
        df.columns = [name]
    merg = lambda x, y: x.merge(y, how='outer', right_index=True,
                                left_index=True)
    info = reduce(merg, cols)
    dat = pd.DataFrame(np.vstack([summ, info]))  # pd.concat better, but error
    dat.columns = summ.columns
    dat.index = pd.Index(summ.index.tolist() + info.index.tolist())
    summ = dat

    summ = summ.fillna('')

    smry = Summary()
    smry.add_df(summ, header=True, align='l')
    smry.add_text('Standard errors in parentheses.')
    if stars:
        smry.add_text('* p<.1, ** p<.05, ***p<.01')

    return smry


def _formatter(element, float_format='%.4f'):
    try:
        out = float_format % element
    except:
        out = str(element)
    return out.strip()


def _df_to_simpletable(df, align='r', float_format="%.4f", header=True,
                       index=True, table_dec_above='-', table_dec_below=None,
                       header_dec_below='-', pad_col=0, pad_index=0):
    dat = df.copy()
    dat = dat.applymap(lambda x: _formatter(x, float_format))
    if header:
        headers = [str(x) for x in dat.columns.tolist()]
    else:
        headers = None
    if index:
        stubs = [str(x) + int(pad_index) * ' ' for x in dat.index.tolist()]
    else:
        dat.iloc[:, 0] = [str(x) + int(pad_index) * ' ' for x in dat.iloc[:, 0]]
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
    float_format = '%.4f'
    if pad_col is None:
        pad_col = [0] * len(tables)
    if pad_index is None:
        pad_index = [0] * len(tables)
    for i, v in enumerate(tables):
        index = settings[i]['index']
        header = settings[i]['header']
        align = settings[i]['align']
        simple_tables.append(_df_to_simpletable(v, align=align,
                                                float_format=float_format,
                                                header=header, index=index,
                                                pad_col=pad_col[i],
                                                pad_index=pad_index[i]))
    return simple_tables
