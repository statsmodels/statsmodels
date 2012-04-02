from urllib2 import urlopen

import statsmodels.api as sm
import pandas
import matplotlib.pyplot as plt
from matplotlib import figure

#################### UTILITY FUNCTIONS ########################


from pandas import DataFrame, Index
import numpy as np
from scipy import stats

#NOTE: these need to take into account weights !

def anova_lm_single(model, **kwargs):
    """
    ANOVA table for one fitted linear model.

    Parameters
    ----------
    model : fitted linear model results instance
        A fitted linear model

    **kwargs**

    scale : float
        Estimate of variance, If None, will be estimated from the largest
    model. Default is None.
        test : str {"F", "Chisq", "Cp"} or None
        Test statistics to provide. Default is "F".

    Notes
    -----
    Use of this function is discouraged. Use anova_lm instead.
    """
    from charlton.desc import INTERCEPT

    test = kwargs.get("test", "F")
    scale = kwargs.get("scale", None)

    endog = model.model.endog
    exog = model.model.exog

    response_name = model.model.endog_names
    model_formula = []
    terms_info = model.model._data._orig_exog.column_info.term_to_columns
    exog_names = model.model.exog_names
    n_rows = len(terms_info) - (INTERCEPT in terms_info) + 1 # for resids

    pr_test = "PR(>%s)" % test
    names = ['df', 'sum_sq', 'mean_sq', test, pr_test]

    #maybe we should rethink using pinv > qr in OLS/linear models?
    #NOTE: to get the full q,r the same as R use scipy.linalg.qr with
    # pivoting
    q,r = np.linalg.qr(exog)
    effects = np.dot(q.T,endog)

    table = DataFrame(np.empty((n_rows, 5)), columns = names)

    index = []
    col_order = []
    if INTERCEPT in terms_info:
        terms_info.pop(INTERCEPT)
    for i, (term, cols) in enumerate(terms_info.iteritems()):
        table.ix[i]['sum_sq'] = np.sum(effects[cols[0]:cols[1]]**2)
        table.ix[i]['df'] = cols[1]-cols[0]
        col_order.append(cols[0])
        index.append(term.name())
    table.index = Index(index + ['Residual'])
    # fill in residual
    table.ix['Residual'][['sum_sq','df']] = model.ssr, model.df_resid

    table = table.ix[np.argsort(col_order + [exog.shape[1]+1])]
    table['mean_sq'] = table['sum_sq'] / table['df']
    if test == 'F':
        table[:n_rows][test] = ((table['sum_sq']/table['df'])/
                                (model.ssr/model.df_resid))
        table[:n_rows][pr_test] = stats.f.sf(table["F"], table["df"],
                                model.df_resid)
    table.ix['Residual'][[test,pr_test]] = np.nan
    return table


def anova_lm(*args, **kwargs):
    """
    ANOVA table for one or more fitted linear models.

    Parmeters
    ---------
    args : fitted linear model results instance
        One or more fitted linear models

    **kwargs**

    scale : float
        Estimate of variance, If None, will be estimated from the largest
    model. Default is None.
        test : str {"F", "Chisq", "Cp"} or None
        Test statistics to provide. Default is "F".

    Returns
    -------
    anova : DataFrame
    A DataFrame containing.

    Notes
    -----
    Model statistics are given in the order of args. Models must have
    a formula_str attribute.

    See Also
    --------
    model_results.compare_f_test, model_results.compare_lm_test
    """
    if len(args) == 1:
        return anova_lm_single(*args, **kwargs)
    test = kwargs.get("test", "F")
    scale = kwargs.get("scale", None)
    n_models = len(args)

    model_formula = []
    pr_test = "PR(>%s)" % test
    names = ['df_resid', 'ssr', 'df_diff', 'ss_diff', test, pr_test]
    table = DataFrame(np.empty((n_models, 6)), columns = names)

    if not scale: # assume biggest model is last
        scale = args[-1].scale

    table["ssr"] = map(getattr, args, ["ssr"]*n_models)
    table["df_resid"] = map(getattr, args, ["df_resid"]*n_models)
    table.ix[1:]["df_diff"] = np.diff(map(getattr, args, ["df_model"]*n_models))
    table["ss_diff"] = -table["ssr"].diff()
    if test == "F":
        table["F"] = table["ss_diff"] / table["df_diff"] / scale
        table[pr_test] = stats.f.sf(table["F"], table["df_diff"],
                             table["df_resid"])

    return table

################## END UTILITY FUNCTIONS #####################

#url = 'http://stats191.stanford.edu/data/salary.table'
#fh = urlopen(url)

#salary_table = pandas.read_table(fh)
salary_table = pandas.read_csv('salary.table')
#salary_table.attach() # require my pandas df-attach branch
#PR rejected but I still think this could be useful...
E = salary_table.E
M = salary_table.M
X = salary_table.X
S = salary_table.S

# Take a look at the data

fig = plt.figure()
ax = fig.add_subplot(111, xlabel='Experience', ylabel='Salary')
#ax.set_xlim(0, 21)
symbols = ['D', '^']
colors = ['r', 'g', 'blue']
factor_groups = salary_table.groupby(['E','M'])
for values, group in factor_groups:
    i,j = values
    ax.scatter(group['X'], group['S'], marker=symbols[j], color=colors[i-1],
               s=144)
ax.axis('tight')
plt.show()

# Fit a linear model


formula = 'S ~ categorical(E) + categorical(M) + X'
lm = sm.OLS.from_formula(formula, salary_table).fit()
print lm.summary()

# Have a look at the created design matrix

lm.model.exog[:20]

# Or since we initially passed in a DataFrame, we have a DataFrame available in

lm.model._data._orig_exog

# We keep a reference to the original untouched data in

lm.model._data.frame

# Get influence statistics
infl = lm.get_outlier_influence()

print infl.summary_table()

# or get a dataframe
df_infl = infl.summary_frame()

#Now plot the reiduals within the groups separately
#there's probably some nice DataFrame trickery for this
resid = lm.resid

fig = plt.figure()
ax = fig.add_subplot(111, xlabel='Group', ylabel='Residuals')
for values, group in factor_groups:
    i,j = values
    group_num = i*2 + j - 1 # for plotting purposes
    x = [group_num] * len(group)
    ax.scatter(x, resid[group.index], marker=symbols[j], color=colors[i-1],
            s=144, edgecolors='black')

ax.axis('tight')
plt.show()

# now we will test some interactions using anova or f_test

interX_lm = sm.OLS.from_formula("S ~ categorical(E) * X + categorical(M)",
                    salary_table).fit()
print interX_lm.summary()

# Do an ANOVA check using https://gist.github.com/2245820
# Will be in statsmodels soon

table1 = anova_lm(lm, interX_lm)
print table1

interM_lm = sm.OLS.from_formula("S ~ X + categorical(E)*categorical(M)",
                                 df=salary_table).fit()
print interM_lm.summary()

table2 = anova_lm(lm, interM_lm)
print table2

# The design matrix as a DataFrame
interM_lm.model._data._orig_exog
# The design matrix as an ndarray
interM_lm.model.exog
interM_lm.model.exog_names

try:
    infl = interM_lm.get_influence()
except: # there was a rename in master
    infl = interM_lm.get_outlier_influence()
resid = infl.resid_studentized_internal

fig = plt.figure()
ax = fig.add_subplot(111, xlabel='X', ylabel='standardized resids')

for values, group in factor_groups:
    i,j = values
    idx = group.index
    ax.scatter(X[idx], resid[idx], marker=symbols[j], color=colors[i-1],
            s=144, edgecolors='black')
ax.axis('tight')
plt.show()

# Looks like one observation is an outlier.
#TODO: do we have Bonferonni outlier test?

drop_idx = abs(resid).argmax()
print drop_idx # zero-based index

lm32 = sm.OLS.from_formula('S ~ categorical(E) + X + categorical(M)',
            df = salary_table.drop([drop_idx])).fit()

print lm32.summary()

interX_lm32 = sm.OLS.from_formula('S ~ categorical(E) * X + categorical(M)',
            df = salary_table.drop([drop_idx])).fit()

print interX_lm32.summary()

table3 = anova_lm(lm32, interX_lm32)
print table3

interM_lm32 = sm.OLS.from_formula('S ~ X + categorical(E) * categorical(M)',
            df = salary_table.drop([drop_idx])).fit()

table4 = anova_lm(lm32, interM_lm32)
print table4

# Replot the residuals
try:
    resid = interM_lm32.get_influence().summary_frame()['standard_resid']
except:
    resid = interM_lm32.get_outlier_influence().summary_frame()['standard_resid']

fig = plt.figure()
ax = fig.add_subplot(111, xlabel='X[~[32]]', ylabel='standardized resids')

for values, group in factor_groups:
    i,j = values
    idx = group.index
    ax.scatter(X[idx], resid[idx], marker=symbols[j], color=colors[i-1],
            s=144, edgecolors='black')
ax.axis('tight')
plt.show()


# Plot the fitted values

lm_final = sm.OLS.from_formula('S ~ X + categorical(E)*categorical(M)',
                    df = salary_table.drop([drop_idx])).fit()
mf = lm_final.model._data._orig_exog
lstyle = ['-','--']

fig = plt.figure()
ax = fig.add_subplot(111, xlabel='Experience', ylabel='Salary')

for values, group in factor_groups:
    i,j = values
    idx = group.index
    ax.scatter(X[idx], S[idx], marker=symbols[j], color=colors[i-1],
            s=144, edgecolors='black')
    # drop NA because there is no idx 32 in the final model
    ax.plot(mf.X[idx].dropna(), lm_final.fittedvalues[idx].dropna(),
            ls=lstyle[j], color=colors[i-1])
ax.axis('tight')
plt.show()

#From our first look at the data, the difference between Master's and PhD in the management group is different than in the non-management group. This is an interaction between the two qualitative variables management,M and education,E. We can visualize this by first removing the effect of experience, then plotting the means within each of the 6 groups using interaction.plot.

U = S - X * interX_lm32.params['X']

def rainbow(n):
    """
    Returns a list of colors sampled at equal intervals over the spectrum.

    Parameters
    ----------
    n : int
        The number of colors to return

    Returns
    -------
    R : (n,3) array
        An of rows of RGB color values

    Notes
    -----
    Converts from HSV coordinates (0, 1, 1) to (1, 1, 1) to RGB. Based on
    the Sage function of the same name.
    """
    from matplotlib import colors
    R = np.ones((1,n,3))
    R[0,:,0] = np.linspace(0, 1, n, endpoint=False)
    #Note: could iterate and use colorsys.hsv_to_rgb
    return colors.hsv_to_rgb(R).squeeze()

import numpy as np
def interaction_plot(x, trace, response, func=np.mean, ax=None, plottype='b',
                     xlabel=None, ylabel=None, colors = [], markers = [],
                     linestyles = [], legendloc='best', legendtitle=None,
                     **kwargs):
    """
    Parameters
    ----------
    x : array-like
        The `x` factor levels are the x-axis
    trace : array-like
        The `trace` factor levels will form the trace
    response : array-like
        The reponse variable
    func : function
        Anything accepted by `pandas.DataFrame.aggregate`. This is applied to
        the response variable grouped by the trace levels.
    plottype : str {'line', 'scatter', 'both'}, optional
        The type of plot to return. Can be 'l', 's', or 'b'
    ax : axes, optional
        Matplotlib axes instance
    xlabel : str, optional
        Label to use for `x`. Default is 'X'. If `x` is a `pandas.Series` it
        will use the series names.
    ylabel : str, optional
        Label to use for `response`. Default is 'func of response'. If
        `response` is a `pandas.Series` it will use the series names.
    colors : list, optional
    linestyles : list, optional
    markers : list, optional
        `colors`, `linestyles`, and `markers` must be lists of the same
        length as the number of unique trace elements. If you want to control
        the overall plotting options, use kwargs.
    kwargs
        These will be passed to the plot command used either plot or scatter.
    """
    from pandas import DataFrame
    if ax is None:
        import matplotlib.pyplot as plt
        fig = plt.figure()
        ax = fig.add_subplot(111)
    else:
        fig = ax.figure

    if ylabel is None:
        try: # did we get a pandas.Series
            response_name = response.name
        except:
            response_name = 'response'
        # py3 compatible?
        ylabel = '%s of %s' % (func.func_name, response_name)

    if xlabel is None:
        try:
            x_name = x.name
        except:
            x_name = 'X'

    if legendtitle is None:
        try:
            legendtitle = trace.name
        except:
            pass

    ax.set_ylabel(ylabel)
    ax.set_xlabel(x_name)


    data = DataFrame(dict(x=x, trace=trace, response=response))
    plot_data = data.groupby(['trace', 'x']).aggregate(func).reset_index()

    # check plot args
    n_trace = len(plot_data['trace'].unique())
    if linestyles:
        try:
            assert len(linestyles) == n_trace
        except AssertionError, err:
            raise ValueError("Must be a linestyle for each trace level")
    else: # set a default
        linestyles = ['-'] * n_trace
    if markers:
        try:
            assert len(markers) == n_trace
        except AssertionError, err:
            raise ValueError("Must be a linestyle for each trace level")
    else: # set a default
        markers = ['.'] * n_trace
    if colors:
        try:
            assert len(colors) == n_trace
        except AssertionError, err:
            raise ValueError("Must be a linestyle for each trace level")
    else: # set a default
        #TODO: how to get n_trace different colors?
        colors = rainbow(n_trace)

    if plottype == 'both' or plottype == 'b':
        for i, (values, group) in enumerate(plot_data.groupby(['trace'])):
            # trace label
            label = str(group['trace'].values[0])
            ax.plot(group['x'], group['response'], color=colors[i],
                    marker=markers[i], label=label,
                    linestyle=linestyles[i], **kwargs)
    elif plottype == 'line' or plottype == 'l':
        for i, (values, group) in enumerate(plot_data.groupby(['trace'])):
            # trace label
            label = str(group['trace'].values[0])
            ax.plot(group['x'], group['response'], color=colors[i],
                    label=label, linestyle=linestyles[i], **kwargs)
    elif plottype == 'scatter' or plottype == 's':
        for i, (values, group) in enumerate(plot_data.groupby(['trace'])):
            # trace label
            label = str(group['trace'].values[0])
            ax.scatter(group['x'], group['response'], color=colors[i],
                    label=label, marker=markers[i], **kwargs)

    else:
        raise ValueError("Plot type %s not understood" % plottype)
    ax.legend(loc=legendloc, title=legendtitle)
    ax.margins(.1)
    return ax

ax = interaction_plot(E, M, U, colors=['red','blue'], markers=['^','D'],
        markersize=10)
plt.show()

# Minority Employment Data
# ------------------------

url = 'http://stats191.stanford.edu/data/minority.table'
minority_table = pandas.read_table(url)

factor_group = minority_table.groupby(['ETHN'])

fig = plt.figure()
ax = fig.add_subplot(111, xlabel='TEST', ylabel='JPERF')
colors = ['purple', 'green']
markers = ['o', 'v']
for factor, group in factor_group:
    ax.scatter(group['TEST'], group['JPERF'], color=colors[factor],
                marker=markers[factor], s=12**2)

plt.show()

min_lm = sm.OLS.from_formula('JPERF ~ TEST', df=minority_table).fit()
print min_lm.summary()

fig = plt.figure()
ax = fig.add_subplot(111, xlabel='TEST', ylabel='JPERF')
for factor, group in factor_group:
    ax.scatter(group['TEST'], group['JPERF'], color=colors[factor],
                marker=markers[factor], s=12**2)

def abline_plot(intercept=None, slope=None, horiz=None, vert=None,
                model_results=None, ax=None, **kwargs):
    """
    Plots a line given an intercept and slope.

    intercept : float
        The intercept of the line
    slope : float
        The slope of the line
    horiz : float or array-like
        Data for horizontal lines on the y-axis
    vert : array-like
        Data for verterical lines on the x-axis
    model_results : statsmodels results instance
        Any object that has a two-value `params` attribute. Assumed that it
        is (intercept, slope)
    ax : axes, optional
        Matplotlib axes instance
    kwargs
        Options passed to matplotlib.pyplot.plt

    Returns
    -------
    axes : Axes
        The axes given by `axes` or a new instance.
    """
    if ax is None:
        import matplotlib.pyplot as plt
        fig = plt.figure()
        ax = fig.add_subplot(111)
    else:
        fig = ax.figure

    if model_results:
        intercept, slope = model_results.params
    else:
        if not (intercept is not None and slope is not None):
            raise ValueError("specify slope and intercepty or model_results")

    origin = [intercept, 0]
    x = ax.get_xlim()
    y = [x[0]*slope+intercept, x[1]*slope+intercept]

    from matplotlib.lines import Line2D

    class ABLine2D(Line2D):

        def update_datalim(self, ax):
            ax.set_autoscale_on(False)

            children = ax.get_children()
            abline = [children[i] for i in range(len(children))
                       if isinstance(children[i], ABLine2D)][0]
            x = ax.get_xlim()
            y = [x[0]*slope+intercept, x[1]*slope+intercept]
            abline.set_data(x,y)
            ax.figure.canvas.draw()

    line = ABLine2D(x, y, **kwargs)
    ax.add_line(line)
    ax.callbacks.connect('xlim_changed', line.update_datalim)
    ax.callbacks.connect('ylim_changed', line.update_datalim)


    if horiz:
        ax.hline(horiz)
    if vert:
        ax.vline(vert)
    return ax

abline_plot(model_results = min_lm, ax=ax)
plt.show()

min_lm2 = sm.OLS.from_formula('JPERF ~ TEST + TEST:ETHN',
        df=minority_table).fit()

print min_lm2.summary()

fig = plt.figure()
ax = fig.add_subplot(111)
for factor, group in factor_group:
    ax.scatter(group['TEST'], group['JPERF'], color=colors[factor],
                marker=markers[factor], s=12**2)

ax = abline_plot(intercept = min_lm2.params['const'],
                 slope = min_lm2.params['TEST'], ax=ax, color='purple')
ax = abline_plot(intercept = min_lm2.params['const'],
        slope = min_lm2.params['TEST'] + min_lm2.params['TEST:ETHN'],
        ax=ax, color='green')
plt.show()


min_lm3 = sm.OLS.from_formula('JPERF ~ TEST + ETHN', df = minority_table).fit()
print min_lm3.summary()

fig = plt.figure()
ax = fig.add_subplot(111)
for factor, group in factor_group:
    ax.scatter(group['TEST'], group['JPERF'], color=colors[factor],
                marker=markers[factor], s=12**2)

ax = abline_plot(intercept = min_lm3.params['const'],
                 slope = min_lm3.params['TEST'], ax=ax, color='purple')
ax = abline_plot(intercept = min_lm3.params['const'] + min_lm3.params['ETHN'],
        slope = min_lm3.params['TEST'], ax=ax, color='green')
plt.show()


min_lm4 = sm.OLS.from_formula('JPERF ~ TEST * ETHN', df = minority_table).fit()
print min_lm4.summary()

fig = plt.figure()
ax = fig.add_subplot(111)
for factor, group in factor_group:
    ax.scatter(group['TEST'], group['JPERF'], color=colors[factor],
                marker=markers[factor], s=12**2)

ax = abline_plot(intercept = min_lm4.params['const'],
                 slope = min_lm4.params['TEST'], ax=ax, color='purple')
ax = abline_plot(intercept = min_lm4.params['const'] + min_lm4.params['ETHN'],
        slope = min_lm4.params['TEST'] + min_lm4.params['TEST:ETHN'],
        ax=ax, color='green')
plt.show()

# is there any effect of ETHN on slope or intercept
table5 = anova_lm(min_lm, min_lm4)
print table5
# is there any effect of ETHN on intercept
table6 = anova_lm(min_lm, min_lm3)
print table6
# is there any effect of ETHN on slope
table7 = anova_lm(min_lm, min_lm2)
print table7
# is it just the slope or both?
table8 = anova_lm(min_lm2, min_lm4)
print table8


# One-way ANOVA
# -------------

#url = 'http://stats191.stanford.edu/data/rehab.csv'
#rehab_table = pandas.read_table(url, delimiter=",")
#rehab_table.save('rehab.table') # save to debug boxplot
rehab_table = pandas.read_csv('rehab.table')

ax = rehab_table.boxplot('Time', 'Fitness')
plt.show()

rehab_lm = sm.OLS.from_formula('Time ~ categorical(Fitness)',
                df=rehab_table).fit()
table9 = anova_lm(rehab_lm)
print table9

print rehab_lm.model._data._orig_exog

print rehab_lm.summary()

# Two-way ANOVA
# -------------

url = 'http://stats191.stanford.edu/data/kidney.table'
# pandas fails on this table
#kidney_table = pandas.read_table(url, delimiter=" ", converters=converters)

kidney_table = np.genfromtxt(urlopen(url), names=True)
kidney_table = pandas.DataFrame.from_records(kidney_table)

kt = kidney_table
interaction_plot(kt['Weight'], kt['Duration'], np.log(kt['Days']+1),
        colors=['red', 'blue'], markers=['D','^'], ms=10)
plt.show()

#I thought we could use the numpy namespace in charlton?
from charlton.builtins import builtins
builtins['np'] = np
kidney_lm = sm.OLS.from_formula(
        'np.log(Days+1) ~ categorical(Duration) * categorical(Weight)',
        df=kt).fit()

table10 = anova_lm(kidney_lm)
