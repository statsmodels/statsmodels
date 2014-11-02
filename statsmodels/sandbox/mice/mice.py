"""
This module implements the Multiple Imputation through Chained
Equations (MICE) approach to handling missing data. The approach has
the following steps:

0. Impute each missing value with the mean of the observed values of
the same variable.

1. For each variable in the data set with missing values, do the
following:

1a. Fit a regression model using only the observed values of the
'focus variable', regressed against the observed and current imputed
values of some or all of the remaining variables.  Then impute the
missing values for the focus variable.  One procedure for doing this
is the 'predictive mean matching' (pmm) procedure.

2. Repeat step 1 for all variables.

3. Once all variables have been imputed, fit the analysis model to the
data set.

4. Repeat steps 1-3 multiple times and combine the results using a
combining rule.

The specific way that each variable is imputed is specified using a
conditional model and formula for each variable.  The default model is
OLS, with a formula specifying main effects for all other variables.

If the goal is only to produce imputed data sets, the MICE_data
class can be used to wrap a data frame, providing facilities for doing
the imputation.  Summary plots are available for assessing the
performance of the imputation.

If the imputed data sets are to be used to specify an additional
'analysis model', a MICE instance can be used.  After specifying the
MICE instance and running it, the results are combined using the
`combine` method.  Results and various summary plots are then
available.

Terminology
-----------
The MICE procedure is determined by a family of conditional models.
There is one conditional model for each variable with missing values.
A conditional model may be conditioned on all or a subset of the
remaining variables, using main effects, transformations,
interactions, etc. as desired.

A 'perturbation method' is a method for setting the parameter estimate
in a conditional model.  The 'gaussian' perturbation method first fits
the model (usually using maximum likelihood, but it could use any
statsmodels fit procedure), then sets the parameter vector equal to a
draw from the Gaussian approximation to the sampling distribution for
the fit.  The 'bootstrap' perturbation method sets the parameter
vector equal to a fitted parameter vector obtained when fitting the
conditional model to a bootstrapped version of the data set.

Class structure
---------------
There are three main classes in the module:

* 'MICE_data' wraps a dataframe (or comparable data container),
  incorporating information about the conditional models for each
  variable with missing values. It can be used to produce multiply
  imputed data sets that are to be further processed or distributed to
  other researchers.  A number of plotting procedures are provided to
  visualize the imputations.  The `history_func` hook allows any
  features of interest of the imputed data sets to be saved for
  further analysis.

* 'MICE_model' takes a MICE_data object along with a specification for
  an additional 'analysis model' which is the central model of
  interest (it may or may not be identical to the conditional model
  for the same outcome variable).  MICE_model carries out multiple
  imputation using the conditional models, and fits the analysis model
  to a subset of these imputed data sets.  It returns the fitted model
  results for these analysis models.  It is structured as an iterator,
  so the analysis model results are obtained using `next` or via any
  standard python iterator pattern.

* 'MICE' takes both a 'MICE_data' object and an analysis model
  specification.  It runs the multiple imputation, fits the analysis
  models, and combines the results to produce a `MICEResults` object.
  The summary method of this results object can be used to see the key
  estimands and inferential quantities..

Notes
-----
By default, to conserve memory 'MICE_data' saves very little
information from one iteration to the next.  The data set passed by
the user is copied on entry, but then is over-written each time new
imputations are produced.  If using 'MICE_model' or 'MICE', the fitted
analysis models and results are saved.  MICE_data includes a
`history_func` hook that allows arbitrary information from the
intermediate datasets to be saved for future use.

References
----------
JL Schafer: 'Multiple Imputation: A Primer', Stat Methods Med Res,
1999.

T E Raghunathan et al.: 'A Multivariate Technique for Multiply
Imputing Missing Values Using a Sequence of Regression Models', Survey
Methodology, 2001.

SAS Institute: 'Predictive Mean Matching Method for Monotone Missing
Data', SAS 9.2 User's Guide, 2014.

A Gelman et al.: 'Multiple Imputation with Diagnostics (mi) in R:
Opening Windows into the Black Box', Journal of Statistical Software,
2009.
"""

#TODO: Add reference http://biomet.oxfordjournals.org/content/86/4/948.full.pdf


import pandas as pd
import numpy as np
import patsy
import statsmodels.api as sm
import statsmodels
from collections import defaultdict

# Can be replaced with scipy in version >= 0.14
class _multivariate_normal(object):

    def __init__(self, mean, cov):
        self.mean = mean
        self.cov = cov
        self.cov_sqrt = np.linalg.cholesky(cov)

    def rvs(self):
        p = len(self.mean)
        z = np.random.normal(size=p)
        return self.mean + np.dot(self.cov_sqrt, z)



_mice_data_example_1 = """
    >>> imp = mice.MICE_data(data)
    >>> imp.set_imputer('x1', formula='x2 + np.square(x2) + x3')
    >>> for j in range(20):
            imp.update_all()
            imp.data.to_csv('data%02d.csv' % j)
"""

class MICE_data(object):

    __doc__ = """\
    Wrap a data set to allow missing data handling for MICE.

    Parameters
    ----------
    data : array-like object
        The data set.
    perturbation_method : string
        The default perturbation method
    history_func : function
        A function that is called after each complete imputation
        cycle.  The return value is appended to `history`.  The
        MICE_data object is passed as the sole argument to
        `history_func`.

    Examples
    --------
    Draw 20 imputations from a data set called `data` and save them in
    separate files with filename pattern `dataXX.csv`.  The variables
    other than `x1` are imputed using linear models fit with OLS, with
    mean structures containing main effects of all other variables in
    `data`.  The variable named `x1` has a condtional mean structure
    that includes an additional term for x2^2.
    %(_mice_data_example_1)s
    Notes
    -----
    Allowed perturbation methods are 'gaussian' (the model parameters
    are set to a draw from the Gaussian approximation to the posterior
    distribution), and 'boot' (the model parameters are set to the
    estimated values obtained when fitting a bootstrapped version of
    the data set).

    `history_func` can be implemented to have side effects such as
    saving the current imputed data set to disk.
    """ % {'_mice_data_example_1': _mice_data_example_1}

    def __init__(self, data, perturbation_method='gaussian',
                 history_func=None):

        # Drop observations where all variables are missing.  This
        # also has the effect of copying the data frame.
        self.data = data.dropna(how='all').reset_index(drop=True)

        self.history_func = history_func
        self.history = []

        self.perturbation_method = defaultdict(lambda :
                                               perturbation_method)

        # Map from variable name to indices of observed/missing
        # values.
        self.ix_obs = {}
        self.ix_miss = {}
        for col in self.data.columns:
            ix_obs, ix_miss = self._split_indices(self.data[col])
            self.ix_obs[col] = ix_obs
            self.ix_miss[col] = ix_miss

        # Most recent model and results for each variable.
        self.models = {}
        self.results = {}

        # Map from variable names to the conditional formula.
        self.conditional_formula = {}

        # Map from variable names to init/fit args of the conditional
        # models.
        self.init_args = defaultdict(lambda : {})
        self.fit_args = defaultdict(lambda : {})

        # Map from variable name to the method used to handle the
        # scale parameter of the conditional model.
        self.scale_method = defaultdict(lambda : "fix")

        # Map from variable names to the model class.
        self.model_class = {}

        # Map from variable names to most recent params update.
        self.params = {}

        # Set default imputers.
        for vname in data.columns:
            self.set_imputer(vname)

        # The order in which variables are imputed in each cycle.
        # Impute variables with the fewest missing values first.
        vnames = data.columns.tolist()
        nmiss = [len(self.ix_miss[v]) for v in vnames]
        nmiss = np.asarray(nmiss)
        ii = np.argsort(nmiss)
        ii = ii[sum(nmiss == 0):]
        self.cycle_order = [vnames[i] for i in ii]

        # Fill missing values with column-wise mean.  This is the
        # starting imputation.
        self.data = self.data.fillna(self.data.mean())

        self.k_pmm = 20

    def _split_indices(self, vec):
        null = pd.isnull(vec)
        ix_obs = np.flatnonzero(~null)
        ix_miss = np.flatnonzero(null)
        if len(ix_obs) == 0:
            raise ValueError("variable to be imputed has no observed values")
        return ix_obs, ix_miss

    def set_imputer(self, endog_name, formula=None, model_class=None,
                    init_args=None, fit_args=None, k_pmm=20,
                    perturbation_method=None, scale_method="fix"):
        """
        Specify the imputation process for a single variable.

        Parameters
        ----------
        endog_name : string
            Name of the variable to be imputed.
        formula : string
            Conditional formula for imputation. Defaults to a formula
            with main effects for all other variables in dataset.  The
            formula should only include an expression for the mean,
            structure, e.g. use 'x1 + x2' not 'x4 ~ x1 + x2'.
        model_class : statsmodels model
            Conditional model for imputation. Defaults to OLS.
        init_args : Dictionary
            Keyword arguments passed to the model init method.
        fit_args : Dictionary
            Keyword arguments passed to the model fit method.
        perturbation_method : string
            Either 'gaussian' or 'bootstrap'. Determines the method
            for perturbing parameters in the conditional model.  If
            None, uses the default specified in init.
        k_pmm : int
            Determines number of neighboring observations from which
            to randomly sample when using predictive mean matching.
        scale_method : string
            Either 'fix' or 'perturb_chi2'.  Governs the type of
            perturbation given to the scale parameter.  Will have no
            effect unless the fitted values depend on the scale
            parameter.  If 'fix', the estimated scale parameter is
            used; if 'perturb_chi2', the scale parameter is updated
            from an approximate chi^2 sampling distribution.
        """

        # TODO: if we only use pmm, do we need scale_method?

        if formula is None:
            main_effects = [x for x in self.data.columns
                            if x != endog_name]
            fml = endog_name + " ~ " + " + ".join(main_effects)
            self.conditional_formula[endog_name] = fml
        else:
            fml = endog_name + " ~ " + formula
            self.conditional_formula[endog_name] = fml

        if model_class is None:
            self.model_class[endog_name] = sm.OLS
        else:
            self.model_class[endog_name] = model_class

        if init_args is not None:
            self.init_args[endog_name] = init_args

        if fit_args is not None:
            self.fit_args[endog_name] = fit_args

        if perturbation_method is not None:
            self.perturbation_method[endog_name] = perturbation_method

        self.k_pmm = k_pmm


    def store_changes(self, col, vals):
        """
        Fill in dataset with imputed values.

        Parameters
        ----------
        col : string
            Name of variable to be filled in.
        vals : array
            Array of imputed values to use in filling in missing values.
        """

        ix = self.ix_miss[col]
        if len(ix) > 0:
            self.data[col].iloc[ix] = vals

    def update_all(self, n_iter=10):
        """
        Perform a specified number of MICE iterations.

        Parameters
        ----------
        n_iter : int
            The number of updates to perform.  Only the result of the
            final update will be available.

        Notes
        -----
        The imputed values are stored in the parent dataset.
        """

        for k in range(n_iter):
            for vname in self.cycle_order:
                self.update(vname)

        if self.history_func is not None:
            hv = self.history_func(self)
            self.history.append(hv)

    def get_split_data(self, vname):
        """
        Use the conditional model formula to construct endog and exog,
        splitting by missingness status.

        Parameters
        ----------
        vname : string
           The variable for which the split data is returned.

        Returns
        -------
        endog_obs : DataFrame
            Observed values of the variable to be imputed.
        exog_obs : DataFrame
            Current values of the predictors where the variable to be
            Imputed is observed.
        exog_miss : DataFrame
            Current values of the predictors where the variable to be
            Imputed is missing.
        """

        formula = self.conditional_formula[vname]
        endog, exog = patsy.dmatrices(formula, self.data,
                                      return_type="dataframe")

        # Rows with observed endog
        ix = self.ix_obs[vname]
        endog_obs = np.asarray(endog.iloc[ix, 0])
        exog_obs = np.asarray(exog.iloc[ix, :])

        # Rows with missing endog
        ix = self.ix_miss[vname]
        exog_miss = np.asarray(exog.iloc[ix, :])

        return endog_obs, exog_obs, exog_miss

    def plot_missing_pattern(self, ax=None, row_order="pattern",
                             column_order="pattern",
                             hide_complete_rows=False,
                             hide_complete_columns=False,
                             color_row_patterns=True):
        """
        Generates an image showing the missing data pattern.

        Parameters
        ----------
        ax : matplotlib axes
            Axes on which to draw the plot.
        row_order : string
            The method for ordering the rows.  Must be one of 'pattern',
            'proportion', or 'raw'.
        column_order : string
            The method for ordering the columns.  Must be one of 'pattern',
            'proportion', or 'raw'.
        hide_complete_rows : bool
            If True, rows with no missing values are not drawn.
        hide_complete_columns : bool
            If True, columns with no missing values are not drawn.

        Returns
        -------
        A figure containing a plot of the missing data pattern.
        """

        # Create an indicator matrix for missing values.
        miss = np.zeros(self.data.shape)
        cols = self.data.columns
        for j, col in enumerate(cols):
            ix = self.ix_miss[col]
            miss[ix, j] = 1

        # Order the columns as requested
        if column_order == "proportion":
            ix = np.argsort(miss.mean(0))
        elif column_order == "pattern":
            cv = np.cov(miss.T)
            u, s, vt = np.linalg.svd(cv, 0)
            ix = np.argsort(cv[:, 0])
        elif column_order == "raw":
            ix = np.arange(len(cols))
        else:
            raise ValueError(column_order + " is not an allowed value for `column_order`.")
        miss = miss[:, ix]
        cols = [cols[i] for i in ix]

        # Order the rows as requested
        if row_order == "proportion":
            ix = np.argsort(miss.mean(1))
        elif row_order == "pattern":
            x = 2**np.arange(miss.shape[1])
            rky = np.dot(miss, x)
            ix = np.argsort(rky)
        elif row_order == "raw":
            ix = np.arange(miss.shape[0])
        else:
            raise ValueError(row_order + " is not an allowed value for `row_order`.")
        miss = miss[ix, :]

        if hide_complete_rows:
            ix = np.flatnonzero((miss == 1).any(1))
            miss = miss[ix, :]

        if hide_complete_columns:
            ix = np.flatnonzero((miss == 1).any(0))
            miss = miss[:, ix]
            cols = [cols[i] for i in ix]

        from statsmodels.graphics import utils as gutils
        from matplotlib.colors import LinearSegmentedColormap

        if ax is None:
            fig, ax = gutils.create_mpl_ax(ax)
        else:
            fig = ax.get_figure()

        if color_row_patterns:
            x = 2**np.arange(miss.shape[1])
            rky = np.dot(miss, x)
            _, rcol = np.unique(rky, return_inverse=True)
            miss *= 1 + rcol[:, None]
            ax.imshow(miss, aspect="auto", interpolation="nearest",
                      cmap='gist_ncar_r')
        else:
            cmap = LinearSegmentedColormap.from_list("_",
                                         ["white", "darkgrey"])
            ax.imshow(miss, aspect="auto", interpolation="nearest",
                      cmap=cmap)

        ax.set_ylabel("Cases")
        ax.set_xticks(range(len(cols)))
        ax.set_xticklabels(cols, rotation=90)

        return fig

    def bivariate_scatterplot(self, col1_name, col2_name,
                              lowess_args={}, lowess_min_n=40,
                              jitter=None, plot_points=True, ax=None):
        """
        Create a scatterplot between two variables, plotting the
        observed and imputed values with different colors.

        Parameters:
        -----------
        col1_name : string
            The variable to be plotted on the horizontal axis.
        col2_name : string
            The variable to be plotted on the vertical axis.
        lowess_args : dictionary
            A dictionary of dictionaries, keys are 'ii', 'io', 'oi'
            and 'oo', where 'o' denotes 'observed' and 'i' denotes
            imputed.  See Notes for details.
        lowess_min_n : integer
            Minimum sample size to plot a lowess fit
        jitter : float or tuple
            Standard deviation for jittering points in the plot.
            Either a single scalar applied to both axes, or a tuple
            containing x-axis jitter and y-axis jitter, respectively.
        plot_points : bool
            If True, the data points are plotted.
        ax : matplotlib axes object
            Axes on which to plot, created if not provided.

        Returns:
        --------
        The matplotlib figure on which the plot id drawn.
        """

        from statsmodels.graphics import utils as gutils
        from statsmodels.nonparametric.smoothers_lowess import lowess

        if ax is None:
            fig, ax = gutils.create_mpl_ax(ax)
        else:
            fig = ax.get_figure()

        ax.set_position([0.1, 0.1, 0.7, 0.8])

        ix1i = self.ix_miss[col1_name]
        ix1o = self.ix_obs[col1_name]
        ix2i = self.ix_miss[col2_name]
        ix2o = self.ix_obs[col2_name]

        ix_ii = np.intersect1d(ix1i, ix2i)
        ix_io = np.intersect1d(ix1i, ix2o)
        ix_oi = np.intersect1d(ix1o, ix2i)
        ix_oo = np.intersect1d(ix1o, ix2o)

        vec1 = np.asarray(self.data[col1_name])
        vec2 = np.asarray(self.data[col2_name])

        if jitter is not None:
            if np.isscalar(jitter):
                jitter = (jitter, jitter)
            vec1 += jitter[0] * np.random.normal(size=len(vec1))
            vec2 += jitter[1] * np.random.normal(size=len(vec2))

        # Plot the points
        keys = ['oo', 'io', 'oi', 'ii']
        lak = {'i': 'imp', 'o': 'obs'}
        ixs = {'ii': ix_ii, 'io': ix_io, 'oi': ix_oi, 'oo': ix_oo}
        color = {'oo': 'grey', 'ii': 'red', 'io': 'orange',
                 'oi': 'lime'}
        if plot_points:
            for ky in keys:
                ix = ixs[ky]
                lab = lak[ky[0]] + "/" + lak[ky[1]]
                ax.plot(vec1[ix], vec2[ix], 'o', color=color[ky],
                        label=lab, alpha=0.6)

        # Plot the lowess fits
        for ky in keys:
            ix = ixs[ky]
            if len(ix) < lowess_min_n:
                continue
            if ky in lowess_args:
                la = lowess_args[ky]
            else:
                la = {}
            ix = ixs[ky]
            lfit = lowess(vec2[ix], vec1[ix], **la)
            if plot_points:
                ax.plot(lfit[:, 0], lfit[:, 1], '-', color=color[ky],
                        alpha=0.6, lw=4)
            else:
                lab = lak[ky[0]] + "/" + lak[ky[1]]
                ax.plot(lfit[:, 0], lfit[:, 1], '-', color=color[ky],
                        alpha=0.6, lw=4, label=lab)

        ha, la = ax.get_legend_handles_labels()
        pad = 0.0001 if plot_points else 0.5
        leg = fig.legend(ha, la, 'center right', numpoints=1,
                         handletextpad=pad)
        leg.draw_frame(False)

        ax.set_xlabel(col1_name)
        ax.set_ylabel(col2_name)

        return fig

    def fit_scatterplot(self, col_name, lowess_args={},
                        lowess_min_n=40, jitter=None,
                        plot_points=True, ax=None):
        """
        Create a scatterplot between the observed or imputed values of
        a variable and the corresponding fitted values.

        Parameters:
        -----------
        col1_name : string
            The variable to be plotted on the horizontal axis.
        lowess_args : dict-like
            Keyword arguments passed to lowess fit.  A dictionary of
            dictionaries, keys are 'o' and 'i' denoting 'observed' and
            'imputed', respectively.
        lowess_min_n : integer
            Minimum sample size to plot a lowess fit
        jitter : float or tuple
            Standard deviation for jittering points in the plot.
            Either a single scalar applied to both axes, or a tuple
            containing x-axis jitter and y-axis jitter, respectively.
        plot_points : bool
            If True, the data points are plotted.
        ax : matplotlib axes object
            Axes on which to plot, created if not provided.

        Returns:
        --------
        The matplotlib figure on which the plot is drawn.
        """

        from statsmodels.graphics import utils as gutils
        from statsmodels.nonparametric.smoothers_lowess import lowess

        if ax is None:
            fig, ax = gutils.create_mpl_ax(ax)
        else:
            fig = ax.get_figure()

        ax.set_position([0.1, 0.1, 0.7, 0.8])

        ixi = self.ix_miss[col_name]
        ixo = self.ix_obs[col_name]

        vec1 = np.asarray(self.data[col_name])

        # Fitted values
        formula = self.conditional_formula[col_name]
        endog, exog = patsy.dmatrices(formula, self.data,
                                      return_type="dataframe")
        results = self.results[col_name]
        vec2 = results.predict(exog=exog)

        if jitter is not None:
            if np.isscalar(jitter):
                jitter = (jitter, jitter)
            vec1 += jitter[0] * np.random.normal(size=len(vec1))
            vec2 += jitter[1] * np.random.normal(size=len(vec2))

        # Plot the points
        keys = ['o', 'i']
        ixs = {'o': ixo, 'i': ixi}
        lak = {'o': 'obs', 'i': 'imp'}
        color = {'o': 'orange', 'i': 'lime'}
        if plot_points:
            for ky in keys:
                ix = ixs[ky]
                ax.plot(vec1[ix], vec2[ix], 'o', color=color[ky],
                        label=lak[ky], alpha=0.6)

        # Plot the lowess fits
        for ky in keys:
            ix = ixs[ky]
            if len(ix) < lowess_min_n:
                continue
            if ky in lowess_args:
                la = lowess_args[ky]
            else:
                la = {}
            ix = ixs[ky]
            lfit = lowess(vec2[ix], vec1[ix], **la)
            if plot_points:
                ax.plot(lfit[:, 0], lfit[:, 1], '-', color=color[ky],
                        alpha=0.6, lw=4)
            else:
                lab = lak[ky[0]] + "/" + lak[ky[1]]
                ax.plot(lfit[:, 0], lfit[:, 1], '-', color=color[ky],
                        alpha=0.6, lw=4, label=lab)

        ha, la = ax.get_legend_handles_labels()
        pad = 0.0001 if plot_points else 0.5
        leg = fig.legend(ha, la, 'center right', numpoints=1,
                         handletextpad=pad)
        leg.draw_frame(False)

        ax.set_xlabel(col_name + " observed or imputed")
        ax.set_ylabel(col_name + " fitted")

        return fig

    def hist(self, col_name, ax=None, imp_hist_args={},
             obs_hist_args={}, all_hist_args={}):
        """
        Produce a set of three overlaid histograms showing the
        marginal distributions of the observed values, imputed
        values, and all values for a given variable.

        Parameters:
        -----------
        col_name : string
            The name of the variable to be plotted.
        ax : matplotlib axes
            An axes on which to draw the histograms.  If not provided,
            one is created.
        imp_hist_args : dict
            Keyword arguments to be passed to pyplot.hist when
            creating the histogram for imputed values.
        obs_hist_args : dict
            Keyword arguments to be passed to pyplot.hist when
            creating the histogram for observed values.
        all_hist_args : dict
            Keyword arguments to be passed to pyplot.hist when
            creating the histogram for all values.

        Returns:
        --------
        The matplotlib figure on which the histograms were drawn
        """

        from statsmodels.graphics import utils as gutils
        from matplotlib.colors import LinearSegmentedColormap

        if ax is None:
            fig, ax = gutils.create_mpl_ax(ax)
        else:
            fig = ax.get_figure()

        ax.set_position([0.1, 0.1, 0.7, 0.8])

        ixm = self.ix_miss[col_name]
        ixo = self.ix_obs[col_name]

        imp = self.data[col_name].iloc[ixm]
        obs = self.data[col_name].iloc[ixo]

        for di in imp_hist_args, obs_hist_args, all_hist_args:
            if 'histtype' not in di:
                di['histtype'] = 'step'

        ha, la = [], []
        if len(imp) > 0:
            h = ax.hist(np.asarray(imp), **imp_hist_args)
            ha.append(h[-1][0])
            la.append("Imp")
        h1 = ax.hist(np.asarray(obs), **obs_hist_args)
        h2 = ax.hist(np.asarray(self.data[col_name]), **all_hist_args)
        ha.extend([h1[-1][0], h2[-1][0]])
        la.extend(["Obs", "All"])

        leg = fig.legend(ha, la, 'center right', numpoints=1)
        leg.draw_frame(False)

        ax.set_xlabel(col_name)
        ax.set_ylabel("Frequency")

        return fig

    def perturb_bootstrap(self, vname):
        """
        Perturbs the model's parameters using a bootstrap.
        """

        endog_obs, exog_obs, exog_miss =\
                   self.get_split_data(vname)

        m = len(endog_obs)
        rix = np.random.randint(0, m, m)
        endog_boot = endog_obs[rix]
        exog_boot = exog_obs[rix, :]
        klass = self.model_class[vname]
        self.models[vname] = klass(endog_boot, exog_boot,
                                  **self.init_args)
        self.results[vname] = self.models[vname].fit(**self.fit_args)
        self.params[vname] = self.results[vname].params

    def perturb_gaussian(self, vname):
        """
        Perturbs the model's parameters by sampling from the Gaussian
        approximation to the sampling distribution of the parameter
        estimates.  Optionally, the scale parameter is perturbed by
        sampling from its asymptotic Chi^2 sampling distribution.
        """

        endog_obs, exog_obs, exog_miss =\
                   self.get_split_data(vname)

        klass = self.model_class[vname]
        formula = self.conditional_formula[vname]
        self.models[vname] = klass.from_formula(formula,
                                            self.data,
                                            **self.init_args[vname])
        self.results[vname] = self.models[vname].fit(
            **self.fit_args[vname])

        if self.scale_method[vname] == "fix":
            scale_pert = 1.
        elif self.scale_method[vname] == "perturb_chi2":
            u = np.random.chisquare(float(self.results.df_resid))
            scale_pert = u / float(self.results.df_resid)
        else:
            raise ValueError("unknown scale perturbation method")

        # Can use scipy here in future
        #from scipy.stats import multivariate_normal
        cov = self.results[vname].cov_params().copy()
        cov *= scale_pert
        norm = _multivariate_normal(mean=self.results[vname].params,
                                    cov=cov)
        self.params[vname] = norm.rvs()

    def perturb_params(self, vname):

        if self.perturbation_method[vname] == "gaussian":
            self.perturb_gaussian(vname)
        elif self.perturbation_method[vname] == "boot":
            self.perturb_bootstrap(vname)
        else:
            raise ValueError("unknown perturbation method")

    def impute(self, vname):
        # Wrap this in case we later add additional imputation
        # methods.
        self.impute_pmm(vname)

    def update(self, vname):
        """
        Update a single variable.  This is a two-step process in which
        first the parameters are perturbed, then the missing values
        are re-imputed.
        """

        self.perturb_params(vname)
        self.impute(vname)

    def impute_pmm(self, vname):
        """
        Use predictive mean matching to impute missing values.

        Notes
        -----
        The `perturb_params` method must be called first to define the
        model.
        """

        k_pmm = self.k_pmm

        endog_obs, exog_obs, exog_miss =\
                   self.get_split_data(vname)

        # Predict imputed variable for both missing and nonmissing
        # observations
        model = self.models[vname]
        pendog_obs = model.predict(self.params[vname], exog_obs)
        pendog_miss = model.predict(self.params[vname],
                                         exog_miss)

        # Jointly sort the observed and predicted endog values for the
        # cases with observed values.
        ii = np.argsort(pendog_obs)
        endog_obs = endog_obs[ii]
        pendog_obs = pendog_obs[ii]

        # Find the closest match to the predicted endog values for
        # cases with missing endog values.
        ix = np.searchsorted(pendog_obs, pendog_miss)

        # Get the indices for the closest k_pmm values on
        # either side of the closest index.
        ixm = ix[:, None] +  np.arange(-k_pmm, k_pmm)[None, :]

        # Account for boundary effects
        msk = np.nonzero((ixm < 0) | (ixm > len(endog_obs) - 1))
        ixm = np.clip(ixm, 0, len(endog_obs) - 1)

        # Get the distances
        dx = pendog_miss[:, None] - pendog_obs[ixm]
        dx = np.abs(dx)
        dx[msk] = np.inf

        # Closest positions in ix, row-wise.
        dxi = np.argsort(dx, 1)[:, 0:k_pmm]

        # Choose a column for each row.
        ir = np.random.randint(0, k_pmm, len(pendog_miss))

        # Unwind the indices
        jj = np.arange(dxi.shape[0])
        ix = dxi[[jj, ir]]
        iz = ixm[[jj, ix]]

        imputed_miss = np.array(endog_obs[iz])

        self.store_changes(vname, imputed_miss)


_mice_model_example_1 = """
    >>> imodel = mice.MICE_model(data, 'y ~ x1 + x2 + x3', sm.OLS)
    >>> results = []
    >>> for j in range(100):
            results.append(imodel.next())
    >>> params1 = [x.params[1] for x in results]
    >>> plt.hist(params1)
"""

class MICE_model(object):

    __doc__ = """\
    An iterator that returns models fit to imputed data sets.

    Parameters
    ----------
    data : MICE_data instance
        The data set, in the form of a MICE_data object.
    analysis_formula : string
        Formula for the analysis model.
    analysis_class : statsmodels model
        Model class to be fit to imputed data sets.
    n_skip : integer
        Number of imputation cycles to perform before fitting
        the analysis models.
    init_args : dict-like
        Additional parameters for statsmodels model instance.
    fit_args : dict-like
        Additional parameters for statsmodels fit instance.

    Examples
    --------
    Fit the model to 100 imputed data sets, then make a histogram
    of the parameter value in position 1.
    %(mice_model_example_1)s
    """ % {'mice_model_example_1': _mice_model_example_1}

    def __init__(self, data, analysis_formula, analysis_class,
                 n_skip=10, init_args={}, fit_args={}):

        if not isinstance(data, MICE_data):
            raise ValueError("data argument must be an instance of MICE_data")
        self.data = data
        self.analysis_formula = analysis_formula

        if not issubclass(analysis_class, statsmodels.base.model.Model):
            raise ValueError("analysis_class must be a statsmodels model")
        self.analysis_class = analysis_class

        self.init_args = init_args
        self.fit_args = fit_args
        self.n_skip = n_skip

    def __iter__(self):
        return self

    def next(self):

        self.data.update_all(n_iter=self.n_skip + 1)

        model = self.analysis_class.from_formula(self.analysis_formula,
                                                 self.data.data,
                                                 **self.init_args)
        result = model.fit(**self.fit_args)

        return result


_mice_example_1 = """
    >>> imp = mice.MICE_data(data)
    >>> fml = 'y ~ x1 + x2 + x3 + x4'
    >>> mice = mice.MICE_model(fml, sm.OLS, imp)
    >>> mice.burnin(10)
    >>> mice.run(10, 5)
    >>> results = mice.combine()
    >>> print results.summary()

                              Results: MICE
    =================================================================
    Method:                    MICE       Sample size:           1000
    Model:                     OLS        Scale                  1.00
    Dependent variable:        y          Num. imputations       10
    -----------------------------------------------------------------
               Coef.  Std.Err.    t     P>|t|   [0.025  0.975]  FMI
    -----------------------------------------------------------------
    Intercept -0.0234   0.0318  -0.7345 0.4626 -0.0858  0.0390 0.0128
    x1         1.0305   0.0578  17.8342 0.0000  0.9172  1.1437 0.0309
    x2        -0.0134   0.0162  -0.8282 0.4076 -0.0451  0.0183 0.0236
    x3        -1.0260   0.0328 -31.2706 0.0000 -1.0903 -0.9617 0.0169
    x4        -0.0253   0.0336  -0.7520 0.4521 -0.0911  0.0406 0.0269
    =================================================================
"""

class MICE(object):

    __doc__ = """\
    Use Multiple Imputation with Chained Equations to fit a model when
    some data values are missing.

    Parameters
    ----------
    model_formula : string
        The model formula to be fit to the imputed data sets.
    model_class : statsmodels model
        The model to be fit to the imputed data sets.
    data : MICE_data instance
        MICE_data object containing the data set for which
        missing values will be imputed

    Examples
    --------
    Simple example using defaults:
    %(mice_example_1)s
    """ % {'mice_example_1' : _mice_example_1}

    def __init__(self, model_formula, model_class, data,
                 init_args={}, fit_args={}):
        self.model_formula = model_formula
        self.model_class = model_class
        self.init_args = init_args
        self.fit_args = fit_args
        self.data = data

        model_chain = MICE_model(data, model_formula,
                                    model_class, 0,
                                    init_args, fit_args)
        self.model_chain = model_chain

    def burnin(self, n_burnin):
        """
        Impute a given number of data sets and discard the results.

        Parameters
        ----------
        n_burnin : int
            The number of update cycles to perform.  Each update cycle
            updates each variable in the data set with one or more
            missing values.
        """

        self.data.update_all(n_iter=n_burnin)

    def run(self, n_imputations=20, n_skip=10):
        """
        Generates analysis model results.

        Parameters
        ----------
        n_imputations : int
            Number of imputed datasets to generate.
        n_skip : int
            Number of imputed datasets to skip between consecutive
            imputed datasets that are used for analysis.
        """

        self.model_chain.n_skip = n_skip

        results_list = []
        for j in range(n_imputations):
            result = self.model_chain.next()
            results_list.append(result)
        self.results_list = results_list

        self.endog_names = result.model.endog_names
        self.exog_names = result.model.exog_names

    def combine(self):
        """
        Pools MICE imputation results to produce overall estimates and
        standard errors.

        Returns a MICEResults instance.
        """

        # Extract a few things from the models that were fit to
        # imputed data sets.
        params_list = []
        cov_list = []
        scale_list = []
        for results in self.results_list:
            params_list.append(np.asarray(results.params))
            cov_list.append(np.asarray(results.cov_params()))
            scale_list.append(results.scale)
        params_list = np.asarray(params_list)
        scale_list = np.asarray(scale_list)

        # The estimated parameters for the MICE analysis
        params = params_list.mean(0)

        # The average of the within-imputation covariances
        cov_within = sum(cov_list) / len(cov_list)

        # The between-imputation covariance
        cov_between = np.cov(params_list.T)

        # The estimated covariance matrix for the MICE analysis
        f = 1 + 1 / float(len(self.results_list))
        cov_params = cov_within + f * cov_between

        # Fraction of missing information
        fmi = f * np.diag(cov_between) / np.diag(cov_params)

        # Set up a results instance
        scale = np.mean(scale_list)
        results = MICEResults(self, params, cov_params / scale)
        results.scale = scale
        results.frac_miss_info = fmi
        results.exog_names = self.exog_names
        results.endog_names = self.endog_names
        results.model_class = self.model_class

        return results


class MICEResults(statsmodels.base.model.LikelihoodModelResults):

    def __init__(self, model, params, normalized_cov_params):

        super(MICEResults, self).__init__(model, params,
                                          normalized_cov_params)

    def summary(self, title=None, alpha=.05):
        """
        Summarize the results of running MICE.

        Parameters
        -----------
        title : string, optional
            Title for the top table. If not None, then this replaces
            the default title
        alpha : float
            Significance level for the confidence intervals

        Returns
        -------
        smry : Summary instance
            This holds the summary tables and text, which can be
            printed or converted to various output formats.
        """

        from statsmodels.iolib import summary2
        from statsmodels.compat.collections import OrderedDict

        smry = summary2.Summary()
        float_format = "%8.3f"

        info = OrderedDict()
        info["Method:"] = "MICE"
        info["Model:"] = self.model_class.__name__
        info["Dependent variable:"] = self.endog_names
        info["Sample size:"] = "%d" % self.model.data.data.shape[0]
        info["Scale"] = "%.2f" % self.scale
        info["Num. imputations"] = "%d" % len(self.model.results_list)

        smry.add_dict(info, align='l', float_format=float_format)

        param = summary2.summary_params(self, alpha=alpha)
        param["FMI"] = self.frac_miss_info

        smry.add_df(param, float_format=float_format)
        smry.add_title(title=title, results=self)

        return smry
