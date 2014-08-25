"""
This module implements the Multiple Imputation through Chained Equations (MICE)
approach to handling missing data. This approach has 3 general steps:

1) Simulate observations using a user specified conditional model.
2) Fit the model of interest to a complete, simulated dataset.
3) Repeat N times and combine the N models according to Rubin's Rules.

Imputer instances, for imputing a single missing variable,
are specified with a (statsmodels) conditional model
(default is OLS with all other variables). A MICE instance is specified with
a model of interest together with its corresponding formulae. The results are
combined using the `combine` method.

Reference for Rubin's Rules and Multiple Imputation:

J L Schafer: "Multiple Imputation: A Primer", Stat Methods Med Res, 1999.

Reference for Gaussian Approximation to the Posterior:

T E Raghunathan et al.: "A Multivariate Technique for Multiply
Imputing Missing Values Using a Sequence of Regression Models",
Survey Methodology, 2001.

Reference for Predictive Mean Matching:

SAS Institute: "Predictive Mean Matching Method for Monotone Missing Data",
SAS 9.2 User's Guide, 2014.

Reference for MICE Design in R package mi:

A Gelman et al.: "Multiple Imputation with Diagnostics (mi) in R: Opening
Windows into the Black Box", Journal of Statistical Software, 2009.

#TODO: Add reference http://biomet.oxfordjournals.org/content/86/4/948.full.pdf
#TODO: Change md to mod, mdf to rslt
"""

import operator
import pandas as pd
import numpy as np
import patsy
import statsmodels.api as sm
import statsmodels
#from statsmodels.tools.decorators import cache_readonly
#from scipy import stats
import copy

class ImputedData(object):
    """
    Stores missing data information and supports functionality for inserting
    values in missing data slots.

    Parameters
    ----------
    data : array-like object
        Needs to support transformation to pandas dataframe. Missing value
        encoding is handled by pandas DataFrame class.
    method : string, dict, or None
        Default imputation method.  Allowed methods are "pmm", "gaussian", and 
        "bootstrap".  If a string, this is the default imputation method for 
        all variables.  If a dictionary, it maps column names to the imputation
        method used for the column.  Defaults to "pmm".

    Note
    ----    
    Can create Imputers directly via new_imputer method. By default,
    imputers are created for each variable using OLS using all other variables
    as predictors.
    """
    
    def __init__(self, data, method=None):
        # May not need to make copies
        self.data = pd.DataFrame(data)
        # Drop observations where all variables are missing.
        self.data = self.data.dropna(how='all').reset_index(drop=True)
#        df.reset_index(drop=True)
        self.columns = {}
        self.implist = []
        for col in self.data.columns:
            self.columns[col] = MissingDataInfo(self.data[col])
            umeth = "pmm"
            if type(method) is dict and col in method:
                umeth = method[col]
            elif type(method) is str:
                umeth = method
            self.new_imputer(col, method=umeth)
        # Fill missing values with column-wise mean.
        self.data = self.data.fillna(self.data.mean())

    def new_imputer(self, endog_name, method="pmm", k_pmm=20, 
                    formula=None, model_class=None, init_args={}, fit_args={}, 
                    perturb_method="gaussian", alt_distribution=None, 
                    scale_method="fix", scale_value=None, transform=None, 
                    inv_transform=None):
        # TODO: Look into kwargs for method details such as k_pmm
        """
        Specify the imputation process for a single variable.

        Parameters
        ----------
        endog_name : string
            Name of the variable to be imputed.
        method : string
            May take on values "pmm", "gaussian", or "bootstrap". Determines
            imputation method, see mice.Imputer.
        k_pmm : int
            Determines number of observations from which to draw imputation
            when using predictive mean matching. See mice.Imputer.
        init_args : Dictionary
            Additional arguments for statsmodels model instance.
        formula : string
            Conditional formula for imputation. Defaults to model with main
            effects for all other variables in dataset.
        model_class : statsmodels model
            Conditional model for imputation. Defaults to OLS.
        alt_distribution : scipy.random instance
            Controls the scale/location family to use as the asymptotic
            distribution of the imputed variable. For use with method
            impute_asymptotic_bayes.
        scale_method : string
            Governs the type of perturbation given to the scale parameter.
        scale_value : float
            Fixed value of scale parameter to use in simulation of data.
        transform : function
            Transformation to apply to endogeneous variable prior to imputation.
            Should be an invertible function on the domain of the variable.
        inv_transform : function
            Functional inverse of `transform`.

        See Also
        --------
        mice.Imputer
        """
        if model_class is None:
            model_class = sm.OLS
        if formula is None:
            main_effects = [x for x in self.data.columns if x != endog_name]
            formula = endog_name + " ~ " + " + ".join(main_effects)
        imp = Imputer(formula, model_class, self, method=method, k_pmm=k_pmm,
                      init_args=init_args, fit_args=fit_args,
                      scale_method=scale_method, scale_value=scale_value,
                      transform=transform, inv_transform=inv_transform)
        self.implist.append(imp)

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

        ix = self.columns[col].ix_miss
        if len(ix) > 0:
            self.data[col].iloc[ix] = vals

    def get_data_from_formula(self, formula):
        """
        Use formula to construct endog and exog split by missing status.

        Parameters
        ----------
        formula : string
            Patsy formula for the Imputer object's conditional model.

        Returns
        -------
        endog_obs : DataFrame
            Observed values of the variable to be imputed.
        exog_obs : DataFrame
            Current values of the predictors where the variable to be Imputed 
            is observed.
        exog_miss : DataFrame
            Current values of the predictors where the variable to be Imputed 
            is missing.
        """
        endog, exog = patsy.dmatrices(formula, self.data, 
                                      return_type="dataframe")
        if len(endog.design_info.term_names) > 1:
            endog_name = tuple(endog.design_info.term_names)
        else:
            endog_name = endog.design_info.term_names[0]
        # Turn endog into a series since it is always 1D
        endog = endog.iloc[:, 0]                                                  
        ix = self.columns[endog_name].ix_obs
        endog_obs = endog[ix]
        exog_obs = exog.iloc[ix,:]
        ix = self.columns[endog_name].ix_miss
        exog_miss = exog.iloc[ix]            
        return endog_obs, exog_obs, exog_miss

    def plot_missing_pattern(self, ax=None, row_order="pattern",
                             column_order="pattern",
                             hide_complete_rows=False,
                             hide_complete_columns=False,
                             color_row_patterns=False):
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
            ix = self.columns[col].ix_miss
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
            # Force rows with no missing to bottom
            ix1 = np.flatnonzero((miss == 0).any(1))
            ix0 = np.flatnonzero((miss != 0).all(1))
            miss_c = miss[ix1, :]
            miss_c -= miss_c.mean(1)[:, None]
            u, s, vt = np.linalg.svd(miss_c, 0)
            ix = np.argsort(u[:, 0])
            ix = np.concatenate((ix1[ix], ix0))
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
                              plot_points=True, ax=None):
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

        ix1i = self.columns[col1_name].ix_miss
        ix1o = self.columns[col1_name].ix_obs
        ix2i = self.columns[col2_name].ix_miss
        ix2o = self.columns[col2_name].ix_obs

        ix_ii = np.intersect1d(ix1i, ix2i)
        ix_io = np.intersect1d(ix1i, ix2o)
        ix_oi = np.intersect1d(ix1o, ix2i)
        ix_oo = np.intersect1d(ix1o, ix2o)

        vec1 = np.asarray(self.data[col1_name])
        vec2 = np.asarray(self.data[col2_name])

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

        ixm = self.columns[col_name].ix_miss
        ixo = self.columns[col_name].ix_obs

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
        
class Imputer(object):
    """
    Object to conduct imputations for a single variable.

    Parameters
    ----------
    formula : string
        Conditional formula used for imputation.
    model_class : statsmodels model
        Conditional model used for imputation.
    data : ImputedData object
        The parent ImputedData object to which this Imputer object is attached
    method : string or None
        May take on values "pmm", "gaussian", or "bootstrap". Determines
        imputation method.
    k_pmm : int
        Determines number of observations from which to draw imputation
        when using predictive mean matching. See method impute_pmm.
    init_args : Dictionary
        Additional parameters to pass to init method when creating model.
    fit_args : Dictionary
        Additional parameters for statsmodels fit instance.
    alt_distribution : scipy.random class
        Controls the scale/location family to use as the asymptotic
        distribution of the imputed variable.    
    scale_method : string
        Governs the type of perturbation given to the scale parameter. See
        method perturb_params.
    scale_value : float
        Fixed value of scale parameter to use in simulation of data. See
        method perturb_params.
    transform : function
        Transformation to apply to endogeneous variable prior to imputation.
        Should be an invertible function on the domain of the variable.
    inv_transform : function
        Functional inverse of `transform`

    Attributes
    ----------
    endog_name : string
        Name of variable to be imputed.
    num_missing : int
        Number of missing values.

    Note
    ----
    All params are saved as attributes.

    """
    
    def __init__(self, formula, model_class, data, method="pmm",
                 perturb_method="gaussian", k_pmm=1, init_args={}, fit_args={},
                 alt_distribution=None, scale_method="fix", scale_value=None,
                 transform=None, inv_transform=None):
        self.data = data
        self.formula = formula
        self.model_class = model_class
        self.init_args = init_args
        self.fit_args = fit_args
        self.endog_name = str(self.formula.split("~")[0].strip())
        self.num_missing = len(self.data.columns[self.endog_name].ix_miss)
        self.alt_distribution = alt_distribution
        self.scale_method = scale_method
        self.scale_value = scale_value
        self.method = method
        self.k_pmm = k_pmm
        self.transform = transform
        self.inv_transform = inv_transform
        self.perturb_method = perturb_method

    def perturb_params(self, mdf):
        """
        Perturbs the model's coefficients and scale parameter.

        Parameters
        ----------
        mdf : Statsmodels results class instance.
            The current fitted conditional model for the variable managed by
            this Imputer instance.

        Returns
        -------
        params_pert : array
            Perturbed model parameters.
        scale_pert : float
            Perturbed scale parameter multiplier.

        Note: Bootstrap perturbation still experimental.
        """

        # TODO: switch to scipy
        if self.perturb_method == "boot":
            endog_obs, exog_obs, exog_miss = self.data.get_data_from_formula(
                                                                self.formula)
            m = len(endog_obs)
            rix = np.random.randint(0, m, m)
            endog_sample = endog_obs[rix]
            exog_sample = exog_obs.iloc[rix,:]
            md = self.model_class(endog_sample, exog_sample, **self.init_args)
            mdf = md.fit(**self.fit_args)
            params_pert = mdf.params
            scale_pert = 1.
        elif self.perturb_method == "gaussian":
            params_pert = mdf.params.copy()
            covmat = mdf.cov_params()
            covmat_sqrt = np.linalg.cholesky(covmat)
            if self.scale_method == "fix":
                if self.scale_value is None:
                    scale_pert = 1.
                else:
                    scale_pert = self.scale_value
            elif self.scale_method == "perturb_chi2":
                u = np.random.chisquare(float(mdf.df_resid))
                scale_pert = float(mdf.df_resid) / float(u)
            p = len(params_pert)
            params_pert += np.dot(covmat_sqrt,
                             np.random.normal(0, np.sqrt(mdf.scale * scale_pert),
                                              p))
        else:
            raise(ValueError("Unknown perturbation method"))

        return params_pert, scale_pert

    def impute_asymptotic_bayes(self):
        """
        Use Gaussian approximation to posterior distribution to simulate data.

        Fills in missing values of input data. User may also choose a different
        approximating location/scale family by specifying alt_distribution in
        Imputer initialization.
        """
        endog_obs, exog_obs, exog_miss = self.data.get_data_from_formula(
                                                                self.formula)
        if self.transform is not None and self.inv_transform is not None:
            endog_obs = self.transform(endog_obs)
        md = self.model_class(endog_obs, exog_obs, **self.init_args)
        mdf = md.fit(**self.fit_args)
        params, scale_per = self.perturb_params(mdf)
        new_rv = md.get_distribution(params=params, exog=exog_miss,
                                     model_class=self.alt_distribution,
                                     scale=np.sqrt(scale_per * mdf.scale))
        new_endog = new_rv.rvs(size=len(exog_miss))
        if self.transform is not None and self.inv_transform is not None:
            new_endog = self.inv_transform(new_endog)
        self.data.store_changes(self.endog_name, new_endog)

    def impute_pmm(self, pmm_neighbors=10):
        """
        Use predictive mean matching to simulate data.

        Parameters
        ----------
        pmm_neighbors : int
            Number of neighbors in prediction space to select imputations from.
            Defaults to 10 (select closest neighbor).

        Note: Fills in missing values of input data. Predictive mean matching
        picks an observation randomly from the observed value of the k-nearest
        predictions of the endogenous variable. Naturally, the candidate
        neighbors must have observed endogenous values.
        """
        endog_obs, exog_obs, exog_miss = self.data.get_data_from_formula(
                                                                self.formula)
        if self.transform is not None and self.inv_transform is not None:
            endog_obs = self.transform(endog_obs)
        md = self.model_class(endog_obs, exog_obs, **self.init_args)
        mdf = md.fit(**self.fit_args)
        params, scale_per = self.perturb_params(mdf)
        # Predict imputed variable for both missing and nonmissing observations
        pendog_obs = md.predict(params, exog_obs)
        pendog_miss = md.predict(params, exog_miss)
        
        # Jointly sort the observed and predicted endog values for the
        # cases with observed values.
        ii = np.argsort(pendog_obs)
        endog_obs = endog_obs[ii]
        pendog_obs = pendog_obs[ii]

        # Find the closest match to the predicted endog values for cases
        # with missing endog values.
        ix = np.searchsorted(pendog_obs, pendog_miss)

        # Get the indices for the closest pmm_neighbors values on either
        # side of the closest index.
        ixm = ix[:, None] +  np.arange(-pmm_neighbors, pmm_neighbors)[None, :]

        # Account for boundary effects
        msk = np.nonzero((ixm < 0) | (ixm > len(endog_obs) - 1))
        ixm = np.clip(ixm, 0, len(endog_obs) - 1)

        # Get the distances
        dx = pendog_miss[:, None] - pendog_obs[ixm]
        dx = np.abs(dx)
        dx[msk] = np.inf

        # Closest positions in ix, row-wise.
        dxi = np.argsort(dx, 1)[:, 0:pmm_neighbors]

        # Choose a column for each row.
        ir = np.random.randint(0, pmm_neighbors, len(pendog_miss))

        # Unwind the indices
        jj = np.arange(dxi.shape[0])
        ix = dxi[[jj, ir]]
        iz = ixm[[jj, ix]]
        
        imputed_miss = np.array(endog_obs[iz])
        if self.transform is not None and self.inv_transform is not None:
            imputed_miss = self.inv_transform(imputed_miss)
        self.data.store_changes(self.endog_name, imputed_miss)

class ImputerChain(object):
    """
    An iterator that returns imputed data sets produced using the MICE
    (multiple imputation by chained equations) procedure.

    Parameters
    ----------
    imputer_list : list
        List of Imputer objects, one for each variable to be imputed.

    Attributes
    ----------
    data : pandas DataFrame
        Underlying data to be modified. Root copy of data is stored in original
        ImputedData object.

    Note
    ----
    All imputers must refer to the same data object. See mice.MICE.run
    for iterator call. This class does imputation and returns the imputed data
    sets, it does not fit the analysis model. See the "next" method for details.
    """
    
    def __init__(self, imputer_list):
        self.imputer_list = imputer_list
        # Impute variable with least missing observations first
        self.imputer_list.sort(key=operator.attrgetter('num_missing'))
        # All imputers must refer to the same data object
        self.data = imputer_list[0].data.data

    def __iter__(self):
        return self

    def next(self):
        """
        Makes this class an iterator that returns imputed datasets after
        cycling through all contained imputers.

        Returns
        -------
        data : pandas DataFrame
            Dataset with imputed values saved after invoking each Imputer
            object in imputer_list.

        Note: Returned datsets are not saved unless specified in the iterator
        call.
        """
        for im in self.imputer_list:
            if im.method=="gaussian":
                im.impute_asymptotic_bayes()
            elif im.method == "pmm":
                im.impute_pmm(im.k_pmm)
            elif im.method == "bootstrap":
                im.impute_bootstrap()
        return self.data

class AnalysisChain(object):
    """
    An iterator that returns the fitted model of interest given a MICE imputed
    dataset.

    Parameters
    ----------
    imputer_chain : ImputerChain instance
    analysis_formula : string
        Pandas formula for model to be analyzed.
    analysis_class : statsmodels model
        Model to be analyzed.
    skipnum : integer
        Number of imputation cycles to perform before storing dataset.
    save : string
        Specifies option for saving. save=full saves all, save=None saves nothing.
    init_args : Dictionary
        Additional parameters for statsmodels model instance.
    fit_args : Dictionary
        Additional parameters for statsmodels fit instance.

    Note
    ----
    See mice.MICE.run for iterator call. Datasets to be used for analysis
    are chosen after an initial burnin period where no imputed data is used and
    also after skipping a set number of imputations for each iteration. See the
    "next" method for details.
    """

    def __init__(self, imputer_chain, analysis_formula, analysis_class,
                 skipnum=10, save=None, init_args={}, fit_args={}):
        self.imputer_chain = imputer_chain
        self.analysis_formula = analysis_formula
        self.analysis_class = analysis_class
        self.init_args = init_args
        self.fit_args = fit_args
        self.skipnum = skipnum
        self.save = save
        self.iter = 0

    def __iter__(self):
        return self

    def next(self):
        """
        Makes this class an iterator that returns the fitted analysis model.

        Returns
        -------
        mdf : statsmodels fitted model
            Fitted model of interest on imputed dataset that has passed all
            skip and burnin criteria

        Note: If save="full", imputed datasets are saved in the format
        "mice_'iteration number'.csv". Handles skipping of imputation
        iterations, burnin period of unconsidered imputation iterations, and
        whether or not to save the datasets to which an analysis model is fit.
        """
        #TODO: Possibly add transform here instead of in Imputer.
        for i in range(self.skipnum + 1):
            data = self.imputer_chain.next()
            print i
        md = self.analysis_class.from_formula(self.analysis_formula,
                                              data, **self.init_args)
        mdf = md.fit(**self.fit_args)
        if self.save=="full":
            fname = "%s_%d.csv" % ('mice_', self.iter)
            data.to_csv(fname, index=False)
            self.iter += 1
        return mdf

class MICE(object):
    """
    Fits the analysis model to each imputed dataset and combines the
    results using Rubin's rule.

    Parameters
    ----------
    analysis_formula : string
        Formula for model of interest to be fitted.
    analysis_class : statsmodels model
        Statsmodels model of interest.
    impdata : mice.ImputedData instance
        ImputedData object which contains an implist that is fully populated
        by the desired Imputer objects.

    Attributes
    ----------
    N : int
        Total number of observations.
    modlist : list
        Contains analysis model fitted on each imputed dataset.
    df : list
        Contains degees of freedom for each parameter in the analysis model.
    fmi : list
        Contains fraction of missing information for each parameter in the
        analysis model.
    exog_names : list
        Contains names of exogenous variables in the analysis model.
    endog_name : string
        Name of endogenous model in the analysis model.

    Example
    --------
    >>>impdata = mice.ImputedData(data)
    >>>impdata.new_imputer("x2", method="pmm", k_pmm=20)
    >>>impdata.new_imputer("x1", method="pmm", model_class=sm.Logit, k_pmm=20)
    >>>impcomb = mice.MICE("x1 ~ x2", sm.Logit,impdata)
    >>>impcomb.run(20, 10)
    >>>p1 = impcomb.combine()
    >>>print p1.summary()

                               Results: MICE
==========================================================================
Method:              MICE              Dependent variable:             x1
Model:               Logit             Sample size:                    978
--------------------------------------------------------------------------
           Coef.  Std.Err.    t     P>|t|   [0.025  0.975]    Df     FMI
--------------------------------------------------------------------------
Intercept  2.2639   0.2182  10.3754 0.0000  1.8322  2.6956 130.0876 0.3412
x2        -3.8863   0.3304 -11.7609 0.5000 -4.5393 -3.2332 145.9336 0.3187
==========================================================================

    Note
    ----
    Calls mice.Imputer_Chain and mice.AnalysisChain to handle imputation
    and fitting of analysis models to the correct imputed datasets,
    respectively.
    """
    
    def __init__(self, analysis_formula, analysis_class, impdata,
                 init_args={}, fit_args={}):
        self.imputer_list = impdata.implist
        self.analysis_formula = analysis_formula
        self.analysis_class = analysis_class
        self.init_args = init_args
        self.fit_args = fit_args
        self.N = len(impdata.implist[0].data.data)

    def run(self, num_ds=20, skipnum=10, burnin=30, save=False, disp=True):
        """
        Generates analysis model results.

        Parameters
        ----------
        num_ds : int
            Number of imputed datasets to fit.
        skipnum : int
            Number of imputed datasets to skip between imputed datasets that
            are used for analysis. This is done to give the conditional
            distribution time to settle down. The literature says that the
            MICE procedure converges in distribution fairly quickly;
            standard practice is to set this number to be around ten.
        burnin : int
            Number of iterations to throw away before ever starting the skipped
            datasets count.
        save : string
            Specifies option for saving. save=full saves all, save=None saves nothing.
        disp : boolean
            Specifies whether to display iteration number.
        """

        self.num_ds = num_ds
        imp_chain = ImputerChain(self.imputer_list)
        for b in range(burnin):
            imp_chain.next()
        analysis_chain = AnalysisChain(imp_chain, self.analysis_formula,
                                       self.analysis_class, skipnum,
                                       save, self.init_args, self.fit_args)
        self.mod_list = []
        for current_iter in range(num_ds):
            achain = copy.deepcopy(analysis_chain)
            model = achain.next()
            self.mod_list.append(model)
            if not hasattr(self, "exog_names"):
                self.exog_names = model.model.exog_names
                self.endog_names = model.model.endog_names
        if disp:
                print current_iter

    def combine(self):
        """
        Pools estimated parameters and covariance matrices of generated
        analysis models according to Rubin's Rule.

        Returns
        -------
        rslt : MICEResults instance
            Contains pooled analysis model results with missing data summaries.
        """

        params_list = []
        cov_list = []
        scale_list = []
        # TODO: Verify calculations for diagnostics
        for md in self.mod_list:
            params_list.append(md.params)
            cov_list.append(np.asarray(md.normalized_cov_params))
            scale_list.append(md.scale)
        scale = np.mean(scale_list)
        params = np.mean(params_list, axis=0)
        # Get average of within-imputation covariances weighted by scale
        full_cov = np.asarray(cov_list) * np.asarray(scale_list)[:, np.newaxis, np.newaxis]
        within_g = np.mean(full_cov, axis=0)
        # Used MLE rather than method of moments between group covariance
        between_g = np.cov(np.array(params_list).T, bias=1)
        cov_params = within_g + (1 + 1. / float(self.num_ds)) * between_g
        gamma = (1. + 1. / float(self.num_ds)) * np.divide(np.diag(between_g), np.diag(cov_params))
        df_approx = (float(self.num_ds) - 1.) * np.square(np.divide(1. , gamma))
        df_obs = (float(self.N) - float(len(params)) + 1.) / (float(self.N) - float(len(params)) + 3.) * (1. - gamma) * (float(self.N) - float(len(params)))
        self.df = np.divide(1. , (np.divide(1. , df_approx) + np.divide(1. , df_obs)))
        self.fmi = gamma
        rslt = MICEResults(self, params, cov_params / scale)
        rslt.scale = scale
        return rslt


class MICEResults(statsmodels.base.model.LikelihoodModelResults):

    def __init__(self, model, params, normalized_cov_params):

        super(MICEResults, self).__init__(model, params,
                                          normalized_cov_params)

    def summary(self, title=None, alpha=.05):
        """
        Summarize the results of running MICE (multiple imputation with chained equations).

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
        info["Model:"] = self.model.analysis_class.__name__
        info["Dependent variable:"] = self.model.endog_names
        info["Sample size:"] = "%d" % self.model.mod_list[0].model.exog.shape[0]


        smry.add_dict(info, align='l', float_format=float_format)

        param = summary2.summary_params(self, alpha=alpha)
        # TODO: Fix diagnostics to be consistent with R
#        param['P>|t|'] = stats.t.sf(np.abs(np.asarray(param['t'])), self.model.df) / 2.
#        ci = np.asarray(stats.t.interval(1-alpha, self.model.df, 
#        loc=np.asarray(param['Coef.']), scale=np.asarray(param['Std.Err.'])))
#        param['[' + str(alpha/2)] = ci[0]
#        param[str(1-alpha/2) + ']'] = ci[1]
#        param['Df'] = self.model.df
#        param['Df'][0] = -100
#        param['FMI'] = self.model.fmi
#        param['FMI'][0] = -100
#        numiss = [0]
#        for value in self.model.exog_names:
#            for x in self.model.imputer_list:
#                if x.endog_name == value :
#                    numiss.append(int(x.num_missing))
#        param['#missing'] = numiss
#        param['#missing'][0] = -100
        smry.add_df(param, float_format=float_format)
        smry.add_title(title=title, results=self)
        return smry

class MissingDataInfo(object):
    """
    Contains all the missing data information from the passed-in data object.

    Parameters
    ----------
    data : pandas DataFrame with missing values.

    Attributes
    ----------
    ix_miss : array
        Indices of missing values for a particular variable.
    ix_obs : array
        Indices of observed values for a particular variable.

    Note
    ----
    A self.columns dictionary entry exists for each column/variable in
    the dataset.
    """

    def __init__(self, data):
        null = pd.isnull(data)
        self.ix_obs = np.flatnonzero(~null)
        self.ix_miss = np.flatnonzero(null)
        if len(self.ix_obs) == 0:
            raise ValueError("Variable to be imputed has no observed values")