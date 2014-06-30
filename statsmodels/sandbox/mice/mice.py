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

"""

import operator
import pandas as pd
import numpy as np
import patsy
import statsmodels.api as sm
import random
import statsmodels
from statsmodels.tools.decorators import cache_readonly
from scipy import stats
import copy
import sys

class ImputedData(object):
    __doc__= """
    Stores missing data information and supports functionality for inserting
    values in missing data slots.

    Can create Imputers directly via new_imputer method.

    %(params)s
    data : array-like object
        Needs to support transformation to pandas dataframe. Missing value
        encoding is handled by pandas DataFrame class.

    **Attributes**

    data : pandas dataframe
        Dataset with missing values. After recording missing data information,
        simple column-wise means are filled into the missing values.
    columns : dictionary
        Stores indices of missing data.

    """
    def __init__(self, data):
        # may not need to make copies
        self.data = pd.DataFrame(data)
        self.columns = {}
        for c in self.data.columns:
            self.columns[c] = MissingDataInfo(self.data[c])
        self.data = self.data.fillna(self.data.mean())

    def new_imputer(self, endog_name, method="gaussian", k_pmm=1, formula=None, model_class=None,
                    init_args={}, fit_args={}, scale_method="fix",
                    scale_value=None):
        """
        Create Imputer instance from our ImputedData instance

        Parameters
        ----------
        endog_name : string
            Name of the variable to be imputed.
        formula : string
            Conditional formula for imputation. Defaults to model with main
            effects for all other variables in dataset.
        model_class : statsmodels model
            Conditional model for imputation. Defaults to OLS.
        scale_method : string
            Governs the type of perturbation given to the scale parameter.
        scale_value : float
            Fixed value of scale parameter to use in simulation of data.

        Returns
        -------
        mice.Imputer object

        See Also
        --------
        mice.Imputer
        """
        if model_class is None:
            model_class = sm.OLS
        if formula is None:
            formula = endog_name + " ~ " + " + ".join(
                            [x for x in self.data.columns if x != endog_name])
        return Imputer(formula, model_class, self, method=method, k_pmm=1, init_args=init_args,
                       fit_args=fit_args, scale_method=scale_method,
                       scale_value=scale_value)

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
        self.data[col].iloc[ix] = vals

    def get_data_from_formula(self, formula):
        """
        Use formula to construct endog and exog split by missing status.

        Called by Imputer before fitting model.

        Parameters
        ----------
        formula : string
            Patsy formula for the Imputer object's conditional model.

        Returns
        -------
        endog_obs : DataFrame
            Observations of the variable to be imputed.
        exog_obs : DataFrame
            Observations of the predictors where the variable to be Imputed is
            observed.
        exog_miss : DataFrame
            Observations of the predictors where the variable to be Imputed is
            missing.
        """
        dmat = patsy.dmatrices(formula, self.data, return_type="dataframe")
        exog = dmat[1]
        endog = dmat[0]
        if len(endog.design_info.term_names) > 1:
            endog_name = tuple(endog.design_info.term_names)
        else:
            endog_name = endog.design_info.term_names[0]
        endog_obs = endog.iloc[self.columns[endog_name].ix_obs]
        exog_obs = exog.iloc[self.columns[endog_name].ix_obs]
        exog_miss = exog.iloc[self.columns[endog_name].ix_miss]
        return endog_obs, exog_obs, exog_miss

class Imputer(object):

    __doc__= """
    Initializes object that imputes values for a single variable
    using a given formula.

    %(params)s

    formula : string
        Conditional formula for imputation.
    model_class : statsmodels model
        Conditional model for imputation.
    data : ImputedData object
        See mice.ImputedData
    return_class : scipy.random class
        Controls the scale/location family to use as the asymptotic
        distribution of the imputed variable. For use in method
        impute_asymptotic_bayes.
    scale : string
        Governs the type of perturbation given to the scale parameter.
    scale_value : float
        Fixed value of scale parameter to use in simulation of data.
    %(extra_params)s

    **Attributes**

    endog_name : string
        Name of variable to be imputed.
    num_missing : int
        Number of missing values.

    Note: all params are saved as attributes.

    """
    def __init__(self, formula, model_class, data,
            method="gaussian", k_pmm=1, init_args={}, fit_args={},
                 return_class=None, scale_method="fix", scale_value=None):
        self.data = data
        self.formula = formula
        self.model_class = model_class
        self.init_args = init_args
        self.fit_args = fit_args
        self.endog_name = str(self.formula.split("~")[0].strip())
        self.num_missing = len(self.data.columns[self.endog_name].ix_miss)
        self.return_class = return_class
        self.scale_method = scale_method
        self.scale_value = scale_value
        self.method = method
        self.k_pmm = k_pmm

    def perturb_params(self, mdf):
        """
        Perturbs the model's scale and fit parameters.

        The user may choose to fix the scale at the estimated quantity or
        perturb it using a chi square approximation.

        Bootstrap perturbation upcoming.

        Parameters
        ----------
        mdf : statsmodels fitted model
            Passed from Imputer object.

        Returns
        -------
        params : array
            Perturbed model parameters.
        scale_per : float
            Perturbed nuisance parameter.
        """
        # switch to scipy
        params = mdf.params.copy()
        covmat = mdf.cov_params()
        covmat_sqrt = np.linalg.cholesky(covmat)
        if self.scale_method == "fix":
            if self.scale_value is None:
                scale_per = 1.
            else:
                scale_per = self.scale_value
        elif self.scale_method == "perturb_chi2":
            u = np.random.chisquare(float(mdf.df_resid))
            scale_per = float(mdf.df_resid) / float(u)
        elif self.scale_method == "perturb_boot":
            raise NotImplementedError
        p = len(params)
        params += np.dot(covmat_sqrt,
                         np.random.normal(0, np.sqrt(mdf.scale * scale_per), p))
        return params, scale_per

    def impute_asymptotic_bayes(self):
        """
        Use Gaussian approximation to posterior distribution to simulate data.

        Fills in missing values of input data. User may also choose a different
        approximating location/scale family by specifying return_class in
        Imputer initialization.
        """
        endog_obs, exog_obs, exog_miss = self.data.get_data_from_formula(
                                                                self.formula)
        md = self.model_class(endog_obs, exog_obs, **self.init_args)
        mdf = md.fit(**self.fit_args)
        params, scale_per = self.perturb_params(mdf)
        new_rv = md.get_distribution(params=params, exog=exog_miss,
                                     model_class=self.return_class,
                                     scale=scale_per * mdf.scale)
        new_endog = new_rv.rvs(size=len(exog_miss))
        self.data.store_changes(self.endog_name, new_endog)

    def impute_pmm(self, pmm_neighbors=1):
        """
        Use predictive mean matching to simulate data.

        Fills in missing values of input data. Predictive mean matching picks
        an observation randomly from the observed value of the k-nearest
        predictions of the endogenous variable. Naturally, the neighbors must
        have observed endogenous values.

        Parameters
        ----------
        pmm_neighbors : int
            Number of neighbors in prediction space to select imputations from.
            Defaults to 1 (select closest neighbor).
        """
        endog_obs, exog_obs, exog_miss = self.data.get_data_from_formula(
                                                                self.formula)
        md = self.model_class(endog_obs, exog_obs, **self.init_args)
        mdf = md.fit(**self.fit_args)
        params, scale_per = self.perturb_params(mdf)
        # Predict imputed variable for both missing and nonmissing observations
        pendog_obs = md.predict(params, exog_obs)
        pendog_miss = md.predict(params, exog_miss)
        
#        imputed_miss = []
#        for val in pendog_miss:
#            dist = abs(pendog_obs - val)
#            dist_ind = np.argsort(dist, axis=0)
#            imputed_miss.append(random.choice(np.array(
#            endog_obs)[dist_ind[0:pmm_neighbors][0]]))
        
        ii = np.argsort(pendog_obs, axis=0)
        pendog_obs = pendog_obs[ii]
        oendog = endog_obs.iloc[ii,:]
        # Get indices of predicted endogs of nonmissing observations that are
        # close to those of missing observations
        ix = np.searchsorted(pendog_obs, pendog_miss)
        np.clip(ix, 0, len(pendog_obs) - 1, out=ix)
        ix_list = []
        for i in range(len(pendog_miss)):
            k = 0
            count_low = 0
            count_high = 0
            ix_list.append([])
            upper = pendog_obs[ix[i]]
            lower = pendog_obs[ix[i] - 1]
            target = pendog_miss[i]
            ixs = ix_list[len(ix_list) - 1]
            limit_low = False
            limit_high = False
            while k < pmm_neighbors:
                if limit_low:
                    count_high += 1
                    ixs.append(ix[i] - 1 + count_high)
                elif limit_high:
                    count_low += 1
                    ixs.append(ix[i] - count_low)
                elif abs(target - upper) >= abs(target - lower):
                    count_low += 1
                    if ix[i] - 1 - count_low >= 0:
                        ixs.append(ix[i] - count_low)
                        target = copy.copy(lower)
                        lower = pendog_obs[ix[i] - 1 - count_low]
                    else:
                        ixs.append(ix[i] - 1 + count_high)
                        limit_low = True
                elif abs(target - upper) < abs(target - lower):
                    count_high += 1
                    if ix[i] + count_high < len(pendog_obs):
                        ixs.append(ix[i] - 1 + count_high)
                        target = copy.copy(upper)
                        upper = pendog_obs[ix[i] + count_high]
                    else:
                        ixs.append(ix[i] - count_low)
                        limit_high = True
                k += 1                    
            ixs = np.clip(ixs, 0, len(oendog))
            ixs = random.choice(ixs)
        # Select a random draw of observed endogenous variable from a set
        # window around the closest predicted value
#        ix += np.random.randint(int(-pmm_neighbors / 2.), int(pmm_neighbors / 2.) + 1, size=len(ix))
#        np.clip(ix, 0, len(oendog), out=ix)
#        imputed_miss = np.array(oendog.iloc[ix,:])
        ix_list = np.squeeze(ix_list)
        imputed_miss = np.array(oendog.iloc[ix_list,:])
        self.data.store_changes(self.endog_name, imputed_miss)

    def impute_bootstrap(self):
        raise NotImplementedError

class ImputerChain(object):
    __doc__= """
    An iterator that returns imputed data sets produced using the MICE
    (multiple imputation by chained equations) procedure.

    This class does imputation and returns the imputed data sets, it does not
    fit the analysis model. See the "next" method for details.

    %(params)s

    imputer_list : list
        List of Imputer objects, one for each variable to be imputed.
    imputer_method : string
        Method used for simulation. See MICE.run.
    pmm_neighbors : int
        Number of neighbors for pmm. See MICE.run.

    **Attributes**

    data : pandas DataFrame
        Underlying data to be modified.

    Note: All imputers must refer to the same data object. See mice.MICE.run
    for iterator call.

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

        Returned datsets are not saved unless specified in the iterator call.

        Returns
        -------
        data : pandas DataFrame
            Dataset with imputed values saved after invoking each Imputer
            object in imputer_list.
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
    __doc__= """
    An iterator that returns the fitted model of interest given a MICE imputed
    dataset.

    Datasets to be used for analysis are chosen after an initial burnin period
    where no imputed data is used and also after skipping a set number of
    imputations for each iteration. See the "next" method for details.

    Note: See mice.MICE.run for iterator call.

    """

    def __init__(self, imputer_chain, analysis_formula, analysis_class,
                 skipnum=10, burnin=5, save=False, init_args={}, fit_args={}):
        self.imputer_chain = imputer_chain
        self.analysis_formula = analysis_formula
        self.analysis_class = analysis_class
        self.init_args = init_args
        self.fit_args = fit_args
        self.skipnum = skipnum
        self.burnin = burnin
        self.save = save
        self.iter = 0
        for b in range(self.burnin):
            self.imputer_chain.next()

    def __iter__(self):
        return self

    def next(self):
        """
        Makes this class an iterator that returns the fitted analysis model.

        Handles skipping of imputation iterations, burnin period of
        unconsidered imputation iterations, and whether or not to save the
        datasets to which an analysis model is fit.

        Returns
        -------
        mdf : statsmodels fitted model
            Fitted model of interest on imputed dataset that has passed all
            skip and burnin criteria

        Note: If save option is True, imputed datasets are saved in the format
        "mice_'iteration number'.csv"
        """
        for i in range(self.skipnum):
            data = self.imputer_chain.next()
            print i
        md = self.analysis_class.from_formula(self.analysis_formula,
                                              data, **self.init_args)
        mdf = md.fit(**self.fit_args)
        if self.save:
            fname = "%s_%d.csv" % ('mice_', self.iter)
            data.to_csv(fname, index=False)
            self.iter += 1
        return mdf

class MICE(object):
    __doc__= """
    Fits the analysis model to each imputed dataset and combines the
    results using Rubin's rule.

    Calls mice.Imputer_Chain and mice.AnalysisChain
    to handle imputation and fitting of analysis models to the correct imputed
    datasets, respectively.

    %(params)s

    analysis_formula : string
        Formula for model of interest to be fitted.
    analysis_class : statsmodels model
        Statsmodels model of interest.
    imputer_list : list
        List of Imputer objects, one for each variable to be imputed.
    %(extra_params)s

    **Attributes**

    data : pandas DataFrame
        Underlying data to be modified.

    Examples
    --------
    >>> import pandas as pd
    >>> import statsmodels.api as sm
    >>> from statsmodels.sandbox.mice import mice
    >>> data = pd.read_csv('directory_here')
    >>> impdata = mice.ImputedData(data)
    >>> m1 = impdata.new_imputer("x2", scale_method="perturb_chi2")
    >>> m2 = impdata.new_imputer("x3", scale_method="perturb_chi2")
    >>> m3 = impdata.new_imputer(
        "x1", model_class=sm.Logit, scale_method="perturb_chi2")
    >>> impcomb = mice.MICE("x1 ~ x2 + x3", sm.Logit,[m1,m2,m3])
    >>> implist = impcomb.run(method="pmm")
    >>> p1 = impcomb.combine(implist)

    The resulting p1 contains an empty sm.Logit instance with MICE-provided
    params and cov_params.

    """
    def __init__(self, analysis_formula, analysis_class, imputer_list,
                 init_args={}, fit_args={}):
        self.imputer_list = imputer_list
        self.analysis_formula = analysis_formula
        self.analysis_class = analysis_class
        self.init_args = init_args
        self.fit_args = fit_args

    def run(self, num_ds=20, skipnum=10, burnin=5, save=False):
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
        save : boolean
            Whether to save the imputed datasets chosen for analysis.
        method : string
            Simulation method to use. May take on values "gaussian", "pmm",
            or "bootstrap".
        k_pmm : int
            Number of neighbors to use for predictive mean matching (pmm).

        Returns
        -------
        md_list : list
            List of length num_ds of fitted analysis models.
        """
        self.num_ds = num_ds
        imp_chain = ImputerChain(self.imputer_list)
        analysis_chain = AnalysisChain(imp_chain, self.analysis_formula,
                                       self.analysis_class, skipnum, burnin,
                                       save, self.init_args, self.fit_args)
        md_list = []
        for current_iter in range(num_ds):
            model = analysis_chain.next()
            md_list.append(model)
            if not hasattr(self, "exog_names"):
                self.exog_names = model.model.exog_names
                self.endog_names = model.model.endog_names
            print current_iter
        self.mod_list = md_list

    def combine(self):
        """
        Pools estimated parameters and covariance matrices of generated
        analysis models according to Rubin's Rule.

        Returns
        -------
        md : statsmodels fitted model
            Altered cov_params and params to be the MICE combined quantities.
        """

        params_list = []
        cov_list = []
        scale_list = []
        for md in self.mod_list:
            params_list.append(md.params)
            cov_list.append(np.array(md.normalized_cov_params))
            scale_list.append(md.scale)
        # Just chose last analysis model instance as a place to store results
        md = self.mod_list[-1]
        scale = np.mean(scale_list)
        params = np.mean(params_list, axis=0)
        within_g = np.mean(cov_list, axis=0)
        # Used MLE rather than method of moments between group covariance
        between_g = np.cov(np.array(params_list).T, bias=1)
        cov_params = within_g * scale + (1 + 1. / float(self.num_ds)) * between_g
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
        smry.add_df(param, float_format=float_format)
        smry.add_title(title=title, results=self)

        return smry
class MissingDataInfo(object):
    __doc__="""
    Contains all the missing data information from the passed-in data object.

    A self.columns dictionary entry exists for each column/variable in the
    dataset.

    %(params)s

    data : pandas DataFrame with missing values.

    **Attributes**

    ix_miss : array
        Indices of missing values for a particular variable.
    ix_obs : array
        Indices of observed values for a particular variable.

    """

    def __init__(self, data):
        null = pd.isnull(data)
        self.ix_obs = np.flatnonzero(~null)
        if np.flatnonzero(null).size is 0:
            self.ix_miss = [False]
        else:
            self.ix_miss = np.flatnonzero(null)
        if len(self.ix_obs) == 0:
            raise ValueError("Variable to be imputed has no observed values")