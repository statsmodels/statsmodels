"""
Intervening Variable Analysis

Computes confidence intervals for the indirect effect using Sobel's method [1]
and bootstrapping methods [2]. Might return to expand further to the methods of
[3].

[1] Sobel, Michael E. (1982). Asymptotic Confidence Intervals for Indirect
Effects in Structural Equation Models. Sociological Methodology, Vol. 13.
pp. 290-312

[2] Mackinnon, David P., Lockwood, Chondra M., and Williams, Jason (2004).
Confidence Limits for the Indirect Effect: Distribution of the Product and
Resampling Methods. Multivariate Behavioral Research.

[3] MacKinnon DP, Lockwood CM, Hoffman JM, West SG, Sheets V (2002).
 A comparison of methods to test mediation and other intervening variable
 effects. Psychol Methods. 2002 Mar;7(1):83-104. doi: 10.1037/1082-989x.7.1.83.
 PMID: 11928892; PMCID: PMC2819363.
"""
import numpy as np
import scipy.stats as st
import pandas as pd


class InterveningVariable:
    """
    Conduct a mediation analysis.

    Parameters
    ----------
    outcome_model : statsmodels model
        Regression model for the outcome.  Exogenous variables must include
        the treatment variable, the mediation variable, and any other
        covariates of interest.
    mediator_model : statsmodels model
        Regression model for the mediator variable. Exogenous variables must
        include the treatement variable. For a rigorous analysis, include the
        same covariates as in outcome_model
    treatment_name : str
        The name of the treatment variable in outcome_model and mediator_model
    outcome_fit_kwargs : dict-like
        Keyword arguments to use when fitting the outcome model.
    mediator_fit_kwargs : dict-like
        Keyword arguments to use when fitting the mediator model.


    Returns a ``InterveningVariableResults`` object.

    Notes
    -----
    The bootstrap standard error is computing using the standard deviation of
    the parameter from the bootstrap samples. This standard error is not used
    in computing the bootstrap confidence intervals, which instead uses the
    appropriate percentiles of the bootstrap sample parameter.

    Examples
    --------


    >>> import statsmodels.api as sm
    >>> from statsmodels.stats.intervening_variable import InterveningVariable
    >>> outcome_model = sm.GLM.from_formula("cong_mesg ~ emo + treat + age
                                            + educ + gender + income", data,
                                            family=
                                            sm.families.Binomial(link=Probit()))
    >>> mediator_model = sm.OLS.from_formula("emo ~ treat + age + educ + gender
                                             + income", data)
    >>> med = InterveningVariable(outcome_model, mediator_model, "treat").fit()
    >>> med.summary()



    References
    ----------
    [1] Sobel, Michael E. (1982). Asymptotic Confidence Intervals for Indirect
    Effects in Structural Equation Models. Sociological Methodology, Vol. 13.
    pp. 290-312

    [2] Mackinnon, David P., Lockwood, Chondra M., and Williams, Jason (2004).
    Confidence Limits for the Indirect Effect: Distribution of the Product and
    Resampling Methods. Multivariate Behavioral Research.

    [3] MacKinnon DP, Lockwood CM, Hoffman JM, West SG, Sheets V (2002).
     A comparison of methods to test mediation and other intervening variable
     effects.Psychol Methods. 2002 Mar;7(1):83-104. doi:
    10.1037/1082-989x.7.1.83.PMID: 11928892; PMCID: PMC2819363.
    """

    def __init__(
        self,
        outcome_model,
        mediator_model,
        treatment_name,
        outcome_fit_kwargs=None,
        mediator_fit_kwargs=None,
    ):

        self.mediator_model = mediator_model
        self.outcome_model = outcome_model

        if treatment_name not in mediator_model.exog_names:
            raise ValueError(
                f"Treatment variable {treatment_name} not in "
                "exogenous variables of mediator_model."
            )
        else:
            self.tx_indx_in_med_model = self.mediator_model.exog_names.index(
                treatment_name
            )
            self.tx_indx_in_outcome_model = (
                self.outcome_model.exog_names.index(treatment_name)
            )
        self.mediator_name = self.mediator_model.endog_names
        self.med_indx_in_outcome_model = self.outcome_model.exog_names.index(
            self.mediator_name
        )

        self._mediator_fit_kwargs = (
            mediator_fit_kwargs if mediator_fit_kwargs is not None else {}
        )
        self._outcome_fit_kwargs = (
            outcome_fit_kwargs if outcome_fit_kwargs is not None else {}
        )

    def _fit_model(self, model, fit_kwargs, boot=False, sample=[]):
        """
        Makes a copy of the model and fits it
        """
        klass = model.__class__
        init_kwargs = model._get_init_kwds()
        endog = model.endog
        exog = model.exog
        if boot:
            endog = endog[sample]
            exog = exog[sample, :]
        outcome_model = klass(endog, exog, **init_kwargs)
        return outcome_model.fit(**fit_kwargs)

    def _get_indirect(self, outcome_model_fitted, mediator_model_fitted):
        treatment_mediator_coeff = mediator_model_fitted.params[
            self.tx_indx_in_med_model
        ]
        mediator_indep_coeff = outcome_model_fitted.params[
            self.med_indx_in_outcome_model
        ]
        treatment_mediator_stderr = mediator_model_fitted.bse[
            self.tx_indx_in_med_model
        ]
        mediator_indep_stderr = outcome_model_fitted.bse[
            self.med_indx_in_outcome_model
        ]

        indirect_effect = treatment_mediator_coeff * mediator_indep_coeff
        indirect_stderr = np.sqrt(
            (treatment_mediator_coeff**2) * (mediator_indep_stderr**2)
            + (mediator_indep_coeff**2) * (treatment_mediator_stderr**2)
        )

        return indirect_effect, indirect_stderr

    def fit(self, n_reps=1000):
        """
        Fit a model to asses the indirect effect of an intervening variable

        Parameters
        ----------
        n_reps : int, optional
            The number of repetition for bootstrapping. The default is 1000.

        Returns an InterveningVariableResults object
        """
        # direct effect and stderr are the same regardless of method
        outcome_model_fitted = self._fit_model(
            self.outcome_model, self._outcome_fit_kwargs
        )
        self.direct_effect = outcome_model_fitted.params[
            self.tx_indx_in_outcome_model
        ]
        self.direct_stderr = outcome_model_fitted.bse[
            self.tx_indx_in_outcome_model
        ]

        mediator_model_fitted = self._fit_model(
            self.mediator_model, self._mediator_fit_kwargs
        )
        (
            self.sobel_indirect_effect,
            self.sobel_indirect_stderr,
        ) = self._get_indirect(outcome_model_fitted, mediator_model_fitted)

        boot_sample_indirect_effects = []
        boot_sample_indirect_proportions = []
        nobs = len(self.outcome_model.endog)
        for _ in range(n_reps):
            sample = np.random.randint(0, nobs, nobs)
            sample_outcome_model_fitted = self._fit_model(
                self.outcome_model,
                self._outcome_fit_kwargs,
                boot=True,
                sample=sample,
            )
            sample_mediator_model_fitted = self._fit_model(
                self.mediator_model, self._mediator_fit_kwargs, sample=sample
            )
            sample_direct_effect = sample_outcome_model_fitted.params[
                self.tx_indx_in_outcome_model
            ]
            (
                sample_indirect_effect,
                sample_indirect_stderr,
            ) = self._get_indirect(
                sample_outcome_model_fitted, sample_mediator_model_fitted
            )
            boot_sample_indirect_effects.append(sample_indirect_effect)
            boot_sample_indirect_proportions.append(
                sample_indirect_effect
                / (sample_direct_effect + sample_indirect_effect)
            )

        self.boot_sample_indirect_effects = np.asarray(
            boot_sample_indirect_effects
        )
        self.boot_sample_indirect_proportions = np.asarray(
            boot_sample_indirect_proportions
        )

        self.boot_indirect_stderr = np.std(self.boot_sample_indirect_effects)

        self.indirect_proportion = self.sobel_indirect_effect / (
            self.direct_effect + self.sobel_indirect_effect
        )
        self.boot_indirect_proportion_stderr = np.std(
            self.boot_sample_indirect_proportions
        )

        return InterveningVariableResults(
            self.direct_effect,
            self.direct_stderr,
            self.sobel_indirect_effect,
            self.sobel_indirect_stderr,
            self.boot_indirect_stderr,
            self.boot_sample_indirect_effects,
            self.indirect_proportion,
            self.boot_indirect_proportion_stderr,
            self.boot_sample_indirect_proportions,
        )


class InterveningVariableResults:
    """
    An object for holding the results of an intervening variable analysis.
    """

    def __init__(
        self,
        direct_effect,
        direct_stderr,
        sobel_indirect_effect,
        sobel_indirect_stderr,
        boot_indirect_stderr,
        boot_sample_indirect_effects,
        indirect_proportion,
        boot_indirect_proportion_stderr,
        boot_sample_indirect_proportions,
    ):

        self.direct_effect = direct_effect
        self.direct_stderr = direct_stderr
        self.sobel_indirect_effect = sobel_indirect_effect
        self.sobel_indirect_stderr = sobel_indirect_stderr
        self.boot_indirect_stderr = boot_indirect_stderr
        self.boot_sample_indirect_effects = boot_sample_indirect_effects
        self.indirect_proportion = indirect_proportion
        self.boot_indirect_proportion_stderr = boot_indirect_proportion_stderr
        self.boot_sample_indirect_proportions = (
            boot_sample_indirect_proportions
        )

    def summary(self, alpha=0.05):
        """
        Provides a summary of the intervening variable analysis.

        Parameters
        ----------
        alpha : int, optional
           1 - (confidence level). I.e. alpha = 0.05 will compute 95%
           confidence intervals. The default is 0.05.

        Returns
        -------
        DataFrame
            A dataframe containing the results of the intervening variable
            analysis.

        """
        z_score = st.norm.ppf(1 - alpha / 2)

        self.direct_effect_lb = self.direct_effect - (
            z_score * self.direct_stderr
        )
        self.direct_effect_ub = self.direct_effect + (
            z_score * self.direct_stderr
        )
        direct_info = [
            self.direct_effect,
            self.direct_stderr,
            self.direct_effect_lb,
            self.direct_effect_ub,
        ]

        self.sobel_indirect_effect_lb = self.sobel_indirect_effect - (
            z_score * self.sobel_indirect_stderr
        )
        self.sobel_indirect_effect_ub = self.sobel_indirect_effect + (
            z_score * self.sobel_indirect_stderr
        )
        sobel_indirect_info = [
            self.sobel_indirect_effect,
            self.sobel_indirect_stderr,
            self.sobel_indirect_effect_lb,
            self.sobel_indirect_effect_ub,
        ]

        self.boot_indirect_effect_lb = np.percentile(
            self.boot_sample_indirect_effects, 100 * alpha / 2
        )
        self.boot_indirect_effect_ub = np.percentile(
            self.boot_sample_indirect_effects, 100 * (1 - alpha / 2)
        )
        boot_indirect_info = [
            self.sobel_indirect_effect,
            self.boot_indirect_stderr,
            self.boot_indirect_effect_lb,
            self.boot_indirect_effect_ub,
        ]

        self.boot_indirect_proportion_lb = np.percentile(
            self.boot_sample_indirect_proportions, 100 * alpha / 2
        )
        self.boot_indirect_proportion_ub = np.percentile(
            self.boot_sample_indirect_proportions, 100 * (1 - alpha / 2)
        )
        boot_indirect_proportion_info = [
            self.indirect_proportion,
            self.boot_indirect_proportion_stderr,
            self.boot_indirect_proportion_lb,
            self.boot_indirect_proportion_ub,
        ]

        columns = ["value", "std err", "lower ci bound", "upper ci bound"]
        index = [
            "direct effect",
            "sobel indirect effect",
            "bootstrap indirect effect",
            "bootstrap indirect/total",
        ]

        return pd.DataFrame(
            [
                direct_info,
                sobel_indirect_info,
                boot_indirect_info,
                boot_indirect_proportion_info,
            ],
            columns=columns,
            index=index,
        )
