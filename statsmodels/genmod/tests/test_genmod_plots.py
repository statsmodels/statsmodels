"""
Tests for the plotting functions in GLMResults and GEEResults.

Set create_pdfs to True to generate a multi-page PDF file containing
many plots.  This variable needs to be set to False in the master
since the CI testing doesn't generate files.
"""

import statsmodels.api as sm
import numpy as np
from numpy.testing import dec

# Set to False in master and releases
pdf_output = False

try:
    import matplotlib.pyplot as plt
    import matplotlib
    if matplotlib.__version__ < '1':
        raise
    have_matplotlib = True
except:
    have_matplotlib = False


def close_or_save(pdf, fig):
    if pdf_output:
        pdf.savefig(fig)
    else:
        plt.close(fig)


@dec.skipif(not have_matplotlib)
def test_glm_added_variable_plot():

    if pdf_output:
        from matplotlib.backends.backend_pdf import PdfPages
        pdf = PdfPages("test_genmod_added_variable_plot.pdf")
    else:
        pdf = None

    n = 100
    p = 4
    exog = np.random.normal(size=(n, p))
    exog[:, 0] = 1
    params = np.r_[0, -1, 1, 0]
    lin_pred = np.dot(exog, params)
    expval = np.exp(lin_pred)
    endog = np.random.poisson(expval, size=n)

    for glm in False, True:

        if glm:
            model = sm.GLM(endog, exog, family=sm.families.Poisson())
            glm_type = "GLM"
        else:
            groups = np.kron(np.arange(n/2), np.ones(2))
            model = sm.GEE(endog, exog, groups, family=sm.families.Poisson())
            glm_type = "GEE"

        result = model.fit()

        for focus_col in 1, 2, 3:
            effect_type = {1: "Negative effect (slope=-1)",
                           2: "Positive effect (slope=1)",
                           3: "No effect (slope=0)"}[focus_col]
            for resid_type in ["resid_response", "resid_pearson",
                               "resid_deviance", "resid_anscombe"]:

                fig = result.added_variable_plot(focus_col, resid_type=resid_type)
                ax = fig.get_axes()[0]
                ax.set_position([0.1, 0.1, 0.8, 0.75])
                ax.set_title(effect_type + "\n" +
                             ("Residual type is " + resid_type) + "\n" +
                             ("Fit by " + glm_type))
                close_or_save(pdf, fig)
                plt.close(fig)

    if pdf_output:
        pdf.close()

@dec.skipif(not have_matplotlib)
def test_glm_partial_residual_plot():

    if pdf_output:
        from matplotlib.backends.backend_pdf import PdfPages
        pdf = PdfPages("test_genmod_partial_residual_plot.pdf")
    else:
        pdf = None

    n = 200
    p = 3
    exog = np.random.normal(size=(n, p))
    exog[:, 0] = 1
    exog[:, 2] = np.random.uniform(-2, 2, size=n)

    # Create a linear predictor involving transforms of the covariates
    # The structure is taken from Cook (1998)
    lin_pred = 4 - exog[:, 1] + exog[:, 2]**2
    expval = np.exp(lin_pred)
    endog = np.random.poisson(expval, size=n)
    model = sm.GLM(endog, exog, family=sm.families.Poisson())
    result = model.fit()

    for glm in False, True:

        if glm:
            model = sm.GLM(endog, exog, family=sm.families.Poisson())
            glm_type = "GLM"
        else:
            groups = np.kron(np.arange(n/2), np.ones(2))
            model = sm.GEE(endog, exog, groups, family=sm.families.Poisson())
            glm_type = "GEE"

        for focus_col in 1, 2:
            effect_type = {1: "True effect is linear (slope = -1)",
                           2: "True effect is quadratic"}[focus_col]

            fig = result.partial_residual_plot(focus_col)
            ax = fig.get_axes()[0]
            ax.set_position([0.1, 0.1, 0.8, 0.8])
            ax.set_title(effect_type + ("\nFit by " + glm_type))
            close_or_save(pdf, fig)
            plt.close(fig)

    if pdf_output:
        pdf.close()

@dec.skipif(not have_matplotlib)
def test_glm_ceres_plot():

    if pdf_output:
        from matplotlib.backends.backend_pdf import PdfPages
        pdf = PdfPages("test_genmod_ceres_plot.pdf")
    else:
        pdf = None

    n = 200
    p = 3
    exog = np.random.normal(size=(n, p))
    exog[:, 0] = 1
    exog[:, 2] = np.random.uniform(-2, 2, size=n)

    # Create a linear predictor involving transforms of the covariates
    # The structure is taken from Cook (1998)
    lin_pred = 4 - exog[:, 1] + exog[:, 2]**2
    expval = np.exp(lin_pred)
    endog = np.random.poisson(expval, size=n)
    model = sm.GLM(endog, exog, family=sm.families.Poisson())
    result = model.fit()

    for glm in False, True:

        if glm:
            model = sm.GLM(endog, exog, family=sm.families.Poisson())
            glm_type = "GLM"
        else:
            groups = np.kron(np.arange(n/2), np.ones(2))
            model = sm.GEE(endog, exog, groups, family=sm.families.Poisson())
            glm_type = "GEE"

        for focus_col in 1, 2:
            effect_type = {1: "True effect is linear (slope = -1)",
                           2: "True effect is quadratic"}[focus_col]
            for simple in 0, 1, 2:
                simple_type = {0: "Estimate conditional means",
                               1: "Set conditional means to focus column",
                               2: "No conditional means"}[simple]

                if simple == 0:
                    cond_means = None
                elif simple == 1:
                    cond_means = exog[:, focus_col][:, None]
                else:
                    cond_means = np.zeros((n, 0))

                fig = result.ceres_plot(focus_col, cond_means=cond_means)
                ax = fig.get_axes()[0]
                ax.set_position([0.1, 0.1, 0.8, 0.75])
                ax.set_title(effect_type + "\n" +
                             "Fit by " + glm_type + "\n" + simple_type)
                close_or_save(pdf, fig)
                plt.close(fig)

    if pdf_output:
        pdf.close()
