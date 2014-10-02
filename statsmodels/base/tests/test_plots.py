"""
Tests for the plotting functions in base.model.

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
def test_covariate_effect_plot():

    if pdf_output:
        from matplotlib.backends.backend_pdf import PdfPages
        pdf = PdfPages("test_covariate_effect_plot.pdf")
    else:
        pdf = None

    n = 200
    p = 3
    exog = np.random.normal(size=(n, p))
    exog[:, 0] = 1
    exog[:, 2] = np.random.uniform(-2, 2, size=n)

    lin_pred = 4 - exog[:, 1] + exog[:, 2]
    expval = np.exp(lin_pred)
    endog = np.random.poisson(expval, size=n)
    model = sm.GLM(endog, exog, family=sm.families.Poisson())
    result = model.fit()

    for focus_col in 1, 2:
        effect_type = {1: "True effect is linear (slope = -1)",
                       2: "True effect is linear (slope = 1)"}[focus_col]
        for show_hist in False, True:
            show_hist_str = {True: "Show histogram",
                             False: "No histogram"}[show_hist]
            for summary_value in 0, 1, 2:
                if summary_value == 0:
                    summary_type = None
                    summary_type_str = "Default summaries"
                elif summary_value == 1:
                    summary_type = [0.75, 0.75, 0.75]
                    summary_type_str = "Summarize with 75th percentile"
                elif summary_value == 2:
                    summary_type = [[0.25, 0.25, 0.25], [0.75, 0.75, 0.75]]
                    summary_type_str = "Summarize with 25th and 75th percentiles"

                fig = result.covariate_effect_plot(focus_col,
                                                   show_hist=show_hist,
                                                   summary_type=summary_type)
                for a in fig.get_axes():
                    a.set_position([0.1, 0.1, 0.8, 0.75])

                if summary_value == 2:
                    ax = fig.get_axes()[0]
                    ha, la = ax.get_legend_handles_labels()
                    la = ["25th pctl", "75th pctl"]
                    title = "Exog %d" % {2: 1, 1: 2}[focus_col]
                    leg = plt.figlegend(ha, la, "center right",
                                        title=title)
                    leg.draw_frame(False)

                    for a in fig.get_axes():
                        a.set_position([0.1, 0.1, 0.68, 0.75])

                ax = fig.get_axes()[0]
                ax.set_title(effect_type + "\n" + show_hist_str + "\n" +
                             summary_type_str)
                close_or_save(pdf, fig)
                plt.close(fig)

    if pdf_output:
        pdf.close()
