import numpy as np
from statsmodels.duration.survfunc import survfunc_right, logrank, plot_survfunc
from numpy.testing import assert_allclose
from numpy.testing import dec

# If true, the output is written to a multi-page pdf file.
pdf_output = False

try:
    import matplotlib.pyplot as plt
    import matplotlib
    have_matplotlib = True
except ImportError:
    have_matplotlib = False


def close_or_save(pdf, fig):
    if pdf_output:
        pdf.savefig(fig)
    else:
        plt.close(fig)



"""
library(survival)
ti1 = c(3, 1, 2, 3, 2, 1, 5, 3)
st1 = c(0, 1, 1, 1, 0, 0, 1, 0)
ti2 = c(1, 1, 2, 3, 7, 1, 5, 3, 9)
st2 = c(0, 1, 0, 0, 1, 0, 1, 0, 1)

ti = c(ti1, ti2)
st = c(st1, st2)
ix = c(rep(1, length(ti1)), rep(2, length(ti2)))
sd = survdiff(Surv(ti, st) ~ ix)

"""


ti1 = np.r_[3, 1, 2, 3, 2, 1, 5, 3]
st1 = np.r_[0, 1, 1, 1, 0, 0, 1, 0]
times1 = np.r_[1, 2, 3, 5]
surv_prob1 = np.r_[0.8750000, 0.7291667, 0.5468750, 0.0000000]
surv_prob_se1 = np.r_[0.1169268, 0.1649762, 0.2005800, np.nan]

ti2 = np.r_[1, 1, 2, 3, 7, 1, 5, 3, 9]
st2 = np.r_[0, 1, 0, 0, 1, 0, 1, 0, 1]
times2 = np.r_[1, 5, 7, 9]
surv_prob2 = np.r_[0.8888889, 0.5925926, 0.2962963, 0.0000000]
surv_prob_se2 = np.r_[0.1047566, 0.2518034, 0.2444320, np.nan]


def test_survfunc1():
    """
    Test where all times have at least 1 event.
    """

    sr = survfunc_right(ti1, st1)

    assert_allclose(sr.surv_prob, surv_prob1, atol=1e-5, rtol=1e-5)
    assert_allclose(sr.surv_prob_se, surv_prob_se1, atol=1e-5, rtol=1e-5)
    assert_allclose(sr.surv_times, times1)


def test_survfunc2():
    """
    Test where some times have no events.
    """

    sr = survfunc_right(ti2, st2)

    assert_allclose(sr.surv_prob, surv_prob2, atol=1e-5, rtol=1e-5)
    assert_allclose(sr.surv_prob_se, surv_prob_se2, atol=1e-5, rtol=1e-5)
    assert_allclose(sr.surv_times, times2)


def test_logrank():

    z, p = logrank(ti1, st1, ti2, st2)
    assert_allclose(z, 2.14673, atol=1e-4, rtol=1e-4)
    assert_allclose(p, 0.14287, atol=1e-4, rtol=1e-4)


def test_quantiles():

    # Regression tests
    sr = survfunc_right(ti1, st1)
    q = sr.quantile(0.5)
    assert_allclose(q, 5)
    lb, ub = sr.quantile_ci(0.5)
    assert_allclose(np.r_[lb, ub], np.r_[2, 5])


@dec.skipif(not have_matplotlib)
def test_plot_km():

    if pdf_output:
        from matplotlib.backends.backend_pdf import PdfPages
        pdf = PdfPages("test_survfunc.pdf")
    else:
        pdf = None

    sr1 = survfunc_right(ti1, st1)
    sr2 = survfunc_right(ti2, st2)

    fig = plot_survfunc(sr1)
    close_or_save(pdf, fig)

    fig = plot_survfunc(sr2)
    close_or_save(pdf, fig)

    fig = plot_survfunc([sr1, sr2])
    close_or_save(pdf, fig)

    if pdf_output:
        pdf.close()
