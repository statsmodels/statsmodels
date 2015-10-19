import numpy as np
from statsmodels.duration.survfunc import SurvfuncRight, survdiff, plot_survfunc
from numpy.testing import assert_allclose
from numpy.testing import dec
import pandas as pd
import os

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


cur_dir = os.path.dirname(os.path.abspath(__file__))
fp = os.path.join(cur_dir, 'results', 'bmt.csv')
bmt = pd.read_csv(fp)


def test_survfunc1():
    """
    Test where all times have at least 1 event.
    """

    sr = SurvfuncRight(ti1, st1)

    assert_allclose(sr.surv_prob, surv_prob1, atol=1e-5, rtol=1e-5)
    assert_allclose(sr.surv_prob_se, surv_prob_se1, atol=1e-5, rtol=1e-5)
    assert_allclose(sr.surv_times, times1)


def test_survfunc2():
    """
    Test where some times have no events.
    """

    sr = SurvfuncRight(ti2, st2)

    assert_allclose(sr.surv_prob, surv_prob2, atol=1e-5, rtol=1e-5)
    assert_allclose(sr.surv_prob_se, surv_prob_se2, atol=1e-5, rtol=1e-5)
    assert_allclose(sr.surv_times, times2)


def test_survdiff_basic():

    # Constants taken from R, code above
    ti = np.concatenate((ti1, ti2))
    st = np.concatenate((st1, st2))
    groups = np.ones(len(ti))
    groups[0:len(ti1)] = 0
    z, p = survdiff(ti, st, groups)
    assert_allclose(z, 2.14673, atol=1e-4, rtol=1e-4)
    assert_allclose(p, 0.14287, atol=1e-4, rtol=1e-4)


def test_simultaneous_cb():

    # The exact numbers here are regression tests, but they are close
    # to page 103 of Klein and Moeschberger.

    df = bmt.loc[bmt["Group"] == "ALL", :]
    sf = SurvfuncRight(df["T"], df["Status"])
    lcb1, ucb1 = sf.simultaneous_cb(transform="log")
    lcb2, ucb2 = sf.simultaneous_cb(transform="arcsin")

    ti = sf.surv_times.tolist()
    ix = [ti.index(x) for x in (110, 122, 129, 172)]
    assert_allclose(lcb1[ix], np.r_[0.43590582, 0.42115592, 0.4035897, 0.38785927])
    assert_allclose(ucb1[ix], np.r_[0.93491636, 0.89776803, 0.87922239, 0.85894181])

    assert_allclose(lcb2[ix], np.r_[0.52115708, 0.48079378, 0.45595321, 0.43341115])
    assert_allclose(ucb2[ix], np.r_[0.96465636,  0.92745068,  0.90885428, 0.88796708])


def test_bmt():
    # All tests against SAS
    # Results taken from here:
    # http://support.sas.com/documentation/cdl/en/statug/68162/HTML/default/viewer.htm#statug_lifetest_details03.htm

    # Confidence intervals for 25% percentile of the survival
    # distribution (for "ALL" subjects), taken from the SAS web site
    cb = {"linear" : [107, 276],
          "cloglog" : [86, 230],
          "log" : [107, 332],
          "asinsqrt" : [104, 276],
          "logit" : [104, 230]}

    dfa = bmt[bmt.Group == "ALL"]

    cur_dir = os.path.dirname(os.path.abspath(__file__))
    fp = os.path.join(cur_dir, 'results', 'bmt_results.csv')
    rslt = pd.read_csv(fp)

    sf = SurvfuncRight(dfa["T"].values, dfa.Status.values)

    assert_allclose(sf.surv_times, rslt.t)
    assert_allclose(sf.surv_prob, rslt.s, atol=1e-4, rtol=1e-4)
    assert_allclose(sf.surv_prob_se, rslt.se, atol=1e-4, rtol=1e-4)

    for method in "linear", "cloglog", "log", "logit", "asinsqrt":
        lcb, ucb = sf.quantile_ci(0.25, method=method)
        assert_allclose(cb[method], np.r_[lcb, ucb])


def test_survdiff():
    # Results come from R survival and survMisc packages (survMisc is
    # used for non G-rho family tests but does not seem to support
    # stratification)

    df = bmt[bmt.Group != "ALL"]

    # Not stratified
    stat, p = survdiff(df["T"], df.Status, df.Group)
    assert_allclose(stat, 13.44556, atol=1e-4, rtol=1e-4)
    stat, p = survdiff(df["T"], df.Status, df.Group, weight_type="gb")
    assert_allclose(stat, 15.38787, atol=1e-4, rtol=1e-4)
    stat, p = survdiff(df["T"], df.Status, df.Group, weight_type="tw")
    assert_allclose(stat, 14.98382, atol=1e-4, rtol=1e-4)
    stat, p = survdiff(df["T"], df.Status, df.Group, weight_type="fh", fh_p=0.5)
    assert_allclose(stat, 14.46866, atol=1e-4, rtol=1e-4)
    stat, p = survdiff(df["T"], df.Status, df.Group, weight_type="fh", fh_p=1)
    assert_allclose(stat, 14.84500, atol=1e-4, rtol=1e-4)

    # 5 strata
    strata = np.arange(df.shape[0]) % 5
    df["strata"] = strata
    stat, p = survdiff(df["T"], df.Status, df.Group, strata=df.strata)
    assert_allclose(stat, 11.97799, atol=1e-4, rtol=1e-4)
    stat, p = survdiff(df["T"], df.Status, df.Group, strata=df.strata,
                       weight_type="fh", fh_p=0.5)
    assert_allclose(stat, 12.6257, atol=1e-4, rtol=1e-4)
    stat, p = survdiff(df["T"], df.Status, df.Group, strata=df.strata,
                       weight_type="fh", fh_p=1)
    assert_allclose(stat, 12.73565, atol=1e-4, rtol=1e-4)

    # 8 strata
    df["strata"] = np.arange(df.shape[0]) % 8
    stat, p = survdiff(df["T"], df.Status, df.Group, strata=df.strata)
    assert_allclose(stat, 12.12631, atol=1e-4, rtol=1e-4)
    stat, p = survdiff(df["T"], df.Status, df.Group, strata=df.strata,
                       weight_type="fh", fh_p=0.5)
    assert_allclose(stat, 12.9633, atol=1e-4, rtol=1e-4)
    stat, p = survdiff(df["T"], df.Status, df.Group, strata=df.strata,
                       weight_type="fh", fh_p=1)
    assert_allclose(stat, 13.35259, atol=1e-4, rtol=1e-4)


@dec.skipif(not have_matplotlib)
def test_plot_km():

    if pdf_output:
        from matplotlib.backends.backend_pdf import PdfPages
        pdf = PdfPages("test_survfunc.pdf")
    else:
        pdf = None

    sr1 = SurvfuncRight(ti1, st1)
    sr2 = SurvfuncRight(ti2, st2)

    fig = plot_survfunc(sr1)
    close_or_save(pdf, fig)

    fig = plot_survfunc(sr2)
    close_or_save(pdf, fig)

    fig = plot_survfunc([sr1, sr2])
    close_or_save(pdf, fig)

    # Plot the SAS BMT data
    gb = bmt.groupby("Group")
    sv = []
    for g in gb:
        s0 = SurvfuncRight(g[1]["T"], g[1]["Status"], title=g[0])
        sv.append(s0)
    fig = plot_survfunc(sv)
    ax = fig.get_axes()[0]
    ax.set_position([0.1, 0.1, 0.64, 0.8])
    ha, lb = ax.get_legend_handles_labels()
    leg = fig.legend([ha[k] for k in (0,2,4)], [lb[k] for k in (0,2,4)],
                     'center right')
    close_or_save(pdf, fig)

    # Simultaneous CB for BMT data
    ii = bmt.Group == "ALL"
    sf = SurvfuncRight(bmt.loc[ii, "T"], bmt.loc[ii, "Status"])
    fig = sf.plot()
    ax = fig.get_axes()[0]
    ax.set_position([0.1, 0.1, 0.64, 0.8])
    ha, lb = ax.get_legend_handles_labels()
    lcb, ucb = sf.simultaneous_cb(transform="log")
    plt.fill_between(sf.surv_times, lcb, ucb, color="lightgrey")
    lcb, ucb = sf.simultaneous_cb(transform="arcsin")
    plt.plot(sf.surv_times, lcb, color="darkgrey")
    plt.plot(sf.surv_times, ucb, color="darkgrey")
    plt.plot(sf.surv_times, sf.surv_prob - 2*sf.surv_prob_se, color="red")
    plt.plot(sf.surv_times, sf.surv_prob + 2*sf.surv_prob_se, color="red")
    plt.xlim(100, 600)
    close_or_save(pdf, fig)

    if pdf_output:
        pdf.close()
