from statsmodels.sandbox.predict_functional import predict_functional, predict_functional_glm
import numpy as np
import pandas as pd
import statsmodels.api as sm
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


def formula_example(plt, pdf):

    np.random.seed(542)
    n = 500
    x1 = np.random.normal(size=n)
    x2 = np.random.normal(size=n)
    x3 = np.random.normal(size=n)
    x4 = np.random.randint(0, 5, size=n)
    x4 = np.asarray(["ABCDE"[i] for i in x4])
    x5 = np.random.normal(size=n)
    y = 0.3*x2**2 + (x4 == "B") + 0.1*(x4 == "B")*x2**2 + x5 + np.random.normal(size=n)

    df = pd.DataFrame({"y": y, "x1": x1, "x2": x2, "x3": x3, "x4": x4, "x5": x5})

    fml = "y ~ x1 + bs(x2, df=4) + x3 + x2*x3 + I(x1**2) + C(x4) + C(x4)*bs(x2, df=4) + x5"
    model = sm.OLS.from_formula(fml, data=df)
    result = model.fit()

    def pctl(q):
        return lambda x : np.percentile(x, 100 *q)

    summaries = {"x1": np.mean, "x3": pctl(0.75), "x5": np.mean}

    values = {"x4": "B"}
    pr1, ci1, fvals1 = predict_functional(result, "x2", summaries, values)

    values = {"x4": "C"}
    pr2, ci2, fvals2 = predict_functional(result, "x2", summaries, values)

    plt.clf()
    fig = plt.figure()
    ax = plt.axes([0.1, 0.1, 0.7, 0.8])
    plt.plot(fvals1, pr1, '-', label='x4=B')
    plt.plot(fvals2, pr2, '-', label='x4=C')
    ha, lb = ax.get_legend_handles_labels()
    plt.figlegend(ha, lb, "center right")
    plt.xlabel("Focus variable", size=15)
    plt.ylabel("Fitted mean", size=15)
    plt.title("Using formula")
    close_or_save(pdf, fig)

    plt.clf()
    fig = plt.figure()
    ax = plt.axes([0.1, 0.1, 0.7, 0.8])
    plt.plot(fvals1, pr1, '-', label='x4=B')
    plt.fill_between(fvals1, ci1[:, 0], ci1[:, 1], color='grey')
    plt.plot(fvals2, pr2, '-', label='x4=C')
    plt.fill_between(fvals2, ci2[:, 0], ci2[:, 1], color='grey')
    ha, lb = ax.get_legend_handles_labels()
    plt.figlegend(ha, lb, "center right")
    plt.xlabel("Focus variable", size=15)
    plt.ylabel("Fitted mean", size=15)
    plt.title("Using formula, pointwise CB")
    close_or_save(pdf, fig)



def scb_example(plt, pdf):

    np.random.seed(473)
    n = 100
    x = np.random.normal(size=(n,4))
    x[:, 0] = 1

    for fam_name in "poisson", "binomial", "gaussian":

        if fam_name == "poisson":
            y = np.random.poisson(20, size=n)
            fam = sm.families.Poisson()
            true_mean = 20
            true_lp = np.log(20)
        elif fam_name == "binomial":
            y = 1 * (np.random.uniform(size=n) < 0.5)
            fam = sm.families.Binomial()
            true_mean = 0.5
            true_lp = 0
        elif fam_name == "gaussian":
            y = np.random.normal(size=n)
            fam = sm.families.Gaussian()
            true_mean = 0
            true_lp = 0

        model = sm.GLM(y, x, family=fam)
        result = model.fit()

        # CB is for linear predictor or mean response
        for linear in False, True:

            true = true_lp if linear else true_mean

            values = {'const': 1, "x2": 0}
            summaries = {"x3": np.mean}
            pred1, cb1, fvals1 = predict_functional_glm(result, "x1",
                        values=values, summaries=summaries, linear=linear)
            pred2, cb2, fvals2 = predict_functional_glm(result, "x1",
                        values=values, summaries=summaries, simultaneous=True,
                        linear=linear)

            plt.clf()
            fig = plt.figure()
            ax = plt.axes([0.1, 0.1, 0.58, 0.8])
            plt.plot(fvals1, pred1, '-', color='black', label='Estimate')
            plt.plot(fvals1, true * np.ones(len(pred1)), '-', color='purple',
                     label='Truth')
            plt.plot(fvals1, cb1[:, 0], color='blue', label='Pointwise CB')
            plt.plot(fvals1, cb1[:, 1], color='blue')
            plt.plot(fvals2, cb2[:, 0], color='green', label='Simultaneous CB')
            plt.plot(fvals2, cb2[:, 1], color='green')
            ha, lb = ax.get_legend_handles_labels()
            leg = plt.figlegend(ha, lb, "center right")
            leg.draw_frame(False)
            plt.xlabel("Focus variable", size=15)
            if linear:
                plt.ylabel("Linear predictor", size=15)
            else:
                plt.ylabel("Fitted mean", size=15)
            plt.title("Using formula, %s family, constant mean" % fam_name.capitalize())

            close_or_save(pdf, fig)


def glm_formula_example(plt, pdf):

    np.random.seed(542)
    n = 500
    x1 = np.random.normal(size=n)
    x2 = np.random.normal(size=n)
    x3 = np.random.randint(0, 3, size=n)
    x3 = np.asarray(["ABC"[i] for i in x3])
    lin_pred = -2 + 0.5*x1**2 + 2*(x3 == "B")
    prob = 1 / (1 + np.exp(-lin_pred))
    y = 1 * (np.random.uniform(size=n) < prob)

    df = pd.DataFrame({"y": y, "x1": x1, "x2": x2, "x3": x3})

    fml = "y ~ bs(x1, df=4) + x2 + C(x3)"
    model = sm.GLM.from_formula(fml, family=sm.families.Binomial(), data=df)
    result = model.fit()

    def pctl(q):
        return lambda x : np.percentile(x, 100 *q)

    summaries = {"x2": np.mean}

    values = {"x3": "B"}
    pr1, ci1, fvals1 = predict_functional_glm(result, "x1", summaries, values)

    values = {"x3": "C"}
    pr2, ci2, fvals2 = predict_functional_glm(result, "x1", summaries, values)

    plt.clf()
    fig = plt.figure()
    ax = plt.axes([0.1, 0.1, 0.7, 0.8])
    plt.plot(fvals1, pr1, '-', label='x3=B')
    plt.plot(fvals2, pr2, '-', label='x3=C')
    ha, lb = ax.get_legend_handles_labels()
    plt.figlegend(ha, lb, "center right")
    plt.xlabel("Focus variable", size=15)
    plt.ylabel("Fitted mean", size=15)
    plt.title("Using formula, Binomial GLM")
    close_or_save(pdf, fig)

    plt.clf()
    fig = plt.figure()
    ax = plt.axes([0.1, 0.1, 0.7, 0.8])
    plt.plot(fvals1, pr1, '-', label='x3=B')
    plt.fill_between(fvals1, ci1[:, 0], ci1[:, 1], color='grey')
    plt.plot(fvals2, pr2, '-', label='x3=C')
    plt.fill_between(fvals2, ci2[:, 0], ci2[:, 1], color='grey')
    ha, lb = ax.get_legend_handles_labels()
    plt.figlegend(ha, lb, "center right")
    plt.xlabel("Focus variable", size=15)
    plt.ylabel("Fitted mean", size=15)
    plt.title("Using formula, Binomial GLM, pointwise CB")
    close_or_save(pdf, fig)


def noformula_example(plt, pdf):

    np.random.seed(6434)
    n = 200
    x1 = np.random.normal(size=n)
    x2 = np.random.normal(size=n)
    x3 = np.random.normal(size=n)
    y = x1 - x2 + np.random.normal(size=n)

    exog = np.vstack((x1, x2, x3)).T

    model = sm.OLS(y, exog)
    result = model.fit()

    def pctl(q):
        return lambda x : np.percentile(x, 100 *q)

    summaries = {"x3": pctl(0.75)}
    values = {"x2": 1}
    pr1, ci1, fvals1 = predict_functional(result, "x1", summaries, values)

    values = {"x2": -1}
    pr2, ci2, fvals2 = predict_functional(result, "x1", summaries, values)

    plt.clf()
    fig = plt.figure()
    ax = plt.axes([0.1, 0.1, 0.7, 0.8])
    plt.plot(fvals1, pr1, '-', label='x2=1', lw=4, alpha=0.6)
    plt.plot(fvals2, pr2, '-', label='x2=-1', lw=4, alpha=0.6)
    ha, lb = ax.get_legend_handles_labels()
    leg = plt.figlegend(ha, lb, "center right")
    leg.draw_frame(False)
    plt.xlabel("Focus variable", size=15)
    plt.ylabel("Fitted mean", size=15)
    plt.title("Not using formula")
    close_or_save(pdf, fig)

    plt.clf()
    fig = plt.figure()
    ax = plt.axes([0.1, 0.1, 0.7, 0.8])
    plt.plot(fvals1, pr1, '-', label='x2=1', lw=4, alpha=0.6)
    plt.fill_between(fvals1, ci1[:, 0], ci1[:, 1], color='grey')
    plt.plot(fvals1, pr2, '-', label='x2=1', lw=4, alpha=0.6)
    plt.fill_between(fvals2, ci2[:, 0], ci2[:, 1], color='grey')
    ha, lb = ax.get_legend_handles_labels()
    plt.figlegend(ha, lb, "center right")
    plt.xlabel("Focus variable", size=15)
    plt.ylabel("Fitted mean", size=15)
    plt.title("Not using formula, pointwise CB")
    close_or_save(pdf, fig)


@dec.skipif(not have_matplotlib)
def test_all():

    if pdf_output:
        from matplotlib.backends.backend_pdf import PdfPages
        pdf = PdfPages("predict_functional.pdf")
    else:
        pdf = None

    scb_example(plt, pdf)
    formula_example(plt, pdf)
    glm_formula_example(plt, pdf)
    noformula_example(plt, pdf)

    if pdf_output:
        pdf.close()
