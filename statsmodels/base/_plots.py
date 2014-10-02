"""
Regression graphics that are applicable to any model (with endog,
exog, and predict).

These functions are not intended to be directly called by users, there
are wrappers in RegressionResults that use these functions.
"""

import numpy as np
from statsmodels.graphics import utils
from statsmodels.nonparametric.smoothers_lowess import lowess

def covariate_effect_plot(results, focus_col, exog=None,
                          summary_type=None, show_hist=True,
                          hist_kwargs=None, ax=None):
    """
    See base.model.results.covariate_effect_plot for documentation.
    """

    fig, ax1 = utils.create_mpl_ax(ax)

    model = results.model

    if exog is None and summary_type is None:
        summary_type = -1
    elif exog is not None and summary_type is not None:
        raise ValueError("Only one of `exog` and `summary_type` may be provided")

    m_exog = model.exog

    # Construct exog from summary functions
    if summary_type is not None:

        from scipy.stats import mode

        def summarize(x, tp):
            if tp == -1:
                return np.mean(x)
            elif tp == -2:
                return mode(x)[0][0]
            else:
                return np.percentile(x, 100 * tp)

        exog = []

        # A common summary for all columns
        if isinstance(summary_type, (int, float)):
            exog = [summarize(x, summary_type) for x in m_exog.T]
            exog = np.asarray(exog)[None, :]

        # A list of summary functions, one for each column
        elif isinstance(summary_type, list) and not isinstance(summary_type[0], list):
            if len(summary_type) != m_exog.shape[1]:
                raise ValueError("wrong number of summary types")
            for j, t in enumerate(summary_type):
                if j != focus_col:
                    exog.append(summarize(m_exog[:, j], t))
                else:
                    exog.append(0.)
            exog = np.asarray(exog)[None, :]

        # A list of lists of summary functions
        else:
            for stype in summary_type:
                if len(stype) != m_exog.shape[1]:
                    raise ValueError("wrong number of summary types")
                exog1 = []
                for j, t in enumerate(stype):
                    if j != focus_col:
                        exog1.append(summarize(m_exog[:, j], t))
                    else:
                        exog1.append(0.)
                exog.append(exog1)
            exog = np.asarray(exog)

    # exog is provided directly
    else:
        if exog.ndim == 1:
            exog = exog[None, :]
        if exog.shape[1] != m_exog.shape[1]:
            raise ValueError("wrong shape for exog")

    npt = 50 # Number of points on each curve
    focus_data = np.linspace(m_exog[:, focus_col].min(),
                             m_exog[:, focus_col].max(), npt)

    # Draw the mean curves
    for j in range(exog.shape[0]):
        new_exog = np.outer(np.ones(npt), exog[j, :])
        new_exog[:, focus_col] = focus_data

        pred_val = results.predict(exog=new_exog)
        ax1.plot(focus_data, pred_val, '-', lw=7,
                 label="mean_curve_%d" % j, alpha=0.7)

    # Darw the histogram
    if show_hist:
        ax2 = ax1.twinx()
        ax2.set_yticks([])
        ax2.set_yticklabels([])
        ha = {"histtype": "step", "normed": True}
        if hist_kwargs is not None:
            ha.update(hist_kwargs)
        _, _, pa = ax2.hist(m_exog[:, focus_col], **ha)
        for p in pa:
            p.set_color("grey")


    # TODO: Use the name of the variable if available
    ax1.set_xlabel("Exog column %d" % focus_col, size=15)
    ax1.set_ylabel("Fitted mean", size=15)

    return fig
