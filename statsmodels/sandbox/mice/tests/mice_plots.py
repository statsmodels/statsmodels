"""
Generate a set of plots testing most of the features of the MICE
plotting methods.
"""

import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.sandbox.mice import mice
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

pdf = PdfPages("mice_plots.pdf")

n = 1000

x1 = np.random.normal(size=n)
x2 = np.random.normal(size=n)
x3 = np.random.normal(size=n)
x4 = np.random.normal(size=n)

endog = x1 - x3 + np.random.normal(size=n)

x1[10:90] = np.nan
x2[50:200] = np.nan
x3[80:150] = np.nan
endog[30:220] = np.nan

data = pd.DataFrame({"x1": x1, "x2": x2, "x3": x3, "x4": x4, "endog": endog})

imp = mice.ImputedData(data)

# Make a plot for all possible argument combinations.
for column_order in "raw", "pattern", "proportion":
    for row_order in "raw", "pattern", "proportion":
        for hide_complete_rows in False, True:
            for hide_complete_columns in False, True:
                for color_row_patterns in False, True:
                    plt.clf()
                    ax = plt.axes([0.1, 0.1, 0.8, 0.8])
                    fig = imp.plot_missing_pattern(ax=ax,
                              column_order=column_order,
                              row_order=row_order,
                              hide_complete_rows=hide_complete_rows,
                              hide_complete_columns=hide_complete_columns,
                              color_row_patterns=color_row_patterns)
                    ti1 = "column_order=%s, row_order=%s\n" % (
                        column_order, row_order)
                    ti2 = "hide_complete_rows=%s, hide_complete_columns=%s\n" % (
                        hide_complete_rows, hide_complete_columns)
                    ti3 = "color_row_patterns=%s" % color_row_patterns
                    ti = ti1 + ti2 + ti3
                    plt.title(ti, size=10)
                    pdf.savefig()


mi = mice.MICE("endog ~ x1 + x2 + x3 + x4", sm.OLS, imp)
mi.run(num_ds=1, skipnum=1, burnin=10)


imp.hist("endog")
pdf.savefig()

imp.hist("x1")
pdf.savefig()

for plot_points in False, True:
    for lowess_args in {}, {"frac": 0.9}:

        plt.clf()
        imp.bivariate_scatterplot("endog", "x1", lowess_args=lowess_args,
                                  plot_points=plot_points)
        pdf.savefig()

        plt.clf()
        imp.bivariate_scatterplot("x2", "x3", plot_points=plot_points,
                                  lowess_args=lowess_args)
        pdf.savefig()

pdf.close()
