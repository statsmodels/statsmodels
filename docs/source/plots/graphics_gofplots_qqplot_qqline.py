'''
    Import the food expenditure dataset.  Plot annual food expendeture on
    x-axis and household income on y-axis.  Use qqline to add regression line
    into the plot.
'''
import statsmodels.api as sm
import matplotlib.pyplot as plt
from statsmodels.graphics.gofplots import qqline

foodexp = sm.datasets.engel.load(as_pandas=False)
x = foodexp.exog
y = foodexp.endog
ax = plt.subplot(111)
plt.scatter(x, y)
ax.set_xlabel(foodexp.exog_name[0])
ax.set_ylabel(foodexp.endog_name)
qqline(ax, 'r', x, y)
plt.show()
