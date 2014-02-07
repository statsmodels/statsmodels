:orphan:

===========
0.6 Release
===========

Release 0.6.0
=============

Release summary.

Major changes:

Addition of Generalized Estimating Equations GEE



Header for Change
~~~~~~~~~~~~~~~~~

Change blurb and example code.

Seasonality Plots
~~~~~~~~~~~~~~~~~

Adding functionality to look at seasonality in plots. Two new functions are :func:`sm.graphics.tsa.month_plot` and :func:`sm.graphics.tsa.quarter_plot`. Another function :func:`sm.graphics.tsa.seasonal_plot` is available for power users.

.. code-block:: python

    import statsmodels.api as sm
    import pandas as pd

    dta = sm.datasets.elnino.load_pandas().data
    dta['YEAR'] = dta.YEAR.astype(int).astype(str)
    dta = dta.set_index('YEAR').T.unstack()
    dates = map(lambda x : pd.datetools.parse('1 '+' '.join(x)),
                                           dta.index.values)

    dta.index = pd.DatetimeIndex(dates, freq='M')
    fig = sm.tsa.graphics.month_plot(dta)


Other important new features
----------------------------

* Added :func:`sm.tsa.arma_order_select_ic`. A convenience function to quickly get the information criteria for use in tentative order selection of ARMA processes.
* Plotting functions for timeseries is now imported under the ``sm.tsa.graphics`` namespace in addition to ``sm.graphics.tsa``.

Major Bugs fixed
----------------

* Bullet list of major bugs
* With a link to its github issue.
* Use the syntax ``:ghissue:`###```.

Backwards incompatible changes and deprecations
-----------------------------------------------

* RegressionResults.norm_resid is now a readonly property, rather than a function.

Development summary and credits
-------------------------------

A blurb about the number of changes and the contributors list.

.. note::

   Obtained by running ``git log v0.5.0..HEAD --format='* %aN <%aE>' | sed 's/@/\-at\-/' | sed 's/<>//' | sort -u``.

