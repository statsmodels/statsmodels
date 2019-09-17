.. module:: statsmodels.duration
   :synopsis: Models for durations

.. currentmodule:: statsmodels.duration


.. _duration:

Methods for Survival and Duration Analysis
==========================================

:mod:`statsmodels.duration` implements several standard methods for
working with censored data.  These methods are most commonly used when
the data consist of durations between an origin time point and the
time at which some event of interest occurred.  A typical example is a
medical study in which the origin is the time at which a subject is
diagnosed with some condition, and the event of interest is death (or
disease progression, recovery, etc.).

Currently only right-censoring is handled.  Right censoring occurs
when we know that an event occurred after a given time `t`, but we do
not know the exact event time.

Survival function estimation and inference
------------------------------------------

The :class:`statsmodels.api.SurvfuncRight` class can be used to
estimate a survival function using data that may be right censored.
``SurvfuncRight`` implements several inference procedures including
confidence intervals for survival distribution quantiles, pointwise
and simultaneous confidence bands for the survival function, and
plotting procedures.  The ``duration.survdiff`` function provides
testing procedures for comparing survival distributions.

Here we create a ``SurvfuncRight`` object using data from the
`flchain` study, which is available through the R datasets repository.
We fit the survival distribution only for the female subjects.


.. code-block:: python

   import statsmodels.api as sm

   data = sm.datasets.get_rdataset("flchain", "survival").data
   df = data.loc[data.sex == "F", :]
   sf = sm.SurvfuncRight(df["futime"], df["death"])

The main features of the fitted survival distribution can be seen by
calling the ``summary`` method:

.. code-block:: python

    sf.summary().head()

We can obtain point estimates and confidence intervals for quantiles
of the survival distribution.  Since only around 30% of the subjects
died during this study, we can only estimate quantiles below the 0.3
probability point:

.. code-block:: python

    sf.quantile(0.25)
    sf.quantile_ci(0.25)

To plot a single survival function, call the ``plot`` method:

.. code-block:: python

    sf.plot()

Since this is a large dataset with a lot of censoring, we may wish
to not plot the censoring symbols:

.. code-block:: python

    fig = sf.plot()
    ax = fig.get_axes()[0]
    pt = ax.get_lines()[1]
    pt.set_visible(False)

We can also add a 95% simultaneous confidence band to the plot.
Typically these bands only plotted for central part of the
distribution.

.. code-block:: python

    fig = sf.plot()
    lcb, ucb = sf.simultaneous_cb()
    ax = fig.get_axes()[0]
    ax.fill_between(sf.surv_times, lcb, ucb, color='lightgrey')
    ax.set_xlim(365, 365*10)
    ax.set_ylim(0.7, 1)
    ax.set_ylabel("Proportion alive")
    ax.set_xlabel("Days since enrollment")

Here we plot survival functions for two groups (females and males) on
the same axes:

.. code-block:: python

    gb = data.groupby("sex")
    ax = plt.axes()
    sexes = []
    for g in gb:
        sexes.append(g[0])
        sf = sm.SurvfuncRight(g[1]["futime"], g[1]["death"])
        sf.plot(ax)
    li = ax.get_lines()
    li[1].set_visible(False)
    li[3].set_visible(False)
    plt.figlegend((li[0], li[2]), sexes, "center right")
    plt.ylim(0.6, 1)
    ax.set_ylabel("Proportion alive")
    ax.set_xlabel("Days since enrollment")

We can formally compare two survival distributions with ``survdiff``,
which implements several standard nonparametric procedures.  The
default procedure is the logrank test:

.. code-block:: python

    stat, pv = sm.duration.survdiff(data.futime, data.death, data.sex)

Here are some of the other testing procedures implemented by survdiff:

.. code-block:: python

    # Fleming-Harrington with p=1, i.e. weight by pooled survival time
    stat, pv = sm.duration.survdiff(data.futime, data.death, data.sex, weight_type='fh', fh_p=1)

    # Gehan-Breslow, weight by number at risk
    stat, pv = sm.duration.survdiff(data.futime, data.death, data.sex, weight_type='gb')

    # Tarone-Ware, weight by the square root of the number at risk
    stat, pv = sm.duration.survdiff(data.futime, data.death, data.sex, weight_type='tw')


Regression methods
------------------

Proportional hazard regression models ("Cox models") are a regression
technique for censored data.  They allow variation in the time to an
event to be explained in terms of covariates, similar to what is done
in a linear or generalized linear regression model.  These models
express the covariate effects in terms of "hazard ratios", meaning the
the hazard (instantaneous event rate) is multiplied by a given factor
depending on the value of the covariates.


.. code-block:: python

   import statsmodels.api as sm
   import statsmodels.formula.api as smf

   data = sm.datasets.get_rdataset("flchain", "survival").data
   del data["chapter"]
   data = data.dropna()
   data["lam"] = data["lambda"]
   data["female"] = (data["sex"] == "F").astype(int)
   data["year"] = data["sample.yr"] - min(data["sample.yr"])
   status = data["death"].values

   mod = smf.phreg("futime ~ 0 + age + female + creatinine + "
                   "np.sqrt(kappa) + np.sqrt(lam) + year + mgus",
                   data, status=status, ties="efron")
   rslt = mod.fit()
   print(rslt.summary())


See :ref:`statsmodels-examples` for more detailed examples.


There are some notebook examples on the Wiki:
`Wiki notebooks for PHReg and Survival Analysis <https://github.com/statsmodels/statsmodels/wiki/Examples#survival-analysis>`_


.. todo::

   Technical Documentation

References
^^^^^^^^^^

References for Cox proportional hazards regression model::

    T Therneau (1996). Extending the Cox model. Technical report.
    http://www.mayo.edu/research/documents/biostat-58pdf/DOC-10027288

    G Rodriguez (2005). Non-parametric estimation in survival models.
    http://data.princeton.edu/pop509/NonParametricSurvival.pdf

    B Gillespie (2006). Checking the assumptions in the Cox proportional
    hazards model.
    http://www.mwsug.org/proceedings/2006/stats/MWSUG-2006-SD08.pdf


Module Reference
----------------

.. module:: statsmodels.duration.survfunc
   :synopsis: Models for Survival Analysis

.. currentmodule:: statsmodels.duration.survfunc

The class for working with survival distributions is:

.. autosummary::
   :toctree: generated/

   SurvfuncRight

.. module:: statsmodels.duration.hazard_regression
   :synopsis: Proportional hazards model for Survival Analysis

.. currentmodule:: statsmodels.duration.hazard_regression

The proportional hazards regression model class is:

.. autosummary::
   :toctree: generated/

   PHReg

The proportional hazards regression result class is:

.. autosummary::
   :toctree: generated/

   PHRegResults
