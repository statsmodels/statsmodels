.. currentmodule:: statsmodels.duration.hazard_regression


.. _duration:

Models for Survival and Duration Analysis
=========================================

Examples
--------

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


Detailed examples can be found here:

.. toctree::
    :maxdepth: 2

    examples/notebooks/generated/


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

The model class is:

.. autosummary::
   :toctree: generated/

   PHReg

The result class is:

.. autosummary::
   :toctree: generated/

   PHRegResults
