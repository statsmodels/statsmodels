.. currentmodule:: statsmodels.duration.hazard_regression


.. _duration:

Models for Survival and Duration Analysis
=========================================

currently contains Cox's Proportional Hazard Model.


Examples
--------

::

  url = "http://vincentarelbundock.github.io/Rdatasets/csv/survival/flchain.csv"
  data = pd.read_csv(url)
  del data["chapter"]
  data = data.dropna()
  data["lam"] = data["lambda"]
  data["female"] = 1*(data["sex"] == "F")
  data["year"] = data["sample.yr"] - min(data["sample.yr"])

  status = np.asarray(data["death"])
  mod = PHreg.from_formula("futime ~ 0 + age + female + creatinine + " +
                           "np.sqrt(kappa) + np.sqrt(lam) + year + mgus",
                           data, status=status, ties="efron")
  rslt = mod.fit()
  print(rslt.summary())

Detailed examples can be found here:

.. toctree::
    :maxdepth: 2

    examples/notebooks/generated/    not yet

There some notebook examples on the Wiki:
`Wiki notebooks for PHReg and Survival Analysis <https://github.com/statsmodels/statsmodels/wiki/Examples#survival-analysis>`_



Technical Documentation
-----------------------

TODO


References
^^^^^^^^^^

References for Cox proportional hazards regression model::

T Therneau (1996).  Extending the Cox model.  Technical report.
http://www.mayo.edu/research/documents/biostat-58pdf/DOC-10027288

G Rodriguez (2005).  Non-parametric estimation in survival models.
http://data.princeton.edu/pop509/NonParametricSurvival.pdf

B Gillespie (2006).  Checking the assumptions in the Cox proportional
hazards model.
http://www.mwsug.org/proceedings/2006/stats/MWSUG-2006-SD08.pdf


Module Reference
----------------

The model class is:

.. autosummary::
   :toctree: generated/

   PHReg

The result classe is:

.. autosummary::
   :toctree: generated/

   PHRegResults
