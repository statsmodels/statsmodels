:hero: statistical models, hypothesis tests, and data exploration

statsmodels documentation
=========================

.. container:: sm-landing-meta

   **Date:** |today| **Version:** |version|

   **Install:** ``python -m pip install statsmodels``

   **Previous versions:** Documentation of previous statsmodels versions is
   available at `statsmodels.org <https://www.statsmodels.org/>`__.

   **Useful links:** `Binary Installers <https://pypi.org/project/statsmodels/>`__
   | `Source Repository <https://github.com/statsmodels/statsmodels/>`__
   | `Issues & Ideas <https://github.com/statsmodels/statsmodels/issues>`__
   | `Q&A Support <https://stackoverflow.com/questions/tagged/statsmodels>`__
   | `Mailing List <https://groups.google.com/forum/?hl=en#!forum/pystatsmodels>`__
   | `DOI <https://doi.org/10.5281/zenodo.593847>`__

.. container:: sm-landing-summary

   ``statsmodels`` provides classes and functions for estimating statistical
   models, running hypothesis tests, and exploring data in Python.

.. raw:: html

   <section class="sm-landing-grid" aria-label="Main documentation sections">
     <a class="sm-landing-card" href="gettingstarted.html">
       <span class="sm-landing-card__icon" aria-hidden="true"><i class="fa-solid fa-rocket"></i></span>
       <span class="sm-landing-card__label">Start here</span>
       <span class="sm-landing-card__title">Getting started</span>
       <span class="sm-landing-card__text">Install statsmodels, fit a first model, and learn the core workflow.</span>
       <span class="sm-landing-card__cta">To the getting started guide</span>
     </a>
     <a class="sm-landing-card" href="user-guide.html">
       <span class="sm-landing-card__icon" aria-hidden="true"><i class="fa-solid fa-book-open"></i></span>
       <span class="sm-landing-card__label">Learn</span>
       <span class="sm-landing-card__title">User Guide</span>
       <span class="sm-landing-card__text">Explore statistical models, tools, diagnostics, and workflows by topic.</span>
       <span class="sm-landing-card__cta">To the user guide</span>
     </a>
     <a class="sm-landing-card" href="examples/index.html">
       <span class="sm-landing-card__icon" aria-hidden="true"><i class="fa-solid fa-chart-line"></i></span>
       <span class="sm-landing-card__label">Practice</span>
       <span class="sm-landing-card__title">Examples</span>
       <span class="sm-landing-card__text">Browse applied notebooks and recipes using real datasets and model outputs.</span>
       <span class="sm-landing-card__cta">To the examples</span>
     </a>
     <a class="sm-landing-card" href="api.html">
       <span class="sm-landing-card__icon" aria-hidden="true"><i class="fa-solid fa-code"></i></span>
       <span class="sm-landing-card__label">Reference</span>
       <span class="sm-landing-card__title">API Reference</span>
       <span class="sm-landing-card__text">Find classes, functions, result objects, and module-level documentation.</span>
       <span class="sm-landing-card__cta">To the API reference</span>
     </a>
   </section>

.. container:: sm-index-logo

   .. image:: images/statsmodels-logo-v2-horizontal.svg
      :alt: statsmodels
      :class: sm-index-logo__image sm-index-logo__image--light only-light

   .. image:: images/statsmodels-logo-v3-horizontal.svg
      :alt: statsmodels
      :class: sm-index-logo__image sm-index-logo__image--dark only-dark

:ref:`statsmodels <about:About statsmodels>` is a Python module that provides classes and functions for the estimation
of many different statistical models, as well as for conducting statistical tests, and statistical
data exploration. An extensive list of result statistics are available for each estimator.
The results are tested against existing statistical packages to ensure that they are correct. The
package is released under the open source Modified BSD (3-clause) license.
The online documentation is hosted at `statsmodels.org <https://www.statsmodels.org/>`__.

Introduction
============

``statsmodels`` supports specifying models using R-style formulas and ``pandas`` DataFrames.
Here is a simple example using ordinary least squares:

.. ipython:: python

    import numpy as np
    import statsmodels.api as sm
    import statsmodels.formula.api as smf

    # Load data
    dat = sm.datasets.get_rdataset("Guerry", "HistData").data

    # Fit regression model (using the natural log of one of the regressors)
    results = smf.ols('Lottery ~ Literacy + np.log(Pop1831)', data=dat).fit()

    # Inspect the results
    print(results.summary())

You can also use ``numpy`` arrays instead of formulas:

.. ipython:: python

    import numpy as np
    import statsmodels.api as sm

    # Generate artificial data (2 regressors + constant)
    nobs = 100
    X = np.random.random((nobs, 2))
    X = sm.add_constant(X)
    beta = [1, .1, .5]
    e = np.random.random(nobs)
    y = np.dot(X, beta) + e

    # Fit regression model
    results = sm.OLS(y, X).fit()

    # Inspect the results
    print(results.summary())

Have a look at `dir(results)` to see available results. Attributes are described in
`results.__doc__` and results methods have their own docstrings.

Citation
========

Please use following citation to cite statsmodels in scientific publications:


Seabold, Skipper, and Josef Perktold. "`statsmodels: Econometric and statistical modeling with
python. <https://proceedings.scipy.org/articles/proceedings-2010.pdf>`_" *Proceedings
of the 9th Python in Science Conference.* 2010.

Bibtex entry::

  @inproceedings{seabold2010statsmodels,
    title={statsmodels: Econometric and statistical modeling with python},
    author={Seabold, Skipper and Perktold, Josef},
    booktitle={9th Python in Science Conference},
    year={2010},
  }

.. toctree::
   :maxdepth: 1

   install
   gettingstarted
   user-guide
   examples/index
   api
   about
   dev/index
   release/index


Index
=====

:ref:`genindex`

:ref:`modindex`
