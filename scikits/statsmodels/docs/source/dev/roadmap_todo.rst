Roadmap to 0.4
==============

Pandas Integration and Improvements
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

* Make models able to take pandas DataFrame (or Panel for panelmod).
  * Started in pandas-integration branch
* Plotting integration of pandas data structures with matplotlib/scikits.timeseries.lib.plotlib

  * Merge with/fork from scikits.timeseries?

* Refactoring of pandas.Panel structures. Find common underlying structure
  for Long and Wide.

Formula Framework
^^^^^^^^^^^^^^^^^

Existing Discussions:

* `R-like formulas - 2-10-2010 <http://groups.google.com/group/pystatsmodels/browse_thread/thread/1f99c1e2a7d9c588/>`__
* `The Return of Formula (?) - 5-16-2010 <http://groups.google.com/group/pystatsmodels/browse_thread/thread/d3a32b834ce153d2/>`__
* `The Return of Formula: The Revenge: The Novel - 6-4-2010 <http://groups.google.com/group/pystatsmodels/browse_thread/thread/9636cb2f8a0d37cf/>`__

Existing Implementations:

* `Jonathan Taylor's Formula <https://github.com/jonathan-taylor/formula>`__

  * `Forked to statsmodels repository <https://github.com/statsmodels/formula>`__
* `Nathaniel Smith's Charlton <https://github.com/charlton>`__

  * `Forked to statsmodels repository <https://github.com/statsmodels/charlton>`__

Open questions:

* What level of integrations with data structures is desirable?
* User API spec.

Core Development
^^^^^^^^^^^^^^^^

* Refactoring models structure. Make sure `DRY <http://en.wikipedia.org/wiki/Don%27t_repeat_yourself>`__ is respected.

Statistics
^^^^^^^^^^

* Bootstrapping, Jackknifing, Re-sampling framework.

Sandbox
^^^^^^^

We currently have a large amount code in the sandbox. The medium term goal
is to move much of this to feature branches as it gets worked on and remove
the sandbox folder. Many of these models and functions are close to done,
however, and we welcome any and all contributions to complete them, including
refactoring, documentation, and tests.

.. toctree::
   :maxdepth: 4

   ../sandbox

.. _todo:

    Fill in upcoming goals.
