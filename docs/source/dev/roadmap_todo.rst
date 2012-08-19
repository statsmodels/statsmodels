Get Involved
============

Google Summer of Code 2012
--------------------------

Statsmodels is participating for the fourth time in `GSoC <http://www.google-melange.com/gsoc/homepage/google/gsoc2012>`__  under the umbrella of the `Python Software Foundation <http://python.org/psf/>`__. We have set up a `wiki page <https://github.com/statsmodels/statsmodels/wiki/GSoC-Ideas>`__ with ideas for projects. Feel free to contribute to the ideas page or contact the mailing list if you are interested in applying so we can coordinate on developing your application and project - the earlier the better.

Where to Start?
---------------

Use grep or download a tool like `grin <pypi.python.org/pypi/grin>`__ to search the code for TODO notes::

    grin -i -I "*.py*" todo

This shows almost 700 TODOs in the code base right now. Feel free to inquire on the mailing list about any of these.

Sandbox
-------

We currently have a large amount code in the :ref:`sandbox`. The medium term goal is to move much of this to feature branches as it gets worked on and remove the sandbox folder. Many of these models and functions are close to done, however, and we welcome any and all contributions to complete them, including refactoring, documentation, and tests. These models include generalized additive models (GAM), information theoretic models such as maximum entropy and empirical likelihood, survival models, systems of equation models, restricted least squares, panel data models, and time series models such as (G)ARCH.

.. .. toctree::
..   :maxdepth: 4
..
..   ../sandbox

Contribute an Example
---------------------

Link to examples documentation. Examples and technical documentation.

Contribute to the Gallery
-------------------------

Link to the Gallery.

Roadmap to 0.5
==============

Work on any of the big picture ideas is very welcome. Implementing these ideas requires some thought and changes will likely affect all the codebase.

Formula Framework
-----------------

Existing Discussions:

* `R-like formulas - 2-10-2010 <http://groups.google.com/group/pystatsmodels/browse_thread/thread/1f99c1e2a7d9c588/>`__
* `The Return of Formula (?) - 5-16-2010 <http://groups.google.com/group/pystatsmodels/browse_thread/thread/d3a32b834ce153d2/>`__
* `The Return of Formula: The Revenge: The Novel - 6-4-2010 <http://groups.google.com/group/pystatsmodels/browse_thread/thread/9636cb2f8a0d37cf/>`__

Existing Implementations:

* `Jonathan Taylor's Formula <https://github.com/jonathan-taylor/formula>`__

  * `Forked to statsmodels repository <https://github.com/statsmodels/formula>`__
* `Nathaniel Smith's Patsy <https://github.com/pydata/patsy>`__

Open questions:

* What level of integrations with data structures is desirable?
* User API spec.

Core Development
----------------

* Refactoring models structure to have consistent variable naming, methods, and signatures. Make sure `DRY <http://en.wikipedia.org/wiki/Don%27t_repeat_yourself>`__ is respected.

Statistics
----------

* Bootstrapping, jackknifing, or re-sampling framework.
