Get Involved
============

Where to Start?
---------------

Use grep or download a tool like `grin <pypi.python.org/pypi/grin>`__ to search the code for TODO notes::

    grin -i -I "*.py*" todo

This shows almost 700 TODOs in the code base right now. Feel free to inquire on the mailing list about any of these.

Sandbox
-------

We currently have a large amount code in the :ref:`sandbox`. The medium term goal is to move much of this to feature branches as it gets worked on and remove the sandbox folder. Many of these models and functions are close to done, however, and we welcome any and all contributions to complete them, including refactoring, documentation, and tests. These models include generalized additive models (GAM), information theoretic models such as maximum entropy, survival models, systems of equation models, restricted least squares, panel data models, and time series models such as (G)ARCH.

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

Roadmap to 0.6
==============

Work on any of the big picture ideas is very welcome. Implementing these ideas requires some thought and changes will likely affect all the codebase.

Core Development
----------------

* Refactoring models structure to have consistent variable naming, methods, and signatures. Make sure `DRY <http://en.wikipedia.org/wiki/Don%27t_repeat_yourself>`__ is respected.

Statistics
----------

* Bootstrapping, jackknifing, or re-sampling framework.
