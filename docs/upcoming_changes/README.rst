Release-note fragments
======================

Pull requests do not edit the release notes in ``docs/source/release``.
Each one adds its release-note bullets in a fragment file in this directory,
so that two pull requests never conflict in the release notes. Before a
release, ``tools/releasing/collect_release_notes.py`` moves the fragments
into the release notes.

File names
----------

``<pr number>.<key>.rst``, for example ``10849.tsa.rst``. The pull request
number is known once the pull request is open, so add the fragment in a
commit right after opening it. A pull request with notes in several
sections adds one file per section, for example ``10889.tsa.rst`` and
``10889.deprecation.rst``.

The key is the section of the release notes:

==========================  ======================================================
Key                         Section
==========================  ======================================================
``deprecation``             Breaking Changes and Deprecations / Deprecations
``behavior``                Breaking Changes and Deprecations / Behavior changes
``feature``                 New Features and Enhancements / New functions and methods
``performance``             New Features and Enhancements / Performance
``regression``              Bug Fixes / Regression and linear models
``glm``                     Bug Fixes / Generalized linear models and discrete models
``mixed``                   Bug Fixes / Mixed models and imputation
``tsa``                     Bug Fixes / Time series analysis
``stats``                   Bug Fixes / Statistics and tests
``treatment``               Bug Fixes / Treatment effects
``robust``                  Bug Fixes / Robust and multivariate statistics
``nonparametric``           Bug Fixes / Nonparametric
``duration``                Bug Fixes / Duration models
``other``                   Bug Fixes / Other
``documentation``           Documentation
``validation-confidence``   Stricter Input Validation / Confidence level and significance
``validation-counts``       Stricter Input Validation / Counts, sample sizes, weights and probabilities
``validation-intervals``    Stricter Input Validation / Intervals, tuning parameters and shapes
``validation-tsa``          Stricter Input Validation / Time series
==========================  ======================================================

Content
-------

The bullets exactly as they will appear in the release notes: each starts
with ``- ``, continuation lines are indented by two spaces, there are no
blank lines, and every bullet ends with its roles, at least
``:pr:`<pr number>```. For example, ``10849.tsa.rst``::

    - ``levinson_durbin`` with ``isacov=True`` treats an (nobs, 1) column as
      the 1-D series instead of raising. :issue:`1234`, :pr:`10849`

Check the fragments with::

    python tools/releasing/collect_release_notes.py --check

At release time
---------------

Move the fragments into the release notes, in the order of the pull request
numbers, and delete them::

    python tools/releasing/collect_release_notes.py --remove docs/source/release/versionX.Y.Z.rst

Each fragment's bullets go to the end of the list of their section.
Review the result, then commit the release notes and the removed fragments
together.
