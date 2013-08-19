Contributing guidelines
=======================

This page explains how you can contribute to the development of `statsmodels`
by submitting patches, statistical tests, new models, or examples. 

`statsmodels` is developed on `Github <https://github.com/statsmodels/statsmodels>`_ 
using the `Git <http://git-scm.com/>`_ version control system. 

Submitting a Bug Report
~~~~~~~~~~~~~~~~~~~~~~~

- Include a short, self-contained code snippet that reproduces the problem
- Specify the statsmodels version used. You can do this with ``sm.version.full_version``
- If the issue looks to involve other dependencies, also include the output of ``sm.show_versions()``

Making Changes to the Code
~~~~~~~~~~~~~~~~~~~~~~~~~~

For a pull request to be accepted, you must meet the below requirements. This greatly helps in keeping the job of maintaining and releasing the software a shared effort.

- Code submissions must always include tests. See our `notes on testing <https://statsmodels.sourceforge.net/devel/dev/test_notes.html>`_.
- Each function, class, method, and attribute needs to be documented using docstrings. We conform to the `numpy docstring standard <https://github.com/numpy/numpy/blob/master/doc/HOWTO_DOCUMENT.rst.txt#docstring-standard>`_.
- If you are adding new functionality, you need to add it to the documentation by editing (or creating) the appropriate file in ``docs/source``.
- Make sure your documentation changes parse correctly. Change into the top-level ``docs/`` directory and type::
  
   make clean
   make html

  Check that the build output does not have *any* warnings due to your changes. 
- Finally, please add your changes to the release notes. Open the ``docs/source/release/versionX.X.rst`` file that has the version number of the next release and add your changes to the appropriate section.

How to Submit a Pull Request
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

So you want to submit a patch to `statsmodels` but aren't too familiar with github? Here are the steps you need to take.

1. `Fork <https://help.github.com/articles/fork-a-repo>`_ the `statsmodels repository <https://github.com/statsmodels/statsmodels>`_ on Github.
2. `Create a new feature branch <http://git-scm.com/book/en/Git-Branching-Basic-Branching-and-Merging>`_. Each branch must be self-contained, with a single new feature or bugfix. 
3. Make sure the test suite passes. This includes testing on Python 3. The easiest way to do this is to either enable `Travis-CI <https://travis-ci.org/>`_ on your fork, or to make a pull request and check there.
4. `Submit a pull request <https://help.github.com/articles/using-pull-requests>`_ 

Mailing List
~~~~~~~~~~~~

Conversations about development take place on the `statsmodels mailing list <http://groups.google.com/group/pystatsmodels?hl=en>`__.

Learn More
~~~~~~~~~~

The ``statsmodels`` documentation's `developer page <http://statsmodels.sourceforge.net/stable/dev/index.html>`_ 
offers much more detailed information about the process.

License
~~~~~~~

Statsmodels is released under the 
`Modified (3-clause) BSD license <http://www.opensource.org/licenses/BSD-3-Clause>`_.
