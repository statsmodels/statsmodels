Developer Page
--------------

This page explains how you can contribute to the development of `statsmodels`
by submitting patches, statistical tests, new models, or examples. 

`statsmodels` is developed on `Github
<https://github.com/statsmodels/statsmodels>`_ using the `Git
<https://git-scm.com/>`_ version control system.

Submitting a Bug Report
~~~~~~~~~~~~~~~~~~~~~~~

- Include a short, self-contained code snippet that reproduces the problem
- Specify the statsmodels version used. You can do this with ``sm.version.full_version``
- If the issue looks to involve other dependencies, also include the output of ``sm.show_versions()``

Making Changes to the Code
~~~~~~~~~~~~~~~~~~~~~~~~~~

For a pull request to be accepted, you must meet the below requirements. This greatly helps the job of maintaining and releasing the software a shared effort.

- **One branch. One feature.** Branches are cheap and github makes it easy to merge and delete branches with a few clicks. Avoid the temptation to lump in a bunch of unrelated changes when working on a feature, if possible. This helps us keep track of what has changed when preparing a release.
- Commit messages should be clear and concise. This means a subject line of less than 80 characters, and, if necessary, a blank line followed by a commit message body. We have an `informal commit format standard <http://www.statsmodels.org/devel/dev/maintainer_notes.html#commit-comments>`_ that we try to adhere to. You can see what this looks like in practice by ``git log --oneline -n 10``. If your commit references or closes a specific issue, you can close it by mentioning it in the `commit message <https://help.github.com/articles/closing-issues-via-commit-messages/>`_.  (*For maintainers*: These suggestions go for Merge commit comments too. These are partially the record for release notes.)
- Code submissions must always include tests. See our notes on :ref:`testing`.
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

1. `Fork <https://help.github.com/articles/fork-a-repo/>`_ the `statsmodels repository <https://github.com/statsmodels/statsmodels>`_ on Github.
2. `Create a new feature branch <https://git-scm.com/book/en/v2/Git-Branching-Basic-Branching-and-Merging>`_. Each branch must be self-contained, with a single new feature or bugfix.
3. Make sure the test suite passes. This includes testing on Python 3. The easiest way to do this is to either enable `Travis-CI <https://travis-ci.org/>`_ on your fork, or to make a pull request and check there.
4. `Submit a pull request <https://help.github.com/articles/about-pull-requests/>`_

Mailing List
~~~~~~~~~~~~

Conversations about development take place on the `statsmodels mailing list <https://groups.google.com/forum/?hl=en#!forum/pystatsmodels>`__.

License
~~~~~~~

Statsmodels is released under
the `Modified (3-clause) BSD license <https://opensource.org/licenses/BSD-3-Clause>`_.

Contents
~~~~~~~~

.. toctree::
   :maxdepth: 3

   git_notes
   maintainer_notes
   test_notes
   naming_conventions
   dataset_notes
   examples
   get_involved
   internal
   testing
