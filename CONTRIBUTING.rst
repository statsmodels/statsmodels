Contributing guidelines
=======================

This page explains how you can contribute to the development of `statsmodels`
by submitting patches, statistical tests, new models, or examples.

`statsmodels` is developed on `Github <https://github.com/statsmodels/statsmodels>`_
using the `Git <https://git-scm.com/>`_ version control system.

Submitting a Bug Report
~~~~~~~~~~~~~~~~~~~~~~~

- Include a short, self-contained code snippet that reproduces the problem
- Specify the statsmodels version used. You can do this with ``sm.version.full_version``
- If the issue looks to involve other dependencies, also include the output of ``sm.show_versions()``

Making Changes to the Code
~~~~~~~~~~~~~~~~~~~~~~~~~~

For a pull request to be accepted, you must meet the below requirements. This greatly helps in keeping the job of maintaining and releasing the software a shared effort.

- **One branch. One feature.** Branches are cheap and github makes it easy to merge and delete branches with a few clicks. Avoid the temptation to lump in a bunch of unrelated changes when working on a feature, if possible. This helps us keep track of what has changed when preparing a release.
- Commit messages should be clear and concise. This means a subject line of less than 80 characters, and, if necessary, a blank line followed by a commit message body. We have an `informal commit format standard <https://www.statsmodels.org/devel/dev/maintainer_notes.html#commit-comments>`_ that we try to adhere to. You can see what this looks like in practice by ``git log --oneline -n 10``. If your commit references or closes a specific issue, you can close it by mentioning it in the `commit message <https://help.github.com/articles/closing-issues-via-commit-messages>`_.  (*For maintainers*: These suggestions go for Merge commit comments too. These are partially the record for release notes.)
- Code submissions must always include tests. See our `notes on testing <https://www.statsmodels.org/devel/dev/test_notes.html>`_.
- Each function, class, method, and attribute needs to be documented using docstrings. We conform to the `numpy docstring standard <https://numpydoc.readthedocs.io/en/latest/format.html#docstring-standard>`_.
- If you are adding new functionality, you need to add it to the documentation by editing (or creating) the appropriate file in ``docs/source``.
- Make sure your documentation changes parse correctly. Change into the top-level ``docs/`` directory and type::

   make clean
   make html

  Check that the build output does not have *any* warnings due to your changes.
- Finally, please add your changes to the release notes. Open the ``docs/source/release/versionX.X.rst`` file that has the version number of the next release and add your changes to the appropriate section.

Linting
~~~~~~~

Due to the way we have the CI builds set up, the linter will not do anything unless the environmental variable $LINT is set to a truthy value.

- On MacOS/Linux

    LINT=true ./lint.sh

- Dependencies: flake8, git

How to Submit a Pull Request
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

So you want to submit a patch to `statsmodels` but are not too familiar with github? Here are the steps you need to take.

1. `Fork <https://help.github.com/articles/fork-a-repo>`_ the `statsmodels repository <https://github.com/statsmodels/statsmodels>`_ on Github.
2. `Create a new feature branch <https://git-scm.com/book/en/Git-Branching-Basic-Branching-and-Merging>`_. Each branch must be self-contained, with a single new feature or bugfix.
3. Make sure the test suite passes. This includes testing on Python 3. The easiest way to do this is to either enable `Travis-CI <https://travis-ci.org/>`_ on your fork, or to make a pull request and check there.
4. Document your changes by editing the appropriate file in ``docs/source/``. If it is a big, new feature add a note and an example to the latest ``docs/source/release/versionX.X.rst`` file. See older versions for examples. If it's a minor change, it will be included automatically in our release notes.
5. Add an example. If it is a big, new feature please submit an example notebook by following `these instructions <https://www.statsmodels.org/devel/dev/examples.html>`_.
6. `Submit a pull request <https://help.github.com/articles/using-pull-requests>`_

Mailing List
~~~~~~~~~~~~

Conversations about development take place on the `statsmodels mailing list <https://groups.google.com/group/pystatsmodels?hl=en>`__.

Learn More
~~~~~~~~~~

The ``statsmodels`` documentation's `developer page <https://www.statsmodels.org/stable/dev/index.html>`_
offers much more detailed information about the process.

License
~~~~~~~

statsmodels is released under the
`Modified (3-clause) BSD license <https://www.opensource.org/licenses/BSD-3-Clause>`_.
