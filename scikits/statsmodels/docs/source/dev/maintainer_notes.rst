Maintainer Notes
================

This is for those with read-write access to upstream. It is recommended to name the upstream
remote something to remind you that it is read-write::

    git remote add upstream-rw git@github.com:statsmodels/statsmodels.git
    git fetch upstream-rw

Git Workflow
------------

Grabbing Changes from Others
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

If you need to push changes from others, you can link to their repository by doing::

    git remote add contrib-name git://github.com/contrib-name/statsmodels.git
    get fetch contrib-name
    git branch shiny-new-feature --track contrib-name/shiny-new-feature
    git checkout shiny-new-feature

The rest of the below assumes you are on your or someone else's branch with the changes you
want to push upstream.

.. _rebasing:

Rebasing
~~~~~~~~

If there are only a few commits, you can rebase to keep a linear history::

    git fetch upstream-rw
    git rebase upstream-rw/master

Rebasing will not automatically close the pull request however, if there is one,
so don't forget to do this.

.. _merging:

Merging
~~~~~~~

If there is a long series of related commits, then you'll want to merge. You may ask yourself,
:ref:`ff-no-ff`? See below for more on this choice. Once decided you can do::

    git fetch upstream-rw
    git merge --no-ff upstream-rw/master

Merging will automaticall close the pull request on github.

Check the History
~~~~~~~~~~~~~~~~~

This is very important. Again, any and all fixes should be made locally before pushing to the
repository::

    git log --oneline --graph

This shows the history in a compact way of the current branch. This::

    git log -p upstream-rw/master..

shows the log of commits excluding those that can be reached from upstream-rw/master, and
including those that can be reached from current HEAD. That is, those changes unique to this
branch versus upstream-rw/master. See :ref:`Pydagogue <pydagogue:git-log-dots>` for more on using
dots with log and also for using :ref:`dots with diff <pydagogue:git-diff-dots>`.

Push Your Feature Branch
~~~~~~~~~~~~~~~~~~~~~~~~

All the changes look good? You can push your feature branch after :ref:`merging` or :ref:`rebasing` by::

    git push upstream-rw shiny-new-feature:master

Cherry-Picking
~~~~~~~~~~~~~~

Say you are interested in some commit in another branch, but want to leave the other ones for now.
You can do this with a cherry-pick. Use `git log --oneline` to find the commit that you want to
cherry-pick. Say you want commit `dd9ff35` from the `shiny-new-feature` branch. You want to apply
this commit to master. You simply do::

    git checkout master
    git cherry-pick dd9ff35

And that's all. This commit is now applied as a new commit in master.

.. _ff-no-ff:

Merging: To Fast-Forward or Not To Fast-Forward
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

By default, `git merge` is a fast-forward merge. What does this mean, and when do you want to avoid
this?

.. figure:: images/git_merge.png
   :alt: git merge diagram
   :scale: 100%
   :align: center

   (source `nvie.com <http://nvie.com>`__, post `"A successful Git branching model" <http://nvie.com/posts/a-successful-git-branching-model/>`__)

The fast-forward merge does not create a merge commit. This means that the existence of the feature
branch is lost in the history. The fast-forward is the default for Git basically because branches are
cheap and, therefore, *usually* short-lived. If on the other hand, you have a long-lived feature
branch or are following an iterative workflow on the feature branch (i.e. merge into master, then
go back to feature branch and add more commits), then it makes sense to include only the merge
in the main branch, rather than all the intermediate commits of the feature branch, so you should
use::

    git merge --no-ff

Handling Pull Requests
~~~~~~~~~~~~~~~~~~~~~~

You can apply a pull request through `fetch <http://www.kernel.org/pub/software/scm/git/docs/git-fetch.html>`__ and `merge <http://www.kernel.org/pub/software/scm/git/docs/git-merge.html>`__. In your local
copy of the main repo::

    git checkout master
    git remote add contrib-name git://github.com/contrib-name/statsmodels.git
    git fetch contrib-name
    git merge contrib-name/shiny-new-feature

Check that the merge applies cleanly and the history looks good. Edit the merge message. Add a short
explanation of what the branch did along with a 'Closes gh-XXX.' string. This will auto-close the pull
request and link the ticket and closing commit. To automatically close the issue, you can use any of::

    gh-XXX
    GH-XXX
    #XXX

in the commit message. Any and all problems need to be taken care of locally before doing::

    git push origin master

Releasing
---------

#. Fix the version number. Open setup.py and set::

    ISRELEASED = True

#. Tag the release. For a release candidate, for example::

    git tag -a v0.3.0rc1 -m "Version 0.3.0 Release Candidate 1" 7b2fb29

#. Upload the source distribution to PyPI::

    python setup.py sdist --formats=gztar,zip register upload

#. Make an announcment

#. Profit


Commit Comments
---------------
Prefix commit messages in the master branch of the main shared repository with the following::

    ENH: Feature implementation
    BUG: Bug fix
    STY: Coding style changes (indenting, braces, code cleanup)
    DOC: Sphinx documentation, docstring, or comment changes
    CMP: Compiled code issues, regenerating C code with Cython, etc.
    REL: Release related commit
    TST: Change to a test, adding a test. Only used if not directly related to a bug.
