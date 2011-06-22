Development Workflow for Maintainers
------------------------------------

Git Workflow
------------

Cherry-Picking
~~~~~~~~~~~~~~

Merging: To Fast-Forward or Not To Fast-Forward
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

By default, `git merge` is a fast-forward merge. What does this mean, and when do you want to avoid 
this?

.. figure:: images/git_merge.png
   :alt: git merge diagram
   :scale: 100%
   :align: center

   (souce `nvie.com <http://nvie.com>`__, post `"A successful Git branchind model" <http://nvie.com/posts/a-successful-git-branching-model/>`__)

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
request and link the ticket and closing commit. All problems need to be taken care of locally 
before doing::

    git push origin master

Releasing
---------

#. Fix the version number. Open setup.py and set::

    ISRELEASED = True

   If it's a release candidate then change to, for example::

    ISRELEASED = True
    QUALIFIER = 'rc1'

#. Tag the release. For example::

    git tag -a v0.3.0rc1 -m "Version 0.3.0 Release Candidate 1" 7b2fb295a421b83a90b04180c8a1678cf9a6ed0d


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
