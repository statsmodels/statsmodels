.. _git_notes:

Working with the statsmodels Code
=================================

Github
------

The `statsmodels` code base is hosted on `Github <https://github.com/statsmodels/statsmodels>`_. To
contribute you will need to `sign up for a free Github account <https://github.com/>`_.

Version Control and Git
-----------------------

We use the `Git <https://git-scm.com/>`_ version control system for development.
Git allows many people to work together on the same project.  In a nutshell, it
allows you to make changes to the code independent of others who may also be
working on the code and allows you to easily contribute your changes to the
codebase. It also keeps a complete history of all changes to the code, so you
can easily undo changes or see when a change was made, by whom, and why.

To install and configure Git, and to setup SSH keys, see
`setting up git <https://help.github.com/articles/set-up-git/>`_.

To learn more about Git, you may want to visit:

+ `Git documentation (book and videos) <https://git-scm.com/documentation>`_
+ `Github help pages <https://help.github.com/>`_
+ `NumPy documentation <https://docs.scipy.org/doc/numpy/dev/index.html>`_
+ `Matthew Brett's Pydagogue <https://matthew-brett.github.io/pydagogue/>`_

Below, we describe the bare minimum git commands you need to contribute to
`statsmodels`.

statsmodels Git/Github Workflow
-------------------------------

Forking and cloning
~~~~~~~~~~~~~~~~~~~

After setting up git, you need to fork the main `statsmodels` repository. To do
this, visit the `statsmodels project page
<https://github.com/statsmodels/statsmodels>`_ and hit the fork button (see
instructions for
`forking a repo <https://help.github.com/articles/fork-a-repo/>`_ for details).
This should take you to your fork's page.

Then, you want to clone the fork to your machine::

    git clone https://github.com/your-user-name/statsmodels
    cd statsmodels
    git remote add upstream https://github.com/statsmodels/statsmodels
    git fetch --all

The third line sets-up a read-only connection to the upstream statsmodels
repository. This will allow you to periodically update your local code with
changes in the upstream.  The final command fetches both your repository and
the upstream statsmodels repository.

Create a Branch
~~~~~~~~~~~~~~~

All changes to the code should be made in a feature branch. To create a branch, type::

    git checkout master
    git rebase upstream/master
    git checkout -b shiny-new-feature

The first two lines ensure you are starting from an up-to-date version of the upstream
statsmodels repository.  The third creates and checkout a new branch.

Doing::

    git branch

will give something like::

    * shiny-new-feature
      master

to indicate that you are now on the `shiny-new-feature` branch.

Making changes
~~~~~~~~~~~~~~

Hack away! Make any changes that you want, but please keep the work in your
branch completely confined to one specific topic, bugfix, or feature
implementation. You can work across multiple files and have many commits, but
the changes should all be related to the feature of the feature branch,
whatever that may be.

Now imagine that you changed the file `foo.py`. You can see your changes by
typing::

    git status

This will print something like::

    # On branch shiny-new-feature
    # Changes not staged for commit:
    #   (use "git add <file>..." to update what will be committed)
    #   (use "git checkout -- <file>..." to discard changes in working directory)
    #
    #       modified:   relative/path/to/foo.py
    #
    no changes added to commit (use "git add" and/or "git commit -a")

Before you can commit these changes, you have to `add`, or `stage`, the
changes. You can do this by typing::

    git add path/to/foo.py

Then check the status to make sure your commit looks okay::

    git status

should give something like::

    # On branch shiny-new-feature
    # Changes to be committed:
    #   (use "git reset HEAD <file>..." to unstage)
    #
    #       modified:   /relative/path/to/foo.py
    #

Pushing your changes
~~~~~~~~~~~~~~~~~~~~

At any time you can push your feature branch (and any changes) to your github
(fork) repository by::

    git push

although the first time you will need to run

    git push --set-upstream origin shiny-new-feature

to instruct git to set the current branch to track its corresponding branch in
your github repository.

You can see the remote repositories by::

    git remote -v

If you added the upstream repository as described above you will see something
like::

    origin  https://github.com/yourname/statsmodels.git (fetch)
    origin  https://github.com/yourname/statsmodels.git (push)
    upstream        https://github.com/statsmodels/statsmodels.git (fetch)
    upstream        https://github.com/statsmodels/statsmodels.git (push)

Before you push any commits, however, it is *highly* recommended that you make
sure what you are pushing makes sense and looks clean. You can review your
change history by::

    git log --oneline --graph

It pays to take care of things locally before you push them to github. So when
in doubt, do not push.  Also see the advice on keeping your history clean in
:ref:`merge-vs-rebase`.

.. _pull-requests:

Pull Requests
~~~~~~~~~~~~~

When you are ready to ask for a code review, we recommend that you file a pull
request. Before you do so you should check your changeset yourself. You can do
this by using `compare view
<https://github.com/blog/612-introducing-github-compare-view>`__ on github.

#. Navigate to your repository on github.
#. Click on `Branch List`
#. Click on the `Compare` button for your feature branch, `shiny-new-feature`.
#. Select the `base` and `compare` branches, if necessary. This will be `master` and
   `shiny-new-feature`, respectively.
#. From here you will see a nice overview of your changes. If anything is amiss, you can fix it.

If everything looks good you are read to make a `pull request <https://help.github.com/articles/about-pull-requests/>`__.

#. Navigate to your repository on github.
#. Click on the `Pull Request` button.
#. You can then click on `Commits` and `Files Changed` to make sure everything looks okay one last time.
#. Write a description of your changes in the `Preview Discussion` tab.
#. Click `Send Pull Request`.

Your request will then be reviewed. If you need to go back and make more
changes, you can make them in your branch and push them to github and the pull
request will be automatically updated.

One last thing to note. If there has been a lot of work in upstream/master
since you started your patch, you might want to rebase. However, you can
probably get away with not rebasing if these changes are unrelated to the work
you have done in the `shiny-new-feature` branch. If you can avoid it, then
do not rebase. If you have to, try to do it once and when you are at the end of
your changes. Read on for some notes on :ref:`merge-vs-rebase`.

Advanced Topics
---------------

.. _merge-vs-rebase:

Merging vs. Rebasing
~~~~~~~~~~~~~~~~~~~~

This is a topic that has been discussed at great length and with considerable
more expertise than we can offer here. This section will provide some resources
for further reading and some advice. The focus, though, will be for those who
wish to submit pull requests for a feature branch. For these cases rebase
should be preferred.

A rebase replays commits from one branch on top of another branch to preserve a
linear history. Recall that your commits were tested against a (possibly) older
version of master from which you started your branch, so if you rebase, you
could introduce bugs. However, if you have only a few commits, this might not
be such a concern. One great place to start learning about rebase is
:ref:`rebasing without tears <pydagogue:actual-rebase>`.  In particular, `heed
the warnings
<https://matthew-brett.github.io/pydagogue/rebase_without_tears.html#safety>`__.
Namely, **always make a new branch before doing a rebase**. This is good
general advice for working with git. I would also add **never use rebase on
work that has already been published**. If another developer is using your
work, do not rebase!!

As for merging, **never merge from trunk into your feature branch**. You will,
however, want to check that your work will merge cleanly into trunk. This will
help out the reviewers. You can do this in your local repository by merging
your work into your master (or any branch that tracks remote master) and
:ref:`run-tests`.

Deleting Branches
~~~~~~~~~~~~~~~~~

Once your feature branch is accepted into upstream, you might want to get rid
of it. First you'll want to merge upstream master into your branch. That way
git will know that it can safely delete your branch::

    git fetch upstream
    git checkout master
    git merge upstream/master

Then you can just do::

    git branch -d shiny-new-feature

Make sure you use a lower-case -d. That way, git will complain if your feature
branch has not actually been merged. The branch will still exist on github
however. To delete the branch on github, do::

    git push origin :shiny-new-feature branch

.. Squashing with Rebase
.. ^^^^^^^^^^^^^^^^^^^^^

.. You have made a bunch of incremental commits, but you think they might be better off together as one
.. commit. You can do this with an interactive rebase. As usual, **only do this when you have local
.. commits. Do not edit the history of changes that have been pushed.**

.. see this reference http://gitready.com/advanced/2009/02/10/squashing-commits-with-rebase.html
