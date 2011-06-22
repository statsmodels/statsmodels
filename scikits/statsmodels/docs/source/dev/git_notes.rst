Working with the Statsmodels Code
---------------------------------

Github
======
Statsmodels code base is hosted on `Github <https://www.github.com/>`_. To
contribute you will need to `sign up for a Github account <https://github.com/signup/free>`_.

**Repository:** https://github.com/statsmodels/statsmodels

**Bug Tracker:**  https://bugs.launchpad.net/statsmodels

Version Control and Git
=======================
We use `Git <http://git-scm.com/>`_ for development. Version control systems such as git allow many
people to work together on the same project.  In a nutshell, it allows you to make changes to the 
code independent of others who may also be working on the code and allows you to easily contribute 
your changes to the codebase. It also keeps a complete history of all changes to the code, so you can 
easily undo changes or see when a change was made, by whom, and why.

There are already a lot of great resources for learning to use git in addition to the comprehensive
`github help pages <http://help.github.com/>`__. Two of the best are `NumPy's documentation <http://docs.scipy.org/doc/numpy/dev/index.html>`__ and 
Matthew Brett's `Pydagogue <http://matthew-brett.github.com/pydagogue/>`__. The below is the bare minimum taken from these resources and applied to working with statsmodels. 
You would do well to have a look at these other resources for more information.

Getting Started with Git
~~~~~~~~~~~~~~~~~~~~~~~~
Instructions for installing git, setting up your SSH key, and configuring git can be found here::

`Linux users <http://help.github.com/linux-set-up-git/>`__.
`Windows users <http://help.github.com/win-set-up-git/>`__.
`Mac users <http://help.github.com/mac-set-up-git/>`__.

Forking
~~~~~~~
After setting up git, you will need your own fork to work on the code. Go to the `statsmodels project page <https://github.com/statsmodels/statsmodels>`__ and hit the fork button. Then you should be taken
to your fork's page. You will want to clone your fork to your machine: ::

    git clone git@github.com:your-user-name/statsmodels.git statsmodels-yourname
    cd statsmodels-yourname
    git remote add upstream git://github.com/statsmodels/statsmodels.git

The first line will create a directory, `statsmodels-yourname`, but you can name it whatever you want.
The last line connects your repository to the upstream statsmodels repository. The name `upstream` is
arbitrary here. Notice that you use git:// instead of git@. You want to connect to the read-only 
URL. You can use this periodically to update your local code with changes in the upstream.

Create a Branch
~~~~~~~~~~~~~~~
Now you are ready to make some changes to the code. You will want to do this in a feature branch. You
want your master branch to remain clean. You always want it to reflect production-ready code. So you
will want to make changes in features branches. For example::

    git branch shiny-new-feature
    git checkout shiny-new-feature
    
Doing::
    
    git branch

will give something like::

    * shiny-new-feature
      master

to indicate that you are now on the `shiny-new-feature` branch.

Making changes
~~~~~~~~~~~~~~

Hack away. Make any changes that you want. Well, not any changes. Keep the work in your branch 
completel confined to one speficic topic, bugfix, or feature implementation. You can work across
multiple files and have many commits, but the changes should all be related to the feature of the 
feature branch, whatever that may be. Now you've made your changes. Say you've changed the file
`foo.py`. You can see your changes typing::

    git status

This will give something like::

    # On branch shiny-new-feature
    # Changes not staged for commit:
    #   (use "git add <file>..." to update what will be committed)
    #   (use "git checkout -- <file>..." to discard changes in working directory)
    #
    #       modified:   relative/path/to/foo.py
    #
    no changes added to commit (use "git add" and/or "git commit -a")

Before you can commit these changes, you have to `add`, or `stage`, the changes. You can do this by 
typing::

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

At any time you can push your feature branch (and any changes) to your repository by::

    git push origin shiny-new-feature

Here `origin` is the default name given to your remote repository. You can see the remote repositories
by::
    
    git remote -v

If you added the upstream repository as described above you will see something like::

    origin  git@github.com:yourname/statsmodels.git (fetch)
    origin  git@github.com:yourname/statsmodels.git (push)
    upstream        git://github.com/statsmodels/statsmodels.git (fetch)
    upstream        git://github.com/statsmodels/statsmodels.git (push)

Before you push any commits, however, it is *highly* recommended that you make sure what you are 
pushing makes sense and looks clean. You can review your change history by::

    git log --oneline --graph

It pays to take care of things locally before you push them to github. So when in doubt, don't push. 
Also see the advice on keeping your history clean in :ref:`merge-vs-rebase`.

.. _pull-requests:

Pull Requests
~~~~~~~~~~~~~
When you are ready to ask for a code review, we recommend that you file a pull request. Before you 
do so you should check your changeset yourself. You can do this by using
`compare view <https://github.com/blog/612-introducing-github-compare-view>`__ on github. 

#. Navigate to your repository on github.
#. Click on `Branch List`
#. Click on the `Compare` button for your feature branch, `shiny-new-feature`.
#. Select the `base` and `compare` branches, if necessary. This will be `master` and 
   `shiny-new-feature`, respectively.
#. From here you will see a nice overview of your changes. If anything is amiss, you can fix it.

If everything looks good you are read to make a `pull request <http://help.github.com/send-pull-requests/>`__.

#. Navigate to your repository on github.
#. Click on the `Pull Request` button.
#. You can then click on `Commits` and `Files Changed` to make sure everything looks okay one last time.
#. Write a description of your changes in the `Preview Discussion` tab.
#. Click `Send Pull Request`.

Your request will then be reviewed. If you need to go back and make more changes, you can make them
in your branch and push them to github and the pull request will be automatically updated.

One last thing to note. If there has been a lot of work in upstream/master since you started your 
patch, you might want to rebase. However, you can probably get away with not rebasing if these changes
are unrelated to the work you have done in the `shiny-new-feature` branch. If you can avoid it, then 
don't rebase. If you have to, try to do it once and when you are at the end of your changes. Read on 
for some notes on :ref:`merge-vs-rebase`.

Advanced Topics
~~~~~~~~~~~~~~~

.. _merge-vs-rebase:

Merging vs. Rebasing
^^^^^^^^^^^^^^^^^^^^
Again, this is a topic that has been discussed at great length and with considerable more expertise 
than I can offer. This section will provide some resources for further reading and some advice. The 
focus, though, will be for those who wish to submit pull requests for a feature branch. For these 
cases rebase should be preferred.

A rebase replays commits from one branch on top of another branch to preserve a linear history. Recall
that your commits were tested against a (possibly) older version of master from which you started
your branch, so if you rebase, you could introduce bugs. However, if you have only a few 
commits, this might not be such a concern. One great place to start learning about rebase is 
:ref:`rebasing without tears <pydagogue:actual-rebase>`. 
In particular, `heed the warnings <http://matthew-brett.github.com/pydagogue/rebase_without_tears.html#safety>`__. Namely, **always make a new branch before doing a rebase**. This is good general advice for
working with git. I would also add **never use rebase on work that has already been published**. If 
another developer is using your work, don't rebase!!

As for merging, **never merge from trunk into your feature branch**. You will, however, want to check
that your work will merge cleanly into trunk. This will help out the reviewers. You can do this 
in your local repository by merging your work into your master (or any branch that tracks remote 
master) and :ref:`run-tests`.

Deleting Branches
^^^^^^^^^^^^^^^^^

Once your feature branch is accepted into upstream, you might want to get rid of it. First you'll want 
to merge upstream master into your branch. That way git will know that it can safely delete your 
branch::

    git fetch upstream
    git checkout master
    git merge upstream/master

Then you can just do::

    git -d shiny-new-feature
 
Make sure you use a lower-case -d. That way, git will complain if your feature branch has not actually
been merged. The branch will still exist on github however. To delete the branch on github, do::

    git push origin :shiny-new-feature branch

Git for Bzr Users
~~~~~~~~~~~~~~~~~

::

    git pull != bzr pull

::

    git pull = git fetch + git merge

Of course, you could::

    git pull --rebase = git fetch + git rebase

::

    git merge != bzr merge
    git merge == bzr merge + bzr commit 
    git merge --no-commit == bzr merge
 
Git Cheat Sheet
~~~~~~~~~~~~~~~

.. todo::
    
    Fill in as needed.
