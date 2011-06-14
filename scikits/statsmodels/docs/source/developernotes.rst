Developer's Notes
-----------------
.. TODO these are intended for developers, we should have separate docs for patches/ pull requests

Mission Statement
=================
Statsmodels is a python package for statistical modelling that is released under
the `BSD license <http://www.opensource.org/licenses/bsd-license.php>`_.

Design
~~~~~~
.. TODO perhaps a flow chart would be the best presentation here?

For the most part, statsmodels is an object-oriented library of statistical
models.  Our working definition of a statistical model is an object that has
both endogenous and exogenous data defined as well as a statistical
relationship.  In place of endogenous and exogenous one can often substitute
the terms left hand side (LHS) and right hand side (RHS), dependent and
independent variables, regressand and regressors, outcome and design, response
variable and explanatory variable, respectively.  The usage is quite often
domain specific; however, we have chosen to use `endog` and `exog` almost
exclusively, since the principal developers of statsmodels have a background
in econometrics, and this feels most natural.  This means that all of the
models are objects with `endog` and `exog` defined, though in some cases
`exog` is None for convenience (for instance, with an autoregressive process).
Each object also defines a `fit` (or similar) method that returns a
model-specific results object.  In addition there are some functions, e.g. for
statistical tests or convenience functions.

Testing
~~~~~~~
We strive to follow a `Test Driven Development (TDD) <http://en.wikipedia.org/wiki/Test-driven_development>`_ pattern.
All models or statistical functions that are added to the main code base are
tested versus an existing statistical package.  All test results are currently obtained from another
statistical package and hard coded.
.. TODO: link to examples of both of these in the test folder

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
`github help pages<http://help.github.com/>`__. Two of the best are `NumPy's documentation <http://docs.scipy.org/doc/numpy/dev/index.html>`__ and 
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

Hack away. Make any changes that you want. Now you've made your changes. Say you've changed the file
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
pushing makes sense. You can review your change history by::

    git log --oneline --graph

It pays to take care of things locally before you push them to github.

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


Advanced Topics
~~~~~~~~~~~~~~~

Merging vs. Rebasing
^^^^^^^^^^^^^^^^^^^^

Deleting branches
^^^^^^^^^^^^^^^^^

.. TODO I'm not positive this is how this works. But I *think* if you have the upstream changes 
.. in your master then it *should* be okay to delete and -d goes through without a hitch

Once your feature branch is accepted into upstream, you might want to get rid of it. First you'll want 
to merge upstream master into your branch. That way git will know that it can safely delete your branch.

Git for Bzr Users
~~~~~~~~~~~~~~~~~

Git cheat sheet
~~~~~~~~~~~~~~~

Mailing List
============

Most of our developer conversations take place on our psystatsmodels
google group mailing list.

**Mailing List:** http://groups.google.com/group/pystatsmodels?hl=en

Related Projects
================

See our :doc:`related projects page <related>`.

Getting Involved and Road Map
=============================

How to Add a Dataset
~~~~~~~~~~~~~~~~~~~~

See the :ref:`notes on adding a dataset <add_data>`.

statsmodels organization
~~~~~~~~~~~~~~~~~~~~~~~~

See the :ref:`Internal Class Guide <model>`.
