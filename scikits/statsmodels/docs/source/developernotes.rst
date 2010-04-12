Developer's Notes
-----------------

Mission Statement
=================
Statsmodels is a python package for statistical modeling that is released under
the `BSD license <http://www.opensource.org/licenses/bsd-license.php>`_.

Design
~~~~~~
.. TODO: perhaps a flow chart would be the best presentation here?
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
tested versus an existing statistical package.  The test results can be
generated at run time using R via RPy or can be obtained from another
statistical package and hard coded.
.. TODO: link to examples of both of these in the test folder

Launchpad
=========
Statsmodels code base is hosted on `Launchpad <https://launchpad.net/>`_. To
contribute you will need to `sign up for a Launchpad account <https://login.launchpad.net/vRDLGvcCNXXjP3F1/+new_account>`_.

**Repository:** http://code.launchpad.net/statsmodels

**Bug Tracker:**  https://bugs.launchpad.net/statsmodels

Version Control and Bzr
=======================
We use the `Bazaar distributed version control system <http://bazaar.canonical.com/en/>`_.  Learning Bazaar (bzr) can be one of the biggest hurdles to
contributing to an open source project, but it is necessary and in the end
makes us all more productive.  Distributed version control systems allow many
people to work together on the same project.  In a nutshell, it allows you to
make changes to the code independent of others who may also be working on the
code and allows you to easily `merge` your changes together in the main
`trunk` branch of the code.  Version control systems also keeps a complete
history of all `revisions` to the code, so you can easily `revert` changes or
see when a change was made, by whom, and why.


Get Bzr
~~~~~~~
Everyone can obtain bzr from their main page `here <http://wiki.bazaar.canonical.com/Download>`_.


Linux users
^^^^^^^^^^^
bzr should be included in your distribution's repository.
On \*Ubuntu: ::

    sudo apt-get install bzr


Windows users
^^^^^^^^^^^^^
Windows users can follow the download instructions `here <http://wiki.bazaar.canonical.com/WindowsDownloads>`_.

Mac Users
^^^^^^^^^
Mac users can follow the download instructions `here <http://wiki.bazaar.canonical.com/MacOSXBundle>`_.


Branching
~~~~~~~~~
The next thing you are going to want to do is create a branch to make changes
to the code.


What is a branch?
^^^^^^^^^^^^^^^^^
You can think of a branch as your own personal copy of the code (repository)
in which you are going to make your changes.


Why branch?
^^^^^^^^^^^
Having a branch of the code allows you to make changes independent of the main
code, but it still maintains a relationship with the main source code.  For
instance, it is easy to keep up with changes in the main trunk while you are
working in your branch, and when your changes are made you can propose to merge
your work back into the trunk easily.


How to branch
^^^^^^^^^^^^^
Given that you have already created a Launchpad account following the link
above, the next step is to create an SSH key.  Follow the instructions `here <https://help.launchpad.net/YourAccount/CreatingAnSSHKeyPair>`_.
If it is not obvious what your launchpad name is you can get and set it `here <https://launchpad.net/people/+me/+edit>`_
We refer to this below as "yourname"

The next step is to tell bzr who you are.  Using your name and e-mail address
type: ::

    > bzr whoami "John Doe <john.doe@gmail.com>"

Check this step by typing: ::

    > bzr whoami

Now you are ready to create and check out a branch.

If you want to register your branch manually on Launchpad go to
    `https://code.launchpad.net/statsmodels <https://code.launchpad.net/statsmodels>`_ and click on Register a branch.  Fill in a name (I will use test-branch).
Click the Hosted radio button.  Choose a status, and click Register Branch.
You will be taken to the branch's web page and there will be a command to
"Update this branch" that shows: ::

    > bzr push lp:~yourname/statsmodels/test-branch

We will come back to this.

Alternatively, you can just create your branch from the command line.  This will
be explained below.

The next step is to get the main trunk branch in order to work in.  I will put
the main trunk into a folder called test-branch.  To do this from the folder
where you want the branch type: ::

    > bzr branch lp:statsmodels test-branch

Now make some changes to the code.  In this case, I will cd to
test-branch/scikits/statsmodels/ and create an empty file called dummy.py.  You
have to tell bzr to put dummy.py under version control by: ::

    > bzr add dummy.py
    adding scikits/statsmodels/dummy.py

We can see what changes are made versus the "parent location" of the branch
(which is still the trunk in this case) by typing (st is short for status): ::

    > bzr st
    added:
      scikits/statsmodels/dummy.py

Next we have to `commit` our changes.  This is how we keep up with what changed
and why.  Committing a change makes a note in the revision history log.  Type: ::

    > bzr commit -m "Added the dummy.py file as an example"

Commits are best done in small increments, so commit often.  We have now
committed our changes locally.  This is fine.  You can continue working and
then commit more changes if you wish.  Eventually you will want to `push` your
changes to Launchpad.  Since this will be the first time pushing we have to tell
bzr that we want to push to a different directory than where we
branched the code from and to use ssh.  If you followed the manual registration
of the branch on Launchpad instructions above, you have to tell bzr that you
are pushing to an already existing location.  We also want to tell bzr that we
are using ssh.  This can be accomplished by typing: ::

    > bzr push bzr+ssh://yourname@bazaar.launchpad.net/~yourname/statsmodels/test-branch --use-existing-dir --remember

If you did not register your branch beforehand, you type almost the exact same
thing: ::

    > bzr push bzr+ssh://yourname@bazaar.launchpad.net/~yourname/statsmodels/test-branch --remember

And bzr will automatically register the branch for you.  You can also tell
bzr your launchpad login by typing: ::

    > bzr launchpad-login yourname

You only need to do this once, then the command above simply becomes: ::

    > bzr push lp:~yourname/statsmodels/test-branch --remember

From now on, you can simply do: ::

    > bzr commit -m "Specific and informative comment about changes"
    > bzr push

And you are good.  I often work on multiple computers.  When I make and push
changes from one and return to another, I have to type: ::

    > bzr pull

And it will pull down all of the changes from your branch.

The last thing to know is that you will want to keep track of changes in trunk.
To do this type: ::

    > bzr merge lp:statsmodels
    > bzr commit -m"Merged with trunk"
    > bzr push

That's basically it.  You should be up and running with bzr now.


A few helpful commands
^^^^^^^^^^^^^^^^^^^^^^

The following a few helpful bzr commands with some common usage:

Commit new changes with a note: ::

    > bzr commit -m "Note"

Push new commits: ::

    > bzr push

Pull from remembered location: ::

    > bzr pull

See the status of changes of new files: ::

    > bzr st

Get diff of current branch versus trunk.  Note that you must be in a folder of
the version controlled branch: ::

    > bzr diff --old lp:statsmodels

Get diff versus existing remembered location: ::

    > bzr diff

Get help for any command.  For diff, for example, type: ::

    > bzr diff --help

There are plenty of resources out there to help you through some more
advanced features of bzr.  Note also that the people #bzr on irc.freenode.net
have always been quite helpful in my experience.

Mailing List
============

Most of our developer conversations take place on our `psystatsmodels
google group mailing list.

**Mailing List:** http://groups.google.com/group/pystatsmodels?hl=en

Related Projects
================

See our `related projects page <related.html>`_.

Getting Involved and Road Map
=============================
Coming Soon.
