===========
0.5 Release
===========

Release 0.5.0
=============

Statsmodels 0.5 contains many new features and a large amount of bug fixes.

See the :ref:`list of fixed issues <issues_list_05>` for specific closed issues.


The following major new features appear in this version.

Support for Model Formulas via Patsy
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
...

New API for importing models with formula support::

    import statsmodels.formula.api as smf

Or each model has a ``from_formula`` classmethod where formulas are available.


ARIMA modeling
~~~~~~~~~~~~~~
...

Improved Pandas integration for Time-Series Analysis
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
...

Empirical Likelihood (Google Summer of Code 2012 project)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
...

Analysis of Variance (ANOVA) Modeling
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
...

Nonparameteric Regression (GSoC 2012 project)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
...

Multivariate Kernel Density Estimators (GSoC 2012 project)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
...


Other important new features
----------------------------
IPython notebook examples
Improved marginal effects for discrete choice models
OLS outlier tests
Expanded probability and diagnostic plots
New datasets and the addition of get_rdataset
Much improved numerical differentiation tools
Quantile Regression
Negative Binomial Regression Model
l1-penalized discrete choice models (optional cvxopt dependency)
Constant handling across models
Missing value handling across models
Ability to write Stata datasets

Major Bugs fixed
----------------
Weighted least squares statistics
Regression through the origin statistics

Backwards incompatible changes
------------------------------
Cython code is now non-optional for building from source

Development summary and credits
-------------------------------

The previous version (statsmodels 0.4.3) was released on July 2, 2012. Since then we have closed a total of 356 issues, 166 pull requests and 190 regular issues. The :ref:`detailed list<issues_list_05>` can be viewed.

This release is a result of the work of the following 36 authors who contributed total of 1962 commits. If for any reason, we've failed to list your name in the below, please contact us:

* Ana Martinez Pardo <anamartinezpardo-at-gmail.com>
* anov <novikova.go.zoom-at-gmail.com>
* avishaylivne <avishay.livne-at-gmail.com>
* Bruno Rodrigues <rodrigues.bruno-at-aquitania.org>
* Carl Vogel <carljv-at-gmail.com>
* Chad Fulton <chad-at-chadfulton.com>
* Christian Prinoth <christian-at-prinoth.name>
* Daniel B. Smith <neuromathdan-at-gmail.com>
* dengemann <denis.engemann-at-gmail.com>
* Dieter Vandenbussche <dvandenbussche-at-axioma.com>
* Dougal Sutherland <dougal-at-gmail.com>
* Enrico Giampieri <enrico.giampieri-at-unibo.it>
* evelynmitchell <efm-github-at-linsomniac.com>
* George Panterov <econgpanterov-at-gmail.com>
* Grayson <graysonbadgley-at-gmail.com>
* Jan Schulz <jasc-at-gmx.net>
* Josef Perktold <josef.pktd-at-gmail.com>
* Justin Grana <jg3705a-at-student.american.edu>
* langmore <ianlangmore-at-gmail.com>
* Matthew Brett <matthew.brett-at-gmail.com>
* Nathaniel J. Smith <njs-at-pobox.com>
* otterb <itoi-at-live.com>
* padarn <padarn-at-wilsonp.anu.edu.au>
* Paul Hobson <pmhobson-at-gmail.com>
* Pietro Battiston <me-at-pietrobattiston.it>
* Ralf Gommers <ralf.gommers-at-googlemail.com>
* Richard T. Guy <richardtguy84-at-gmail.com>
* Robert Cimrman <cimrman3-at-ntc.zcu.cz>
* Skipper Seabold <jsseabold-at-gmail.com>
* Thomas Haslwanter <thomas.haslwanter-at-fh-linz.at>
* timmie <timmichelsen-at-gmx-topmail.de>
* Tom Augspurger <thomas-augspurger-at-uiowa.edu>
* Trent Hauck <trent.hauck-at-gmail.com>
* tylerhartley <tyleha-at-gmail.com>
* Vincent Arel-Bundock <varel-at-umich.edu>
* VirgileFritsch <virgile.fritsch-at-gmail.com>

.. note:: 

   Obtained by running ``git log v0.4.3..HEAD --format='* %aN <%aE>' | sed 's/@/\-at\-/' | sed 's/<>//' | sort -u``.

