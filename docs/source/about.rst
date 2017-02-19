.. module:: statsmodels
   :synopsis: Statistical analysis in Python


.. currentmodule:: statsmodels

*****************
About Statsmodels
*****************

Background
----------

The ``models`` module of ``scipy.stats`` was originally written by Jonathan 
Taylor. For some time it was part of scipy but was later removed. During
the Google Summer of Code 2009, ``statsmodels`` was corrected, tested,
improved and released as a new package. Since then, the statsmodels 
development team has continued to add new models, plotting tools, and statistical methods.

Testing
-------

Most results have been verified with at least one other statistical package:
R, Stata or SAS. The guiding principle for the initial rewrite and for 
continued development is that all numbers have to be verified. Some 
statistical methods are tested with Monte Carlo studies. While we strive to
follow this test driven approach, there is no guarantee that the code is 
bug-free and always works. Some auxiliary function are still insufficiently 
tested, some edge cases might not be correctly taken into account, and the 
possibility of numerical problems is inherent to many of the statistical 
models. We especially appreciate any help and reports for these kind of 
problems so we can keep improving the existing models.

Code Stability
~~~~~~~~~~~~~~

The existing models are mostly settled in their user interface and we do not
expect many large changes going forward. For the existing code, although 
there is no guarantee yet on API stability, we have long deprecation periods 
in all but very special cases, and we try to keep changes that require 
adjustments by existing users to a minimal level. For newer models we might
adjust the user interface as we gain more experience and obtain feedback. 
These changes will always be noted in our release notes available in the
documentation.

Financial Support
-----------------

We are grateful for the financial support that we obtained for the
development of statsmodels:

 Google `www.google.com <https://www.google.com/>`_ : Google Summer of Code
 (GSOC) 2009-2013.

 AQR `www.aqr.com <http://www.aqr.com/>`_ : financial sponsor for the work on
 Vector Autoregressive Models (VAR) by Wes McKinney

We would also like to thank our hosting providers, `github
<https://github.com/>`_ for the public code repository, `github.io
<http://statsmodels.github.io/>`_ for hosting our documentation and `python.org
<https://python.org>`_ for making our downloads available on PyPi.
