:orphan:

===========
0.9 Release
===========

Release 0.9.0
=============

Release summary
---------------

Statsmodels is using now github to store the updated documentation which
is available under
http://www.statsmodels.org/stable for the last release, and
http://www.statsmodels.org/devel/ for the development version.


**Warning**

API stability is not guaranteed for new features, although even in
this case changes will be made in a backwards compatible way if
possible. The stability of a new feature depends on how much time it
was already in statsmodels master and how much usage it has already
seen.  If there are specific known problems or limitations, then they
are mentioned in the docstrings.


The following major new features appear in this version.


Documentation
-------------



Other important improvements
----------------------------




Major Bugs fixed
----------------

* see github issues

While most bugs are usability problems, there is now a new label `type-bug-wrong`
for bugs that cause that silently incorrect numbers are returned.
https://github.com/statsmodels/statsmodels/issues?q=label%3Atype-bug-wrong+is%3Aclosed



Backwards incompatible changes and deprecations
-----------------------------------------------

* In MixedLM, names for the random effects variance and covariance
  parameters have changed from, e.g. G RE to G Var or G x F Cov.  This
  impacts summary output, and also may require modifications to user
  code that extracted these parameters from the fitted results object
  by name.

Development summary and credits
-------------------------------

Besides receiving contributions for new and improved features and for bugfixes,
important contributions to general maintenance came from

* Kevin Sheppard
* Pierre Barbier de Reuille
* Tom Augsburger

and the general maintainer and code reviewer

* Josef Perktold

Additionally, many users contributed by participation in github issues and
providing feedback.

Thanks to all of the contributors for the 0.8 release (based on git log):

.. note::

   * Ashish
   * Brendan
   * Brendan Condon
   * BrianLondon
   * Chad Fulton
   * Chris Fonnesbeck
   * Christian Lorentzen
   * Christoph T. Weidemann
   * James Kerns
   * Josef Perktold
   * Kerby Shedden
   * Kevin Sheppard
   * Leoyzen
   * Matthew Brett
   * Niels Wouda
   * Paul Hobson
   * Pierre Barbier de Reuille
   * Pietro Battiston
   * Ralf Gommers
   * Roman Ring
   * Skipper Seabold
   * Soren Fuglede Jorgensen
   * Thomas Cokelaer
   * Tom Augspurger
   * ValeryTyumen
   * Vanessa
   * Yaroslav Halchenko
   * dhpiyush
   * joesnacks
   * kokes
   * matiumerca
   * rlan
   * ssktotoro
   * thequackdaddy
   * vegcev

Thanks to all of the contributors for the 0.7 release:

.. note::

   * Alex Griffing
   * Antony Lee
   * Chad Fulton
   * Christoph Deil
   * Daniel Sullivan
   * Hans-Martin von Gaudecker
   * Jan Schulz
   * Joey Stockermans
   * Josef Perktold
   * Kerby Shedden
   * Kevin Sheppard
   * Kiyoto Tamura
   * Louis-Philippe Lemieux Perreault
   * Padarn Wilson
   * Ralf Gommers
   * Saket Choudhary
   * Skipper Seabold
   * Tom Augspurger
   * Trent Hauck
   * Vincent Arel-Bundock
   * chebee7i
   * donbeo
   * gliptak
   * hlin117
   * jerry dumblauskas
   * jonahwilliams
   * kiyoto
   * neilsummers
   * waynenilsen

These lists of names are automatically generated based on git log, and may not be
complete.
