:orphan:

==============
Release 0.15.1
==============

This release is under development. These notes describe changes since 0.15.0.

Bug fixes
---------

- KDEUnivariate centers and scales quadrature for unbounded-kernel entropy,
  and uses intervals around the observations to locate shifted, narrow or
  separated kernel components without truncating the infinite tails.
  :issue:`1313`
