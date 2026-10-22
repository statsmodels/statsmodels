:orphan:

==============
Release 0.15.1
==============

This release is under development. These notes describe changes since 0.15.0.

Bug fixes
---------

- GEE's autoregressive dependence update correctly brackets interior minima
  when the fitted correlation is high, instead of raising an error while
  searching for the right bracket. :issue:`8122`
