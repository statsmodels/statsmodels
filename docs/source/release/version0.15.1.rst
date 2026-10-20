:orphan:

==============
Release 0.15.1
==============

Development release
===================

Bug fixes
---------

- Corrected ``graphics.functional.banddepth`` with ``method="BD2"`` for
  crossing curves and tied values. Band containment now preserves the identity
  of both bounding curves across all sampled points (:issue:`7105`).
