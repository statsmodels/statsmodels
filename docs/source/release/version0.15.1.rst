:orphan:

==============
Release 0.15.1
==============

This release is under development. These notes describe changes since 0.15.0.

Bug fixes
---------

- ConditionalLogit evaluates the conditional likelihood and its score without
  recursion, allowing groups larger than Python's recursion limit.
  :issue:`6668`
