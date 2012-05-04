.. currentmodule:: statsmodels.iolib

.. _iolib:

Input-Output :mod:`iolib`
=========================

Introduction
------------

Some functions for input and output.

Module Reference
----------------

This contains a reader for STATA files, a class for generating tables for
printing in several formats and two helper functions for pickling.

.. autosummary::
   :toctree: generated/

   foreign.StataReader
   foreign.genfromdta
   foreign.savetxt
   table.SimpleTable
   table.csv2st
   smpickle.save_pickle
   smpickle.load_pickle
