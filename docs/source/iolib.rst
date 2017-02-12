.. currentmodule:: statsmodels.iolib

.. _iolib:

Input-Output :mod:`iolib`
=========================

``statsmodels`` offers some functions for input and output. These include a
reader for STATA files, a class for generating tables for printing in several
formats and two helper functions for pickling.

Users can also leverage the powerful input/output functions provided by :ref:`pandas.io <pandas:io>`. Among other things, ``pandas`` (a ``statsmodels`` dependency) allows reading and writing to Excel, CSV, and HDF5 (PyTables).

Examples
--------

    `SimpleTable: Basic example <examples/notebooks/generated/wls.html#ols-vs-wls>`__

Module Reference
----------------

.. module:: statsmodels.iolib
   :synopsis: Tools for reading datasets and producing summary output

.. autosummary::
   :toctree: generated/

   foreign.StataReader
   foreign.StataWriter
   foreign.genfromdta
   foreign.savetxt
   table.SimpleTable
   table.csv2st
   smpickle.save_pickle
   smpickle.load_pickle


The following are classes and functions used to return the summary of
estimation results, and mostly intended for internal use. There are currently
two versions for creating summaries.

.. autosummary::
   :toctree: generated/

   summary.Summary
   summary2.Summary
