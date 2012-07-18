.. currentmodule:: statsmodels.iolib

.. _iolib:

Input-Output :mod:`iolib`
=========================

``statsmodels`` offers some functions for input and output. These include a
reader for STATA files, a class for generating tables for printing in several
formats and two helper functions for pickling.

Users can also leverage the powerful input/output functions provided by ``pandas`` (a ``statsmodels`` dependency). Among other things, ``pandas`` allows reading and writing to Excel, CSV, and HDF5 (PyTables). http://pandas.sourceforge.net/io.html

Examples
--------

    `SimpleTable: Basic example <examples/generated/example_wls.html#ols-vs-wls>`_

Module Reference
----------------


.. autosummary::
   :toctree: generated/

   foreign.StataReader
   foreign.genfromdta
   foreign.savetxt
   table.SimpleTable
   table.csv2st
   smpickle.save_pickle
   smpickle.load_pickle
