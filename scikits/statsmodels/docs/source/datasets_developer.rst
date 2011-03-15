.. _add_data:

Datasets
~~~~~~~~

For details about the datasets, please see the :ref:`datasets page <datasets>`.

Adding a dataset
================

First, if the data is not in the public domain or listed with a BSD-compatible
license, we must obtain permission from the original author.

To take an example, I will use the Nile River data that measures the volume of
the discharge of the Nile River at Aswan for the years 1871 to 1970. The data
are copied from the paper of Cobb (1978).

Create a directory `datasets/nile/`.  Add `datasets/nile/nile.csv` and
`datasets/__init__.py` that contains ::

    from data import *

If the data will be cleaned before it is in the form included in the datasets
package then create a `nile/src` directory and include the original raw data
there. In this case, it's not necessary.

Next, copy the template_data.py to nile and rename it data.py. Edit the data.py
as follows.  Fill in the strings for COPYRIGHT, TITLE, SOURCE, DESCRSHORT,
DESCLONG, and NOTE. ::

    COPYRIGHT   = """This is public domain."""
    TITLE       = """Nile River Data"""
    SOURCE      = """
    Cobb, G.W. 1978. The Problem of the Nile: Conditional Solution to a Changepoint
        Problem. Biometrika. 65.2, 243-251,
    """

    DESCRSHORT  = """Annual Nile River Volume at Aswan, 1871-1970""

    DESCRLONG   = """AAnnual Nile River Volume at Aswan, 1871-1970. The units of
    measurement are 1e9 m^{3}, and there is an apparent changepoint near 1898."""

    NOTE        = """
    Number of observations: 100
    Number of variables: 2
    Variable name definitions:
        year - Year of observation
        volume - Nile River volume at Aswan

    The data were originally used in Cobb (1987, See SOURCE). The author
    acknowledges that the data were originally compiled from various sources by
    Dr. Barbara Bell, Center for Astrophysics, Cambridge, Massachusetts. The data
    set is also used as an example in many textbooks and software packages.
    """

Next we edit the `load` function. You only need to edit the docstring to
specify which dataset will be loaded. You should also edit the path and the
indices for the `endog` and `exog` attributes. In this case, there is no
`exog`, so everything referencing `exog` is not used. The `year` variable is
also not used.

Lastly, edit the datasets/__init__.py to import the directory.

That's it!
