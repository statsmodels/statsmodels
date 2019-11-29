.. _add_data:

Datasets
========

For a list of currently available datasets and usage instructions, see the
:ref:`datasets page <datasets>`.

License
-------

To be considered for inclusion in `statsmodels`, a dataset must be in the
public domain, distributed under a BSD-compatible license, or we must obtain
permission from the original author.

Adding a dataset: An example
----------------------------

The Nile River data measures the volume of the discharge of the Nile River at
Aswan for the years 1871 to 1970. The data are copied from the paper of Cobb
(1978).

**Step 1**: Create a directory `datasets/nile/`

**Step 2**: Add `datasets/nile/nile.csv` and  a new file `datasets/__init__.py` which contains ::

    from data import *

**Step 3**: If `nile.csv` is a transformed/cleaned version of the original data, create a `nile/src` directory and include the original raw data there. In the `nile` case, this step is not necessary.

**Step 4**: Copy `datasets/template_data.py` to `nile/data.py`. Edit `nile/data.py` by filling-in strings for COPYRIGHT, TITLE, SOURCE, DESCRSHORT, DESCLONG, and NOTE. ::

    COPYRIGHT   = """This is public domain."""
    TITLE       = """Nile River Data"""
    SOURCE      = """
    Cobb, G.W. 1978. The Problem of the Nile: Conditional Solution to a Changepoint
        Problem. Biometrika. 65.2, 243-251,
    """

    DESCRSHORT  = """Annual Nile River Volume at Aswan, 1871-1970""

    DESCRLONG   = """Annual Nile River Volume at Aswan, 1871-1970. The units of
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

**Step 5:** Edit the docstring of the `load` function in `data.py` to specify
which dataset will be loaded. Also edit the path and the indices for the
`endog` and `exog` attributes. In the `nile` case, there is no `exog`, so
everything referencing `exog` is not used. The `year` variable is also not
used.

**Step 6:** Edit the `datasets/__init__.py` to import the directory.

That's it! The result can be found `here
<https://github.com/statsmodels/statsmodels/tree/master/statsmodels/datasets/nile>`_
for reference.
