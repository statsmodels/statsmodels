.. _examples:

Examples
========

Examples are invaluable for new users who hope to get up and running quickly
with `statsmodels`, and they are extremely useful to those who wish to explore
new features of `statsmodels`. We hope to provide documentation and tutorials
for as many models and use-cases as possible! Please consider submitting an
example with any PR that introduces new functionality.

User-contributed examples/tutorials/recipes can be placed on the
`statsmodels examples wiki page <https://github.com/statsmodels/statsmodels/wiki/Examples>`_
That wiki page is freely editable. Please post your cool tricks,
examples, and recipes on there!

If you would rather have your example file officially accepted to the
`statsmodels` distribution and posted on this website, you will need to go
through the normal `patch submission process <index.html#submitting-a-patch>`_
and follow the instructions that follow.

File Format
-----------

Examples are best contributed as IPython notebooks. Save your notebook with all
output cells cleared in ``examples/notebooks``. From the notebook save the pure
Python output to ``examples/python``. The first line of the Notebook *must* be
a header cell that contains a title for the notebook, if you want the notebook
to be included in the documentation.


The Example Gallery
-------------------

We have a gallery of example notebooks available
`here <https://www.statsmodels.org/devel/examples/index.html>`_. If you would
like your example to show up in this gallery, add a link to the notebook in
``docs/source/examples/landing.json``. For the thumbnail, take a screenshot of
what you think is the best "hook" for the notebook. The image will be displayed
at 360 x 225 (W x H). It's best to save the image as a PNG with a resolution
that is some multiple of 360 x 225 (720 x 450 is preferred).

Please remember to shrink the PNG file, if you can.
`This website <https://tinypng.com>`_ can help with that.


Before submitting a PR
----------------------

To save you some time and to make the new examples nicely fit into the
existing ones consider the following points.

**Look at examples source code** to get a feel for how statsmodels examples
should look like.

**Build the docs** by running `make html` from the docs directory to see how
your example looks in the fully rendered html pages.
