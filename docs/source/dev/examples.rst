.. _examples:

Statsmodels Examples
====================

Examples go in the top-level examples directory. Let's try to have documentation
and tutorials for as many models and code uses as possible! These are invaluable
for new users to get up and running. You can also include cookbook type recipes 
here if For the most part these are just runnable
example scripts. However, when the documentation is built, these are converted 
into ReST files and included in the documentation. There is a bit of magic that
can be used to make these look nice.

reStructured Text
~~~~~~~~~~~~~~~~~

Every example file must have a module level docstring. This docstring should contain 
the tile of the example, and that's it. You can include ReST markup in the files as 
comments. Anything that is commented out will be rendered as ReST with a few 
exceptions noted below. If you want a true comment in the outputed file, then you 
should use ``#..``. The hash symbol is stripped leaving ``..``, ReST markup for a 
comment line.

Code Snippets
~~~~~~~~~~~~~

Code snippets are rendered using the :ref:`ipython_directive` for Sphinx. See
the documentation for explaining its usage in greater detail. Some of it is 
explained in :ref:`special_markup`.

.. _special_markup:

Special Markup
~~~~~~~~~~~~~~

Pretty much anything you can do with the IPython directive is supported for the
example scripts. To suppress a line in the built documentation, follow it with
a semicolon. To save a figure, prepend the line directly before the pyplot 
command with ``#@savefig file_name.png width=4in``, for example. You don't
need to call show or close. You can also call IPython magic functions. So if 
you wanted to include some timings you could have a line ``#%timeit X = 
np.empty((1000,1000))``.
