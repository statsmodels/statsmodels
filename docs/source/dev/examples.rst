.. _examples:

Examples
========

Examples are invaluable for new users who hope to get up and running quickly
with `statsmodels`, and they are extremely useful to those who wish to explore
new features of `statsmodels`. We hope to provide documentation and tutorials
for as many models and use-cases as possible!

Most user-contributed examples/tutorials/recipes should be placed on the
`statsmodels examples wiki page
<https://github.com/statsmodels/statsmodels/wiki/Examples:-user-contributions>`_
That wiki page is freely editable. Please post your cool tricks,
examples, and recipes on there! 

If you would rather have your example file officially accepted to the
`statsmodels` distribution and posted on this website, you will need to go
through the normal `patch submission process <index.html#submitting-a-patch>`_.  

File Format
~~~~~~~~~~~

Examples are simple runnable python scripts that go in the top-level examples
directory. We use the `ipython_directive for Sphinx
<http://ipython.org/ipython-doc/dev/development/ipython_directive.html>`_  to
convert them automatically to `reStructuredText
<http://docutils.sourceforge.net/rst.html>`_ and html at build time. 

Each line of the script is executed; both the python code and the printed
results are shown in the output file. Lines that are commented out using the
hash symbol ``#`` are rendered as reST markup. 

**Comments**: "True" comments that should not appear in the output file should be written on lines that start with ``#..``. 

**Error handling**: Syntax errors in pure Python will raise an error during the build process. If you need to show a SyntaxError, an alternative would be to provide a verbatim copy of an IPython session encased in a ReST code block instead of pure Python code. 

**Suppressing lines**: To suppress a line in the built documentation, follow it with a semicolon. 

**Figures**: To save a figure, prepend the line directly before the plotting command with ``#@savefig file_name.png width=4in``, for example. You do not need to call show or close.

**IPython magics**: You can use IPython magics by writing a line like this: ``#%timeit X = np.empty((1000,1000))``.


Make Life Easier
~~~~~~~~~~~~~~~~

To save you some time and to make the new examples nicely fit into the existing
ones consider the following points.

**Look at examples source code** to get a feel for how statsmodels examples should look like.

**PEP8 syntax checker** install a [PEP8] http://pypi.python.org/pypi/pep8 syntax checker for you editor. It will not only make your code look nicer but also serves as `pre-debugger`. Note that some of doc directives explained above imply pep8 violations. Also, for the sake of readability it's a local convention not to add white spaces around power operators, e.g. `x * 2 + y**2 + z`. 

**build docs** run `make html` from the docs directory to see how your example looks in the fully rendered html pages.
