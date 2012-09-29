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

