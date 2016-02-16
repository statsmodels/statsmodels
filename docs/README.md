# Documentation Documentation

We use a combination of sphinx and Jupyter notebooks for the documentation.
Jupyter notebooks should be used for longer, self-contained examples demonstrating
a topic.
Sphinx is nice because we get the tables of contents and API documentation.

## Build Process

Building the docs requires a few additional dependencies. You can get most
of these with

```bash

   pip install -e .[docs]

```

From the root of the project.
Some of the examples rely on `rpy2` to execute R code from the notebooks.
It's not included in the setup requires since it's known to be difficult to
install.

To generate the HTML docs, run ``make html`` from the ``docs`` directory.
This executes a few distinct builds

1. datasets
2. notebooks
3. sphinx

# Notebook Builds

We're using `nbconvert` to execute the notebooks, and then convert them
to HTML. The conversion is handled by `statsmodels/tools/nbgenerate.py`.
The default python kernel (embedded in the notebook) is `python3`.
You need at least `nbconvert==4.2.0` to specify a non-default kernel,
which can be passed in the Makefile.
