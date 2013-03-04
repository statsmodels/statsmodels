Using without installing
------------------------

To use this script from an R session: 

    source('statsmodels/tools/R2nparray/R/R2nparray.R')

The 3 most useful commands are: cat_items(), convert_items() and
apply_functions(). To print the content of an R regression to file in a format readable in python, you can:

    sink(file='tmp.py')
    data(iris)
    mod = lm(Sepal.Length ~ Sepal.Width, data=iris)
    cat_items(mod)
    sink()

Install the R2nparray package
-----------------------------

To make full use of the package's functions (including reading docs), you can
install R2nparray in your R library. This first requires installing the
devtools and roxygen2 packages: 

    install.packages(c('devtools', 'roxygen2'))

Then, you need to create a directory to host the documentation. The ``man``
directory should be located here: 

    mkdir statsmodels/tools/R2nparray/man

Finally, run ``R`` inside the ``statsmodels.tools.R2nparray`` folder, and execute:

    library(devtools)
    document()
    install()
