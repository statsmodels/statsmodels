.. _datasets:

The Datasets Package
====================

Original Proposal
~~~~~~~~~~~~~~~~~

The idea for a datasets package was originally proposed by David Cournapeau and
can be found :ref:`here <dataset_proposal>` with updates by me (Skipper
Seabold).

Available Datasets
~~~~~~~~~~~~~~~~~~

.. toctree::
   :maxdepth: 1
   :glob:

   generated/*

Main Usage
~~~~~~~~~~

To load a dataset do the following ::

    >>> import scikits.statsmodels.api as sm
    >>> data = sm.datasets.longley.load()

The `Dataset` object follows the bunch pattern as explain in the
:ref:`proposal <dataset_proposal>`.

Most datasets have two attributes of particular interest to users for examples ::

    >>> data.endog
    array([ 60323.,  61122.,  60171.,  61187.,  63221.,  63639.,  64989.,
            63761.,  66019.,  67857.,  68169.,  66513.,  68655.,  69564.,
            69331.,  70551.])
    >>> data.exog
    array([[     83. ,  234289. ,    2356. ,    1590. ,  107608. ,    1947. ],
           [     88.5,  259426. ,    2325. ,    1456. ,  108632. ,    1948. ],
           [     88.2,  258054. ,    3682. ,    1616. ,  109773. ,    1949. ],
           [     89.5,  284599. ,    3351. ,    1650. ,  110929. ,    1950. ],
           [     96.2,  328975. ,    2099. ,    3099. ,  112075. ,    1951. ],
           [     98.1,  346999. ,    1932. ,    3594. ,  113270. ,    1952. ],
           [     99. ,  365385. ,    1870. ,    3547. ,  115094. ,    1953. ],
           [    100. ,  363112. ,    3578. ,    3350. ,  116219. ,    1954. ],
           [    101.2,  397469. ,    2904. ,    3048. ,  117388. ,    1955. ],
           [    104.6,  419180. ,    2822. ,    2857. ,  118734. ,    1956. ],
           [    108.4,  442769. ,    2936. ,    2798. ,  120445. ,    1957. ],
           [    110.8,  444546. ,    4681. ,    2637. ,  121950. ,    1958. ],
           [    112.6,  482704. ,    3813. ,    2552. ,  123366. ,    1959. ],
           [    114.2,  502601. ,    3931. ,    2514. ,  125368. ,    1960. ],
           [    115.7,  518173. ,    4806. ,    2572. ,  127852. ,    1961. ],
           [    116.9,  554894. ,    4007. ,    2827. ,  130081. ,    1962. ]])

Univariate datasets, however, do not have an `exog` attribute. You can find
out the variable names by doing ::

    >>> data.endog_name
    'TOTEMP'
    >>> data.exog_name
    ['GNPDEFL', 'GNP', 'UNEMP', 'ARMED', 'POP', 'YEAR']

If the dataset does not have a clear interpretation of what should be an
`endog` and `exog`, then you can always access the `data` or `raw_data`
attributes. This is the case for the `macrodata` dataset, which is a collection
of US macroeconomic data rather than a dataset with a specific example in mind.
The `data` attribute contains a record array of the full dataset and the
`raw_data` attribute contains an ndarray with the names of the columns given
by the `names` attribute. ::

    >>> type(data.data)
    numpy.core.records.recarray
    >>> type(data.raw_data)
    numpy.ndarray
    >>> data.names
    ['TOTEMP', 'GNPDEFL', 'GNP', 'UNEMP', 'ARMED', 'POP', 'YEAR']


Extra Information
~~~~~~~~~~~~~~~~~

If you want to know more about the dataset itself, you can access the
following, again using the Longley dataset as an example ::

    >>> dir(sm.datasets.longley)[:6]
    ['COPYRIGHT', 'DESCRLONG', 'DESCRSHORT', 'NOTE', 'SOURCE', 'TITLE']

How to Add a Dataset
~~~~~~~~~~~~~~~~~~~~

See the :ref:`notes on adding a dataset <add_data>`.
