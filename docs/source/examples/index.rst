:orphan:

.. _statsmodels-examples:

Examples
========

This page provides a series of examples, tutorials and recipes to help you get
started with ``statsmodels``. Each of the examples shown here is made available
as an IPython Notebook and as a plain python script on the `statsmodels github
repository <https://github.com/statsmodels/statsmodels/tree/master/examples>`_.

We also encourage users to submit their own examples, tutorials or cool
`statsmodels` trick to the `Examples wiki page
<https://github.com/statsmodels/statsmodels/wiki/Examples>`_

{# This content is white space sensitive. Do not reformat #}

{% for category in examples%}
{% set underscore = "-" * (category.header | length) %}
{{ category.header }}
{{ underscore }}

.. toctree::
   :maxdepth: 1
   :hidden:

{% for notebook in category.links  %}   {{ notebook.target | replace('.html','') }}
{% endfor %}

{%- for notebook in category.links  %}
{% set heading = "`" ~ notebook.text ~ " <" ~ notebook.target|e ~ ">`_" %}
{% set subunderscore = "~" * (heading | length) %}
{{ heading }}
{{ subunderscore }}
.. image:: {{ notebook.img }}
   :target: {{ notebook.target }}
   :width: 360px

{%- endfor %}

{% endfor %}
