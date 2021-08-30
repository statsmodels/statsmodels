{{ fullname | escape | underline}}

.. currentmodule:: {{ module }}

.. autoclass:: {{ objname }}
   :exclude-members:

   {% block attributes %}
   {% if attributes %}
   .. rubric:: Properties

   .. autosummary::
      :toctree:

   {% for item in attributes %}
   {%- if not item.startswith('_') or item in ['__call__'] %}   ~{{ name }}.{{ item }}
   {% endif %}
   {%- endfor %}
   {% endif %}
   {% endblock %}
