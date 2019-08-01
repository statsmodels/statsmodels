{{ fullname | escape | underline}}

.. currentmodule:: {{ module }}

.. autoclass:: {{ objname }}
   :exclude-members:

   {% block methods %}
   {% if methods %}
   .. rubric:: Methods

   .. autosummary::
      :toctree: generated/

   {% for item in methods %}
   {%- if not item.startswith('_') or item in ['__call__'] %}   ~{{ name }}.{{ item }}
   {% endif %}
   {%- endfor %}
   {% endif %}
   {% endblock %}
   {% block attributes %}
   {% if attributes %}
   .. rubric:: Properties

   .. autosummary::
      :toctree: generated/

   {% for item in attributes %}
   {%- if not item.startswith('_') or item in ['__call__'] %}   ~{{ name }}.{{ item }}
   {% endif %}
   {%- endfor %}
   {% endif %}
   {% endblock %}
