{{ fullname }}
{{ underline }}

.. currentmodule:: {{ module }}

.. autoclass:: {{ objname }}

   {% block methods %}

   {% if methods %}
   .. rubric:: Methods

   .. autosummary::
      :toctree:
   {% for item in methods %}
    {% if item != '__init__' %}
      ~{{ name }}.{{ item }}
    {% endif %}
   {%- endfor %}
   {% endif %}
   {% endblock %}
