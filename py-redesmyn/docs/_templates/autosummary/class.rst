{{ objname | escape | underline}}

.. currentmodule:: {{ module }}

.. autoclass:: {{ objname }}

   {% block methods %}
   {% if methods | map("string") | map("first") | reject("equalto", "_") | list %}

   .. rubric:: {{ _('Methods') }}

   .. autosummary::
      :toctree:
   {% for item in methods %}
   {%- if not item.startswith('_') and (item not in inherited_members or fullname not in no_inherited_members) %}
      ~{{ name }}.{{ item }}
   {%- endif -%}
   {%- endfor %}
   {% endif %}
   {% endblock %}

   {% block attributes %}
   {% set all_attributes = attributes + special.get(fullname, []) %}
   {% if all_attributes %}
   .. rubric:: {{ _('Attributes') }}

   .. autosummary::
   {% for item in all_attributes %}
      ~{{ name }}.{{ item }}
   {%- endfor -%}

   {% endif %}
   {% endblock %}
