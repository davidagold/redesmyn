{{ name | escape | underline}}


.. currentmodule:: {{ module }}

{%- if fullname in decorators %}
.. autodecorator:: {{ objname }}
{% else %}
.. autofunction:: {{ objname }}
{% endif -%}
