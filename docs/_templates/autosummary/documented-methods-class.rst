.. Custom class template without class attributes.
   sphinx has a bug with inherited attributes (https://github.com/sphinx-doc/sphinx/issues/9884)
   that results in a ton of warnings. You can use this template to remove those
   warnings and check that the documentation does not have any other errors.
   Just rename this file into `class.rst` and it will be used by default.

{{ fullname | escape | underline}}

.. currentmodule:: {{ module }}

.. autoclass:: {{ objname }}

   {% block methods %}
   .. automethod:: __init__

   {% if methods %}
   .. rubric:: {{ _('Methods') }}

   .. autosummary::
      :toctree:
   {% for item in methods %}
      ~{{ name }}.{{ item }}
   {%- endfor %}
   {% endif %}
   {% endblock %}

   {% block attributes %}
   {% if attributes %}
   .. rubric:: {{ _('Attributes') }}

   .. autosummary::
   {% for item in attributes %}
      ~{{ name }}.{{ item }}
   {%- endfor %}
   {% endif %}
   {% endblock %}

