# Documentation

Documentation types:
- gitbook: this corresponds to [avalanche.continualai.org/](avalanche.continualai.org/).
- apidoc: [https://avalanche-api.continualai.org](https://avalanche-api.continualai.org)
- notebooks: notebook folder. Also mirrored at [avalanche.continualai.org/](avalanche.continualai.org/).

## API Doc
Built with sphinx usig the ReST markup language, autosummary, and a bunch of other sphinx extensions.

command to build the documentation (it must be executed in the `docs` folder):
```
sphinx-build . _build
```

### Doc coverage
it is possible to check the class coverage, i.e. whether some missing classes and for syntax errors using the command:
```
sphinx-build -b coverage . _build
```
in `conf.py` you find a list `undocumented_classes_to_ignore`. This keeps track of the classes that we don't want to add to the apidoc. Add a class here if you believe it should not be documented in the apidoc. Ideally, this should be kept to a minimum.

### Check for errors
Due to a bug in sphinx, we have lots of warnings. You can silence them by changing the class template by renaming `_templates/autosummary/documented-methods-class.rst` into `_templates/autosummary/class.rst` to disable the documentation of attributes.