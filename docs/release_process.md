# Release Process Documentation
This document describes the steps required to publish a release of Avalanche in PyPi.

Place yourself in the Avalanche project directory.

1. Create the release statement with the new candidate tag version

2. Change the version in avalanche/__init__.py

3. Execute the following commands via bash:
    ```bash 
    python -m pip install build twine # install build tools
    python -m build # build the binaries
    twine check dist/* # check that build went smoothly
    twine upload -r testpypi dist/* # upload package to test.pypi (need credentials)
    ```
   
4. After uploading to test.pypi, install the package in a fresh environment with 
    ```bash
    python -m pip install --extra-index-url https://pypi.python.org/simple -i https://test.pypi.org/simple/ avalanche-lib`
    ```
    The `--extra-index-url` is required to install dependencies from pypi and not from test.pypi where they are not all available.

5. Upload to PyPi
    ```bash
    twine upload dist/*
    ```

6. Update default ReadTheDocs and GitBook version.   
For ReadTheDocs, activate the new tag and make it default.  
For GitBook, create a copy of the current space. The copy will not be synced with Github, while the original one will
still be synced. Make it the default collection. 