Project Structure
-----------------

- [avalanche](avalanche): Main avalanche directory.
    - [Benchmarks](avalanche/benchmarks): All the benchmarks code here.
        -  [cdata_loaders](avalanche/benchmarks/cdata_loaders): CData Loaders
         stands for Continual Data Loader. These data loaders should respect the
          same API and are basically iterators providing new batch/task data
           on demand.
        - [datasets](avalanche/benchmarks/datasets): Since there
         may be multiple CData Loaders (i.e. settings/scenarios) for the same
          orginal dataset, in this directory we maintain all the basic
           utilities functions and classes relative to each dataset/environment.
        - [utils.py](avalanche/benchmarks/utils.py): All the utility function
         related to datasets and cdata_loaders.
    - [Training](avalanche/training): In this module we maintain all the
     continual learning strategies.
    - [Evaluation](avalanche/evaluation): In this module we maintain all the
     evaluation part: protocols, metrics computation and logging.
    - [Extra](avalanche/extras): Other artifacts that may be useful (i.e. pre-trained models, etc.)
- [Examples](examples): In this directory we need to provide a lot of
 examples for every avalanche feature.
- [Tests](tests): A lot of unit tests to cover the entire code. We will also
 need to add a few integration tests.
- [docs](docs): Avalanche versioned documentation with Sphinx and Trevis CI
 to build it automatically on the gh-pages of the github repo.
- [How to Contribute](CONTRIBUTE.md): check this for making the contributions.

# How to Contribute
We welcome your pull requests.

1. Fork the repo and create your branch from `master`.
2. Create or Assign an existing to yourself.
3. Make a pull request

## The following rules should be respected
* Do no push commits directly to the master
* I will merge your PRs into the master
* Always pull before pushing a commit

## Coding Style  
* 4 spaces for indentation rather than tabs
* 80 character line length
* PEP8 formatting


