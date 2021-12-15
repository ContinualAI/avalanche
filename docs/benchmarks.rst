Benchmarks module
============================

| This module provides popular continual learning benchmarks and generic facilities to build custom benchmarks.

* Popular benchmarks (like SplitMNIST, PermutedMNIST, SplitCIFAR, ...) are contained in the ``classic`` sub-module.
* Dataset implementations are available in the ``datasets`` sub-module.
* One can create new benchmarks by using the utilities found in the ``generators`` sub-module.
* Avalanche uses custom dataset and dataloader implementations contained in the ``utils`` sub-module. More info can be found in this couple of How-Tos `here <https://avalanche.continualai.org/how-tos/dataloading_buffers_replay>`_ and `here <https://avalanche.continualai.org/how-tos/avalanchedataset>`_.


avalanche.benchmarks
----------------------------------------

.. contents::
    :depth: 2
    :local:
    :backlinks: top

.. currentmodule:: avalanche.benchmarks.classic

Classic Benchmarks
^^^^^^^^^^^^^^^^^^^^^^^^^^

| The **classic benchmarks** sub-module covers all mainstream benchmarks. Expect this list to grow over time!


CORe50-based benchmarks
............................
Benchmarks based on the `CORe50 <https://vlomonaco.github.io/core50/>`_ dataset.

.. autosummary::
    :toctree: generated

    CORe50


CIFAR-based benchmarks
............................
Benchmarks based on the `CIFAR-10 and CIFAR-100 <https://www.cs.toronto.edu/~kriz/cifar.html>`_ datasets.

.. autosummary::
    :toctree: generated

    SplitCIFAR10
    SplitCIFAR100
    SplitCIFAR110


CUB200-based benchmarks
............................
Benchmarks based on the `Caltech-UCSD Birds 200 <http://www.vision.caltech.edu/visipedia/CUB-200.html>`_ dataset.

.. autosummary::
    :toctree: generated

    SplitCUB200


EndlessCLSim-based benchmarks
............................
Benchmarks based on the `EndlessCLSim <https://zenodo.org/record/4899267>`_ derived datasets.

.. autosummary::
    :toctree: generated

    EndlessCLSim


FashionMNIST-based benchmarks
............................
Benchmarks based on the `Fashion MNIST <https://github.com/zalandoresearch/fashion-mnist>`_ dataset.

.. autosummary::
    :toctree: generated

    SplitFMNIST


ImageNet-based benchmarks
............................
Benchmarks based on the `ImageNet ILSVRC-2012 <https://www.image-net.org/>`_ dataset.

.. autosummary::
    :toctree: generated

    SplitImageNet
    SplitTinyImageNet


iNaturalist-based benchmarks
............................
Benchmarks based on the `iNaturalist-2018 <https://www.kaggle.com/c/inaturalist-2018/>`_ dataset.

.. autosummary::
    :toctree: generated

    SplitInaturalist


MNIST-based benchmarks
............................
Benchmarks based on the `MNIST <http://yann.lecun.com/exdb/mnist/>`_ dataset.

.. autosummary::
    :toctree: generated

    SplitMNIST
    PermutedMNIST
    RotatedMNIST


Omniglot-based benchmarks
............................
Benchmarks based on the `Omniglot <https://github.com/brendenlake/omniglot>`_ dataset.

.. autosummary::
    :toctree: generated

    SplitOmniglot
    PermutedOmniglot
    RotatedOmniglot


OpenLORIS-based benchmarks
............................
Benchmarks based on the `OpenLORIS <https://lifelong-robotic-vision.github.io/dataset/scene.html>`_ dataset.

.. autosummary::
    :toctree: generated

    OpenLORIS


Stream51-based benchmarks
............................
Benchmarks based on the `Stream-51, <https://github.com/tyler-hayes/Stream-51>`_ dataset.

.. autosummary::
    :toctree: generated

    CLStream51


.. currentmodule:: avalanche.benchmarks.datasets

Datasets
^^^^^^^^^^^^^^^^^^^^^^^^^^

| The **datasets** sub-module provides PyTorch dataset implementations for datasets missing from the torchvision/audio/* libraries. These datasets can also be used in a standalone way!

.. autosummary::
    :toctree: generated

    CORe50Dataset
    CUB200
    EndlessCLSimDataset
    INATURALIST2018
    MiniImageNetDataset
    Omniglot
    OpenLORIS
    Stream51
    TinyImagenet

.. currentmodule:: avalanche.benchmarks.generators

Benchmark Generators
^^^^^^^^^^^^^^^^^^^^^^^^^^
| The **generators** sub-module provides a lot of functions that can be used to create a new benchmark.
| This set of functions tries to cover most common use cases (Class/Task-Incremental, Domain-Incremental, ...) but it also allows for the creation of entirely custom benchmarks (based on lists of tensors, on file lists, ...).


Generators for Class/Task/Domain-incremental benchmarks
............................

.. autosummary::
    :toctree: generated

    nc_benchmark
    ni_benchmark


Starting from tensor lists, file lists, PyTorch datasets
............................

.. autosummary::
    :toctree: generated

    dataset_benchmark
    filelist_benchmark
    paths_benchmark
    tensors_benchmark


Misc (make data-incremental, add a validation stream, ...)
............................

| Avalanche offers utilities to adapt a previously instantiated benchmark object.
| More utilities to come!

.. autosummary::
    :toctree: generated

    data_incremental_benchmark
    benchmark_with_validation_stream

.. currentmodule:: avalanche.benchmarks.utils

Utils (Data Loading and AvalancheDataset)
^^^^^^^^^^^^^^^^^^^^^^^^^^
| The custom dataset and dataloader implementations contained in this sub-module are described in more detailed in the How-Tos `here <https://avalanche.continualai.org/how-tos/dataloading_buffers_replay>`_ and `here <https://avalanche.continualai.org/how-tos/avalanchedataset>`_.


.. currentmodule:: avalanche.benchmarks.utils.data_loader

Data Loaders
............................
.. autosummary::
    :toctree: generated

    TaskBalancedDataLoader
    GroupBalancedDataLoader
    ReplayDataLoader
    GroupBalancedInfiniteDataLoader


.. currentmodule:: avalanche.benchmarks.utils.avalanche_dataset

AvalancheDataset
............................
.. autosummary::
    :toctree: generated

    AvalancheDataset
    AvalancheSubset
    AvalancheTensorDataset
    AvalancheConcatDataset
