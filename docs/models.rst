Models module
============================

| This module provides models and building blocks to design continual learning architectures.

models
----------------------------------------

.. contents::
    :depth: 2
    :local:
    :backlinks: top

.. currentmodule:: avalanche.models

Dynamic Modules
^^^^^^^^^^^^^^^^^^^^

Dynamic Modules are Pytorch modules that can be incrementally expanded
to allow architectural modifications (multi-head classifiers, progressive
networks, ...).

.. autosummary::
    :toctree: generated

    MultiTaskModule
    IncrementalClassifier
    MultiHeadClassifier
    TrainEvalModel

Models
^^^^^^^^^^^^^^^^^^^^^^^

.. autosummary::
    :toctree: generated

    PNN
    make_icarl_net
    SimpleMLP_TinyImageNet
    SimpleCNN
    MTSimpleCNN
    SimpleMLP
    MTSimpleMLP
    MobilenetV1
    NCMClassifier
    SLDAResNetModel
    pytorchcv_wrapper.get_model


Model Utilities
^^^^^^^^^^^^^^^^^^^^

.. autosummary::
    :toctree: generated

    avalanche_forward
    as_multitask
    initialize_icarl_net

Dynamic optimizer utilities
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Utilities to handle optimizer's update when using dynamic architectures.
Dynamic Modules (e.g. multi-head) can change their parameters dynamically
during training, which usually requires to update the optimizer to learn
the new parameters or freeze the old ones.

.. currentmodule:: avalanche.models.dynamic_optimizers

.. autosummary::
    :toctree: generated

    reset_optimizer
    update_optimizer
    add_new_params_to_optimizer

