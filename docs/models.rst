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
^^^^^^^^^^^^^^^

Dynamic Modules are Pytorch modules that can be incrementally expanded
to allow architectural modifications (multi-head classifiers, progressive
networks, ...).

.. autosummary::
    :toctree: generated

    DynamicModule
    MultiTaskModule
    IncrementalClassifier
    MultiHeadClassifier


Models
^^^^^^^^^^^^^^^^^^^^^^^

| Neural network architectures that can be used as backbones for CL experiments.

.. autosummary::
    :toctree: generated

    MLP
    make_icarl_net
    IcarlNet
    SimpleMLP_TinyImageNet
    SimpleCNN
    MTSimpleCNN
    SimpleMLP
    MTSimpleMLP
    SimpleSequenceClassifier
    MTSimpleSequenceClassifier
    MobilenetV1
    NCMClassifier
    SLDAResNetModel  
    MlpVAE
    LeNet5
    SlimResNet18
    MTSlimResNet18


Progressive Neural Networks
^^^^^^^^^^^^^^^^^^^^^^^^^^^
Modules that implement progressive neural networks models, layers, and adapters.

.. autosummary::
    :toctree: generated
    
    PNN
    PNNLayer
    PNNColumn    
    LinearAdapter
    MLPAdapter


Model Wrappers and Utilities
^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Wrappers and functions that add utility support to your models.

.. autosummary::
    :toctree: generated

    TrainEvalModel
    FeatureExtractorBackbone
    BaseModel
    avalanche_forward
    as_multitask
    initialize_icarl_net
    pytorchcv_wrapper.get_model

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

