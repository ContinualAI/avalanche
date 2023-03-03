Training module
============================

.. currentmodule:: avalanche.training

training
----------------------------------------

.. contents::
    :depth: 2
    :local:
    :backlinks: top

Training Templates
------------------

Templates define the training/eval loop for each setting (supervised CL, online CL, RL, ...). Each template supports a set of callback that can be used by a plugin to execute code inside the training/eval loops.

Templates
"""""""""

Templates are defined in the `avalanche.training.templates` module.

.. currentmodule:: avalanche.training.templates

.. autosummary::
    :toctree: generated

    BaseTemplate
    BaseSGDTemplate
    SupervisedTemplate
    OnlineSupervisedTemplate


Plugins ABCs
""""""""""""

ABCs for plugins are available in `avalanche.core`.

.. currentmodule:: avalanche.core

.. autosummary::
    :toctree: generated

    BasePlugin
    BaseSGDPlugin
    SupervisedPlugin


.. currentmodule:: avalanche.training

Training Strategies
----------------------------------------

Ready-to-use continual learning strategies.


.. autosummary::
    :toctree: generated

    Cumulative
    JointTraining
    Naive
    AR1
    StreamingLDA
    ICaRL
    PNNStrategy
    CWRStar
    Replay
    GSS_greedy
    GDumb
    LwF
    AGEM
    GEM
    EWC
    SynapticIntelligence
    CoPE
    LFL
    GenerativeReplay
    LaMAML
    MAS
    BiC
    MIR

Replay Buffers and Selection Strategies
----------------------------------------

Buffers to store past samples according to different policies and selection strategies.

Buffers
"""""""

.. autosummary::
    :toctree: generated

    ExemplarsBuffer
    ReservoirSamplingBuffer
    BalancedExemplarsBuffer
    ExperienceBalancedBuffer
    ClassBalancedBuffer
    ParametricBuffer


Selection strategies
""""""""""""""""""""

.. autosummary::
    :toctree: generated

    ExemplarsSelectionStrategy
    RandomExemplarsSelectionStrategy
    FeatureBasedExemplarsSelectionStrategy
    HerdingSelectionStrategy
    ClosestToCenterSelectionStrategy


Loss Functions
----------------------------------------

.. autosummary::
    :toctree: generated

    ICaRLLossPlugin
    RegularizationMethod
    LearningWithoutForgetting


Training Plugins
----------------------------------------

Plugins can be added to any CL strategy to support additional behavior.

Utilities
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Utilities in `avalanche.training.plugins`.

.. currentmodule:: avalanche.training.plugins

.. autosummary::
    :toctree: generated

    EarlyStoppingPlugin
    EvaluationPlugin
    LRSchedulerPlugin


Strategies
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Strategy implemented as plugins in `avalanche.training.plugins`.

.. currentmodule:: avalanche.training.plugins

.. autosummary::
    :toctree: generated

    AGEMPlugin
    CoPEPlugin
    CWRStarPlugin
    EWCPlugin
    GDumbPlugin
    GEMPlugin
    GSS_greedyPlugin
    LFLPlugin
    LwFPlugin
    ReplayPlugin
    SynapticIntelligencePlugin
    MASPlugin
    TrainGeneratorAfterExpPlugin
    RWalkPlugin
    GenerativeReplayPlugin
    BiCPlugin
    MIRPlugin
