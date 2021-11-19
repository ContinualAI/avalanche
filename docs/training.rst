Training module
============================

Training Strategies
----------------------------------------

Ready-to-use continual learning strategies.

.. currentmodule:: avalanche.training

.. autosummary::
    :toctree: generated

    BaseStrategy
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


Replay Buffers
----------------------------------------

Buffers to store past samples according to different policies and selection strategies.

.. autosummary::
    :toctree: generated

    ExemplarsBuffer
    ReservoirSamplingBuffer
    BalancedExemplarsBuffer
    ExperienceBalancedBuffer
    ClassBalancedBuffer
    ParametricBuffer


Selection strategies
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

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


Training Plugins
----------------------------------------

Plugins can be added to any CL strategy to support additional behavior.

Utilities
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. currentmodule:: avalanche.training.plugins

.. autosummary::
    :toctree: generated

    StrategyPlugin
    EarlyStoppingPlugin
    EvaluationPlugin
    LRSchedulerPlugin


Strategies
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

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
