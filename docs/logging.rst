Logging module
============================

| This module provides a number of automatic logging facilities to monitor continual learning experiments.
| Loggers should be provided as input to the :py:class:`EvaluationPlugin` class.

logging
----------------------------------------

.. contents::
    :depth: 2
    :local:
    :backlinks: top

.. currentmodule:: avalanche.logging

Loggers
^^^^^^^^^^^^^^^^^^^^

.. autosummary::
    :toctree: generated

    InteractiveLogger
    TensorboardLogger
    WandBLogger
    TextLogger
    CSVLogger

All the loggers inherit from the base class :py:class:`StrategyLogger`.

.. autosummary::
    :toctree: generated

    StrategyLogger


