Evaluation module
============================

| This module provides a number of metrics to monitor the continual learning performance.
| Metrics subclass the :py:class:`PluginMetric` class, which provides all the callbacks needed to include custom metric logic in specific points of the continual learning workflow.

evaluation.metrics
----------------------------------------

.. contents::
    :depth: 2
    :local:
    :backlinks: top

.. currentmodule:: avalanche.evaluation.metrics

Metrics helper functions
^^^^^^^^^^^^^^^^^^^^^^^^^^

| High-level functions to get specific plugin metrics objects (to be passed to the :py:class:`EvaluationPlugin`).
| **This is the recommended way to build metrics. Use these functions when available.**

.. autosummary::
    :toctree: generated

    accuracy_metrics
    loss_metrics
    bwt_metrics
    forgetting_metrics
    forward_transfer_metrics
    confusion_matrix_metrics
    cpu_usage_metrics
    disk_usage_metrics
    gpu_usage_metrics
    ram_usage_metrics
    timing_metrics
    MAC_metrics
    image_samples_metrics
    labels_repartition_metrics
    mean_scores_metrics


Stream Metrics
^^^^^^^^^^^^^^^^^^^^

| Stream metrics work at eval time only. Stream metrics return the average of metric results over all the experiences present in the evaluation stream.
| Slicing the evaluation stream during test (e.g., strategy.eval(benchmark.test_stream[0:2])) will not include sliced-out experiences in the average.

.. autosummary:: Stream metrics
   :toctree: generated

    StreamAccuracy
    TrainedExperienceAccuracy
    StreamLoss
    StreamBWT
    StreamForgetting
    StreamForwardTransfer
    StreamConfusionMatrix
    WandBStreamConfusionMatrix
    StreamCPUUsage
    StreamDiskUsage
    StreamTime
    StreamMaxRAM
    StreamMaxGPU
    MeanScoresEvalPluginMetric

Experience Metrics
^^^^^^^^^^^^^^^^^^^^

| Experience metrics work at eval time only. Experience metrics return the average metric results over all the patterns in the experience.

.. autosummary:: Experience metrics
   :toctree: generated

    ExperienceAccuracy
    ExperienceLoss
    ExperienceBWT
    ExperienceForgetting
    ExperienceForwardTransfer
    ExperienceCPUUsage
    ExperienceDiskUsage
    ExperienceTime
    ExperienceMAC
    ExperienceMaxRAM
    ExperienceMaxGPU

Epoch Metrics
^^^^^^^^^^^^^^^^^^^^

| Epoch metrics work at train time only. Epoch metrics return the average metric results over all the patterns in the training dataset.

.. autosummary:: Epoch metrics
   :toctree: generated

    EpochAccuracy
    EpochLoss
    EpochCPUUsage
    EpochDiskUsage
    EpochTime
    EpochMAC
    EpochMaxRAM
    EpochMaxGPU
    MeanScoresTrainPluginMetric

RunningEpoch Metrics
^^^^^^^^^^^^^^^^^^^^^^

| Running Epoch metrics work at train time only. RunningEpoch metrics return the average metric results over all the patterns encountered up to the current iteration in the training epoch.

.. autosummary:: RunningEpoch metrics
   :toctree: generated

    RunningEpochAccuracy
    RunningEpochLoss
    RunningEpochCPUUsage
    RunningEpochTime

Minibatch Metrics
^^^^^^^^^^^^^^^^^^^^

| Minibatch metrics work at train time only. Minibatch metrics return the average metric results over all the patterns in the current minibatch.

.. autosummary:: Minibatch metrics
   :toctree: generated

    MinibatchAccuracy
    MinibatchLoss
    MinibatchCPUUsage
    MinibatchDiskUsage
    MinibatchTime
    MinibatchMAC
    MinibatchMaxRAM
    MinibatchMaxGPU


evaluation.metric_definitions
-------------------------------

General interfaces on which metrics are built.

.. contents:: avalanche.evaluation.metric_definitions
    :depth: 2
    :local:
    :backlinks: top

.. currentmodule:: avalanche.evaluation.metric_definitions

.. autosummary::
    :toctree: generated

    Metric
    PluginMetric
    GenericPluginMetric