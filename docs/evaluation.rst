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
    class_accuracy_metrics
    amca_metrics
    topk_acc_metrics
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
    images_samples_metrics
    labels_repartition_metrics
    mean_scores_metrics


Stream Metrics
^^^^^^^^^^^^^^^^^^^^

| Stream metrics work at eval time only. Stream metrics return the average of metric results over all the experiences present in the evaluation stream.
| Slicing the evaluation stream during test (e.g., strategy.eval(benchmark.test_stream[0:2])) will not include sliced-out experiences in the average.

.. autosummary::
   :toctree: generated

    StreamAccuracy
    StreamClassAccuracy
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
    StreamTopkAccuracy

Experience Metrics
^^^^^^^^^^^^^^^^^^^^

| Experience metrics compute values that are updated after each experience. Most of them are only updated at eval time and return the average metric results over all the patterns in the experience.

.. autosummary::
   :toctree: generated

    ExperienceAccuracy
    ExperienceClassAccuracy
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
    ExperienceTopkAccuracy
    WeightCheckpoint
    ImagesSamplePlugin

Epoch Metrics
^^^^^^^^^^^^^^^^^^^^

| Epoch metrics work at train time only. Epoch metrics return the average metric results over all the patterns in the training dataset.

.. autosummary::
   :toctree: generated

    EpochAccuracy
    EpochClassAccuracy
    EpochLoss
    EpochCPUUsage
    EpochDiskUsage
    EpochTime
    EpochMAC
    EpochMaxRAM
    EpochMaxGPU
    EpochTopkAccuracy

RunningEpoch Metrics
^^^^^^^^^^^^^^^^^^^^^^

| Running Epoch metrics work at train time only. RunningEpoch metrics return the average metric results over all the patterns encountered up to the current iteration in the training epoch.

.. autosummary::
   :toctree: generated

    RunningEpochAccuracy
    RunningEpochClassAccuracy
    RunningEpochTopkAccuracy
    RunningEpochLoss
    RunningEpochCPUUsage
    RunningEpochTime

Minibatch Metrics
^^^^^^^^^^^^^^^^^^^^

| Minibatch metrics work at train time only. Minibatch metrics return the average metric results over all the patterns in the current minibatch.

.. autosummary::
   :toctree: generated

    MinibatchAccuracy
    MinibatchClassAccuracy
    MinibatchLoss
    MinibatchCPUUsage
    MinibatchDiskUsage
    MinibatchTime
    MinibatchMAC
    MinibatchMaxRAM
    MinibatchMaxGPU
    MinibatchTopkAccuracy

Other Plugin Metrics
^^^^^^^^^^^^^^^^^^^^
.. autosummary::
   :toctree: generated

    WeightCheckpoint

Standalone Metrics
^^^^^^^^^^^^^^^^^^

| Standalone metrics define the metric computation itself. Unlike metric plugins, they cannot be used in Avalanche strategies directly. However, they can be easily used without Avalanche.

.. autosummary::
   :toctree: generated

    Accuracy
    LossMetric
    TaskAwareAccuracy
    TaskAwareLoss
    AverageMeanClassAccuracy
    BWT
    CPUUsage
    ClassAccuracy
    ConfusionMatrix
    DiskUsage
    ElapsedTime
    Forgetting
    ForwardTransfer
    LabelsRepartition
    MAC
    MaxGPU
    MaxRAM
    Mean
    MeanNewOldScores
    MeanScores
    MultiStreamAMCA
    Sum
    TopkAccuracy
    TrainedExperienceTopkAccuracy



evaluation.metrics.detection
----------------------------------------

| Metrics for Object Detection tasks. Please, take a look at the examples in the `examples` folder of Avalanche to better understand how to use these metrics.

.. contents::
    :depth: 2
    :local:
    :backlinks: top

.. currentmodule:: avalanche.evaluation.metrics.detection

.. autosummary::
    :toctree: generated

    make_lvis_metrics
    get_detection_api_from_dataset
    DetectionMetrics


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


evaluation.metric_results
-------------------------------

Metric result types

.. contents:: avalanche.evaluation.metric_results
    :depth: 2
    :local:
    :backlinks: top

.. currentmodule:: avalanche.evaluation.metric_results

.. autosummary::
    :toctree: generated

    MetricValue
    LoggingType
