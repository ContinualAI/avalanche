---
description: Understand the Avalanche Structure
---

# Introduction

![](../.gitbook/assets/avalanche\_api.png)

Welcome to the "_Introduction_" tutorial of the "_From Zero to Hero_" series. We will start our journey by taking a quick look at _Avalanche_ main modules and its **general architecture**.

_Avalanche_ is organized in **five main modules**: Benchmarks, Training, Evaluation, Models, Logging.

#### Benchmarks

This module provides a uniform API for data handling: mostly generating a stream of data from one or more datasets. It contains all the major CL benchmarks (similar to what has been done for [torchvision](https://pytorch.org/docs/stable/torchvision/index.html)). Its main components are:

* **Benchmarks**: these are complete benchmarks, providing access to train/test streams.
* **Streams**: a sequential list of learning experiences.
* **Experience**: a bunch of data available at a specific point in time.
* **AvalancheDataset**: a dataset that provides support for train/eval transformations, concatenation and subsampling, and other operations needed to manipulate data in continual learning strategies.

#### Training

This module provides all the necessary utilities concerning model training. This includes simple and efficient ways of implement new _continual learning_ strategies as well as a set pre-implemented CL baselines and state-of-the-art algorithms you will be able to use for comparison!

* **BaseStrategy** provides the default training and eval loops.
* **Plugins** extend the basic loops with additional functionality.

#### **`Evaluation`**

This module provides all the utilities and metrics that can help evaluate a CL algorithm with respect to all the factors we believe to be important for a continually learning system.

* **EvaluationPlugin**: the plugin that connects metrics and the `BaseStrategy.`
* **MetricPlugins**: plugins that provide a bridge between metrics and the evaluation plugin. They emit `MetricValue`s that will be collected by the `EvaluationPlugin` and serialized by the loggers.
* **Metric**: basic logic to compute a metric. Provides the `update`, `result` and `reset` operations used to compute and retrieve the metric value.

#### Models

In this module you'll find model architectures, pre-trained models, and utilities to implement continual learning models. We provide everything you need to implement architectural strategies, task-aware models, and dynamic model expansion.

#### Logging

Metrics are automatically logger using native _stdout_, _files_ and [Tensorboard](https://www.tensorflow.org/tensorboard) support (How cool it is to have a complete, interactive dashboard, tracking your experiment metrics in real-time with a single line of code?)

## File Structure

{% code title="Avalanche Main Modules and Sub-Modules" %}
```
Avalanche
├── Benchmarks
│   ├── Classic
│   ├── Datasets
│   ├── Generators
│   ├── Scenarios
│   └── Utils
├── Evaluation
│   ├── Metrics
│   ├── Tensorboard
|   └── Utils
├── Training
│   ├── Strategies
│   ├── Plugins
|   └── Utils
├── Models
└── Loggers
```
{% endcode %}

In this series of tutorials, you'll get the chance to learn in-depth all the features offered by each module and sub-module of _Avalanche_, how to put them together and how to master _Avalanche_, for a **stress-free continual learning prototyping experience**!

{% hint style="info" %}
In the following tutorials we will assume you have already installed _Avalanche on your computer_ or server. If you haven't yet, check out how you can do it following our [How to Install](../getting-started/how-to-install.md) guide.
{% endhint %}

## 🤝 Run it on Google Colab

You can run _this chapter_ and play with it on Google Colaboratory: [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ContinualAI/avalanche/blob/master/notebooks/from-zero-to-hero-tutorial/01\_introduction.ipynb)
