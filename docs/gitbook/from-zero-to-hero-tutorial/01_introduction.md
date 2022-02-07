---
description: Understand the Avalanche Package Structure
---

# Introduction

Welcome to the "_Introduction_" tutorial of the "_From Zero to Hero_" series. We will start our journey by taking a quick look at the _Avalanche_ main modules to understand its **general architecture**.

As hinted in the getting started introduction _Avalanche_ is organized in **five main modules**:

* **`Benchmarks`**: This module maintains a uniform API for data handling: mostly generating a stream of data from one or more datasets. It contains all the major CL benchmarks \(similar to what has been done for [torchvision](https://pytorch.org/docs/stable/torchvision/index.html)\).
* **`Training`**: This module provides all the necessary utilities concerning model training. This includes simple and efficient ways of implement new _continual learning_ strategies as well as a set pre-implemented CL baselines and state-of-the-art algorithms you will be able to use for comparison!
* **`Evaluation`**: This module provides all the utilities and metrics that can help evaluate a CL algorithm with respect to all the factors we believe to be important for a continually learning system. It also includes advanced logging and plotting features, including native [Tensorboard](https://www.tensorflow.org/tensorboard) support.
* **`Models`**: In this module you'll find several model architectures and pre-trained models that can be used for your continual learning experiment \(similar to what has been done in [torchvision.models](https://pytorch.org/docs/stable/torchvision/index.html)\). Furthermore, we provide everything you need to implement architectural strategies, task-aware models, and dynamic model expansion.
* **`Logging`**: It includes advanced logging and plotting features, including native _stdout_, _file_ and [Tensorboard](https://www.tensorflow.org/tensorboard) support \(How cool it is to have a complete, interactive dashboard, tracking your experiment metrics in real-time with a single line of code?\)

{% code title="Avalanche Main Modules and Sub-Modules" %}
```text
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

You can run _this chapter_ and play with it on Google Colaboratory: [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ContinualAI/avalanche/blob/master/notebooks/from-zero-to-hero-tutorial/01_introduction.ipynb)
