---
description: Understand the Avalanche Package Structure
---

# Introduction

Welcome to the "_Modules_" tutorial of the "_From Zero to Hero_" series. We will start our journey by taking a quick look at the _Avalanche_ main modules to understand its **general architecture**.

As hinted in the introduction _Avalanche_ is organized in **four main modules**:

* **`Benchmarks`**: This module maintains a uniform API for data handling: mostly generating a stream of data from one or more datasets. It contains all the major CL benchmarks \(similar to what has been done for [torchvision](https://pytorch.org/docs/stable/torchvision/index.html)\).
* **`Training`**: This module provides all the necessary utilities concerning model training. This includes simple and efficient ways of implement new _continual learning_ strategies as well as a set pre-implemented CL baselines and state-of-the-art algorithms you will be able to use for comparison!
* **`Evaluation`**: This modules provides all the utilities and metrics that can help evaluate a CL algorithm with respect to all the factors we believe to be important for a continually learning system. It also includes advanced logging and plotting features, including native [Tensorboard](https://www.tensorflow.org/tensorboard) support.
* **`Extras`**: In the extras module you'll be able to find several useful utilities and building blocks that will help you create your continual learning experiments with ease. This includes configuration files for quick reproducibility and model building functions for example.

{% code title="Avalanche Main Modules and Sub-Modules" %}
```text
Avalanche
├── Benchmarks
│   ├── Classic
│   ├── Datasets
│   ├── Generators
│   ├── Scenarios
│   └── Utils
├── Evaluation
│   ├── Metrics
│   ├── Tensorboard
|   └── Utils
├── Training
│   ├── Strategies
│   ├── Plugins
|   └── Utils
└── Extras
    ├── Configs
    └── Models
```
{% endcode %}

In this series of tutorials, you'll get the chance to learn in-depth all the features offered by each module and sub-module of _Avalanche_, how to put them together and how to master _Avalanche_, for a **stress-free continual learning prototyping experience**!



{% hint style="info" %}
In the following tutorials we will assume you have already installed _Avalanche on your computer_ or server. If you haven't yet, check out how you can do it following our [How to Install](../getting-started-1/1.-how-to-install.md) guide.
{% endhint %}

## 🤝 Run it on Google Colab

You can run _this chapter_ and play with it on Google Colaboratory:

{% hint style="danger" %}
TODO: add link here.
{% endhint %}

