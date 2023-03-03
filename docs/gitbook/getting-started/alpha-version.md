---
description: 'Avalnche Features: Benchmarks, Strategies & Metrics'
---

# Current Release

_Avalanche_ is a framework in constant development. Thanks to the support of the [ContinualAI](https://www.continualai.org) community and its active members we plan to **extend its features** and **improve its usability** based on the demands of our research community!\
\
A the moment, _Avalanche_ is in **Beta (v0.3.1).** We support a large number of _Benchmarks_, _Strategies_ and _Metrics_, that makes it, we believe, **the best tool out there for your continual learning research!** 💪

{% hint style="success" %}
You can find the **full list of available features** on the [API documentation](https://avalanche-api.continualai.org).
{% endhint %}

{% hint style="warning" %}
Do you think we are missing some important features? Please [let us know](../questions-and-issues/request-a-feature.md)! We deeply value [your feedback](../questions-and-issues/give-feedback.md)!
{% endhint %}

## Benchmarks and Datasets

You find a complete list of the features on the [benchmarks API documentation](https://avalanche-api.continualai.org/en/latest/benchmarks.html).

### 🖼️ Datasets

Avalanche supports all the most popular computer vision datasets used in _Continual Learning_. Some of them are available in [_Torchvision_](https://pytorch.org/docs/stable/torchvision/index.html), while other have been integrated by us. Most datasets can be automatically downloaded by Avalanche.

* **Toy datasets**: MNIST, Fashion MNIST, KMNIST, EMNIST, QMNIST.
* **CIFAR:** CIFAR10, CIFAR100.
* **ImageNet**: TinyImagenet, MiniImagenet, Imagenet.
* **Others**: EndlessCLDataset, CUB200, OpenLORIS, Stream-51, INATURALIST2018, Omniglot, CLEARImage, ...

### 📚 Benchmarks

All the major continual learning benchmarks are available and ready to use. Benchmarks split the datasets and create the train and test streams:

* **MNIST**: SplitMNIST, RotatedMNIST, PermutedMNIST, SplitFashionMNIST.
* **CIFAR10**: SplitCIFAR10, SplitCIFAR100, SplitCIFAR110.
* **CORe50**: all the CORe50 scenarios are supported.
* **Others**: SplitCUB200, CLStream51, CLEAR.

## 📈 Continual Learning Strategies

Avalanche provides _Continual Learning_ algorithms (_strategies_). We are continuously expanding the library with new algorithms and making sure they can reproduce seminal papers results in the sibling project [CL-Baselines](https://github.com/ContinualAI/continual-learning-baselines).

* **Baselines**: Naive, JointTraining, Cumulative.
* **Rehearsal**: Replay with reservoir sampling and balanced buffers, GSS greedy, CoPE, Generative Replay.
* **Regularization**: EWC, LwF, GEM, AGEM, CWR\*, Synaptic Intelligence, MAS.
* **Architectural**: Progressive Neural Networks, multi-head, incremental classifier.
* **Others**: GDumb, iCaRL, AR1, Streaming LDA, LFL.

## Models

Avalanche uses and extends pytorch `nn.Module` to define continual learning models:

* support for `nn.Module`s and `torchvision` models.
* Dynamic output heads for class-incremental scenarios and multi heads for task-incremental scenarios.
* support for architectural strategies and dynamically expanding models such as progressive neural networks.

## 📊 Metrics and Evaluation

Avalanche provides continuous evaluation of CL strategies with a large set of **Metrics**. They are collected and logged automatically by the strategy during the training and evaluation loops.

* **Standard Performance Metrics**: accuracy, loss, confusion (averaged over streams or experiences).
* **CL-Metrics**: backward/forward transfer, forgetting.
* **Computational Resources**: CPU and RAM usage, MAC, execution times.

and [many more](https://avalanche-api.continualai.org/en/latest/evaluation.html).
