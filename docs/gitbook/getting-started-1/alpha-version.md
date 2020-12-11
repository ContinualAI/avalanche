---
description: 'Currently Supported Benchmarks, Strategies & Metrics'
---

# Alpha Version

_Avalanche_ is a framework in constant development. Thanks to the support of the [ContinualAI](https://www.continualai.org/) community and its active members we plan to **extend its features** and **improve its usability**, based on the always new demands of our research community!  
  
A the moment, _Avalanche_ is in **Alpha \(v0.0.1\)**, but we already support a number of _Benchmarks_, _Strategies_ and Metrics, that makes it, we believe, **the best tool out there for your continual learning research!** 💪

{% hint style="info" %}
Check out below what we support in details, and please let us know if you think [we are missing out something important](../questions-and-issues/request-a-feature.md)! We deeply value [your feedback](../questions-and-issues/give-feedback.md)!
{% endhint %}

## 📚 Supported Benchmarks

In the Table below, we list all the major benchmarks used in _Continual Learning_ \(along with some references\) and indicating if we **support** them in _Avalanche_ or not. 

_"Dataset Support"_ is checked if an easy-to-use PyTorch version of the dataset is available, whereas _"Benchmark Support"_ is checked if the actual _continual learning benchmark_ \(which sequentialize the data\) is also provided.

| Name | Dataset Support | Benchmark Support | References |
| :--- | :--- | :--- | :--- |
| **CORe50** | ✔️ | ✔️ | [\[1\]](http://proceedings.mlr.press/v78/lomonaco17a.html) |
| **RotatedMNIST** | ✔️ | ✔️ |  |
| **PermutedMNIST** | ✔️ | ✔️ |  |
| **SplitMNIST** | ✔️ | ✔️ |  |
| **FashionMNIST** | ✔️ | ✔️ |  |
| **CIFAR-10** | ✔️ | ✔️ |  |
| **CIFAR-100** | ✔️ | ✔️ |  |
| **CIFAR-110** | ✔️ | ✔️ |  |
| **TinyImagenet** | ✔️ | ✔️ |  |
| **CUB200** | ✔️ | ✔️ |  |
| **SplitImagenet** | ✔️ | ✔️ |  |
| **CRIB** | ❌ | ❌ |  |
| **OpenLORIS** | ❌ | ❌ |  |
| **MiniImagenet** | ❌ | ❌ |  |
| **Stream-51** | ❌ | ❌ |  |
| **KMNIST** | ✔️ | ❌ | N.A. |
| **EMNIST** | ✔️ | ❌ | N.A. |
| **QMNIST** | ✔️ | ❌ | N.A. |
| **FakeData** | ✔️ | ❌ | N.A. |
| **CocoCaption** | ✔️ | ❌ | N.A. |
| **CocoDetection** | ✔️ | ❌ | N.A. |
| **LSUN** | ✔️ | ❌ | N.A. |
| **STL10** | ✔️ | ❌ | N.A. |
| **SVHN** | ✔️ | ❌ | N.A. |
| **PhotoTour** | ✔️ | ❌ | N.A. |
| **SBU** | ✔️ | ❌ | N.A. |
| **Flickr8k** | ✔️ | ❌ | N.A. |
| **Flickr30k** | ✔️ | ❌ | N.A. |
| **VOCDetection** | ✔️ | ❌ | N.A. |
| **VOCSegmentation** | ✔️ | ❌ | N.A. |
| **Cityscapes** | ✔️ | ❌ | N.A. |
| **SBDataset** | ✔️ | ❌ | N.A. |
| **USPS** | ✔️ | ❌ | N.A. |
| **Kinetics400** | ✔️ | ❌ | N.A. |
| **HMDB51** | ✔️ | ❌ | N.A. |
| **UCF101** | ✔️ | ❌ | N.A. |
| **CelebA** | ✔️ | ❌ | N.A. |

## 📈 Supported Strategies

In the Table below, we list all the _Continual Learning_ algorithms \(also known as _strategies_\) we currently support in _Avalanche_. 

_"Strategy Support"_ is checked if the algorithm is already available in _Avalanche_, whereas _"Plugin Support"_ is checked if the **corresponding plugin** of the strategy \(that can be used in conjunction with other strategies\) is is also provided.

| Name | Strategy Support | Plugin Support | References |
| :--- | :--- | :--- | :--- |
| **Naive \(a.k.a. "Finetuning"\)** | ✔️ | ❌ |  |
| **Naive Multi-Head** | ✔️ | ✔️ |  |
| **Joint Training \(a.k.a. "Multi-Task"\)** | ✔️ | ❌ |  |
| **Cumulative** | ✔️ | ❌ |  |
| **GDumb** | ✔️ | ✔️ |  |
| **Experience Replay \(a.k.a. "Rehearsal"\)** | ✔️ | ✔️ |  |
| **EWC** | ✔️ | ✔️ |  |
| **LWF** | ✔️ | ✔️ |  |
| **GEM** | ✔️ | ✔️ |  |
| **AGEM** | ✔️ | ✔️ |  |
| **CWR** | ✔️ | ✔️ |  |
| **SI** | ❌ | ❌ |  |
| **iCaRL** | ❌ | ❌ |  |
| **AR1** | ❌ | ❌ |  |

## 📊 Supported Metrics

In the Table below, we list all the _Continual Learning_ **Metrics** we currently support in _Avalanche_. All the metrics by default can be **collected** during runtime, **logged on stdout** or on **log file**.

With _"Tensorboard"_ is checked if the metrics can be also visualized in **Tensorboard** is already available in _Avalanche_, whereas _"Wandb"_ is checked if the metrics can be visualized in **Wandb**.

| Name | Support | Tensorboard | Wandb | References |
| :--- | :--- | :--- | :--- | :--- |
| **Accuracy** | ✔️ | ✔️ | ❌ | \*\*\*\* |
| **ACC** | ❌ | ❌ | ❌ | [\(Lopez-Paz, 2017\)](https://arxiv.org/pdf/1706.08840.pdf) |
| **BWT** | ❌ | ❌ | ❌ | [\(Lopez-Paz, 2017\)](https://arxiv.org/pdf/1706.08840.pdf) |
| **FWT** | ❌ | ❌ | ❌ | [\(Lopez-Paz, 2017\)](https://arxiv.org/pdf/1706.08840.pdf) |
| **Catastrophic Forgetting** | ✔️ | ✔️ | ❌ | \*\*\*\* |
| **Remembering** | ❌ | ❌ | ❌ |  |
| **A** | ❌ | ❌ | ❌ | [\(Díaz-Rodríguez, 2018\)](https://arxiv.org/pdf/1810.13166.pdf) |
| **MS** | ❌ | ❌ | ❌ | [\(Díaz-Rodríguez, 2018\)](https://arxiv.org/pdf/1810.13166.pdf) |
| **SSS** | ❌ | ❌ | ❌ | [\(Díaz-Rodríguez, 2018\)](https://arxiv.org/pdf/1810.13166.pdf) |
| **CE** | ❌ | ❌ | ❌ | [\(Díaz-Rodríguez, 2018\)](https://arxiv.org/pdf/1810.13166.pdf) |
| **Confusion Matrix** | ✔️ | ✔️ | ❌ | \*\*\*\* |
| **MAC** | ✔️ | ✔️ | ❌ | \*\*\*\* |
| **CPU Usage** | ✔️ | ✔️ | ❌ | \*\*\*\* |
| **Disk Usage** | ✔️ | ✔️ | ❌ | \*\*\*\* |
| **GPU Usage** | ✔️ | ✔️ | ❌ | \*\*\*\* |
| **RAM Usage** | ✔️ | ✔️ | ❌ | \*\*\*\* |
| **Running Time** | ✔️ | ✔️ | ❌ |  |
| **CLScore** | ❌ | ❌ | ❌ | [\(Díaz-Rodríguez, 2018\)](https://arxiv.org/pdf/1810.13166.pdf) |
| **CLStability** | ❌ | ❌ | ❌ | [\(Díaz-Rodríguez, 2018\)](https://arxiv.org/pdf/1810.13166.pdf) |

