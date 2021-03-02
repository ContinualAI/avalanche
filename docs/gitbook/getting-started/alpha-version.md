---
description: 'Supported Benchmarks, Strategies & Metrics'
---

# Current Release

_Avalanche_ is a framework in constant development. Thanks to the support of the [ContinualAI](https://www.continualai.org/) community and its active members we plan to **extend its features** and **improve its usability** based on the demands of our research community!  
  
A the moment, _Avalanche_ is in **Alpha \(v0.0.1\)**, but we already support a number of _Benchmarks_, _Strategies_ and _Metrics_, that makes it, we believe, **the best tool out there for your continual learning research!** 💪

{% hint style="info" %}
Check out below what we support in details, and please let us know if you think [we are missing out something important](../questions-and-issues/request-a-feature.md)! We deeply value [your feedback](../questions-and-issues/give-feedback.md)!
{% endhint %}

This doc is out-of-date. Check the [full tutorial](../from-zero-to-hero-tutorial/introduction.md) for more details.

## 🖼️ Supported Datasets

In the Table below, we list all the Pytorch datasets used in _Continual Learning_ \(along with some references\) and indicating if we **support** them in _Avalanche_ or not. Some of them were already available in [_Torchvision_](https://pytorch.org/docs/stable/torchvision/index.html), while other have been integrated by us.

| Name | Dataset Support | From Torch Vision | Automatic Download | References |
| :--- | :--- | :--- | :--- | :--- |
| **CORe50** | ✔️ | ✔️ | ✔️ | [\[1\]](http://proceedings.mlr.press/v78/lomonaco17a.html) |
| **MNIST** | ✔️ | ✔️ | ✔️ | n.a. |
| **CIFAR-10** | ✔️ | ✔️ | ✔️ | n.a. |
| **CIFAR-100** | ✔️ | ✔️ | ✔️ | n.a. |
| **FashionMNIST** | ✔️ | ✔️ | ✔️ | n.a. |
| **TinyImagenet** | ✔️ | ✔️ | ✔️ | n.a. |
| **MiniImagenet** | ✔️ | ❌ | ❌ | n.a. |
| **Imagenet** | ✔️ | ✔️ | ❌ | n.a. |
| **CUB200** | ✔️ | ❌ | ❌ | n.a. |
| **CRIB** | ❌ | ❌ | ❌ | n.a. |
| **OpenLORIS** | ✔️ | ❌ | ❌ | n.a. |
| **Stream-51** | ✔️ | ❌ | ✔️ | n.a. |
| **KMNIST** | ✔️ | ✔️ | ✔️ | n.a. |
| **EMNIST** | ✔️ | ✔️ | ✔️ | n.a. |
| **QMNIST** | ✔️ | ✔️ | ✔️ | n.a. |
| **FakeData** | ✔️ | ✔️ | ✔️ | n.a. |
| **CocoCaption** | ✔️ | ✔️ | ❌ | n.a. |
| **CocoDetection** | ✔️ | ❌ | ❌ | n.a. |
| **LSUN** | ✔️ | ❌ | ❌ | n.a. |
| **STL10** | ✔️ | ❌ | ✔️ | n.a. |
| **SVHN** | ✔️ | ❌ | ✔️ | n.a. |
| **PhotoTour** | ✔️ | ❌ | ✔️ | n.a. |
| **SBU** | ✔️ | ✔️ | ✔️ | n.a. |
| **Flickr8k** | ✔️ | ✔️ | ❌ | n.a. |
| **Flickr30k** | ✔️ | ✔️ | ❌ | n.a. |
| **VOCDetection** | ✔️ | ✔️ | ✔️ | n.a. |
| **VOCSegmentation** | ✔️ | ✔️ | ✔️ | n.a. |
| **Cityscapes** | ✔️ | ✔️ | ❌ | n.a. |
| **SBDataset** | ✔️ | ✔️ | ✔️ | n.a. |
| **USPS** | ✔️ | ✔️ | ✔️ | n.a. |
| **Kinetics400** | ✔️ | ✔️ | ❌ | n.a. |
| **HMDB51** | ✔️ | ✔️ | ❌ | n.a. |
| **UCF101** | ✔️ | ✔️ | ❌ | n.a. |
| **CelebA** | ✔️ | ✔️ | ✔️ | n.a. |
| **Caltech101** | ❌ | ❌ | ❌ | n.a. |
| **Caltech256** | ❌ | ❌ | ❌ | n.a. |

## 📚 Supported Benchmarks

In the Table below, we list all the major benchmarks used in _Continual Learning_ \(along with some references\) and indicating if we **support** them in _Avalanche_ or not. 

_"Dataset Support"_ is checked if an easy-to-use PyTorch version of the dataset is available, whereas _"Benchmark Support"_ is checked if the actual _continual learning benchmark_ \(which sequentialize the data\) is also provided.

| Name | Benchmark Support | References |
| :--- | :--- | :--- |
| **CORe50** | ✔️ | [\[1\]](http://proceedings.mlr.press/v78/lomonaco17a.html) |
| **RotatedMNIST** | ✔️ | n.a. |
| **PermutedMNIST** | ✔️ | n.a. |
| **SplitMNIST** | ✔️ | n.a. |
| **FashionMNIST** | ✔️ | n.a. |
| **CIFAR-10** | ✔️ | n.a. |
| **CIFAR-100** | ✔️ | n.a. |
| **CIFAR-110** | ✔️ | n.a. |
| **TinyImagenet** | ✔️ | n.a. |
| **CUB200** | ✔️ | n.a. |
| **SplitImagenet** | ✔️ | n.a. |
| **CRIB** | ❌ | n.a. |
| **OpenLORIS** | ✔️ | n.a. |
| **MiniImagenet** | ❌ | n.a. |
| **Stream-51** | ✔️ | n.a. |

## 📈 Supported Strategies

In the Table below, we list all the _Continual Learning_ algorithms \(also known as _strategies_\) we currently support in _Avalanche_. 

_"Strategy Support"_ is checked if the algorithm is already available in _Avalanche_, whereas _"Plugin Support"_ is checked if the **corresponding plugin** of the strategy \(that can be used in conjunction with other strategies\) is is also provided.

| Name | Strategy Support | Plugin Support | References |
| :--- | :--- | :--- | :--- |
| **Naive \(a.k.a. "Finetuning"\)** | ✔️ | ❌ | n.a. |
| **Naive Multi-Head** | ✔️ | ✔️ | n.a. |
| **Joint Training \(a.k.a. "Multi-Task"\)** | ✔️ | ❌ | n.a. |
| **Cumulative** | ✔️ | ❌ | n.a. |
| **GDumb** | ✔️ | ✔️ | n.a. |
| **Experience Replay \(a.k.a. "Rehearsal"\)** | ✔️ | ✔️ | n.a. |
| **EWC** | ✔️ | ✔️ | n.a. |
| **LWF** | ✔️ | ✔️ | n.a. |
| **GEM** | ✔️ | ✔️ | n.a. |
| **AGEM** | ✔️ | ✔️ | n.a. |
| **CWR** | ✔️ | ✔️ | n.a. |
| **SI** | ✔️ | ✔️ | n.a. |
| **iCaRL** | ❌ | ❌ | n.a. |
| **AR1** | ✔️ | ❌ | n.a. |

## 📊 Supported Metrics

In the Table below, we list all the _Continual Learning_ **Metrics** we currently support in _Avalanche_. All the metrics by default can be **collected** during runtime, **logged on stdout** or on **log file**.

With _"Tensorboard"_ is checked if the metrics can be also visualized in **Tensorboard** is already available in _Avalanche_, whereas _"Wandb"_ is checked if the metrics can be visualized in **Wandb**.

| Name | Support | Tensorboard | Wandb | References |
| :--- | :--- | :--- | :--- | :--- |
| **Accuracy** | ✔️ | ✔️ | ❌ | n.a. |
| **Loss** | ✔️ | ✔️ | ❌ | n.a. |
| **ACC** | ❌ | ❌ | ❌ | [\(Lopez-Paz, 2017\)](https://arxiv.org/pdf/1706.08840.pdf) |
| **BWT** | ❌ | ❌ | ❌ | [\(Lopez-Paz, 2017\)](https://arxiv.org/pdf/1706.08840.pdf) |
| **FWT** | ❌ | ❌ | ❌ | [\(Lopez-Paz, 2017\)](https://arxiv.org/pdf/1706.08840.pdf) |
| **Catastrophic Forgetting** | ✔️ | ✔️ | ❌ | n.a. |
| **Remembering** | ❌ | ❌ | ❌ | n.a. |
| **A** | ❌ | ❌ | ❌ | [\(Díaz-Rodríguez, 2018\)](https://arxiv.org/pdf/1810.13166.pdf) |
| **MS** | ❌ | ❌ | ❌ | [\(Díaz-Rodríguez, 2018\)](https://arxiv.org/pdf/1810.13166.pdf) |
| **SSS** | ❌ | ❌ | ❌ | [\(Díaz-Rodríguez, 2018\)](https://arxiv.org/pdf/1810.13166.pdf) |
| **CE** | ❌ | ❌ | ❌ | [\(Díaz-Rodríguez, 2018\)](https://arxiv.org/pdf/1810.13166.pdf) |
| **Confusion Matrix** | ✔️ | ✔️ | ❌ | n.a. |
| **MAC** | ✔️ | ✔️ | ❌ | n.a. |
| **CPU Usage** | ✔️ | ✔️ | ❌ | n.a. |
| **Disk Usage** | ✔️ | ✔️ | ❌ | n.a. |
| **GPU Usage** | ✔️ | ✔️ | ❌ | n.a. |
| **RAM Usage** | ✔️ | ✔️ | ❌ | n.a. |
| **Running Time** | ✔️ | ✔️ | ❌ | n.a. |
| **CLScore** | ❌ | ❌ | ❌ | [\(Díaz-Rodríguez, 2018\)](https://arxiv.org/pdf/1810.13166.pdf) |
| **CLStability** | ❌ | ❌ | ❌ | [\(Díaz-Rodríguez, 2018\)](https://arxiv.org/pdf/1810.13166.pdf) |

