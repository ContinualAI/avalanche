---
description: A Short Guide for Researchers on the Run
---

# Learn Avalanche in 5 Minutes

_Avalanche_ is mostly about making the life of a continual learning researcher easier. 

> #### What are the **three pillars** of any respectful continual learning research project?

* **Benchmarks**: Machine learning researchers, need datasets and benchmarks with efficient and easy-to-use data handling utils to design and prototype new algorithms. Quantitative results on ever-changing benchmarks has been one of the driving forces of _Deep Learning_.
* **Training**: ****Efficient implementation and training of continual learning algorithms; comparisons with other baselines and state-of-the-art methods become fundamental to asses the quality of an original algorithmic proposal.
* **Evaluation**: ****_Training_ utils and _Benchmarks_ are not enough alone to push continual learning research forward. Comprehensive and sound _evaluation protocols_ and _metrics_ need to be employed as well.

> #### _With Avalanche, you can find all these three fundamental pieces together and more, in a single and coherent codabase._

Indeed, these are the _three main modules_ offered in the _Avalanche_ package.

## 📦 How to Install Avalanche

_Avalanche_ can be installed as simply as:

```text
pip install git+https://vlomonaco:****@github.com/vlomonaco/avalanche.git
```

You can check if the installation was successful importing the package in python with:

```text
import avalanche
avalanche.__version__
```

## 📚 Benchamarks

The benchamark module offers three main features: 

1. **Datasets**: a comprehensive list of Pytorch Datasets ready to use \(It includes all the _Torchvision_ Datasets and more!\).
2. **Classic Benchmarks**: a set of classic Continual Learning Benchmarks ready to be used.
3. **Generators**: a set of functions you can use to generate your own benchmark starting from any Pytorch Dataset!

### Datasets

Datasets can be imported in _Avalanche_ as simply as:

```python
# the ones already supported in TorchVision
from avalanche.benchmarks.datasets import MNIST, FashionMNIST, KMNIST, EMNIST, \
QMNIST, FakeData, CocoCaptions, CocoDetection, LSUN, ImageNet, CIFAR10, \
CIFAR100, STL10, SVHN, PhotoTour, SBU, Flickr8k, Flickr30k, VOCDetection, \
VOCSegmentation, Cityscapes, SBDataset, USPS, Kinetics400, HMDB51, UCF101, \
CelebA 

# ...and additiona ones!
from avalanche.benchmarks.datasets import CORe50, TinyImagenet, CUB200

# ... or generic constructors
from avalanche.benchmarks.datasets import ImageFolder, DatasetFolder, FilelistDataset
```

Of course, you can use them as you would use any _Pythorch Dataset_. 

### **Classic Benchmarks**

_Continual Learning_ is all concerned with learning over a stream of data. _Avalache_ maintains a set of commonly used benchmarks build on top of one of multiple datasets, that simulate that stream.

```python
from avalanche.benchmarks.classic import CORe50, SplitTinyImageNet, \
SplitCIFAR10, SplitCIFAR100, SplitCIFAR110, SplitMNIST, RotatedMNIST, PermutedMNIST

# creating PermutedMNIST with
clscenario = PermutedMNIST(
    incremental_steps=2,
    seed=1234,
)
```

### Benchmarks Generators

What if we want to create a new benchmark that is not present in the "_Classic_" ones? Well, in that case _Avalanche_ offer a number of utilities that you can use to create your own benchmark with maximum flexibilty!

The _specific_ scenario generators are useful when starting from one or multiple pytorch datasets you want to create a "**New Instances**" or "**New Classes**" benchmark:

```python
from avalanche.benchmarks.generators import NIScenario, NCScenario

ni_scenario = NIScenario(
    mnist_train, mnist_test, n_batches=10, shuffle=True, seed=1234,
    balance_batches=True
)

nc_scenario = NCScenario(
    mnist_train, mnist_test, n_steps=10, shuffle=True, seed=1234,
    multi_task=False
)
```

Finally, if you cannot create your ideal benchmark since it does not fit well in the aforementioned SIT-NI, SIT-NC or MT-NC scenarios, you can always use our **generic generators**:

* **FilelistScenario**
* **DatasetScenario**
* **TensorScenario**

```python
from avalanche.benchmarks.generators import FilelistScenario, DatasetScenario, \
                                            TensorScenario
```

## 🤝 Run it on Google Colab

You can run _this chapter_ and play with it on Google Colaboratory:

{% hint style="danger" %}
TODO: add link here.
{% endhint %}



