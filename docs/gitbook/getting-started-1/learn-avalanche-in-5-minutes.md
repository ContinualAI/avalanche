---
description: A Short Guide for Researchers on the Run
---

# Learn Avalanche in 5 Minutes

{% hint style="danger" %}
This doc is out-of-date. Check the [full tutorial](../from-zero-to-hero-tutorial/introduction.md) for more details.
{% endhint %}

This doc is out-of-date. Check the [corresponding notebook](https://colab.research.google.com/drive/1vjLrdYEHWGH9Rz0cQZzb63BO2yCAsUIT#scrollTo=ADOrYmNXak23) for the latest version.

_Avalanche_ is mostly about making the life of a continual learning researcher easier.

> #### What are the **three pillars** of any respectful continual learning research project?

* **Benchmarks**: Machine learning researchers need multiple benchmarks with efficient data handling utils to design and prototype new algorithms. Quantitative results on ever-changing benchmarks has been one of the driving forces of _Deep Learning_.
* **Training**: Efficient implementation and training of continual learning algorithms; comparisons with other baselines and state-of-the-art methods become fundamental to asses the quality of an original algorithmic proposal.
* **Evaluation**: _Training_ utils and _Benchmarks_ are not enough alone to push continual learning research forward. Comprehensive and sound _evaluation protocols_ and _metrics_ need to be employed as well.

> #### _With Avalanche, you can find all these three fundamental pieces together and much more, in a single and coherent codabase._

Let's take a quick tour on how you can use Avalanche for your research projects with a **5-minutes guide**, for _researchers on the run_!

{% hint style="info" %}
In this short guide we assume you have already installed _Avalanche_. If you haven't yet, check out how you can do it following our [How to Install](1.-how-to-install.md) guide.
{% endhint %}

## 🏛️ General Architecture

_Avalanche_ is organized in **four main modules**:

* **`Benchmarks`**: This module maintains a uniform API for data handling: mostly generating a stream of data from one or more datasets. It contains all the major CL benchmarks \(similar to what has been done for [torchvision](https://pytorch.org/docs/stable/torchvision/index.html)\).
* **`Training`**: This module provides all the necessary utilities concerning model training. This includes simple and efficient ways of implement new _continual learning_ strategies as well as a set pre-implemented CL baselines and state-of-the-art algorithms you will be able to use for comparison!
* **`Evaluation`**: This modules provides all the utilities and metrics that can help evaluate a CL algorithm with respect to all the factors we believe to be important for a continually learning system. It also includes advanced logging and plotting features, including native [Tensorboard](https://www.tensorflow.org/tensorboard) support.
* **`Extras`**: In the extras module you'll be able to find several useful utilities and building blocks that will help you create your continual learning experiments with ease. This includes configuration files for quick reproducibility and model building functions for example.

In the graphic below, you can see how _Avalanche_ sub-modules are available and organized as well:

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

We will learn more about each of them during this tutorial series, but keep in mind that the [Avalanche API documentation](https://vlomonaco.github.io/avalanche) is your friend as well!

All right, let's start with the _benchmarks_ module right away 👇

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

# ...and additional ones!
from avalanche.benchmarks.datasets import CORe50, TinyImagenet, CUB200

# ... or generic constructors
from avalanche.benchmarks.datasets import ImageFolder, DatasetFolder, FilelistDataset
```

Of course, you can use them as you would use any _Pythorch Dataset_.

### Benchmarks Basics

The _Avalanche_ benchmarks \(instances of the _Scenario_ class\), contains several attributes that characterize the bechmark. However, the most important ones are the `train` and `test streams`.

In _Avalanche_ we often suppose to have access to these **two parallel stream of data** \(even though some benchmarks may not provide such feature, but contain just a unique test set\).

Each of these `streams` are _iterable_, _indexable_ and _sliceable_ objects that are composed of **steps**. Steps are batch of data \(or "_tasks_"\) that can be provided with or without a specific task label.

### **Classic Benchmarks**

_Avalanche_ maintains a set of commonly used benchmarks build on top of one of multiple datasets, that simulate that stream.

```python
from avalanche.benchmarks.classic import CORe50, SplitTinyImageNet, \
SplitCIFAR10, SplitCIFAR100, SplitCIFAR110, SplitMNIST, RotatedMNIST, PermutedMNIST, \
SplitCUB200

# creating the benchmark (scenario object)
perm_mnist = PermutedMNIST(
    n_steps=3,
    seed=1234,
)

# recovering the train and test streams
train_stream = perm_mnist.train_stream
test_stream = perm_mnist.test_stream

# iterating over the train stream
for step in train_stream:

    print("Start of task ", step.task_label)
    print('Classes in this task:', step.classes_in_this_step)

    # The current Pytorch training set can be easily recovered through the step
    current_training_set = step.dataset
    # ...as well as the task_label
    print('Task {}'.format(step.task_label))
    print('This task contains', len(current_training_set), 'training examples')

    # we can recover the corresponding test step in the test stream
    current_test_set = test_stream[step.current_step].dataset
    print('This task contains', len(current_test_set), 'test examples')
```

### Benchmarks Generators

What if we want to create a new benchmark that is not present in the "_Classic_" ones? Well, in that case _Avalanche_ offer a number of utilites that you can use to create your own benchmark with maximum flexibilty: the **benchmarks generators**!

 The _specific_ scenario generators are useful when starting from one or multiple pytorch datasets you want to create a "**New Instances**" or "**New Classes**" benchmark: i.e. it supports the easy and flexible creation of a _Domain-Incremental_, _Class-Incrementa_l or _Task-Incremental_ scenarios among others.

```python
from avalanche.benchmarks.generators import nc_scenario, ni_scenario

scenario = ni_scenario(
    mnist_train, mnist_test, n_steps=10, shuffle=True, seed=1234,
    balance_steps=True
)
scenario = nc_scenario(
    mnist_train, mnist_test, n_steps=10, shuffle=True, seed=1234,
    task_labels=False
)
```

Finally, if you cannot create your ideal benchmark since it does not fit well in the aforementioned _Domain-Incremental_, _Class-Incrementa_l or _Task-Incremental_  scenarios, you can always use our **generic generators**:

* **filelist\_scenario**
* **dataset\_scenario**
* **tensor\_scenario**

```python
from avalanche.benchmarks.generators import filelist_scenario, dataset_scenario, \
                                            tensor_scenario
```

You can read more about how to use them the full _Benchmarks_ module tutorial!

{% page-ref page="../from-zero-to-hero-tutorial/2.-benchmarks.md" %}

## 💪Training

The `training` module in _Avalanche_ is build on modularity and its main goals are two:

1. provide a set of standard **continual learning baselines** that can be easily run for comparison;
2. provide the necessary utilities to **create and run your own strategy** as efficiently and easy as possible with building blocks we already prepared for you.

### Strategies

If you want to compare your strategy with other classic continual learning algorithms or baselines, in _Avalanche_ this is as simply as instantiate an object:

```python
from avalanche.extras.models import SimpleMLP
from avalanche.training.strategies import Naive, CWRStar, Replay, GDumb, 
Cumulative, LwF

model = SimpleMLP(num_classes=10)
cl_strategy = Naive(
    model, 'classifier', SGD(model.parameters(), lr=0.001, momentum=0.9),
    CrossEntropyLoss(), train_mb_size=100, train_epochs=4, test_mb_size=100
)
```

### Create your own Strategy

The simplest way to build your own strategy is to create a python class that implements the main `train` and `test` methods.

Let's define our Continual Learning algorithm "_MyStrategy_" as a simple python class:

```python
class MyStrategy():
    """My Basic Strategy"""

    def __init__(self, model, optimizer, criterion):
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
    
    def train(self, step):
        # here you can implement your own training loop for each step (i.e. 
        # batch or task).

        train_dataset = step.dataset
        t = step.task_label
        train_data_loader = DataLoader(
            train_dataset, num_workers=4, batch_size=128
        )

        for epoch in range(1):
            for mb in train_data_loader:
                # you magin here...
                pass

    def test(self, step):
        # here you can implement your own test loop for each step (i.e. 
        # batch or task).

        test_dataset = step.dataset
        t = step.task_label
        test_data_loader = DataLoader(
            test_dataset, num_workers=4, batch_size=128
        )

        # test here
```

Then, we can use our strategy as we would do for the pre-implemented ones:

```python
# MODEL CREATION
model = SimpleMLP(num_classes=scenario.n_classes)

# CREATE THE STRATEGY INSTANCE (NAIVE)
cl_strategy = MyStrategy(
    model, SGD(model.parameters(), lr=0.001, momentum=0.9),
    CrossEntropyLoss())

# TRAINING LOOP
print('Starting experiment...')
 
for step in scenario.train_stream:
    print("Start of step ", step.current_step)

    cl_strategy.train(step)
    print('Training completed')

    print('Computing accuracy on the whole test set')
    cl_strategy.test(scenario.test_stream[step.current_step])
```

While this is the easiest possible way to add your own stratey to _Avalanche_ we support more sophisticated modalities that let you write **more neat and reusable** **code**, inheriting functionality from a parent classes and using **pre-implemented plugins** shared with other strategies.

Check out more details about what Avalanche can offer in this module following the "_Training_" chapter of the **"**_**From Zero to Hero**_**"** tutorial!

{% page-ref page="../from-zero-to-hero-tutorial/3.-training.md" %}

## 📈 Evaluation

The `evaluation` module is quite straightforward at the moment and offers 3 main submodules:

* **Metrics**: a set of classes \(one for metric\) which implement the main continual learning matrics computation like _accuracy_, _forgetting_, _memory usage_, _running times_, etc.
* **TensorboardLogging**: offers a main class to handles Tensorboard logging configuration with nice plots updated on-the-fly to control and "_babysit_" your experiment easily.
* **Evaluation Protocols**: this module provides a single point of entry to the evaluation methodology configuration and in charge of hangling the metrics computation, tensorboard or avanced console logging.

### Metrics

In _Avalanche_ we offer at the moment a number of pre-implemented metrics you can use for your own experiments. We made sure to include all the major accuracy-based matrics but also the ones related to computation and memory.

The metrics already available \(soon to be expanded\) are:

* **Accuracy** \(`ACC`\): Accuracy over time \(Total average or per task\).
* **Catastrophic Forgetting** \(`CF`\): Forgetting as defined in \(Lopez-paz 2017\).
* **RAM Usage** \(`RAMU`\): RAM usage by the process over time.
* **Confusion Matrix** \(`CM`\): Confusion matrix over time.
* **CPU Usage** \(`CPUUsage`\): CPU usage by the process over time.
* **GPU Usage** \(`GPUUsage`\): GPU usage by the process over time.
* **Disk Usage** \(`DiskUsage`\): Disk usage by the process over time.
* **Time Usage** \(`TimeUsage`\): Running time of the python process.

```python
from avalanche.evaluation.metrics import ACC, CF, RAMU, CM, CPUUsage, GPUUsage,\
 DiskUsage, TimeUsage
```

While each metric can be directly managed within and _Evaluation Protocol_ \(see next section\), we can use each metric directly, being them simply python classes. For example the accuracy metric works as follows:

```python
real_y = np.asarray([1, 2])
predicted_y = np.asarray([1, 0])
acc_metric = ACC()
acc, acc_x_class = acc_metric.compute([real_y], [predicted_y])

print("Average Accuracy:", acc)
print("Accuracy per class:", acc_x_class)
```

### Tensorboard

Tensorboard has consolidated its position as the go-to **visualization toolkit** for deep learning experiments for both pytorch and tensorflow. In _Avalanche_ we decided the different the different metrics \(and their eventual change over time\) using the standard Pytorch version.

At the moment we implemented just the `TensorboardLogging` object that can be used to specify Tensorboard behavious with eventual customizations.

```python
from avalanche.evaluation.tensorboard import TensorboardLogging

# a simple example of tensorboard instantiation
tb_logging = TensorboardLogging(tb_logdir=".")
```

### Evaluation Protocol

The **Evaluation Protocol**, is the object in charge of configuring and controlling the evaluation procedure. This object can be passed to a Strategy that will automatically call its main functionalities when the **training and testing flows** are activated.

```python
from avalanche.evaluation.metrics import ACC, CF, RAMU, CM
from avalanche.extras.models import SimpleMLP
from avalanche.training.strategies import Naive
from avalanche.evaluation import EvalProtocol

# load the model with PyTorch for example
model = SimpleMLP()

# Eval Protocol
evalp = EvalProtocol(
    metrics=[ACC(), CF(), RAMU(), CM()], tb_logdir='.'
)

# adding the CL strategy
clmodel = Naive(model, eval_protocol=evalp)
```

For more details about the evaluation module, check out the extended guide in the "_Evaluation_" chapter of the **"**_**From Zero to Hero**_**"** Avalanche tutorial!

{% page-ref page="../from-zero-to-hero-tutorial/4.-evaluation.md" %}

## Putting all Together

You've learned how to install _Avalanche,_ how to create benchmarks that can suit your needs, how you can create your own continual learning algorithm and how you can evaluate its performance.

Here we show how you can use all these modules together to **design your experiments** as quantitative supporting evidence for your research project or paper.

```python
import torch
from torch.nn import CrossEntropyLoss
from torch.optim import SGD

from avalanche.benchmarks.classic import PermutedMNIST
from avalanche.evaluation import EvalProtocol
from avalanche.evaluation.metrics import ACC
from avalanche.extras.models import SimpleMLP
from avalanche.training.strategies import Naive

# Config
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# model
model = SimpleMLP(num_classes=10)

# CL Benchmark Creation
perm_mnist = PermutedMNIST(n_steps=3)
train_stream = perm_mnist.train_stream
test_stream = perm_mnist.test_stream

# Prepare for training & testing
optimizer = SGD(model.parameters(), lr=0.001, momentum=0.9)
criterion = CrossEntropyLoss()
evaluation_protocol = EvalProtocol(metrics=[ACC(num_class=10)])

# Continual learning strategy
cl_strategy = Naive(
    model, optimizer, criterion, train_mb_size=32, train_epochs=2, 
    test_mb_size=32, evaluation_protocol=evaluation_protocol, device=device)

# train and test loop
results = []
for train_task in train_stream:
    cl_strategy.train(train_task, num_workers=4)
    results.append(cl_strategy.test(test_stream))
```

## 🤝 Run it on Google Colab

You can run _this chapter_ and play with it on Google Colaboratory:

{% hint style="danger" %}
TODO: add link here.
{% endhint %}

