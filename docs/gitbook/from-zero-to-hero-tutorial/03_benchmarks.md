---
description: Create your Continual Learning Benchmark and Start Prototyping
---

# Benchmarks

Welcome to the "_benchmarks_" tutorial of the "_From Zero to Hero_" series. In this part we will present the functionalities offered by the `Benchmarks` module.


```python
%pip install avalanche-lib==0.6
```

## 🎯 Nomenclature

Avalanche Benchmarks provide the data that you will for training and evaluating your model. Benchmarks have the following structure:
- A `Benchmark` is a collection of streams. Most benchmarks have at least a `train_stream` and a `test_stream`;
- A `Stream` is a sequence of `Experience`s. It can be a list or a generator;
- An `Experience` contains all the information available at a certain time `t`;
- `AvalancheDataset` is a wrapper of PyTorch datasets. It provides functionalities used by the training module, such as concatenation, subsampling, and management of augmentations.

### 📚 The Benchmarks Module

The `bechmarks` module offers:

* **Datasets**: Pytorch datasets are wrapped in an `AvalancheDataset` to provide additional functionality.
* **Classic Benchmarks**: classic benchmarks used in CL litterature ready to be used with great flexibility.
* **Benchmarks Generators**: a set of functions you can use to create your own benchmark and streams starting from any kind of data and scenario, such as class-incremental or task-incremental streams.

But let's see how we can use this module in practice!

## 🖼️ Datasets

Let's start with the `Datasets`. When using _Avalanche_, your code will manipulate `AvalancheDataset`s. It is a wrapper compatible with pytorch and torchvision map-style datasets.


```python
import torch
import torchvision
from avalanche.benchmarks.datasets import MNIST
from avalanche.benchmarks.datasets.dataset_utils import default_dataset_location
from avalanche.benchmarks.utils import as_classification_dataset, AvalancheDataset

# Most datasets in Avalanche are automatically downloaded the first time you use them
# and stored in a default location. You can change this folder by calling
# avalanche.benchmarks.utils.set_dataset_root(new_location)
datadir = default_dataset_location('mnist')

# As we would simply do with any Pytorch dataset we can create the train and 
# test sets from it. We could use any of the above imported Datasets, but let's
# just try to use the standard MNIST.
train_MNIST = MNIST(datadir, train=True, download=True)
test_MNIST = MNIST(datadir, train=False, download=True)

# transformations are managed by the AvalancheDataset
train_transforms = torchvision.transforms.ToTensor()
eval_transforms = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Resize((32, 32))
])

# wrap datasets into Avalanche datasets
# notice that AvalancheDatasets have multiple transform groups
# `train` and `eval` are the default ones, but you can add more (e.g. replay-specific transforms)
train_MNIST = as_classification_dataset(
    train_MNIST,
    transform_groups={
        'train': train_transforms, 
        'eval': eval_transforms
    }
)
test_MNIST = as_classification_dataset(
    test_MNIST,
    transform_groups={
        'train': train_transforms, 
        'eval': eval_transforms
    }
)

# we can iterate the examples as we would do with any Pytorch dataset
for i, example in enumerate(train_MNIST):
    print(f"Sample {i}: {example[0].shape} - {example[1]}")
    break

# or use a Pytorch DataLoader
train_loader = torch.utils.data.DataLoader(
    train_MNIST, batch_size=32, shuffle=True
)
for i, (x, y) in enumerate(train_loader):
    print(f"Batch {i}: {x.shape} - {y.shape}")
    break

# we can also switch between train/eval transforms
train_MNIST.train()
print(train_MNIST[0][0].shape)

train_MNIST.eval()
print(train_MNIST[0][0].shape)

```

In this example we created a classification dataset. Avalanche expects an attribute `targets` for classification dataset, which is provided by MNIST and most classification datasets.
Avalanche provides concatenation and subsampling, which also keep the dataset attributes consistent.


```python
print(len(train_MNIST))  # 60k
print(len(train_MNIST.concat(train_MNIST)))  # 120k

# subsampling is often used to create streams or replay buffers!
dsub = train_MNIST.subset([0, 1, 2, 3, 4])
print(len(dsub))  # 5
# targets are preserved when subsetting
print(list(dsub.targets))
```

## 🏛️ Classic Benchmarks

Most benchmarks will provide two streams: the `train_stream` and `test_stream`.
Often, these are two parallel streams of the same length, where each experience is sampled from the same distribution (e.g. same set of classes). 
Some benchmarks may have a single test experience with the whole test dataset.

Experiences provide all the information needed to update the model, such as the new batch of data, and they may be decorated with attributes that are helpful for training or logging purposes.
Long streams can be generated on-the-fly to reduce memory requirements and avoiding long preprocessing time during the benchmark creation step.

We will use `SplitMNIST`, a popular CL benchmark which is the class-incremental version of `MNIST`.


```python
from avalanche.benchmarks.classic import SplitMNIST

bm = SplitMNIST(
    n_experiences=5,  # 5 incremental experiences
    return_task_id=True,  # add task labels
    seed=1  # you can set the seed for reproducibility. This will fix the order of classes
)

# streams have a name, used for logging purposes
# each metric will be logged with the stream name
print(f'--- Stream: {bm.train_stream.name}')
# each stream is an iterator of experiences
for exp in bm.train_stream:
    # experiences have an ID that denotes its position in the stream
    # this is used only for logging (don't rely on it for training!)
    eid = exp.current_experience
    # for classification benchmarks, experiences have a list of classes in this experience
    clss = exp.classes_in_this_experience
    # you may also have task labels
    tls = exp.task_labels
    print(f"EID={eid}, classes={clss}, tasks={tls}")
    # the experience provides a dataset
    print(f"data: {len(exp.dataset)} samples")

for exp in bm.test_stream:
    print(f"EID={exp.current_experience}, classes={exp.classes_in_this_experience}, task={tls}")

```

## 🐣 Benchmarks Generators

The most basic way to create a benchmark is to use the `benchmark_from_datasets` method. It takes a list of datasets for each stream and returns a benchmark with the specified streams.


```python

from avalanche.benchmarks.datasets.torchvision_wrapper import CIFAR10
from avalanche.benchmarks.scenarios.dataset_scenario import benchmark_from_datasets


datadir = default_dataset_location('mnist')
train_MNIST = as_classification_dataset(MNIST(datadir, train=True, download=True))
test_MNIST = as_classification_dataset(MNIST(datadir, train=False, download=True))

datadir = default_dataset_location('cifar10')
train_CIFAR10 = as_classification_dataset(CIFAR10(datadir, train=True, download=True))
test_CIFAR10 = as_classification_dataset(CIFAR10(datadir, train=False, download=True))

bm = benchmark_from_datasets(
    train=[train_MNIST, train_CIFAR10],
    test=[test_MNIST, test_CIFAR10]
)

print(f"{bm.train_stream.name} - len {len(bm.train_stream)}")
print(f"{bm.test_stream.name} - len {len(bm.test_stream)}")
```

we can also split a validation stream from the training stream


```python
from avalanche.benchmarks.scenarios.dataset_scenario import benchmark_with_validation_stream


print(f"original training samples = {len(bm.train_stream[0].dataset)}")

bm = benchmark_with_validation_stream(bm, validation_size=0.25)
print(f"new training samples = {len(bm.train_stream[0].dataset)}")
print(f"validation samples = {len(bm.valid_stream[0].dataset)}")
```

### Experience Attributes

The Continual Learning nomenclature is overloaded and quite confusing. Avalanche has its own nomenclature to provide consistent naming across the library.
For example:
- **Task-awareness**: a model is task-aware if it requires task labels. Avalanche benchmarks can have task labels to support this use case;
- **Online**: online streams are streams with small experiences (e.g. 10 samples). They look exactly like their "large batches" counterpart, except for the fact that `len(experience.dataset)` is small;
- **Boundary-awareness**: a model is boundary-aware if it requires boundary labels. Boundary-free models are also called task-free in the literature (there is not accepted nomenclature for "boundary-aware" models). We don't use this nomenclature because task and boundaries are different concepts in Avalanche. Avalanche benchmarks can have boundary labels to support this use case. Even for boundary-free models, Avalanche benchmarks can provide boundary labels to support evaluation metrics that require them;
- **Classification**: classification is the most common CL setting. Avalanche adds class labels to experience to simplify the code of the user. Similarly, Avalanche datasets keep track of `targets` to support this use case.

Avalanche experiences can be decorated with different attributes depending on the specific setting.
Classic benchmarks already provide the attributes you need. We will see some examples of attributes and generators in the remaining part of this tutorial.

One general aspects of experience attributes is that they may not always be available. Sometimes, a model can use task labels during training but not at evaluation time. Other times, the model should never use task lavels but you may still need them for evaluation purposes (to compute task-aware metrics). Avalanche experience have different modalities:
- training mode
- evaluation mode
- logging mode

Each modality can provide access or mask some of the experience attributes. This mechanism allows you to easily add private attributes to the experience for logging purposes while ensuring that the model will not cheat by using that information.


```python
from avalanche.benchmarks.scenarios.generic_scenario import MaskedAttributeError


bm = SplitMNIST(n_experiences=5)

exp = bm.train_stream[0]
# current experience is the position of the experience in the stream.
# It must never be used during training or evaluation
# if you try to use it will fail with a MaskedAttributeError

try:
    # exp.train() returns the experience in training mode
    print(f"Experience {exp.train().current_experience}")
except MaskedAttributeError as e:
    print("can't access current_experience during training")

try:
    # exp.eval() returns the experience in evaluation mode
    print(f"Experience {exp.eval().current_experience}")
except MaskedAttributeError as e:
    print("can't access current_experience during evaluation")

# exp.logging() returns the experience in logging mode
# everything is available during logging
print(f"Experience {exp.logging().current_experience}")
```

#### Classification

classification benchmarks follow the `ClassesTimeline` protocol and provide attributes about the classes in the stream. 


```python
from avalanche.benchmarks.scenarios.supervised import class_incremental_benchmark

datadir = default_dataset_location('mnist')
train_MNIST = as_classification_dataset(MNIST(datadir, train=True, download=True))
test_MNIST = as_classification_dataset(MNIST(datadir, train=False, download=True))

# a class-incremental split
# 5 experiences, 2 classes per experience
bm = class_incremental_benchmark({'train': train_MNIST, 'test': test_MNIST}, num_experiences=5)

exp = bm.train_stream[0]
print(f"Experience {exp.current_experience}")
print(f"Classes in this experience: {exp.classes_in_this_experience}")
print(f"Previous classes: {exp.classes_seen_so_far}")
print(f"Future classes: {exp.future_classes}")
```

#### Task Labels

task-aware benchmarks add task labels, following the `TaskAware` protocol.


```python
from avalanche.benchmarks.scenarios.supervised import class_incremental_benchmark
from avalanche.benchmarks.scenarios.task_aware import task_incremental_benchmark

bm = class_incremental_benchmark({'train': train_MNIST, 'test': test_MNIST}, num_experiences=5)

# we take the class-incremental benchmark defined above and
# add an incremental task label to each experience
# each sample will have its own task label
bm = task_incremental_benchmark(bm)

for exp in bm.train_stream:
    print(f"Experience {exp.current_experience}")

    # in Avalanche an experience may have multiple task labels
    # if the samples in its dataset come from different tasks
    # here we just have one task label per experience
    print(f"\tTask labels: {exp.task_labels}")

    # samples are now triplets <x, y, task_id>
    print(f"\tSample: {exp.dataset[0]}")
```

#### Online

To define online streams we need two things:
- a mechanism to split a larger stream
- attribute that indicate the boundaries (if necessary)

This is how you do it in Avalanche:


```python
from avalanche.benchmarks.scenarios.online import split_online_stream


bm = class_incremental_benchmark({'train': train_MNIST, 'test': test_MNIST}, num_experiences=5)

# we split the training stream into online experiences
# we don't need to split validation/test streams
# it's actually more convenient to keep them whole to compute the metrics
online_train_stream = split_online_stream(bm.train_stream, experience_size=10)

for exp in online_train_stream:
    print(f"Experience {exp.current_experience}")
    print(f"\tsize: {len(exp.dataset)}")

    # in a training loop, here you would train on the online_train_stream
    # here you would test on bm.valid_stream or bm.test_stream 
```

This completes the "_Benchmark_" tutorial for the "_From Zero to Hero_" series. We hope you enjoyed it!

## 🤝 Run it on Google Colab

You can run _this chapter_ and play with it on Google Colaboratory: [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ContinualAI/avalanche/blob/master/notebooks/from-zero-to-hero-tutorial/03_benchmarks.ipynb)
