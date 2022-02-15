---
description: Create your Continual Learning Benchmark and Start Prototyping
---

# Benchmarks

Welcome to the "_benchmarks_" tutorial of the "_From Zero to Hero_" series. In this part we will present the functionalities offered by the `Benchmarks` module.


```python
!pip install avalanche-lib
```

## 🎯 Nomenclature

First off, let's clarify a bit the nomenclature we are going to use, introducing the following terms: `Datasets`, `Scenarios`, `Benchmarks` and `Generators`.

* By `Dataset` we mean a **collection of examples** that can be used for training or testing purposes but not already organized to be processed as a stream of batches or tasks. Since Avalanche is based on Pytorch, our Datasets are [torch.utils.Datasets](https://pytorch.org/docs/stable/_modules/torch/utils/data/dataset.html#Dataset) objects.
* By `Scenario` we mean a **particular setting**, i.e. specificities about the continual stream of data, a continual learning algorithm will face.
* By `Benchmark` we mean a well-defined and carefully thought **combination of a scenario with one or multiple datasets** that we can use to asses our continual learning algorithms.
* By `Generator` we mean a function that **given a specific scenario and a dataset can generate a Benchmark**.

## 📚 The Benchmarks Module

The `bechmarks` module offers 3 types of utils:

* **Datasets**: all the Pytorch datasets plus additional ones prepared by our community and particularly interesting for continual learning.
* **Classic Benchmarks**: classic benchmarks used in CL litterature ready to be used with great flexibility.
* **Benchmarks Generators**: a set of functions you can use to create your own benchmark starting from any kind of data and scenario. In particular, we distinguish two type of generators: `Specific` and `Generic`. The first ones will let you create a benchmark based on a clear scenarios and Pytorch dataset\(s\); the latters, instead, are more generic and flexible, both in terms of scenario definition then in terms of type of data they can manage.
  * _Specific_:
    * **nc\_benchmark**: given one or multiple datasets it creates a benchmark instance based on scenarios where _New Classes_ \(NC\) are encountered over time. Notable scenarios that can be created using this utility include _Class-Incremental_, _Task-Incremental_ and _Task-Agnostic_ scenarios.
    * **ni\_benchmark**: it creates a benchmark instance based on scenarios where _New Instances_ \(NI\), i.e. new examples of the same classes are encountered over time. Notable scenarios that can be created using this utility include _Domain-Incremental_ scenarios.
  * _Generic_:
    * **filelist\_benchmark**: It creates a benchmark instance given a list of filelists.
    * **paths\_benchmark**:  It creates a benchmark instance given a list of file paths and class labels.
    * **tensors\_benchmark**: It creates a benchmark instance given a list of tensors.
    * **dataset\_benchmark**: It creates a benchmark instance given a list of pytorch datasets.

But let's see how we can use this module in practice!

## 🖼️ Datasets

Let's start with the `Datasets`. As we previously hinted, in _Avalanche_ you'll find all the standard Pytorch Datasets available in the torchvision package as well as a few others that are useful for continual learning but not already officially available within the Pytorch ecosystem.


```python
import torch
import torchvision
from avalanche.benchmarks.datasets import MNIST, FashionMNIST, KMNIST, EMNIST, \
QMNIST, FakeData, CocoCaptions, CocoDetection, LSUN, ImageNet, CIFAR10, \
CIFAR100, STL10, SVHN, PhotoTour, SBU, Flickr8k, Flickr30k, VOCDetection, \
VOCSegmentation, Cityscapes, SBDataset, USPS, Kinetics400, HMDB51, UCF101, \
CelebA, CORe50Dataset, TinyImagenet, CUB200, OpenLORIS

# As we would simply do with any Pytorch dataset we can create the train and 
# test sets from it. We could use any of the above imported Datasets, but let's
# just try to use the standard MNIST.
train_MNIST = MNIST(
    './data/mnist', train=True, download=True, transform=torchvision.transforms.ToTensor()
)
test_MNIST = MNIST(
    './data/mnist', train=False, download=True, transform=torchvision.transforms.ToTensor()
)

# Given these two sets we can simply iterate them to get the examples one by one
for i, example in enumerate(train_MNIST):
    pass
print("Num. examples processed: {}".format(i))

# or use a Pytorch DataLoader
train_loader = torch.utils.data.DataLoader(
    train_MNIST, batch_size=32, shuffle=True
)
for i, (x, y) in enumerate(train_loader):
    pass
print("Num. mini-batch processed: {}".format(i))
```

Of course also the basic utilities `ImageFolder` and `DatasetFolder` can be used. These are two classes that you can use to create a Pytorch Dataset directly from your files \(following a particular structure\). You can read more about these in the Pytorch official documentation [here](https://pytorch.org/vision/stable/datasets.html#torchvision.datasets.ImageFolder).

We also provide an additional `FilelistDataset` and `AvalancheDataset` classes. The former to construct a dataset from a filelist [\(caffe style\)](https://ceciliavision.wordpress.com/2016/03/08/caffedata-layer/) pointing to files anywhere on the disk. The latter to augment the basic Pytorch Dataset functionalities with an extention to better deal with a stack of transformations to be used during train and test.


```python
from avalanche.benchmarks.utils import ImageFolder, DatasetFolder, FilelistDataset, AvalancheDataset
```

## 🛠️ Benchmarks Basics

The _Avalanche_ benchmarks \(instances of the _Scenario_ class\), contains several attributes that characterize the benchmark. However, the most important ones are the `train` and `test streams`.

In _Avalanche_ we often suppose to have access to these **two parallel stream of data** \(even though some benchmarks may not provide such feature, but contain just a unique test set\).

Each of these `streams` are _iterable_, _indexable_ and _sliceable_ objects that are composed of unique **experiences**. Experiences are batch of data \(or "_tasks_"\) that can be provided with or without a specific task label.

#### Efficiency

It is worth mentioning that all the data belonging to a _stream_ are not loaded into the RAM beforehand. Avalanche actually loads the data when a specific _mini-batches_ are requested at training/test time based on the policy defined by each `Dataset` implementation.

This means that memory requirements are very low, while the speed is guaranteed by a multi-processing data loading system based on the one defined in Pytorch.

#### Scenarios

So, as we have seen, each `scenario` object in _Avalanche_ has several useful attributes that characterizes the benchmark, including the two important `train` and `test streams`. Let's check what you can get from a scenario object more in details:


```python
from avalanche.benchmarks.classic import SplitMNIST
split_mnist = SplitMNIST(n_experiences=5, seed=1)

# Original train/test sets
print('--- Original datasets:')
print(split_mnist.original_train_dataset)
print(split_mnist.original_test_dataset)

# A list describing which training patterns are assigned to each experience.
# Patterns are identified by their id w.r.t. the dataset found in the
# original_train_dataset field.
print('--- Train patterns assignment:')
print(split_mnist.train_exps_patterns_assignment)

# A list describing which test patterns are assigned to each experience.
# Patterns are identified by their id w.r.t. the dataset found in the
# original_test_dataset field
print('--- Test patterns assignment:')
print(split_mnist.test_exps_patterns_assignment)

# the task label of each experience.
print('--- Task labels:')
print(split_mnist.task_labels)

# train and test streams
print('--- Streams:')
print(split_mnist.train_stream)
print(split_mnist.test_stream)

# A list that, for each experience (identified by its index/ID),
# stores a set of the (optionally remapped) IDs of classes of patterns
# assigned to that experience.
print('--- Classes in each experience:')
print(split_mnist.original_classes_in_exp)
```

#### Train and Test Streams

The _train_ and _test streams_ can be used for training and testing purposes, respectively. This is what you can do with these streams:


```python
# each stream has a name: "train" or "test"
train_stream = split_mnist.train_stream
print(train_stream.name)

# we have access to the scenario from which the stream was taken
train_stream.benchmark

# we can slice and reorder the stream as we like!
substream = train_stream[0]
substream = train_stream[0:2]
substream = train_stream[0,2,1]

len(substream)
```

#### Experiences

Each stream can in turn be treated as an iterator that produces a unique `experience`, containing all the useful data regarding a _batch_ or _task_ in the continual stream our algorithms will face. Check out how can you use these experiences below:


```python
# we get the first experience
experience = train_stream[0]

# task label and dataset are the main attributes
t_label = experience.task_label
dataset = experience.dataset

# but you can recover additional info
experience.current_experience
experience.classes_in_this_experience
experience.classes_seen_so_far
experience.previous_classes
experience.future_classes
experience.origin_stream
experience.benchmark

# As always, we can iterate over it normally or with a pytorch
# data loader.
# For instance, we can use tqdm to add a progress bar.
from tqdm import tqdm
for i, data in enumerate(tqdm(dataset)):
  pass
print("\nNumber of examples:", i + 1)
print("Task Label:", t_label)
```

## 🏛️ Classic Benchmarks

Now that we know how our benchmarks work in general through scenarios, streams and experiences objects, in this section we are going to explore **common benchmarks** already available for you with one line of code yet flexible enough to allow proper tuning based on your needs:


```python
from avalanche.benchmarks.classic import CORe50, SplitTinyImageNet, \
SplitCIFAR10, SplitCIFAR100, SplitCIFAR110, SplitMNIST, RotatedMNIST, \
PermutedMNIST, SplitCUB200, SplitImageNet

# creating PermutedMNIST (Task-Incremental)
perm_mnist = PermutedMNIST(
    n_experiences=2,
    seed=1234,
)
```

Many of the classic benchmarks will download the original datasets they are based on automatically and put it under the `"~/.avalanche/data"` directory.

### How to Use the Benchmarks

Let's see now how we can use the classic benchmark or the ones that you can create through the generators \(see next section\). For example, let's try out the classic `PermutedMNIST` benchmark \(_Task-Incremental_ scenario\).


```python
# creating the benchmark instance (scenario object)
perm_mnist = PermutedMNIST(
  n_experiences=3,
  seed=1234,
)

# recovering the train and test streams
train_stream = perm_mnist.train_stream
test_stream = perm_mnist.test_stream

# iterating over the train stream
for experience in train_stream:
  print("Start of task ", experience.task_label)
  print('Classes in this task:', experience.classes_in_this_experience)

  # The current Pytorch training set can be easily recovered through the
  # experience
  current_training_set = experience.dataset
  # ...as well as the task_label
  print('Task {}'.format(experience.task_label))
  print('This task contains', len(current_training_set), 'training examples')

  # we can recover the corresponding test experience in the test stream
  current_test_set = test_stream[experience.current_experience].dataset
  print('This task contains', len(current_test_set), 'test examples')
```

## 🐣 Benchmarks Generators

What if we want to create a new benchmark that is not present in the "_Classic_" ones? Well, in that case _Avalanche_ offer a number of utilites that you can use to create your own benchmark with maximum flexibility: the **benchmarks generators**!

### Specific Generators

The _specific_ scenario generators are useful when starting from one or multiple Pytorch datasets you want to create a "**New Instances**" or "**New Classes**" benchmark: i.e. it supports the easy and flexible creation of a _Domain-Incremental_, _Class-Incremental or Task-Incremental_ scenarios among others.

For the **New Classes** scenario you can use the following function:

* `nc_benchmark`

for the **New Instances**:

* `ni_benchmark`


```python
from avalanche.benchmarks.generators import nc_benchmark, ni_benchmark
```

Let's start by creating the MNIST dataset object as we would normally do in Pytorch:


```python
from torchvision.transforms import Compose, ToTensor, Normalize, RandomCrop
train_transform = Compose([
    RandomCrop(28, padding=4),
    ToTensor(),
    Normalize((0.1307,), (0.3081,))
])

test_transform = Compose([
    ToTensor(),
    Normalize((0.1307,), (0.3081,))
])

mnist_train = MNIST(
    './data/mnist', train=True, download=True, transform=train_transform
)
mnist_test = MNIST(
    './data/mnist', train=False, download=True, transform=test_transform
)
```

Then we can, for example, create a new benchmark based on MNIST and the classic _Domain-Incremental_ scenario:


```python
scenario = ni_benchmark(
    mnist_train, mnist_test, n_experiences=10, shuffle=True, seed=1234,
    balance_experiences=True
)

train_stream = scenario.train_stream

for experience in train_stream:
    t = experience.task_label
    exp_id = experience.current_experience
    training_dataset = experience.dataset
    print('Task {} batch {} -> train'.format(t, exp_id))
    print('This batch contains', len(training_dataset), 'patterns')
```

Or, we can create a benchmark based on MNIST and the _Class-Incremental_ \(what's commonly referred to as "_Split-MNIST_" benchmark\):


```python
scenario = nc_benchmark(
    mnist_train, mnist_test, n_experiences=10, shuffle=True, seed=1234,
    task_labels=False
)

train_stream = scenario.train_stream

for experience in train_stream:
    t = experience.task_label
    exp_id = experience.current_experience
    training_dataset = experience.dataset
    print('Task {} batch {} -> train'.format(t, exp_id))
    print('This batch contains', len(training_dataset), 'patterns')
```

### Generic Generators

Finally, if you cannot create your ideal benchmark since it does not fit well in the aforementioned _new classes_ or _new instances_ scenarios, you can always use our **generic generators**:

* **filelist\_benchmark**
* **paths\_benchmark**
* **dataset\_benchmark**
* **tensors\_benchmark**


```python
from avalanche.benchmarks.generators import filelist_benchmark, dataset_benchmark, \
                                            tensors_benchmark, paths_benchmark
```

Let's start with the `filelist_benchmark` utility. This function is particularly useful when it is important to preserve a particular order of the patterns to be processed \(for example if they are frames of a video\), or in general if we have data scattered around our drive and we want to create a sequence of batches/tasks providing only a txt file containing the list of their paths.

For _Avalanche_ we follow the same format of the _Caffe_ filelists \("_path_ _class\_label_"\):

/path/to/a/file.jpg 0  
/path/to/another/file.jpg 0  
...  
/path/to/another/file.jpg M  
/path/to/another/file.jpg M  
...  
/path/to/another/file.jpg N  
/path/to/another/file.jpg N  


So let's download the classic "_Cats vs Dogs_" dataset as an example:


```python
!wget -N --no-check-certificate \
    https://storage.googleapis.com/mledu-datasets/cats_and_dogs_filtered.zip
!unzip -q -o cats_and_dogs_filtered.zip
```

You can now see in the `content` directory on colab the image we downloaded. We are now going to create the filelists and then use the `filelist_benchmark` function to create our benchmark:


```python
import os
# let's create the filelists since we don't have it
dirpath = "cats_and_dogs_filtered/train"

for filelist, rel_dir, t_label in zip(
        ["train_filelist_00.txt", "train_filelist_01.txt"],
        ["cats", "dogs"],
        [0, 1]):
    # First, obtain the list of files
    filenames_list = os.listdir(os.path.join(dirpath, rel_dir))

    # Create the text file containing the filelist
    # Filelists must be in Caffe-style, which means
    # that they must define path in the format:
    #
    # relative_path_img1 class_label_first_img
    # relative_path_img2 class_label_second_img
    # ...
    #
    # For instance:
    # cat/cat_0.png 1
    # dog/dog_54.png 0
    # cat/cat_3.png 1
    # ...
    # 
    # Paths are relative to a root path
    # (specified when calling filelist_benchmark)
    with open(filelist, "w") as wf:
        for name in filenames_list:
            wf.write(
                "{} {}\n".format(os.path.join(rel_dir, name), t_label)
            )

# Here we create a GenericCLScenario ready to be iterated
generic_scenario = filelist_benchmark(
   dirpath,  
   ["train_filelist_00.txt", "train_filelist_01.txt"],
   ["train_filelist_00.txt"],
   task_labels=[0, 0],
   complete_test_set_only=True,
   train_transform=ToTensor(),
   eval_transform=ToTensor()
)
```

In the previous cell we created a benchmark instance starting from file lists. However, `paths_benchmark` is a better choice if you already have the list of paths directly loaded in memory:


```python
train_experiences = []
for rel_dir, label in zip(
        ["cats", "dogs"],
        [0, 1]):
    # First, obtain the list of files
    filenames_list = os.listdir(os.path.join(dirpath, rel_dir))

    # Don't create a file list: instead, we create a list of 
    # paths + class labels
    experience_paths = []
    for name in filenames_list:
      instance_tuple = (os.path.join(dirpath, rel_dir, name), label)
      experience_paths.append(instance_tuple)
    train_experiences.append(experience_paths)

# Here we create a GenericCLScenario ready to be iterated
generic_scenario = paths_benchmark(
   train_experiences,
   [train_experiences[0]],  # Single test set
   task_labels=[0, 0],
   complete_test_set_only=True,
   train_transform=ToTensor(),
   eval_transform=ToTensor()
)
```

Let us see how we can use the `dataset_benchmark` utility, where we can use several PyTorch datasets as different batches or tasks. This utility expectes a list of datasets for the train, test (and other custom) streams. Each dataset will be used to create an experience:


```python
train_cifar10 = CIFAR10(
    './data/cifar10', train=True, download=True
)
test_cifar10 = CIFAR10(
    './data/cifar10', train=False, download=True
)

generic_scenario = dataset_benchmark(
    [train_MNIST, train_cifar10],
    [test_MNIST, test_cifar10]
)
```

Adding task labels can be achieved by wrapping each datasets using `AvalancheDataset`. Apart from task labels, `AvalancheDataset` allows for more control over transformations and offers an ever growing set of utilities (check the documentation for more details).


```python
# Alternatively, task labels can also be a list (or tensor)
# containing the task label of each pattern

train_MNIST_task0 = AvalancheDataset(train_cifar10, task_labels=0)
test_MNIST_task0 = AvalancheDataset(test_cifar10, task_labels=0)

train_cifar10_task1 = AvalancheDataset(train_cifar10, task_labels=1)
test_cifar10_task1 = AvalancheDataset(test_cifar10, task_labels=1)

scenario_custom_task_labels = dataset_benchmark(
    [train_MNIST_task0, train_cifar10_task1],
    [test_MNIST_task0, test_cifar10_task1]
)

print('Without custom task labels:',
      generic_scenario.train_stream[1].task_label)

print('With custom task labels:',
      scenario_custom_task_labels.train_stream[1].task_label)
```

And finally, the `tensors_benchmark` generator:


```python
pattern_shape = (3, 32, 32)

# Definition of training experiences
# Experience 1
experience_1_x = torch.zeros(100, *pattern_shape)
experience_1_y = torch.zeros(100, dtype=torch.long)

# Experience 2
experience_2_x = torch.zeros(80, *pattern_shape)
experience_2_y = torch.ones(80, dtype=torch.long)

# Test experience
# For this example we define a single test experience,
# but "tensors_benchmark" allows you to define even more than one!
test_x = torch.zeros(50, *pattern_shape)
test_y = torch.zeros(50, dtype=torch.long)

generic_scenario = tensors_benchmark(
    train_tensors=[(experience_1_x, experience_1_y), (experience_2_x, experience_2_y)],
    test_tensors=[(test_x, test_y)],
    task_labels=[0, 0],  # Task label of each train exp
    complete_test_set_only=True
)
```

This completes the "_Benchmark_" tutorial for the "_From Zero to Hero_" series. We hope you enjoyed it!

## 🤝 Run it on Google Colab

You can run _this chapter_ and play with it on Google Colaboratory: [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ContinualAI/avalanche/blob/master/notebooks/from-zero-to-hero-tutorial/03_benchmarks.ipynb)
