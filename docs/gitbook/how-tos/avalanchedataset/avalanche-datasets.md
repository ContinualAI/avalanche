---
description: Converting PyTorch Datasets to Avalanche Dataset
---

# Avalanche Datasets
Datasets are a fundamental data structure for continual learning. Unlike offline training, in continual learning we often need to manipulate datasets to create streams, benchmarks, or to manage replay buffers. High-level utilities and predefined benchmarks already take care of the details for you, but you can easily manipulate the data yourself if you need to. These how-to will explain:

1. PyTorch datasets and data loading
2. How to instantiate Avalanche Datasets
3. AvalancheDataset features

In Avalanche, the `AvalancheDataset` is everywhere:
- The dataset carried by the `experience.dataset` field is always an *AvalancheDataset*.
- Many benchmark creation functions accept *AvalancheDataset*s to create benchmarks.
- Avalanche benchmarks are created by manipulating *AvalancheDataset*s.
- Replay buffers also use `AvalancheDataset` to easily concanate data and handle transformations.


## 📚 PyTorch Dataset: general definition

In PyTorch, **a `Dataset` is a class** exposing two methods:
- `__len__()`, which returns the amount of instances in the dataset (as an `int`). 
- `__getitem__(idx)`, which returns the data point at index `idx`.

In other words, a Dataset instance is just an object for which, similarly to a list, one can simply:
- Obtain its length using the Python `len(dataset)` function.
- Obtain a single data point using the `x, y = dataset[idx]` syntax.

The content of the dataset can be either loaded in memory when the dataset is instantiated (like the torchvision MNIST dataset does) or, for big datasets like ImageNet, the content is kept on disk, with the dataset keeping the list of files in an internal field. In this case, data is loaded from the storage on-the-fly when `__getitem__(idx)` is called. The way those things are managed is specific to each dataset implementation.

### Quick note on the IterableDataset class
A variation of the standard `Dataset` exist in PyTorch: the [IterableDataset](https://pytorch.org/docs/stable/data.html#iterable-style-datasets). When using an `IterableDataset`, one can load the data points in a sequential way only (by using a tape-alike approach). The `dataset[idx]` syntax and `len(dataset)` function are not allowed. **Avalanche does NOT support `IterableDataset`s.** You shouldn't worry about this because, realistically, you will never encounter such datasets (at least in torchvision). If you need `IterableDataset` let us know and we will consider adding support for them.


## How to Create an AvalancheDataset
To create an `AvalancheDataset` from a PyTorch you only need to pass the original data to the constructor as follows


```python
!pip install avalanche-lib
```


```python
import torch
from torch.utils.data.dataset import TensorDataset
from avalanche.benchmarks.utils import AvalancheDataset

# Create a dataset of 100 data points described by 22 features + 1 class label
x_data = torch.rand(100, 22)
y_data = torch.randint(0, 5, (100,))

# Create the Dataset
torch_data = TensorDataset(x_data, y_data)

avl_data = AvalancheDataset(torch_data)
```

The dataset is equivalent to the original one:


```python
print(torch_data[0])
print(avl_data[0])
```

### Classification Datasets

most of the time, you can also use one of the utility function in [benchmark utils](https://avalanche-api.continualai.org/en/latest/benchmarks.html#utils-data-loading-and-avalanchedataset) that also add attributes such as class and task labels to the dataset. For example, you can create a classification dataset using `make_classification_dataset`.

Classification dataset
- returns triplets of the form <x, y, t>, where t is the task label (which defaults to 0).
- The wrapped dataset must contain a valid **targets** field.

Avalanche provides some utility functions to create supervised classification datasets such as:
- `make_tensor_classification_dataset` for tensor datasets
all of these will automatically create the `targets` and `targets_task_labels` attributes.


```python
from avalanche.benchmarks.utils import make_classification_dataset

# first, we add targets to the dataset. This will be used by the AvalancheDataset
# If possible, avalanche tries to extract the targets from the dataset.
# most datasets in torchvision already have a targets field so you don't need this step.
torch_data.targets = torch.randint(0, 5, (100,)).tolist()
tls = [0 for _ in range(100)] # one task label for each sample
sup_data = make_classification_dataset(torch_data, task_labels=tls)
```

## DataLoader
Avalanche provides some [custom dataloaders](https://avalanche-api.continualai.org/en/latest/benchmarks.html#utils-data-loading-and-avalanchedataset) to sample in a task-balanced way or to balance the replay buffer and current data, but you can also use the standard pytorch `DataLoader`.


```python
from torch.utils.data.dataloader import DataLoader

my_dataloader = DataLoader(avl_data, batch_size=10, shuffle=True)

# Run one epoch
for x_minibatch, y_minibatch in my_dataloader:
    print('Loaded minibatch of', len(x_minibatch), 'instances')
# Output: "Loaded minibatch of 10 instances" x10 times
```

## Dataset Operations: Concatenation and SubSampling
While PyTorch provides two different classes for concatenation and subsampling (`ConcatDataset` and `Subset`), Avalanche implements them as dataset methods. These operations return a new dataset, leaving the original one unchanged.


```python
cat_data = avl_data.concat(avl_data)
print(len(cat_data))  # 100 + 100 = 200
print(len(avl_data))  # 100, original data stays the same

sub_data = avl_data.subset(list(range(50)))
print(len(sub_data))  # 50
print(len(avl_data))  # 100, original data stays the same
```

## Dataset Attributes
AvalancheDataset allows to add attributes to datasets. Attributes are named arrays that carry some information that is propagated by concatenation and subsampling operations.
For example, classification datasets use this functionality to manage class and task labels.


```python
tls = [0 for _ in range(100)] # one task label for each sample
sup_data = make_classification_dataset(torch_data, task_labels=tls)
print(sup_data.targets.name, len(sup_data.targets._data))
print(sup_data.targets_task_labels.name, len(sup_data.targets_task_labels._data))

# after subsampling
sub_data = sup_data.subset(range(10))
print(sub_data.targets.name, len(sub_data.targets._data))
print(sub_data.targets_task_labels.name, len(sub_data.targets_task_labels._data))

# after concat
cat_data = sup_data.concat(sup_data)
print(cat_data.targets.name, len(cat_data.targets._data))
print(cat_data.targets_task_labels.name, len(cat_data.targets_task_labels._data))
```

Thanks to `DataAttribute`s, you can freely operate on your data (e.g. to manage a replay buffer) without losing class or task labels. This makes it easy to manage multi-task datasets or to balance datasets by class.

## Transformations
Most datasets from the *torchvision* libraries (as well as datasets found "in the wild") allow for a `transformation` function to be passed to the dataset constructor. The support for transformations is not mandatory for a dataset, but it is quite common to support them. The transformation is used to process the X value of a data point before returning it. This is used to normalize values, apply augmentations, etcetera.

`AvalancheDataset` implements a very rich and powerful set of functionalities for managing transformation. You can learn more about it in the [Advanced Transformations How-To](https://avalanche.continualai.org/how-tos/avalanchedataset/advanced-transformations).

## Next steps
With these notions in mind, you can start start your journey on understanding the functionalities offered by the AvalancheDatasets by going through the *Mini How-To*s.

Please refer to the [list of the *Mini How-To*s regarding AvalancheDatasets](https://avalanche.continualai.org/how-tos/avalanchedataset) for a complete list. It is recommended to start with the **"Creating AvalancheDatasets"** *Mini How-To*.

## 🤝 Run it on Google Colab

You can run _this chapter_ and play with it on Google Colaboratory by clicking here: [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ContinualAI/avalanche/blob/master/notebooks/how-tos/avalanchedataset/preamble-pytorch-datasets.ipynb)
