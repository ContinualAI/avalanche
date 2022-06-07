---
description: Few words about PyTorch Datasets
---

# Preamble: PyTorch Datasets
This short preamble will briefly go through the basic notions of Dataset offered natively by PyTorch. A solid grasp of these notions are needed to understand:
1. How PyTorch data loading works in general
2. How AvalancheDatasets differs from PyTorch Datasets

## 📚 Dataset: general definition

In PyTorch, **a `Dataset` is a class** exposing two methods:
- `__len__()`, which returns the amount of instances in the dataset (as an `int`). 
- `__getitem__(idx)`, which returns the data point at index `idx`.

In other words, a Dataset instance is just an object for which, similarly to a list, one can simply:
- Obtain its length using the Python `len(dataset)` function.
- Obtain a single data point using the `x, y = dataset[idx]` syntax.

The content of the dataset can be either loaded in memory when the dataset is instantiated (like the torchvision MNIST dataset does) or, for big datasets like ImageNet, the content is kept on disk, with the dataset keeping the list of files in an internal field. In this case, data is loaded from the storage on-the-fly when `__getitem__(idx)` is called. The way those things are managed is specific to each dataset implementation.

## PyTorch Datasets
The PyTorch library offers 4 Dataset implementations:
- `Dataset`: an interface defining the `__len__` and `__getitem__` methods.
- `TensorDataset`: instantiated by passing X and Y tensors. Each row of the X and Y tensors is interpreted as a data point. The `__getitem__(idx)` method will simply return the `idx`-th row of X and Y tensors.
- `ConcatDataset`: instantiated by passing a list of datasets. The resulting dataset is a concatenation of those datasets.
- `Subset`: instantiated by passing a dataset and a list of indices. The resulting dataset will only contain the data points described by that list of indices.

As explained in the mini *How-To*s, Avalanche offers a customized version for all these 4 datasets.

## Transformations
Most datasets from the *torchvision* libraries (as well as datasets found "in the wild") allow for a `transformation` function to be passed to the dataset constructor. The support for transformations is not mandatory for a dataset, but it is quite common to support them. The transformation is used to process the X value of a data point before returning it. This is used to normalize values, apply augmentations, etcetera.

As explained in the mini *How-To*s, the `AvalancheDataset` class implements a very rich and powerful set of functionalities for managing transformations.

## Quick note on the IterableDataset class
A variation of the standard `Dataset` exist in PyTorch: the [IterableDataset](https://pytorch.org/docs/stable/data.html#iterable-style-datasets). When using an `IterableDataset`, one can load the data points in a sequential way only (by using a tape-alike approach). The `dataset[idx]` syntax and `len(dataset)` function are not allowed. **Avalanche does NOT support `IterableDataset`s.** You shouldn't worry about this because, realistically, you will never encounter such datasets.

## DataLoader
The `Dataset` is a very simple object that only returns one data point given its index. In order to create minibatches and speed-up the data loading process, a `DataLoader` is required.

The PyTorch `DataLoader` class is a very efficient mechanism that, given a `Dataset`, will return **minibatches** by optonally **shuffling** data brefore each epoch and by **loading data in parallel** by using multiple workers.

## Preamble wrap-up
To wrap-up, let's see how the native, *non-Avalanche*, PyTorch components work in practice. In the following code we create a `TensorDataset` and then we load it in minibatches using a `DataLoader`.


```python
import torch
from torch.utils.data.dataset import TensorDataset
from torch.utils.data.dataloader import DataLoader

# Create a dataset of 100 data points described by 22 features + 1 class label
x_data = torch.rand(100, 22)
y_data = torch.randint(0, 5, (100,))

# Create the Dataset
my_dataset = TensorDataset(x_data, y_data)

# Create the DataLoader
my_dataloader = DataLoader(my_dataset, batch_size=10, shuffle=True, num_workers=4)

# Run one epoch
for x_minibatch, y_minibatch in my_dataloader:
    print('Loaded minibatch of', len(x_minibatch), 'instances')
# Output: "Loaded minibatch of 10 instances" x10 times
```

## Next steps
With these notions in mind, you can start start your journey on understanding the functionalities offered by the AvalancheDatasets by going through the *Mini How-To*s.

Please refer to the [list of the *Mini How-To*s regarding AvalancheDatasets](https://avalanche.continualai.org/how-tos/avalanchedataset) for a complete list. It is recommended to start with the **"Creating AvalancheDatasets"** *Mini How-To*.

## 🤝 Run it on Google Colab

You can run _this chapter_ and play with it on Google Colaboratory by clicking here: [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ContinualAI/avalanche/blob/master/notebooks/how-tos/avalanchedataset/preamble-pytorch-datasets.ipynb)
