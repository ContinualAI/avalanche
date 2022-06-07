---
description: Creation and manipulation of AvalancheDatasets and its subclasses.
---

# Creating AvalancheDatasets

The *AvalancheDataset* is an implementation of the PyTorch Dataset class which comes with many out-of-the-box functionalities. The *AvalancheDataset* (an its few subclass) are extensively used through the whole Avalanche library as the reference way to manipulate datasets:

- The dataset carried by the `experience.dataset` field is always an *AvalancheDataset*.
- Benchmark creation functions accept *AvalancheDataset*s to create benchmarks where a finer control over task labels is required.
- Internally, benchmarks are created by manipulating *AvalancheDataset*s.

This first *Mini How-To* will guide through the main ways you can use to **instantiate an _AvalancheDataset_** while the **other Mini How-Tos ([complete list here](https://avalanche.continualai.org/how-tos/avalanchedataset)) will show how to use its functionalities**.

It is warmly recommended to **run this page as a notebook** using Colab (info at the bottom of this page).

Let's start by installing avalanche:


```python
!pip install avalanche-lib
```

## AvalancheDataset vs PyTorch Dataset
This mini How-To will guide you through the main ways used to instantiate an *AvalancheDataset*.

First thing: the base class `AvalancheDataset` is a **wrapper for existing datasets**. Only two things must be considered when wrapping an existing dataset:

- Apart from the x and y values, the resulting AvalancheDataset will also return a third value: the task label (which defaults to 0).
- The wrapped dataset must contain a valid **targets** field.

The **targets field** is available is nearly all *torchvision* datasets. It must be a list containing the label for each data point (usually the y value). In this way, Avalanche can use that field when instantiating benchmarks like the "Class/Task-Incremental* and *Domain-Incremental* ones.

Avalanche exposes 4 classes of *AvalancheDataset*s which map exactly the 4 *Dataset* classes offered by PyTorch:
- `AvalancheDataset`: the base class, which acts a wrapper to existing *Dataset* instances.
- `AvalancheTensorDataset`: equivalent to PyTorch `TesnsorDataset`.
- `AvalancheSubset`: equivalent to PyTorch `Subset`.
- `AvalancheConcatDataset`: equivalent to PyTorch `ConcatDataset`.

## 🛠️ Create an AvalancheDataset
Given a dataset (like MNIST), an *AvalancheDataset* can be instantiated as follows:


```python
from avalanche.benchmarks.utils import AvalancheDataset
from torchvision.datasets import MNIST

# Instantiate the MNIST train dataset from torchvision
mnist_dataset = MNIST('mnist_data', download=True)

# Create the AvalancheDataset
mnist_avalanche_dataset = AvalancheDataset(mnist_dataset)
```

Just like any other Dataset, a data point can be obtained using the `x, y = dataset[idx]` syntax. **When obtaining a data point from an AvalancheDataset, an additional third value (the task label) will be returned**:


```python
# Obtain the first instance from the original dataset
x, y = mnist_dataset[0]
print(f'x={x}, y={y}')
# Output: "x=<PIL.Image.Image image mode=L size=28x28 at 0x7FBEDFDB2430>, y=5"

# Obtain the first instance from the AvalancheDataset
x, y, t = mnist_avalanche_dataset[0]
print(f'x={x}, y={y}, t={t}')
# Output: "x=<PIL.Image.Image image mode=L size=28x28 at 0x7FBEEFD3A850>, y=5, t=0"
```

**Useful tip:** if you are not sure if you are dealing with a PyTorch *Dataset* or an *AvalancheDataset*, or if you want to ignore task labels, you can use this syntax:


```python
# You can use "x, y, *_" to manage both kinds of Datasets
x, y, *_ = mnist_dataset[0]  # OK
x, y, *_ = mnist_avalanche_dataset[0]  # OK
```

## The AvalancheTensorDataset
The PyTorch *TensorDataset* is one of the most useful Dataset classes as it can be used to quickly prototype the data loading part of your code.

A *TensorDataset* can be wrapped in an AvalancheDataset just like any Dataset, but this is not much convenient, as shown below:


```python
import torch
from torch.utils.data import TensorDataset


# Create 10 instances described by 7 features 
x_data = torch.rand(10, 7)

# Create the class labels for the 10 instances
y_data = torch.randint(0, 5, (10,))

# Create the tensor dataset
tensor_dataset = TensorDataset(x_data, y_data)

# Wrap it in an AvalancheDataset
wrapped_tensor_dataset = AvalancheDataset(tensor_dataset)

# Obtain the first instance from the dataset
x, y, t = wrapped_tensor_dataset[0]
print(f'x={x}, y={y}, t={t}')
# Output: "x=tensor([0.6329, 0.8495, 0.1853, 0.7254, 0.7893, 0.8079, 0.1106]), y=4, t=0"
```

**Instead, it is recommended to use the AvalancheTensorDataset** class to get the same result. In this way, you can just skip one intermediate step.


```python
from avalanche.benchmarks.utils import AvalancheTensorDataset

# Create the tensor dataset
avl_tensor_dataset = AvalancheTensorDataset(x_data, y_data)

# Obtain the first instance from the AvalancheTensorDataset
x, y, t = avl_tensor_dataset[0]
print(f'x={x}, y={y}, t={t}')
# Output: "x=tensor([0.6329, 0.8495, 0.1853, 0.7254, 0.7893, 0.8079, 0.1106]), y=4, t=0"
```

In both cases, **AvalancheDataset will automatically populate its _targets_ field by using the values from the second Tensor** (which usually contains the Y values). This behaviour can be customized by passing a custom `targets` constructor parameter (by either passing a list of targets or the index of the Tensor to use).

The cell below shows the content of the target field of the dataset created in the cell above. Notice that the *targets* field has been filled with the content of the second Tensor (*y\_data*).


```python
# Check the targets field
print('y_data=', y_data)
 # Output: "y_data= tensor([4, 3, 3, 2, 0, 1, 3, 3, 3, 2])"

print('targets field=', avl_tensor_dataset.targets)
# Output: "targets field= [tensor(4), tensor(3), tensor(3), tensor(2), 
#          tensor(0), tensor(1), tensor(3), tensor(3), tensor(3), tensor(2)]"
```

## The AvalancheSubset and AvalancheConcatDataset classes
Avalanche offers the `AvalancheSubset` and `AvalancheConcatDataset` implementations that extend the functionalities of PyTorch *Subset* and *ConcatDataset*.

Regarding the subsetting operation, `AvalancheSubset` behaves in the same way the PyTorch `Subset` class does: both implementations accept a dataset and a list of indices as parameters. The resulting Subset is not a copy of the dataset, it's just a view. This is similar to creating a view of a NumPy array by passing a list of indexes using the `numpy_array[list_of_indices]` syntax. This can be used to both *create a smaller dataset* and to *change the order of data points* in the dataset.

Here we create a toy dataset in which each X and Y values are *int*s. We then obtain a subset of it by creating an **AvalancheSubset**:


```python
from avalanche.benchmarks.utils import AvalancheSubset

# Define the X values of 10 instances (each instance is an int)
x_data_toy = [50, 51, 52, 53, 54, 55, 56, 57, 58, 59]

# Define the class labels for the 10 instances
y_data_toy = [10, 11, 12, 13, 14, 15, 16, 17, 18, 19]

# Create  the tensor dataset
# Note: AvalancheSubset can also be applied to PyTorch TensorDataset directly!
# However, note that PyTorch TensorDataset doesn't support Python lists...
# ... (it only supports Tensors) while AvalancheTensorDataset does.
toy_dataset = AvalancheTensorDataset(x_data_toy, y_data_toy) 

# Define the indices for the subset
# Here we want to obtain a subset containing only the data points...
# ... at indices 0, 5, 8, 2 (in this specific order)
subset_indices = [0, 5, 8, 2]

# Create the subset
avl_subset = AvalancheSubset(toy_dataset, indices=subset_indices)
print('The subset contains', len(avl_subset), 'instances.')
# Output: "The subset contains 4 instances."

# Obtain instances from the AvalancheSubset
for x, y, t in avl_subset:
    print(f'x={x}, y={y}, t={t}')
# Output:
# x=50, y=10, t=0
# x=55, y=15, t=0
# x=58, y=18, t=0
# x=52, y=12, t=0
```

Concatenation is even simpler. Just like with PyTorch *ConcatDataset*, one can easily concatentate datasets with **AvalancheConcatDataset**.

Both *AvalancheConcatDataset* and PyTorch *ConcatDataset* accept a list of datasets to concatenate.


```python
from avalanche.benchmarks.utils import AvalancheConcatDataset

# Define the 2 datasets to be concatenated
x_data_toy_1 = [50, 51, 52, 53, 54]
y_data_toy_1 = [10, 11, 12, 13, 14]
x_data_toy_2 = [60, 61, 62, 63, 64]
y_data_toy_2 = [20, 21, 22, 23, 24]

# Create the datasets
toy_dataset_1 = AvalancheTensorDataset(x_data_toy_1, y_data_toy_1) 
toy_dataset_2 = AvalancheTensorDataset(x_data_toy_2, y_data_toy_2) 

# Create the concat dataset
avl_concat = AvalancheConcatDataset([toy_dataset_1, toy_dataset_2])
print('The concat dataset contains', len(avl_concat), 'instances.')
# Output: "The concat dataset contains 10 instances."

# Obtain instances from the AvalancheConcatDataset
for x, y, t in avl_concat:
    print(f'x={x}, y={y}, t={t}')
# Output:
# x=51, y=11, t=0
# x=52, y=12, t=0
# x=53, y=13, t=0
# x=54, y=14, t=0
# x=60, y=20, t=0
# x=61, y=21, t=0
# x=62, y=22, t=0
# x=63, y=23, t=0
# x=64, y=24, t=0
```

## Dataset Creation wrap-up
This *Mini How-To* showed you how to **create instances of AvalancheDataset (and its subclasses)**.

Other *Mini How-To*s will guide you through the functionalities offered by AvalancheDataset. The list of *Mini How-To*s can be found [here](https://avalanche.continualai.org/how-tos/avalanchedataset).

## 🤝 Run it on Google Colab

You can run _this chapter_ and play with it on Google Colaboratory by clicking here: [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ContinualAI/avalanche/blob/master/notebooks/how-tos/avalanchedataset/creating-avalanchedatasets.ipynb)
