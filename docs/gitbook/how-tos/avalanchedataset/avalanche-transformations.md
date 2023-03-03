---
description: Dealing with transformations (groups, appending, replacing, freezing).
---

# Advanced Transformations
While torchvision (and other) datasets typically have a fixed set of transformations, AvalancheDataset also provides some additional functionalities. `AvalancheDataset`s can:
1. Have multiple **transformation "groups"** in the same dataset (like separate train and eval transformations).
2. Manipulate transformation by **freezing, replacing and removing** them.

The following sub-sections show examples on how to use these features.
It is warmly recommended to **run this page as a notebook** using Colab (info at the bottom of this page).

Let's start by installing Avalanche:


```python
!pip install avalanche-lib
```

## Transformation groups
AvalancheDatasets can contain multiple **transformation groups**. This can be useful to keep train and test transformations in the same dataset and to have different sets of transformations. For instance, you can easily add ad-hoc transformations to using for replay data.

For classification dataset, we follow torchvision conventions. Therefore, `make_classification_dataset` supports `transform`, which is applied to input (X) values, and `target_transform`, which is applied to class labels (Y). The latter is rarely used. This means that **a transformation group is a pair of transformations to be applied to the X and Y values** of each instance returned by the dataset. In both torchvision and Avalanche implementations, **a transformation must be a function (or other callable object)** that accepts one input (the X or Y value) and outputs its transformed version. A comprehensive guide on transformations can be found in the [torchvision documentation](https://pytorch.org/vision/stable/transforms.html).

In the following example, a MNIST dataset is created and then wrapped in an AvalancheDataset. When creating the AvalancheDataset, we can set *train* and *eval* transformations by passing a *transform\_groups* parameter. Train transformations usually include some form of random augmentation, while eval transformations usually include a sequence of deterministic transformations only. Here we define the sequence of train transformations as a random rotation followed by the ToTensor operation. The eval transformations only include the ToTensor operation.


```python
from torchvision import transforms
from torchvision.datasets import MNIST
from avalanche.benchmarks.utils import make_classification_dataset

mnist_dataset = MNIST('mnist_data', download=True)

# Define the training transformation for X values
train_transformation = transforms.Compose([
    transforms.RandomRotation(45),
    transforms.ToTensor(),
])
# Define the training transformation for Y values (rarely used)
train_target_transformation = None

# Define the test transformation for X values
eval_transformation = transforms.ToTensor()
# Define the test transformation for Y values (rarely used)
eval_target_transformation = None

transform_groups = {
    'train': (train_transformation, train_target_transformation),
    'eval': (eval_transformation, eval_target_transformation)
}

avl_mnist_transform = make_classification_dataset(mnist_dataset, transform_groups=transform_groups)
```

Of course, one can also just use the `transform` and `target_transform` constructor parameters to set the transformations for both the *train* and the *eval* groups. However, it is recommended to use the approach based on *transform\_groups* (shown in the code above) as it is much more flexible.


```python
# Not recommended: use transform_groups instead
avl_mnist_same_transforms =  make_classification_dataset(mnist_dataset, transform=train_transformation)
```

### Using `.train()` and `.eval()`

**The default behaviour of the AvalancheDataset is to use transformations from the _train_ group.** However, one can easily obtain a version of the dataset where the *eval* group is used. Note: when obtaining the dataset of experiences from the test stream, those datasets will already be using the *eval* group of transformations so you don't need to switch to the eval group ;).

You can switch between the *train* and *eval* groups using the `.train()` and `.eval()` methods to obtain a copy (view) of the dataset with the proper transformations enabled. As a general rule, **methods that manipulate the AvalancheDataset fields (and transformations) always create a view of the dataset. The original dataset is never changed.**

In the following cell we use the *avl\_mnist\_transform* dataset created in the cells above. We first obtain a view of it in which *eval* transformations are enabled. Then, starting from this view, we obtain a version of it in which *train* transformations are enabled. We want to double-stress that `.train()` and `.eval()` never change the group of the dataset on which they are called: they always create a view.

One can check that the correct transformation group is in use by looking at the content of the *transform/target_transform* fields.


```python
# Obtain a view of the dataset in which eval transformations are enabled
avl_mnist_eval = avl_mnist_transform.eval()

# Obtain a view of the dataset in which we get back to train transforms
# Basically, avl_mnist_transform ~= avl_mnist_train
avl_mnist_train = avl_mnist_eval.train()

# we are looking inside the dataset to check the transformations.
# in real code, you never need to do this ;)
cgroup = avl_mnist_train._transform_groups.current_group
print("Original dataset transformations: (train group by default)")
# notice that the original transform are unchanged.
print(avl_mnist_train._transform_groups.transform_groups[cgroup])

print("\neval mode dataset transformations:")
cgroup = avl_mnist_eval._transform_groups.current_group
print(avl_mnist_eval._transform_groups.transform_groups[cgroup])

print("\ntrain mode dataset transformations:")
cgroup = avl_mnist_train._transform_groups.current_group
print(avl_mnist_train._transform_groups.transform_groups[cgroup])
```

### Custom transformation groups
In *AvalancheDataset*s the **_train_ and _eval_ transformation groups are always available**. However, *AvalancheDataset* also supports **custom transformation groups**.

The following example shows how to create an AvalancheDataset with an additional group named *replay*. We define the replay transformation as a random crop followed by the ToTensor operation.


```python
from avalanche.benchmarks.utils import AvalancheDataset

replay_transform = transforms.Compose([
    transforms.RandomCrop(28, padding=4),
    transforms.ToTensor()
])

replay_target_transform = None

transform_groups_with_replay = {
    'train': (None, None),
    'eval': (None, None),
    'replay': (replay_transform, replay_target_transform)
}

AvalancheDataset(mnist_dataset, transform_groups=transform_groups_with_replay)
```

However, once created the dataset will use the *train* group. You can switch to the group using the `.with_transforms(group_name)` method. The `.with_transforms(group_name)` method behaves in the same way `.train()` and `.eval()` do by creating a view of the original dataset.


```python
avl_mnist_custom_transform_not_enabled = AvalancheDataset(
    mnist_dataset,
    transform_groups=transform_groups_with_replay)

avl_mnist_custom_transform_2 = avl_mnist_custom_transform_not_enabled.with_transforms('replay')
cgroup = avl_mnist_custom_transform_2._transform_groups.current_group
print(avl_mnist_custom_transform_2._transform_groups.transform_groups[cgroup])

# prints output:
# Compose(
#     RandomCrop(size=(28, 28), padding=4)
#     ToTensor()
# )
```

## Replacing transformations

The replacement operation follows the same idea (and benefits) of the append one. By using `.replace_current_transform_group(transform, target_transform)` one can obtain a view of the original dataset in which the **transformaations for the current group** are replaced with the given ones. One may also change tranformations for other groups by passing the name of the group as the optional parameter `group`. As with any transform-related operation, the original dataset is not affected. 

Note: one can use `.replace_transforms(...)` to remove previous transformations (by passing `None` as the new transform).

The following cell shows how to use `.replace_transforms(...)` to replace the transformations of the current group:


```python
avl_mnist = make_classification_dataset(mnist_dataset, transform_groups=transform_groups)
new_transform = transforms.RandomCrop(size=(28, 28), padding=4)

# Append a transformation. Simple as:
transform = (new_transform, None)
avl_mnist_replaced_transform = avl_mnist.replace_current_transform_group(transform)

cgroup = avl_mnist_replaced_transform._transform_groups.current_group
print('With replaced transform:', avl_mnist_replaced_transform._transform_groups.transform_groups[cgroup])
# Prints: "With replaces transforms: RandomCrop(size=(28, 28), padding=4)"

# Check that the original dataset was not affected:
cgroup = avl_mnist._transform_groups.current_group
print('Original dataset:', avl_mnist._transform_groups.transform_groups[cgroup])
# Prints: "Original dataset: ToTensor()"
```

## Freezing transformations

One last functionality regarding transformations is the ability to "freeze" transformations. Freezing transformations menas **permanently glueing transformations to the dataset so that they can't be replaced or changed in any way** (usually by mistake). Frozen transformations cannot be changed by using `.replace_transforms(...)`.

One may wonder when this may come in handy... in fact, you will probably rarely need to freeze transformations. However, imagine having to instantiate the PermutedMNIST benchmark. You want the permutation transformation to not be changed by mistake. However, the end users do not know how the internal implementations of the benchmark works, so they may end up messing with those transformations. By freezing the permutation transformation, users cannot mess with it.

Transformations for all transform groups can be frozen at once by using `.freeze_transforms()`. As always, those methods return a view of the original dataset.

The cell below shows a simplified excerpt from the [PermutedMNIST benchmark implementation](https://github.com/ContinualAI/avalanche/blob/master/avalanche/benchmarks/classic/cmnist.py). First, a *PixelsPermutation* instance is created. That instance is a transformation that will permute the pixels of the input image. We then create the train end test sets. Once created, transformations for those datasets are frozen using `.freeze_transforms()`.


```python
from avalanche.benchmarks.classic.cmnist import PixelsPermutation
import numpy as np
import torch

# Instantiate MNIST train and test sets
mnist_train = MNIST('mnist_data', train=True, download=True)
mnist_test = MNIST('mnist_data', train=False, download=True)
    
# Define the transformation used to permute the pixels
rng_seed = 4321
rng_permute = np.random.RandomState(rng_seed)
idx_permute = torch.from_numpy(rng_permute.permutation(784)).type(torch.int64)
permutation_transform = PixelsPermutation(idx_permute)

# Define the transforms group
perm_group_transforms = dict(
    train=(permutation_transform, None),
    eval=(permutation_transform, None)
)

# Create the datasets and freeze transforms
# Note: one can call "freeze_transforms" on constructor result
# or you can do this in 2 steps. The result is the same (obviously).
# The next part show both ways:

# Train set
permuted_train_set = AvalancheDataset(
    mnist_train, 
    transform_groups=perm_group_transforms).freeze_transforms()

# Test set
permuted_test_set = AvalancheDataset(mnist_test, transform_groups=perm_group_transforms).eval()
permuted_test_set = permuted_test_set.freeze_transforms()
```

In this way, that transform can't be removed. However, remember that one can always append other transforms atop of frozen transforms.

The cell below shows that `replace_transforms` can't remove frozen transformations:


```python
# First, show that the image pixels are permuted
print('Before replace_transforms:')
display(permuted_train_set[0][0].resize((192, 192), 0))

# Try to remove the permutation
with_removed_transforms = permuted_train_set.replace_current_transform_group((None, None))

print('After replace_transforms:')
display(permuted_train_set[0][0].resize((192, 192), 0))
display(with_removed_transforms[0][0].resize((192, 192), 0))
```

## Transformations wrap-up
This completes the *Mini How-To* for the functionalities of the *AvalancheDataset* related to **transformations**. 

Here you learned how to use **transformation groups** and how to **append/replace/freeze transformations** in a simple way.

Other *Mini How-To*s will guide you through the other functionalities offered by the *AvalancheDataset* class. The list of *Mini How-To*s can be found [here](https://avalanche.continualai.org/how-tos/avalanchedataset).

## 🤝 Run it on Google Colab

You can run _this chapter_ and play with it on Google Colaboratory by clicking here: [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ContinualAI/avalanche/blob/master/notebooks/how-tos/avalanchedataset/advanced-transformations.ipynb)
