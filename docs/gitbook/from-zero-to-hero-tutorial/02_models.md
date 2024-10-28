---
description: 'First things first: let''s start with a good model!'
---

# Models

Welcome to the "**Models**" tutorial of the "_From Zero to Hero_" series. In this notebook we will talk about the features offered by the `models` _Avalanche_ sub-module.

### Support for pytorch Modules

Every continual learning experiment needs a model to train incrementally. You can use any `torch.nn.Module`, even pretrained models.  The `models` sub-module provides the most commonly used architectures in the CL literature.

You can use any model provided in the [Pytorch](https://pytorch.org/) official ecosystem models as well as the ones provided by [pytorchcv](https://pypi.org/project/pytorchcv/)!


```python
!pip install avalanche-lib==0.6
```


```python
from avalanche.models import SimpleCNN
from avalanche.models import SimpleMLP
from avalanche.models import SimpleMLP_TinyImageNet
from avalanche.models import MobilenetV1

model = SimpleCNN()
print(model)
```

## Dynamic Model Expansion
A continual learning model may change over time. As an example, a classifier may add new units for previously unseen classes, while progressive networks add a new set units after each experience. Avalanche provides `DynamicModule`s to support these use cases. `DynamicModule`s are `torch.nn.Module`s that provide an addition method, `adaptation`, that is used to update the model's architecture. The method takes a single argument, the data from the current experience.

For example, an IncrementalClassifier updates the number of output units:


```python
from avalanche.benchmarks import SplitMNIST
from avalanche.models import IncrementalClassifier

benchmark = SplitMNIST(5, shuffle=False, class_ids_from_zero_in_each_exp=False)
model = IncrementalClassifier(in_features=784)

print(model)
for exp in benchmark.train_stream:
    model.adaptation(exp)
    print(model)
```

As you can see, after each call to the `adaptation` method, the model adds 2 new units to account for the new classes. Notice that no learning occurs at this point since the method only modifies the model's architecture.

Keep in mind that when you use Avalanche strategies you don't have to call the adaptation yourself. Avalanche strategies automatically call the model's adaptation and update the optimizer to include the new parameters.

## Multi-Task models

Some models, such as multi-head classifiers, are designed to exploit task labels. In Avalanche, such models are implemented as `MultiTaskModule`s. These are dynamic models (since they need to be updated whenever they encounter a new task) that have an additional `task_labels` argument in their `forward` method. `task_labels` is a tensor with a task id for each sample.


```python
from avalanche.benchmarks import SplitMNIST
from avalanche.models import MultiHeadClassifier

benchmark = SplitMNIST(5, shuffle=False, return_task_id=True, class_ids_from_zero_in_each_exp=True)
model = MultiHeadClassifier(in_features=784)

print(model)
for exp in benchmark.train_stream:
    model.adaptation(exp)
    print(model)
```

When you use a `MultiHeadClassifier`, a new head is initialized whenever a new task is encountered. Avalanche strategies automatically recognize multi-task modules and provide task labels to them.

### How to define a multi-task Module
If you want to define a custom multi-task module you need to override two methods: `adaptation` (if needed), and `forward_single_task`. The `forward` method of the base class will split the mini-batch by task-id and provide single task mini-batches to `forward_single_task`.


```python
from avalanche.models import MultiTaskModule

class CustomMTModule(MultiTaskModule):
    def __init__(self, in_features, initial_out_features=2):
        super().__init__()

    def adaptation(self, dataset):
        super().adaptation(dataset)
        # your adaptation goes here

    def forward_single_task(self, x, task_label):
        # your forward goes here.
        # task_label is a single integer
        # the mini-batch is split by task-id inside the forward method.
        pass
```

Alternatively, if you only want to convert a single-head model into a multi-head model, you can use the `as_multitask` wrapper, which converts the model for you.


```python
from avalanche.models import as_multitask

model = SimpleCNN()
print(model)

mt_model = as_multitask(model, 'classifier')
print(mt_model)
```

### Nested Dynamic Modules
Whenever one or more dynamic modules are nested one inside the other, you must call the `recursive_adaptation` method, and if they are nested inside a normal pytorch module (non dynamic), you can call the `avalanche_model_adaptation` function. Avalanche strategies will by default adapt the models before training on each experience by calling `avalanche_model_adaptation`


```python
benchmark = SplitMNIST(5, shuffle=False, class_ids_from_zero_in_each_exp=True, return_task_id=True)

model = SimpleCNN(num_classes=1)
mt_model = as_multitask(model, 'classifier')

print(mt_model)
for exp in benchmark.train_stream:
    mt_model.recursive_adaptation(exp)
print(mt_model)
```


```python
from avalanche.models.utils import avalanche_model_adaptation

benchmark = SplitMNIST(5, shuffle=False, class_ids_from_zero_in_each_exp=False)

model = SimpleCNN(num_classes=1)
model.classifier = IncrementalClassifier(model.classifier[0].in_features, 1)

for exp in benchmark.train_stream:
    avalanche_model_adaptation(model, exp)
    
print(model)
```

## 🤝 Run it on Google Colab

You can run _this chapter_ and play with it on Google Colaboratory: [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ContinualAI/avalanche/blob/master/notebooks/from-zero-to-hero-tutorial/02_models.ipynb)


