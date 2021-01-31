---
description: 'First things first: let''s start with a good model!'
---

# Models

Welcome to the "**Models**" tutorial of the "_From Zero to Hero_" series. In this notebook we will talk about the features offered by the `models` _Avalanche_ sub-module.

### Models Support

Every continual learning experiment needs a model to train incrementally. The `models` sub-module provide ready-to-use **randomly initialized** and **pre-trained** models you can use _off-the-shelf_.

At the moment we support only the following architectures:

```python
from avalanche.models import SimpleCNN
from avalanche.models import SimpleMLP
from avalanche.models import SimpleMLP_TinyImageNet
from avalanche.models import MobilenetV1
```

However, we plan to support in the near future all the models provided in the [Pytorch](https://pytorch.org/) official ecosystem models as well as the ones provided by [pytorchcv](https://pypi.org/project/pytorchcv/)!

## 🤝 Run it on Google Colab

You can run _this chapter_ and play with it on Google Colaboratory:

{% embed url="https://colab.research.google.com/drive/179B9Jav8BcF3jOZZ95tV0IJE1htvbxg-?usp=sharing" %}

