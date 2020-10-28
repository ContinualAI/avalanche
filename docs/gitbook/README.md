---
description: Powered by ContinualAI
---

# Avalanche: an End-to-End Framework for Continual Learning Research

![](.gitbook/assets/avalanche_logo_with_clai.png)

**Avalanche** is an _end-to-end Continual Learning research_ framework based on [**Pytorch**](https://pytorch.org/), born within [**ContinualAI**](https://www.continualai.org/) with the unique goal of providing a **shared** and **collaborative** open-source **codebase** for _fast prototyping_, _training_ and _reproducible_ _evaluation_ of continual learning algorithms. 

Avalanche can help _Continual Learning_ researchers in several ways:

* _Shared, easy-to-use & coherent codebase_
* _Writing less code, prototype faster & introduce less bugs_
* _Improve reproducibility_
* _Improve modularity and reusability_
* _Increased efficiency & portability_
* _Augment impact and usability of your research products_

The framework is than split in three main modules:

* **`Benchmarks`**: This module maintains a uniform API for data handling: mostly generating a stream of data from one or more datasets. It contains all the major CL benchmarks \(similar to what has been done for [torchvision](https://pytorch.org/docs/stable/torchvision/index.html)\).
* **`Training`**: This module provides all the necessary utilities concerning model training. This includes simple and efficient ways of implement new _continual learning_ strategies as well as a set pre-implemented CL baselines and state-of-the-art algorithms you will be able to use for comparison!
* **`Evaluation`**: This modules provides all the utilities and metrics that can help evaluate a CL algorithm with respect to all the factors we believe to be important for a continually learning system. It also includes advanced logging and plotting features, including native [Tensorboard](https://www.tensorflow.org/tensorboard) support.

_Avalanche_ is one of the first experiments of a **End-to-end Research Framework** for reproducible m_achine learning_ research where you can find _benchmarks_, _algorithms_ and _evaluation protocols_ **in the same place**.  
  
Let's make it together 👫 a wonderful ride! 🎈

Check out _how you code changes_ when you start using _Avalanche_! 👇

{% tabs %}
{% tab title="Without Avalanche" %}
```python
# Fill here...
```
{% endtab %}

{% tab title="With Avalanche" %}
```python
import torch
from torch.nn import CrossEntropyLoss
from torch.optim import SGD
from torchvision import transforms
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor, RandomCrop

from avalanche.benchmarks.scenarios import DatasetPart, \
    create_nc_single_dataset_sit_scenario, NCBatchInfo
from avalanche.evaluation import EvalProtocol
from avalanche.evaluation.metrics import ACC, CF, RAMU, CM
from avalanche.extras.models import SimpleMLP
from avalanche.training.strategies.new_strategy_api.cl_naive import Naive


mnist_train = MNIST('./data/mnist', train=True, download=True)
mnist_test = MNIST('./data/mnist', train=False, download=True)
    
nc_scenario = NCScenario(mnist_train, mnist_test, n_batches, shuffle=True, seed=1234)

# Create your model
model = SimpleMLP(num_classes=nc_scenario.n_classes)

# Define the Evaluation Protocol
evaluation_protocol = EvalProtocol(
    metrics=[ACC(num_class=nc_scenario.n_classes),  # Accuracy metric
             CF(num_class=nc_scenario.n_classes),  # Catastrophic forgetting
             RAMU(),  # Ram usage
             CM()],  # Confusion matrix
    tb_logdir='../logs/mnist_test_sit'
)

# Creat your Strategy (Naive)
cl_strategy = Naive(
    model, 'classifier', SGD(model.parameters(), lr=0.001, momentum=0.9),
    CrossEntropyLoss(), train_mb_size=100, train_epochs=4, test_mb_size=100,
    evaluation_protocol=evaluation_protocol
)

# Training loop
print('Starting experiment...')
results = []

for batch_info in nc_scenario:
    print("Start of step ", batch_info.current_step)

    cl_strategy.train(batch_info)
    results.append(cl_strategy.test(batch_info)
```
{% endtab %}
{% endtabs %}

## 🚦 Getting Started

We know that learning a new tool _may be tough at first_. This is why we made _Avalanche_ as easy as possible to learn with a set of resources that will help you along the way.  
  
For example, you may start with our _**5-minutes**_ **guide** that will let you acquire the basics about _Avalanche_ and how you can use it in your research project:

We have also prepared for you a large set of _**examples & snippets**_ you can plug-in directly into your code and play with:

Having completed these two sections, you will already feel with _superpowers_ ⚡, this is why we have also created an **in-depth tutorial** that will cover all the aspect of _Avalanche_ in details and make you a true _Continual Learner_! 👨‍🎓️

## 🗂️ Maintained by ContinualAI Research

![](.gitbook/assets/continualai_research_logo.png)

_Avalanche_ is the flagship open-source collaborative project of [**ContinuaAI**](https://www.continualai.org/#home): _a non profit research organziation and the largest open community on Continual Learning for AI_. __

The _Avalanche_ project is maintained by the collaborative research team [_**ContinualAI Research \(CLAIR\)**_](https://www.continualai.org/research/)_._ We are always looking for new _awesome members_, so check out our official website if you want to learn more about us and our activities.

Learn more about the [_**Avalanche Team and all the people who made it great**_](contacts-and-links/the-team.md)_!_

