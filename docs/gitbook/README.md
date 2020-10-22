---
description: Powered by ContinualAI
---

# Avalanche: an End-to-End Framework for Continual Learning Research

![](.gitbook/assets/big_logo%20%281%29.png)

**Avalanche** is meant to provide a set of tools and resources for easily prototype new continual learning algorithms and assess them in a comprehensive way without effort. This can also help standardize training and evaluation protocols in continual learning.

In order to achieve this goal the _Avalanche_ framework should be general enough to quickly incorporate new CL strategies as well as new benchmarks and metrics. While it would be great to be DL framework independent, for simplicity I believe we should stick to Pytorch which today is becoming the standard de-facto for machine learning research.

The framework is than split in three main modules:

* **Benchmarks**: This module should maintain a uniform API for processing data in a stream and contain all the major CL datasets \(similar to what has been done for Pytorch-vision\). 
* **Training**: This module should provide all the utilities as well as a standard interface to implement and add a new continual learning strategy. All major CL baselines should be provided here. 
* **Evaluation**: This modules should provide all the utilities and metrics that can help evaluate a CL strategy with respect to all the factors we think are important for CL.

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

# MODEL CREATION
model = SimpleMLP(num_classes=nc_scenario.n_classes)

# DEFINE THE EVALUATION PROTOCOL
evaluation_protocol = EvalProtocol(
    metrics=[ACC(num_class=nc_scenario.n_classes),  # Accuracy metric
             CF(num_class=nc_scenario.n_classes),  # Catastrophic forgetting
             RAMU(),  # Ram usage
             CM()],  # Confusion matrix
    tb_logdir='../logs/mnist_test_sit'
)

# CREATE THE STRATEGY INSTANCE (NAIVE)
cl_strategy = Naive(
    model, 'classifier', SGD(model.parameters(), lr=0.001, momentum=0.9),
    CrossEntropyLoss(), train_mb_size=100, train_epochs=4, test_mb_size=100,
    evaluation_protocol=evaluation_protocol
)

# TRAINING LOOP
print('Starting experiment...')
results = []

for batch_info in nc_scenario:
    print("Start of step ", batch_info.current_step)

    cl_strategy.train(batch_info)
    results.append(cl_strategy.test(batch_info)
```

## 🚦 Getting Started

We know that learning a new tool may be tough at first. This is why we made Avalanche as easy as possible to learn with a set of resources that will help you along the way.  
  
For example, you may start with our _5-minutes_ guide that will let you acquire the basics about _Avalanche_ and how you can use it in your research project:

We have also prepared for you a large set of _examples & snippets_ you can plug-in directly into your code and play with:

Having completed these two sections, you will already feel with superpowers ⚡, this is why we have also created an in-depth tutorial that will cover all the aspect of Avalanche in details and make you a true Continual Learner! 👨‍🎓️

## 🗂️ Maintained by ContinualAI Research

![](.gitbook/assets/continualai_research_logo.png)

_Avalanche_ is the flagship open-source collaborative project of [**ContinuaAI**](https://www.continualai.org/#home): _a non profit research organziation and the largest open community on Continual Learning for AI_. __

The _Avalanche_ project is maintained by the collaborative research team [_**ContinualAI Research \(CLAIR\)**_](https://www.continualai.org/research/)_._ We are always looking for new _awesome members_, so check out our official website if you want to learn more about us and our activities.

Learn more about the [_**Avalanche Team**_](contacts-and-links/the-team.md) _****and all the people who made it great!_

