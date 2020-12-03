# Avalanche: a Comprehensive Framework for Continual Learning Research
**[Avalanche Website]()** | **[Getting Started]()** | **[Examples]()** | **[Tutorial]()** | **[API Doc]()**
<br>
![issue](https://img.shields.io/github/issues/vlomonaco/core50)
![pr](https://img.shields.io/github/issues-pr/vlomonaco/core50)

<p align="center">
<img src="https://www.dropbox.com/s/90thp7at72sh9tj/avalanche_logo_with_clai.png?raw=1"/>
</p>

**Avalanche** is an *end-to-end Continual Learning research framework* based on **Pytorch**, born within ContinualAI with the unique goal of providing a shared and collaborative 
open-source (MIT licensed) codebase for fast prototyping, training and reproducible evaluation of continual learning algorithms. 
Avalanche can help Continual Learning researchers in several ways:

- *Write less code, prototype faster & reduce errors*
- *Improve reproducibility*
- *Improve modularity and reusability*
- *Increase code efficiency, scalability & portability*
- *Augment impact and usability of your research products*

The framework is organized in four main modules:

- [Benchmarks](avalanche/benchmarks): This module maintains a uniform API for data handling: mostly generating a stream of data from one or more datasets. It contains all the major CL benchmarks (similar to what has been done for torchvision).
- [Training](avalanche/training): This module provides all the necessary utilities concerning model training. This includes simple and efficient ways of implement new continual learning strategies as well as a set pre-implemented CL baselines and state-of-the-art algorithms you will be able to use for comparison!
- [Evaluation](avalanche/training): This modules provides all the utilities and metrics that can help evaluate a CL algorithm with respect to all the factors we believe to be important for a continually learning system. It also includes advanced logging and plotting features, including native Tensorboard support.
- [Extras](avalanche/extras): In the extras module you'll be able to find several useful utilities and building blocks that will help you create your continual learning experiments with ease. This includes configuration files for quick reproducibility and model building functions for example.
Avalanche is one of the first experiments of a End-to-end Research Framework for reproducible machine learning research where you can find benchmarks, algorithms and evaluation protocols in the same place.

Let's make it together :people_holding_hands: a wonderful ride! :balloon:

Check out below how you can start using Avalanche! :point_down:

Quick Example
----------------

```python
import torch
from torch.nn import CrossEntropyLoss
from torch.optim import SGD

from avalanche.benchmarks.classic import PermutedMNIST
from avalanche.evaluation import EvalProtocol
from avalanche.evaluation.metrics import ACC
from avalanche.extras.models import SimpleMLP
from avalanche.training.strategies import Naive

# Config
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# model
model = SimpleMLP(num_classes=10)

# CL Benchmark Creation
perm_mnist = PermutedMNIST(incremental_steps=3)
train_stream = perm_mnist.train_stream
test_stream = perm_mnist.test_stream

# Prepare for training & testing
optimizer = SGD(model.parameters(), lr=0.001, momentum=0.9)
criterion = CrossEntropyLoss()
evaluation_protocol = EvalProtocol(metrics=[ACC(num_class=10)])

# Continual learning strategy
cl_strategy = Naive(
    model, optimizer, criterion, train_mb_size=32, train_epochs=2, 
    test_mb_size=32, evaluation_protocol=evaluation_protocol, device=device)

# train and test loop
results = []
for train_task in train_stream:
    cl_strategy.train(train_task, num_workers=4)
    results.append(cl_strategy.test(test_stream))
```

Getting Started
----------------

We know that learning a new tool may be tough at first. This is why we made Avalanche as easy as possible to learn with a set of resources that will help you along the way.
For example, you may start with our 5-minutes guide that will let you acquire the basics about Avalanche and how you can use it in your research project:

- [Getting Stated Guide]()

We have also prepared for you a large set of examples & snippets you can plug-in directly into your code and play with:

- [Avalanche Examples]()

Having completed these two sections, you will already feel with superpowers ⚡, this is why we have also created an in-depth tutorial that will cover all the aspect of Avalanche in 
details and make you a true Continual Learner! :woman_student:

- [From Zero to Hero Tutorial]()

Cite Avalanche
----------------
If you used Avalanche in your research project, please remember to cite our white paper. 
This will help us make Avalanche better known in the machine learning community, ultimately making a better tool for everyone:

```
@article{lomonaco2020,
   title = {Avalanche: an End-to-End Framework for Continual Learning Research},
   author = {...},
   journal = {Arxiv preprint arXiv:xxxx.xxxx},
   year = {2020}
}
```

Maintained by ContinualAI Lab
----------------

*Avalanche* is the flagship open-source collaborative project of ContinuaAI: a non profit research organziation and the largest open community on Continual Learning for AI. 

Do you have a question, do you want to report an issue or simply ask for a new feture? Check out the [Questions & Issues center](). Do you want to improve Avalanche yourself? Follow these simple rules on [How to Contribute]().

The Avalanche project is maintained by the collaborative research team [ContinualAI Lab](). We are always looking for new awesome members, so check out our official website if you want to learn more about us and our activities.

Learn more about the [Avalanche Team and all the people who made it great]()!
