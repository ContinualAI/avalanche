<div align="center">
    
# Avalanche: an End-to-End Library for Continual Learning
**[Avalanche Website](https://avalanche.continualai.org)** | **[Getting Started](https://avalanche.continualai.org/getting-started)** | **[Examples](https://avalanche.continualai.org/examples)** | **[Tutorial](https://avalanche.continualai.org/from-zero-to-hero-tutorial)** | **[API Doc](https://avalanche-api.continualai.org)** | **[Paper](https://arxiv.org/abs/2104.00405)** | **[Twitter](https://twitter.com/AvalancheLib)**

[![unit test](https://github.com/ContinualAI/avalanche/actions/workflows/unit-test.yml/badge.svg)](https://github.com/ContinualAI/avalanche/actions/workflows/unit-test.yml)
[![syntax checking](https://github.com/ContinualAI/avalanche/actions/workflows/syntax.yml/badge.svg)](https://github.com/ContinualAI/avalanche/actions/workflows/syntax.yml)
[![PEP8 checking](https://github.com/ContinualAI/avalanche/actions/workflows/pep8.yml/badge.svg)](https://github.com/ContinualAI/avalanche/actions/workflows/pep8.yml)
[![docstring coverage](https://github.com/ContinualAI/avalanche-report/blob/main/badge/interrogate-badge.svg)](https://github.com/ContinualAI/avalanche-report/blob/main/docstring_coverage/documentation-coverage.txt)
[![Coverage Status](https://coveralls.io/repos/github/ContinualAI/avalanche/badge.svg)](https://coveralls.io/github/ContinualAI/avalanche)
</div>

<p align="center">
    <img src="https://www.dropbox.com/s/90thp7at72sh9tj/avalanche_logo_with_clai.png?raw=1"/>
</p>



**Avalanche** is an *end-to-end Continual Learning library* based on **Pytorch**, born within [ContinualAI](https://www.continualai.org/) with the unique goal of providing a shared and collaborative 
open-source (MIT licensed) codebase for fast prototyping, training and reproducible evaluation of continual learning algorithms. 
Avalanche can help Continual Learning researchers in several ways:

- *Write less code, prototype faster & reduce errors*
- *Improve reproducibility*
- *Improve modularity and reusability*
- *Increase code efficiency, scalability & portability*
- *Augment impact and usability of your research products*

The library is organized into four main modules:

- [Benchmarks](avalanche/benchmarks): This module maintains a uniform API for data handling: mostly generating a stream of data from one or more datasets. It contains all the major CL benchmarks (similar to what has been done for torchvision).
- [Training](avalanche/training): This module provides all the necessary utilities concerning model training. This includes simple and efficient ways of implement new continual learning strategies as well as a set of pre-implemented CL baselines and state-of-the-art algorithms you will be able to use for comparison!
- [Evaluation](avalanche/evaluation): This module provides all the utilities and metrics that can help evaluate a CL algorithm with respect to all the factors we believe to be important for a continually learning system. It also includes advanced logging and plotting features, including native Tensorboard support.
- [Models](avalanche/models): This module provides utilities to implement model expansion and task-aware models, along with a set of pre-trained models and popular architectures that can be used for your continual learning experiment (similar to what has been done in torchvision.models).
- [Logging](avalanche/logging): It includes advanced logging and plotting features, including native stdout, file and TensorBoard support (How cool it is to have a complete, interactive dashboard, tracking your experiment metrics in real-time with a single line of code?)

_Avalanche_ the first experiment of an **End-to-end Library** for reproducible continual learning research & development where you can find benchmarks, algorithms, evaluation metrics and much more, in the same place.

Let's make it together :people_holding_hands: a wonderful ride! :balloon:

Check out below how you can start using Avalanche! :point_down:

Quick Example
----------------

```python
import torch
from torch.nn import CrossEntropyLoss
from torch.optim import SGD

from avalanche.benchmarks.classic import PermutedMNIST
from avalanche.models import SimpleMLP
from avalanche.training import Naive


# Config
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# model
model = SimpleMLP(num_classes=10)

# CL Benchmark Creation
perm_mnist = PermutedMNIST(n_experiences=3)
train_stream = perm_mnist.train_stream
test_stream = perm_mnist.test_stream

# Prepare for training & testing
optimizer = SGD(model.parameters(), lr=0.001, momentum=0.9)
criterion = CrossEntropyLoss()

# Continual learning strategy
cl_strategy = Naive(
    model, optimizer, criterion, train_mb_size=32, train_epochs=2,
    eval_mb_size=32, device=device)

# train and test loop over the stream of experiences
results = []
for train_exp in train_stream:
    cl_strategy.train(train_exp)
    results.append(cl_strategy.eval(test_stream))
```

Current Release
----------------

Avalanche is a framework in constant development. Thanks to the support of the [ContinualAI](https://www.continualai.org/) community and its active members we are quickly extending its features and improve its usability based on the demands of our research community!

A the moment, Avalanche is in [**Beta (v0.2.0)**](https://github.com/ContinualAI/avalanche/releases/tag/v0.2.0). We support [several *Benchmarks*, *Strategies* and *Metrics*](https://avalanche.continualai.org/getting-started/alpha-version), that make it, we believe, the best tool out there for your continual learning research! 💪

**You can install Avalanche by running `pip install avalanche-lib`.**  
This will install the core Avalanche package. You can install Avalanche with extra packages to enable more functionalities.  
Look [here](https://avalanche.continualai.org/getting-started/how-to-install) for a more complete guide on the different ways available to install Avalanche.

Getting Started
----------------

We know that learning a new tool may be tough at first. This is why we made Avalanche as easy as possible to learn with a set of resources that will help you along the way.
For example, you may start with our 5-minutes guide that will let you acquire the basics about Avalanche and how you can use it in your research project:

- [Getting Started Guide](https://avalanche.continualai.org/getting-started)

We have also prepared for you a large set of examples & snippets you can plug-in directly into your code and play with:

- [Avalanche Examples](https://avalanche.continualai.org/examples)

Having completed these two sections, you will already feel with superpowers ⚡, this is why we have also created an in-depth tutorial that will cover all the aspects of Avalanche in 
detail and make you a true Continual Learner! :woman_student:

- [From Zero to Hero Tutorial](https://avalanche.continualai.org/from-zero-to-hero-tutorial)

Cite Avalanche
----------------
If you used Avalanche in your research project, please remember to cite our reference paper published at the [CLVision @ CVPR2021](https://sites.google.com/view/clvision2021/overview?authuser=0) workshop: ["Avalanche: an End-to-End Library for Continual Learning"](https://arxiv.org/abs/2104.00405). 
This will help us make Avalanche better known in the machine learning community, ultimately making a better tool for everyone:

```
@InProceedings{lomonaco2021avalanche,
    title={Avalanche: an End-to-End Library for Continual Learning},
    author={Vincenzo Lomonaco and Lorenzo Pellegrini and Andrea Cossu and Antonio Carta and Gabriele Graffieti and Tyler L. Hayes and Matthias De Lange and Marc Masana and Jary Pomponi and Gido van de Ven and Martin Mundt and Qi She and Keiland Cooper and Jeremy Forest and Eden Belouadah and Simone Calderara and German I. Parisi and Fabio Cuzzolin and Andreas Tolias and Simone Scardapane and Luca Antiga and Subutai Amhad and Adrian Popescu and Christopher Kanan and Joost van de Weijer and Tinne Tuytelaars and Davide Bacciu and Davide Maltoni},
    booktitle={Proceedings of IEEE Conference on Computer Vision and Pattern Recognition},
    series={2nd Continual Learning in Computer Vision Workshop},
    year={2021}
}
```

Maintained by ContinualAI Lab
----------------

*Avalanche* is the flagship open-source collaborative project of [ContinualAI](https://www.continualai.org/): a non-profit research organization and the largest open community on Continual Learning for AI.

Do you have a question, do you want to report an issue or simply ask for a new feature? Check out the [Questions & Issues](https://avalanche.continualai.org/questions-and-issues/ask-your-question) center. Do you want to improve Avalanche yourself? Follow these simple rules on [How to Contribute](https://app.gitbook.com/@continualai/s/avalanche/~/drafts/-MMtZhFEUwjWE4nnEpIX/from-zero-to-hero-tutorial/6.-contribute-to-avalanche).

The Avalanche project is maintained by the collaborative research team [ContinualAI Lab](https://www.continualai.org/lab/) and used extensively by the Units of the [ContinualAI Research (CLAIR)](https://www.continualai.org/research/) consortium, a research network of the major continual learning stakeholders around the world.

We are always looking for new awesome members willing to join the ContinualAI Lab, so check out our [official website](https://www.continualai.org/lab/) if you want to learn more about us and our activities, or [contact us](https://avalanche.continualai.org/contacts-and-links/the-team#contacts).

Learn more about the [Avalanche team and all the people who made it great](https://avalanche.continualai.org/contacts-and-links/the-team)!

<br>
<p align="left">
<a href="https://github.com/ContinualAI/avalanche/graphs/contributors">
 <img width="700" src="https://contrib.rocks/image?repo=ContinualAI/avalanche" />
</a>
</p>
