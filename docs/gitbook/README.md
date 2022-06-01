---
description: Powered by ContinualAI
---

# Avalanche: an End-to-End Library for Continual Learning

![](../../.gitbook/assets/avalanche\_logo\_with\_clai.png)

**Avalanche** is an _End-to-End Continual Learning Library_ based on [**PyTorch**](https://pytorch.org), born within [**ContinualAI**](https://www.continualai.org) with the unique goal of providing a **shared** and **collaborative** open-source (MIT licensed) **codebase** for _fast prototyping_, _training_ and [_reproducible_ _evaluation_](https://github.com/ContinualAI/reproducible-continual-learning) of continual learning algorithms.

Avalanche can help _Continual Learning_ researchers and practitioners in several ways:

* _Write less code, prototype faster & reduce errors_
* _Improve reproducibility_
* _Improve modularity and reusability_
* _Increase code efficiency, scalability & portability_
* _Augment impact and usability of your research products_

The library is organized in five main modules:

* **`Benchmarks`**: This module maintains a uniform API for data handling: mostly generating a stream of data from one or more datasets. It contains all the major CL benchmarks (similar to what has been done for [torchvision](https://pytorch.org/docs/stable/torchvision/index.html)).
* **`Training`**: This module provides all the necessary utilities concerning model training. This includes simple and efficient ways of implement new _continual learning_ strategies as well as a set pre-implemented CL baselines and state-of-the-art algorithms you will be able to use for comparison!
* **`Evaluation`**: This modules provides all the utilities and metrics that can help evaluate a CL algorithm with respect to all the factors we believe to be important for a continually learning system.
* **`Models`**: In this module you'll be able to find several model architectures and pre-trained models that can be used for your continual learning experiment (similar to what has been done in [torchvision.models](https://pytorch.org/docs/stable/torchvision/index.html)).
* **`Logging`**: It includes advanced logging and plotting features, including native _stdout_, _file_ and [TensorBoard](https://www.tensorflow.org/tensorboard) support (How cool it is to have a complete, interactive dashboard, tracking your experiment metrics in real-time with a single line of code?)

_Avalanche_ the first experiment of a **End-to-end Library** for [reproducible continual learning](https://github.com/ContinualAI/reproducible-continual-learning) research & development where you can find _benchmarks_, _algorithms,_ _evaluation metrics_ and much more, **in the same place**.

Let's make it together 👫 a wonderful ride! 🎈

Check out _how your code changes_ when you start using _Avalanche_! 👇

{% tabs %}
{% tab title="With Avalanche" %}
```python
import torch
from torch.nn import CrossEntropyLoss
from torch.optim import SGD

from avalanche.benchmarks.classic import PermutedMNIST
from avalanche.training.plugins import EvaluationPlugin
from avalanche.evaluation.metrics import accuracy_metrics
from avalanche.models import SimpleMLP
from avalanche.training.strategies import Naive

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
eval_plugin = EvaluationPlugin(
    accuracy_metrics(minibatch=True, epoch=True, epoch_running=True, 
                     experience=True, stream=True))

# Continual learning strategy
cl_strategy = Naive(
    model, optimizer, criterion, train_mb_size=32, train_epochs=2, 
    eval_mb_size=32, evaluator=eval_plugin, device=device)

# train and test loop
results = []
for train_task in train_stream:
    cl_strategy.train(train_task, num_workers=4)
    results.append(cl_strategy.eval(test_stream))
```
{% endtab %}

{% tab title="Without Avalanche" %}
```python
import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss
from torch.optim import SGD
from torchvision import transforms
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor, RandomCrop
from torch.utils.data import DataLoader
import numpy as np
from copy import copy

# Config
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# model
class SimpleMLP(nn.Module):

    def __init__(self, num_classes=10, input_size=28*28):
        super(SimpleMLP, self).__init__()

        self.features = nn.Sequential(
            nn.Linear(input_size, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(),
        )
        self.classifier = nn.Linear(512, num_classes)
        self._input_size = input_size

    def forward(self, x):
        x = x.contiguous()
        x = x.view(x.size(0), self._input_size)
        x = self.features(x)
        x = self.classifier(x)
        return x
model = SimpleMLP(num_classes=10)

# CL Benchmark Creation
list_train_dataset = []
list_test_dataset = []
rng_permute = np.random.RandomState(0)
train_transform = transforms.Compose([
    RandomCrop(28, padding=4),
    ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])
test_transform = transforms.Compose([
    ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

# permutation transformation
class PixelsPermutation(object):
    def __init__(self, index_permutation):
        self.permutation = index_permutation

    def __call__(self, x):
        return x.view(-1)[self.permutation].view(1, 28, 28)

def get_permutation():
    return torch.from_numpy(rng_permute.permutation(784)).type(torch.int64)

# for every incremental step
permutations = []
for i in range(3):
    # choose a random permutation of the pixels in the image
    idx_permute = get_permutation()
    current_perm = PixelsPermutation(idx_permute)
    permutations.append(idx_permute)

    # add the permutation to the default dataset transformation
    train_transform_list = train_transform.transforms.copy()
    train_transform_list.append(current_perm)
    new_train_transform = transforms.Compose(train_transform_list)

    test_transform_list = test_transform.transforms.copy()
    test_transform_list.append(current_perm)
    new_test_transform = transforms.Compose(test_transform_list)

    # get the datasets with the constructed transformation
    permuted_train = MNIST(root='./data/mnist',
                           download=True, transform=new_train_transform)
    permuted_test = MNIST(root='./data/mnist',
                    train=False,
                    download=True, transform=new_test_transform)
    list_train_dataset.append(permuted_train)
    list_test_dataset.append(permuted_test)

# Train
optimizer = SGD(model.parameters(), lr=0.001, momentum=0.9)
criterion = CrossEntropyLoss()

for task_id, train_dataset in enumerate(list_train_dataset):

    train_data_loader = DataLoader(
        train_dataset, num_workers=4, batch_size=32)
    
    for ep in range(2):
        for iteration, (train_mb_x, train_mb_y) in enumerate(train_data_loader):
            optimizer.zero_grad()
            train_mb_x = train_mb_x.to(device)
            train_mb_y = train_mb_y.to(device)

            # Forward
            logits = model(train_mb_x)
            # Loss
            loss = criterion(logits, train_mb_y)
            # Backward
            loss.backward()
            # Update
            optimizer.step()

# Test
acc_results = []
for task_id, test_dataset in enumerate(list_test_dataset):
    
    test_data_loader = DataLoader(
        test_dataset, num_workers=4, batch_size=32)
    
    correct = 0
    for iteration, (test_mb_x, test_mb_y) in enumerate(test_data_loader):

        # Move mini-batch data to device
        test_mb_x = test_mb_x.to(device)
        test_mb_y = test_mb_y.to(device)

        # Forward
        test_logits = model(test_mb_x)

        # Loss
        test_loss = criterion(test_logits, test_mb_y)

        # compute acc
        correct += test_mb_y.eq(test_logits.argmax(dim=1)).sum().item()
    
    acc_results.append(correct / len(test_dataset))
```
{% endtab %}
{% endtabs %}

## 🚦 Getting Started

We know that learning a new tool _may be tough at first_. This is why we made _Avalanche_ as easy as possible to learn with a set of resources that will help you along the way.

For example, you may start with our _**5-minutes**_ **guide** that will let you acquire the basics about _Avalanche_ and how you can use it in your research project:

{% content-ref url="getting-started/learn-avalanche-in-5-minutes.md" %}
[learn-avalanche-in-5-minutes.md](getting-started/learn-avalanche-in-5-minutes.md)
{% endcontent-ref %}

We have also prepared for you a large set of _**examples & snippets**_ you can plug-in directly into your code and play with:

{% content-ref url="broken-reference/" %}
[broken-reference](broken-reference/)
{% endcontent-ref %}

Having completed these two sections, you will already feel with _superpowers_ ⚡, this is why we have also created an **in-depth tutorial** that will cover all the aspect of _Avalanche_ in details and make you a true _Continual Learner_! 👨‍🎓️

{% content-ref url="broken-reference/" %}
[broken-reference](broken-reference/)
{% endcontent-ref %}

## 📑 Cite Avalanche

If you used _Avalanche_ in your research project, please remember to cite our reference paper [**"Avalanche: an End-to-End Library for Continual Learning"**](https://arxiv.org/abs/2104.00405). This will help us make _Avalanche_ better known in the machine learning community, ultimately making a better tool for everyone:

```
@InProceedings{lomonaco2021avalanche,
    title={Avalanche: an End-to-End Library for Continual Learning},
    author={Vincenzo Lomonaco and Lorenzo Pellegrini and Andrea Cossu and Antonio Carta and Gabriele Graffieti and Tyler L. Hayes and Matthias De Lange and Marc Masana and Jary Pomponi and Gido van de Ven and Martin Mundt and Qi She and Keiland Cooper and Jeremy Forest and Eden Belouadah and Simone Calderara and German I. Parisi and Fabio Cuzzolin and Andreas Tolias and Simone Scardapane and Luca Antiga and Subutai Amhad and Adrian Popescu and Christopher Kanan and Joost van de Weijer and Tinne Tuytelaars and Davide Bacciu and Davide Maltoni},
    booktitle={Proceedings of IEEE Conference on Computer Vision and Pattern Recognition},
    series={2nd Continual Learning in Computer Vision Workshop},
    year={2021}
}
```

## 🗂️ Maintained by ContinualAI Lab

![](<../../.gitbook/assets/continualai\_lab\_logo (1).png>)

_Avalanche_ is the flagship open-source collaborative project of [**ContinualAI**](https://www.continualai.org/#home): _a non profit research organization and the largest open community on Continual Learning for AI._

Do you have a question, do you want to report an issue or simply ask for a new feature? Check out the [Questions & Issues](questions-and-issues/ask-your-question.md) center. Do you want to improve _Avalanche_ yourself? Follow these simple rules on [How to Contribute](https://app.gitbook.com/@continualai/s/avalanche/\~/drafts/-MMtZhFEUwjWE4nnEpIX/from-zero-to-hero-tutorial/6.-contribute-to-avalanche).

The _Avalanche_ project is maintained by the collaborative research team [_**ContinualAI Lab**_](https://www.continualai.org/lab/) _and used extensively by the Units_ of the [_**ContinualAI Research (CLAIR)**_](https://www.continualai.org/research/) consortium, a research network of the major continual learning stakeholders around the world.

We are always looking for new _awesome members_ willing to join the _ContinualAI Lab_, so check out our [official website](https://www.continualai.org/lab/) if you want to learn more about us and our activities, or [contact us](contacts-and-links/the-team.md#contacts).

Learn more about the [_**Avalanche team and all the people who made it great**_](contacts-and-links/the-team.md)_!_

![](https://contrib.rocks/image?repo=ContinualAI/avalanche)
