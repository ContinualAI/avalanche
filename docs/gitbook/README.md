---
description: Powered by ContinualAI
---

# Avalanche: an End-to-End Framework for Continual Learning Research

![](.gitbook/assets/avalanche_logo_with_clai.png)

**Avalanche** is an _end-to-end Continual Learning research_ framework based on [**Pytorch**](https://pytorch.org/), born within [**ContinualAI**](https://www.continualai.org/) with the unique goal of providing a **shared** and **collaborative** open-source \(MIT licensed\) **codebase** for _fast prototyping_, _training_ and _reproducible_ _evaluation_ of continual learning algorithms.

Avalanche can help _Continual Learning_ researchers in several ways:

* _Write less code, prototype faster & reduce errors_
* _Improve reproducibility_
* _Improve modularity and reusability_
* _Increase code efficiency, scalability & portability_
* _Augment impact and usability of your research products_

The framework is organized in five main modules:

* **`Benchmarks`**: This module maintains a uniform API for data handling: mostly generating a stream of data from one or more datasets. It contains all the major CL benchmarks \(similar to what has been done for [torchvision](https://pytorch.org/docs/stable/torchvision/index.html)\).
* **`Training`**: This module provides all the necessary utilities concerning model training. This includes simple and efficient ways of implement new _continual learning_ strategies as well as a set pre-implemented CL baselines and state-of-the-art algorithms you will be able to use for comparison!
* **`Evaluation`**: This modules provides all the utilities and metrics that can help evaluate a CL algorithm with respect to all the factors we believe to be important for a continually learning system. 
* **`Models`**: In this module you'll be able to find several model architectures and pre-trained models that can be used for your continual learning experiment \(similar to what has been done in [torchvision.models](https://pytorch.org/docs/stable/torchvision/index.html)\). 
* **`Logging`**: It includes advanced logging and plotting features, including native _stdout_, _file_ and [Tensorboard](https://www.tensorflow.org/tensorboard) support \(How cool it is to have a complete, interactive dashboard, tracking your experiment metrics in real-time with a single line of code?\)

_Avalanche_ is one of the first experiments of a **End-to-end Research Framework** for reproducible _machine learning_ research where you can find _benchmarks_, _algorithms,_ _evaluation protocols_ and much more_,_ **in the same place**.

Let's make it together 👫 a wonderful ride! 🎈

Check out _how you code changes_ when you start using _Avalanche_! 👇

{% tabs %}
{% tab title="With Avalanche" %}
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
perm_mnist = PermutedMNIST(n_steps=3)
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
{% endtab %}

{% tab title="Without Avalanche" %}
```python
import torch
from torch.nn import CrossEntropyLoss
from torch.optim import SGD
from torchvision import transforms
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor, RandomCrop

# Config
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
n_tasks = 5
n_classes = 10
train_ep = 2
mb_size = 32

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
model = SimpleMLP(num_classes=n_classes)

# CL Benchmark Creation
list_train_dataset = []
list_test_dataset = []
rng_permute = np.random.RandomState(seed)
train_transform = transforms.Compose([
    RandomCrop(28, padding=4),
    ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])
test_transform = transforms.Compose([
    ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

# for every incremental step
for _ in range(n_tasks):
    # choose a random permutation of the pixels in the image
    idx_permute = torch.from_numpy(rng_permute.permutation(784)).type(torch.int64)

    # add the permutation to the default dataset transformation
    train_transform_list = train_transform.transforms.copy()
    train_transform_list.append(
        transforms.Lambda(lambda x: x.view(-1)[idx_permute].view(1, 28, 28))
    )
    new_train_transform = transforms.Compose(train_transform_list)

    test_transform_list = test_transform.transforms.copy()
    test_transform_list.append(
        transforms.Lambda(lambda x: x.view(-1)[idx_permute].view(1, 28, 28))
    )
    new_test_transform = transforms.Compose(test_transform_list)

    # get the datasets with the constructed transformation
    permuted_train = MNIST(root='./data/mnist',
                           download=True, transform=train_transformation)
    permuted_test = MNIST(root='./data/mnist',
                    train=False,
                    download=True, transform=test_transformation)
    list_train_dataset.append(permuted_train)
    list_test_dataset.append(permuted_test)

# Train
optimizer = SGD(model.parameters(), lr=0.001, momentum=0.9)
criterion = CrossEntropyLoss()

for task_id, train_dataset in enumerate(list_train_dataset):

    train_data_loader = DataLoader(
        train_dataset, num_workers=num_workers, batch_size=train_mb_size)

    for ep in range(train_ep):
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

    train_data_loader = DataLoader(
        train_dataset, num_workers=num_workers, batch_size=train_mb_size)

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
        correct += (test_mb_y.eq(test_logits.long())).sum()

    acc_results.append(len(test_dataset)/correct)
```
{% endtab %}
{% endtabs %}

## 🚦 Getting Started

We know that learning a new tool _may be tough at first_. This is why we made _Avalanche_ as easy as possible to learn with a set of resources that will help you along the way.

For example, you may start with our _**5-minutes**_ **guide** that will let you acquire the basics about _Avalanche_ and how you can use it in your research project:

We have also prepared for you a large set of _**examples & snippets**_ you can plug-in directly into your code and play with:

Having completed these two sections, you will already feel with _superpowers_ ⚡, this is why we have also created an **in-depth tutorial** that will cover all the aspect of _Avalanche_ in details and make you a true _Continual Learner_! 👨‍🎓️

## 📑 Cite Avalanche

If you used _Avalanche_ in your research project, please remember to cite our white paper. This will help us make _Avalanche_ better known in the machine learning community, ultimately making a better tool for everyone:

```text
@article{lomonaco2020,
   title = {Avalanche: an End-to-End Framework for Continual Learning Research},
   author = {...},
   journal = {Arxiv preprint arXiv:xxxx.xxxx},
   year = {2020}
}
```

## 🗂️ Maintained by ContinualAI Lab

![](.gitbook/assets/continualai_lab_logo%20%281%29.png)

_Avalanche_ is the flagship open-source collaborative project of [**ContinuaAI**](https://www.continualai.org/#home): _a non profit research organization and the largest open community on Continual Learning for AI._ 

Do you have a question, do you want to report an issue or simply ask for a new feature? Check out the [Questions & Issues](questions-and-issues/ask-your-question.md) center. Do you want to improve _Avalanche_ yourself? Follow these simple rules on [How to Contribute](https://app.gitbook.com/@continualai/s/avalanche/~/drafts/-MMtZhFEUwjWE4nnEpIX/from-zero-to-hero-tutorial/6.-contribute-to-avalanche).

The _Avalanche_ project is maintained by the collaborative research team [_**ContinualAI Lab**_](https://www.continualai.org/lab/) _****_and used extensively by the _Units_ of the _****_[_**ContinualAI Research \(CAIR\)**_](https://www.continualai.org/research/) consortium, a research network of the major continual learning stakeholders around the world_._ 

We are always looking for new _awesome members_ willing to join the _ContinualAI Lab_, so check out our [official website](https://www.continualai.org/lab/) if you want to learn more about us and our activities, or [contact us](contacts-and-links/the-team.md#contacts). 

Learn more about the [_**Avalanche team and all the people who made it great**_](contacts-and-links/the-team.md)_!_

