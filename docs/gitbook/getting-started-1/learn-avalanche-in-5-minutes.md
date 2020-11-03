---
description: A Short Guide for Researchers on the Run
---

# Learn Avalanche in 5 Minutes

_Avalanche_ is mostly about making the life of a continual learning researcher easier. 

> #### What are the **three pillars** of any respectful continual learning research project?

* **Benchmarks**: Machine learning researchers need multiple benchmarks with efficient data handling utils to design and prototype new algorithms. Quantitative results on ever-changing benchmarks has been one of the driving forces of _Deep Learning_.
* **Training**: ****Efficient implementation and training of continual learning algorithms; comparisons with other baselines and state-of-the-art methods become fundamental to asses the quality of an original algorithmic proposal.
* **Evaluation**: ****_Training_ utils and _Benchmarks_ are not enough alone to push continual learning research forward. Comprehensive and sound _evaluation protocols_ and _metrics_ need to be employed as well.

> #### _With Avalanche, you can find all these three fundamental pieces together and more, in a single and coherent codabase._

Indeed, these are the _three main modules_ offered in the _Avalanche_ package.

## 📦 How to Install Avalanche

_Avalanche_ can be installed as simply as:

```text
pip install git+https://vlomonaco:****@github.com/vlomonaco/avalanche.git
```

You can check if the installation was successful importing the package in Python with:

```text
import avalanche
avalanche.__version__
```

## 📚 Benchamarks

The benchamark module offers three main features: 

1. **Datasets**: a comprehensive list of Pytorch Datasets ready to use \(It includes all the _Torchvision_ Datasets and more!\).
2. **Classic Benchmarks**: a set of classic Continual Learning Benchmarks ready to be used.
3. **Generators**: a set of functions you can use to generate your own benchmark starting from any Pytorch Dataset!

### Datasets

Datasets can be imported in _Avalanche_ as simply as:

```python
# the ones already supported in TorchVision
from avalanche.benchmarks.datasets import MNIST, FashionMNIST, KMNIST, EMNIST, \
QMNIST, FakeData, CocoCaptions, CocoDetection, LSUN, ImageNet, CIFAR10, \
CIFAR100, STL10, SVHN, PhotoTour, SBU, Flickr8k, Flickr30k, VOCDetection, \
VOCSegmentation, Cityscapes, SBDataset, USPS, Kinetics400, HMDB51, UCF101, \
CelebA 

# ...and additional ones!
from avalanche.benchmarks.datasets import CORe50, TinyImagenet, CUB200

# ... or generic constructors
from avalanche.benchmarks.datasets import ImageFolder, DatasetFolder, FilelistDataset
```

Of course, you can use them as you would use any _Pythorch Dataset_. 

### **Classic Benchmarks**

_Continual Learning_ is all concerned with learning over a stream of data. _Avalache_ maintains a set of commonly used benchmarks build on top of one of multiple datasets, that simulate that stream.

```python
from avalanche.benchmarks.classic import CORe50, SplitTinyImageNet, \
SplitCIFAR10, SplitCIFAR100, SplitCIFAR110, SplitMNIST, RotatedMNIST, PermutedMNIST, \
SplitCUB200

# creating PermutedMNIST with
perm_mnist = PermutedMNIST(
    incremental_steps=3,
    seed=1234,
)

for step in perm_mnist:
    print("Start of task ", step.current_task)
    print('Classes in this task:', step.classes_in_this_task)

    # Here's what you can do with the NIStepInfo object
    current_training_set = step.current_training_set()
    training_dataset, t = current_training_set
    print('Task {} batch {} -> train'.format(t, step.current_task))
    print('This task contains', len(training_dataset), 'patterns')

    complete_test_set = step.complete_test_sets()
```

### Benchmarks Generators

What if we want to create a new benchmark that is not present in the "_Classic_" ones? Well, in that case _Avalanche_ offer a number of utilities that you can use to create your own benchmark with maximum flexibilty!

The _specific_ scenario generators are useful when starting from one or multiple pytorch datasets you want to create a "**New Instances**" or "**New Classes**" benchmark:

```python
from avalanche.benchmarks.generators import NIScenario, NCScenario

ni_scenario = NIScenario(
    mnist_train, mnist_test, n_steps=10, shuffle=True, seed=1234,
    balance_steps=True
)

nc_scenario = NCScenario(
    mnist_train, mnist_test, n_steps=10, shuffle=True, seed=1234,
    task_labels=False
)
```

Finally, if you cannot create your ideal benchmark since it does not fit well in the aforementioned SIT-NI, SIT-NC or MT-NC scenarios, you can always use our **generic generators**:

* **FilelistScenario**
* **DatasetScenario**
* **TensorScenario**

```python
from avalanche.benchmarks.generators import FilelistScenario, DatasetScenario, \
                                            TensorScenario
```

You can read more about how to use them the full _Benchmarks_ module tutorial:

{% page-ref page="../from-zero-to-hero-tutorial/2.-benchmarks.md" %}

## 💪Training

The `training` module in _Avalanche_ is build heavily on modularity and its main goals are two:

1. provide a set of standard **continual learning baselines** that can be easily run for comparison; 
2. provide the necessary utilities to **create and run your own strategy** as efficiently and easy as possible.

### Strategies

If you want to compare your strategy with other classic continual learning algorithms or baselines, in _Avalanche_ this is as simply as instantiate an object. Please note that at the moment only the _Naive, Cumulative_ and _GDumb_ baselines are supported.

```python
from avalanche.extras.models import SimpleMLP
from avalanche.strategies import Naive, Cumulative, GDumb

model = SimpleMLP(num_classes=10)
cl_strategy = Naive(
    model, 'classifier', SGD(model.parameters(), lr=0.001, momentum=0.9),
    CrossEntropyLoss(), train_mb_size=100, train_epochs=4, test_mb_size=100
)
```

### Create your own Strategy

The simplest way to build your own strategy is to create a python class that implements the main `train` and `test` methods. 

Let's define our Continual Learning algorithm "_MyStrategy_" as a simple python class:

```python
class MyStrategy():
    """My Basic Strategy"""

    def __init__(self, model, optimizer, criterion):
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
    
    def train(self, step_info):
        # here you can implement your own training loop for each step (i.e. 
        # batch or task).

        train_dataset, t = step_info.current_training_set()
        train_data_loader = DataLoader(
            train_dataset, num_workers=4, batch_size=128
        )

        for epoch in range(1):
            for mb in train_data_loader:
                # you magin here...
                pass

    def test(self, step_info):
        # here you can implement your own test loop for each step (i.e. 
        # batch or task).

        test_dataset, t = step_info.step_specific_test_set(0)
        test_data_loader = DataLoader(
            test_dataset, num_workers=4, batch_size=128
        )

        # test here
```

Then, we can use our strategy as we would do for the pre-implemented ones:

```python
# CREATE THE STRATEGY 
cl_strategy = MyStrategy(
    model, SGD(model.parameters(), lr=0.001, momentum=0.9),
    CrossEntropyLoss())

# TRAINING LOOP
print('Starting experiment...')
results = []
batch_info: NCBatchInfo
for batch_info in nc_scenario:
    print("Start of step ", batch_info.current_step)

    cl_strategy.train(batch_info)
    print('Training completed')

    print('Computing accuracy on the whole test set')
    results.append(cl_strategy.test(batch_info))
```

While this is the easiest possible way to add your own stratey to _Avalanche_ we support more sophisticated modalities that let you write **more neat and reusable** **code**, inheriting functionality from a parent classes and using **pre-implemented plugins** shared with other strategies.

Check out more details about what Avalanche can offer in this module following the "Training" chapter of the "From Zero to Hero" tutorial:

{% page-ref page="../from-zero-to-hero-tutorial/3.-training.md" %}

## 📈 Evaluation

The `evaluation` module is quite straightforward at the moment and offers 3 main submodules:

* **Metrics**: a set of classes \(one for metric\) which implement the main continual learning matrics computation like _accuracy_, _forgetting_, _memory usage_, _running times_, etc.
* **TensorboardLogging**: offers a main class to handles Tensorboard logging configuration with nice plots updated on-the-fly to control and "_babysit_" your experiment easily.
* **Evaluation Protocols**: this module provides a single point of entry to the evaluation methodology configuration and in charge of hangling the metrics computation, tensorboard or avanced console logging.

### Metrics

In _Avalanche_ we offer at the moment a number of pre-implemented metrics you can use for your own experiments. We made sure to include all the major accuracy-based matrics but also the ones related to computation and memory.

The metrics already available \(soon to be expanded\) are:

* **Accuracy** \(`ACC`\): Accuracy over time \(Total average or per task\).
* **Catastrophic Forgetting** \(`CF`\): Forgetting as defined in \(Lopez-paz 2017\).
* **RAM Usage** \(`RAMU`\): RAM usage by the process over time.
* **Confusion Matrix** \(`CM`\): Confusion matrix over time.
* **CPU Usage** \(`CPUUsage`\): CPU usage by the process over time.
* **GPU Usage** \(`GPUUsage`\): GPU usage by the process over time.
* **Disk Usage** \(`DiskUsage`\): Disk usage by the process over time.
* **Time Usage** \(`TimeUsage`\): Running time of the python process.

```python
from avalanche.evaluation.metrics import ACC, CF, RAMU, CM, CPUUsage, GPUUsage,\
 DiskUsage, TimeUsage
```

While each metric can be directly managed within and _Evaluation Protocol_ \(see next section\), we can use each metric directly, being them simply python classes. For example the accuracy metric works as follows:

```python
real_y = np.asarray([1, 2])
predicted_y = np.asarray([1, 0])
acc_metric = ACC()
acc, acc_x_class = acc_metric.compute([real_y], [predicted_y])

print("Average Accuracy:", acc)
print("Accuracy per class:", acc_x_class)
```

### Tensorboard

Tensorboard has consolidated its position as the go-to **visualization toolkit** for deep learning experiments for both pytorch and tensorflow. In _Avalanche_ we decided the different the different metrics \(and their eventual change over time\) using the standard Pytorch version.

At the moment we implemented just the `TensorboardLogging` object that can be used to specify Tensorboard behavious with eventual customizations.

```python
from avalanche.evaluation.tensorboard import TensorboardLogging

# a simple example of tensorboard instantiation
tb_logging = TensorboardLogging(tb_logdir=".")
```

### Evaluation Protocol

The **Evaluation Protocol**, is the object in charge of configuring and controlling the evaluation procedure. This object can be passed to a Strategy that will automatically call its main functionalities when the **training and testing flows** are activated.

```python
from avalanche.evaluation.metrics import ACC, CF, RAMU, CM
from avalanche.extras.models import SimpleMLP
from avalanche.training.strategies import Naive
from avalanche.evaluation import EvalProtocol

# load the model with PyTorch for example
model = SimpleMLP()

# Eval Protocol
evalp = EvalProtocol(
    metrics=[ACC(), CF(), RAMU(), CM()], tb_logdir='.'
)

# adding the CL strategy
clmodel = Naive(model, eval_protocol=evalp)
```

For more details about the evaluation module, check out the extended guide in the "Evaluation" chapter of the "_From Zero to Hero_" Avalanche tutorial:

{% page-ref page="../from-zero-to-hero-tutorial/4.-evaluation.md" %}

## Putting all Together

You've learned how to install _Avalanche,_ how to create benchmarks that can suit your needs, how you can create your own continual learning algorithm and how you can evaluate its performance.

Here we show how you can use all these modules together to **design your experiments** as quantitative supporting evidence for your research project or paper.

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

## 🤝 Run it on Google Colab

You can run _this chapter_ and play with it on Google Colaboratory:

{% hint style="danger" %}
TODO: add link here.
{% endhint %}



