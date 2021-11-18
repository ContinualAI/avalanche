---
description: "Logging... logging everywhere! \U0001F52E"
---

# Loggers

Welcome to the _"Logging"_ tutorial of the _"From Zero to Hero"_ series. In this part we will present the functionalities offered by the _Avalanche_ `logging` module.


```python
!pip install git+https://github.com/ContinualAI/avalanche.git
```

### 📑 The Logging Module

In the previous tutorial we have learned how to evaluate a continual learning algorithm in _Avalanche_, through different metrics that can be used _off-the-shelf_ via the _Evaluation Plugin_ or stand-alone. However, computing metrics and collecting results, may not be enough at times.

While running complex experiments with long _waiting times_, **logging** results over-time is fundamental to "_babysit_" your experiments in real-time, or even understand what went wrong in the aftermath.

This is why in Avalanche we decided to put a strong emphasis on logging and **provide a number of loggers** that can be used with any set of metrics!

### Loggers

_Avalanche_ at the moment supports three main Loggers:

* **InteractiveLogger**: This logger provides a nice progress bar and displays real-time metrics results in an interactive way \(meant for `stdout`\).
* **TextLogger**: This logger, mostly intended for file logging, is the plain text version of the `InteractiveLogger`. Keep in mind that it may be very verbose.
* **TensorboardLogger**: It logs all the metrics on [Tensorboard](https://www.tensorflow.org/tensorboard) in real-time. Perfect for real-time plotting.

In order to keep track of when each metric value has been logged, we leverage a `global counter`. You can see the `global counter` reported in the x axis of the logged plots.

The `global counter` is an ever-increasing value which starts from 0 and it is increased by one each time a training or evaluation iteration is performed (i.e. after each training or evaluation minibatch).
The `global counter` is updated automatically by the strategy. It should be reset by creating a new instance of the strategy.

#### How to use Them


```python
from torch.optim import SGD
from torch.nn import CrossEntropyLoss
from avalanche.benchmarks.classic import SplitMNIST
from avalanche.evaluation.metrics import forgetting_metrics, \
accuracy_metrics, loss_metrics, timing_metrics, cpu_usage_metrics, \
confusion_matrix_metrics, disk_usage_metrics
from avalanche.models import SimpleMLP
from avalanche.logging import InteractiveLogger, TextLogger, TensorboardLogger
from avalanche.training.plugins import EvaluationPlugin
from avalanche.training.strategies import Naive

scenario = SplitMNIST(n_experiences=5)

# MODEL CREATION
model = SimpleMLP(num_classes=scenario.n_classes)

# DEFINE THE EVALUATION PLUGIN and LOGGERS
# The evaluation plugin manages the metrics computation.
# It takes as argument a list of metrics, collectes their results and returns
# them to the strategy it is attached to.

# log to Tensorboard
tb_logger = TensorboardLogger()

# log to text file
text_logger = TextLogger(open('log.txt', 'a'))

# print to stdout
interactive_logger = InteractiveLogger()

eval_plugin = EvaluationPlugin(
    accuracy_metrics(minibatch=True, epoch=True, experience=True, stream=True),
    loss_metrics(minibatch=True, epoch=True, experience=True, stream=True),
    timing_metrics(epoch=True, epoch_running=True),
    cpu_usage_metrics(experience=True),
    forgetting_metrics(experience=True, stream=True),
    confusion_matrix_metrics(num_classes=scenario.n_classes, save_image=False,
                             stream=True),
    disk_usage_metrics(minibatch=True, epoch=True, experience=True, stream=True),
    loggers=[interactive_logger, text_logger, tb_logger]
)

# CREATE THE STRATEGY INSTANCE (NAIVE)
cl_strategy = Naive(
    model, SGD(model.parameters(), lr=0.001, momentum=0.9),
    CrossEntropyLoss(), train_mb_size=500, train_epochs=1, eval_mb_size=100,
    evaluator=eval_plugin)

# TRAINING LOOP
print('Starting experiment...')
results = []
for experience in scenario.train_stream:
    print("Start of experience: ", experience.current_experience)
    print("Current Classes: ", experience.classes_in_this_experience)

    # train returns a dictionary which contains all the metric values
    res = cl_strategy.train(experience)
    print('Training completed')

    print('Computing accuracy on the whole test set')
    # test also returns a dictionary which contains all the metric values
    results.append(cl_strategy.eval(scenario.test_stream))
```

### Create your Logger

If the available loggers are not sufficient to suit your needs, you can always implement a custom logger by specializing the behaviors of the `StrategyLogger` base class.

This completes the "_Logging_" tutorial for the "_From Zero to Hero_" series. We hope you enjoyed it!

## 🤝 Run it on Google Colab

You can run _this chapter_ and play with it on Google Colaboratory: [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ContinualAI/avalanche/blob/master/notebooks/from-zero-to-hero-tutorial/06_loggers.ipynb)
