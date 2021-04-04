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
* **WandBLogger**: It logs all the metrics along with Confusion Matrix on a centralized dashboard on [Weights & Biases](https://wandb.ai/site) in real-time. Best for interactive logging, dynamic visualizations and easy analysis. 

#### How to use Them

```python
from torch.optim import SGD
from torch.nn import CrossEntropyLoss
from avalanche.benchmarks.classic import SplitMNIST
from avalanche.evaluation.metrics import ExperienceForgetting, \
accuracy_metrics, loss_metrics, timing_metrics, cpu_usage_metrics, \
StreamConfusionMatrix, disk_usage_metrics
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

# log to Weights & Biases
# For all initialization parameters available for WandBLogger, 
# please visit the WandB docs at https://docs.wandb.ai/ref/run/init
init_kwargs = {"project":' Avalanche', "run": 'continual_learning 1'}
wb_logger = WandBLogger(init_kwargs = init_kwargs)

# log to text file
text_logger = TextLogger(open('log.txt', 'a'))

# print to stdout
interactive_logger = InteractiveLogger()

eval_plugin = EvaluationPlugin(
    accuracy_metrics(minibatch=True, epoch=True, experience=True, stream=True),
    loss_metrics(minibatch=True, epoch=True, experience=True, stream=True),
    timing_metrics(epoch=True, epoch_running=True),
    cpu_usage_metrics(experience=True),
    StreamConfusionMatrix(num_classes=scenario.n_classes, save_image=False),
    disk_usage_metrics(minibatch=True, epoch=True, experience=True, stream=True),
    loggers=[interactive_logger, text_logger, tb_logger, wb_logger]
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

You can run _this chapter_ and play with it on Google Colaboratory: [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ContinualAI/colab/blob/master/notebooks/avalanche/loggers.ipynb)

