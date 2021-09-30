---
description: Automatic Evaluation with Pre-implemented Metrics
---

# Evaluation

Welcome to the "_Evaluation_" tutorial of the "_From Zero to Hero_" series. In this part we will present the functionalities offered by the `evaluation` module.

```python
!pip install git+https://github.com/ContinualAI/avalanche.git
```

## 📈 The Evaluation Module

The `evaluation` module is quite straightforward: it offers all the basic functionalities to evaluate and keep track of a continual learning experiment.

This is mostly done through the **Metrics**: a set of classes which implement the main continual learning metrics computation like A_ccuracy_, F_orgetting_, M_emory Usage_, R_unning Times_, etc. At the moment, in _Avalanche_ we offer a number of pre-implemented metrics you can use for your own experiments. We made sure to include all the major accuracy-based metrics but also the ones related to computation and memory.

Each metric comes with a standalone class and a set of plugin classes aimed at emitting metric values on specific moments during training and evaluation.

### Standalone metric

As an example, the standalone `Accuracy` class can be used to monitor the average accuracy over a stream of `<input,target>` pairs. The class provides an `update` method to update the current average accuracy, a `result` method to print the current average accuracy and a `reset` method to set the current average accuracy to zero. The call to `result`does not change the metric state.

```python
import torch
from avalanche.evaluation.metrics import Accuracy

# create an instance of the standalone Accuracy metric
# initial accuracy is 0
acc_metric = Accuracy()
print("Initial Accuracy: ", acc_metric.result()) #  output 0

# two consecutive metric updates
real_y = torch.tensor([1, 2]).long()
predicted_y = torch.tensor([1, 0]).float()
acc_metric.update(real_y, predicted_y)
acc = acc_metric.result()
print("Average Accuracy: ", acc) # output 0.5
predicted_y = torch.tensor([1,2]).float()
acc_metric.update(real_y, predicted_y)
acc = acc_metric.result()
print("Average Accuracy: ", acc) # output 0.75

# reset accuracy to 0
acc_metric.reset()
print("After reset: ", acc_metric.result()) # output 0
```

### Plugin metric

If you want to integrate the available metrics automatically in the training and evaluation flow, you can use plugin metrics, like `EpochAccuracy` which logs the accuracy after each training epoch, or `ExperienceAccuracy` which logs the accuracy after each evaluation experience. Each of these metrics emits a **curve** composed by its values at different points in time \(e.g. on different training epochs\). In order to simplify the use of these metrics, we provided utility functions with which you can create different plugin metrics in one shot. The results of these functions can be passed as parameters directly to the `EvaluationPlugin`\(see below\).

{% hint style="info" %}
We recommend to use the helper functions when creating plugin metrics.
{% endhint %}

```python
from avalanche.evaluation.metrics import accuracy_metrics, \
    loss_metrics, forgetting_metrics, bwt_metrics,\
    confusion_matrix_metrics, cpu_usage_metrics, \
    disk_usage_metrics, gpu_usage_metrics, MAC_metrics, \
    ram_usage_metrics, timing_metrics

# you may pass the result to the EvaluationPlugin
metrics = accuracy_metrics(epoch=True, experience=True)
```

The metrics currently available in the current _Avalanche_ release are:

```python
from avalanche.evaluation.metrics import Accuracy, \
MinibatchAccuracy, EpochAccuracy, RunningEpochAccuracy, \
ExperienceAccuracy, StreamAccuracy, \
Loss, MinibatchLoss, EpochLoss, RunningEpochLoss, \
ExperienceLoss, StreamLoss, \
Forgetting, ExperienceForgetting, StreamForgetting, \
BWT, ExperienceBWT, StreamBWT, \
ConfusionMatrix, StreamConfusionMatrix, WandBStreamConfusionMatrix, \
CPUUsage, MinibatchCPUUsage, EpochCPUUsage, RunningEpochCPUUsage, \
ExperienceCPUUsage, StreamCPUUsage, \
DiskUsage, MinibatchDiskUsage, EpochDiskUsage, \
ExperienceDiskUsage, StreamDiskUsage, \
MaxGPU, MinibatchMaxGPU, EpochMaxGPU, ExperienceMaxGPU, \
StreamMaxGPU,\
MAC, MinibatchMAC, EpochMAC, ExperienceMAC, \
MaxRAM, MinibatchMaxRAM, EpochMaxRAM, ExperienceMaxRAM, \
StreamMaxRAM, \
ElapsedTime, MinibatchTime, EpochTime, RunningEpochTime, \
ExperienceTime, StreamTime
```

## 📐Evaluation Plugin

The **Evaluation Plugin**, is the object in charge of configuring and controlling the evaluation procedure. This object can be passed to a Strategy as a "special" plugin through the evaluator attribute.

The Evaluation Plugin accepts as inputs the plugin metrics you want to track. In addition, you can add one or more loggers to print the metrics in different ways \(on file, on standard output, on Tensorboard...\).

```python
from torch.nn import CrossEntropyLoss
from torch.optim import SGD
from avalanche.benchmarks.classic import SplitMNIST
from avalanche.evaluation.metrics import forgetting_metrics, \
accuracy_metrics, loss_metrics, timing_metrics, cpu_usage_metrics, \
confusion_matrix_metrics, disk_usage_metrics
from avalanche.models import SimpleMLP
from avalanche.training.plugins import EvaluationPlugin
from avalanche.training.strategies import Naive

scenario = SplitMNIST(n_experiences=5)

# MODEL CREATION
model = SimpleMLP(num_classes=scenario.n_classes)

# DEFINE THE EVALUATION PLUGIN
# The evaluation plugin manages the metrics computation.
# It takes as argument a list of metrics, collectes their results and returns
# them to the strategy it is attached to.

eval_plugin = EvaluationPlugin(
    accuracy_metrics(minibatch=True, epoch=True, experience=True, stream=True),
    loss_metrics(minibatch=True, epoch=True, experience=True, stream=True),
    timing_metrics(epoch=True),
    forgetting_metrics(experience=True, stream=True),
    cpu_usage_metrics(experience=True),
    confusion_matrix_metrics(num_classes=scenario.n_classes, save_image=False, stream=True),
    disk_usage_metrics(minibatch=True, epoch=True, experience=True, stream=True)
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

## Implement your own metric

To implement a **standalone metric**, you have to subclass `Metric` class.

```python
from avalanche.evaluation import Metric


# a standalone metric implementation
class MyStandaloneMetric(Metric[float]):
    """
    This metric will return a `float` value
    """
    def __init__(self):
        """
        Initialize your metric here
        """
        super().__init__()
        pass

    def update(self):
        """
        Update metric value here
        """
        pass

    def result(self) -> float:
        """
        Emit the metric result here
        """
        return 0

    def reset(self):
        """
        Reset your metric here
        """
        pass
```

To implement a **plugin metric** you have to subclass `MetricPlugin` class

```python
from avalanche.evaluation import PluginMetric
from avalanche.evaluation.metrics import Accuracy


class MyPluginMetric(PluginMetric[float]):
    """
    This metric will return a `float` value after
    each training epoch
    """

    def __init__(self):
        """
        Initialize the metric
        """
        super().__init__()

        self._accuracy_metric = Accuracy()
        # current x values for the metric curve
        self.x_coord = 0

    def reset(self) -> None:
        """
        Reset the metric
        """
        self._accuracy_metric.reset()

    def result(self) -> float:
        """
        Emit the result
        """
        return self._accuracy_metric.result()

    def after_training_iteration(self, strategy: 'PluggableStrategy') -> None:
        """
        Update the accuracy metric with the current
        predictions and targets
        """
        self._accuracy_metric.update(strategy.mb_y,
                                     strategy.logits)

    def before_training_epoch(self, strategy: 'PluggableStrategy') -> None:
        """
        Reset the accuracy before the epoch begins
        """
        self.reset()

    def after_training_epoch(self, strategy: 'PluggableStrategy'):
        """
        Emit the result
        """
        value = self._accuracy_metric.result()
        self.x_coord += 1 # increment x value
        return [MetricValue(self, 'metric_full_name', value,
                            self.x_coord)]

    def __str__(self):
        """
        Here you can specify the name of your metric
        """
        return "Top1_Acc_Epoch"
```

## Accessing metric values

If you want to access all the metrics computed during training and evaluation, you have to make sure that `collect_all=True` is set when creating the `EvaluationPlugin` \(default option is `True`\). This option maintains an updated version of all metric results in the plugin, which can be retrieved by calling `evaluation_plugin.get_all_metrics()`. You can call this methods whenever you need the metrics.

The result is a dictionary with full metric names as keys and a tuple of two lists as values. The first list stores all the `x` values recorded for that metric. Each `x` value represents the time step at which the corresponding metric value has been computed. The second list stores metric values associated to the corresponding `x` value.

```python
eval_plugin = EvaluationPlugin(
    accuracy_metrics(minibatch=True, epoch=True, experience=True, stream=True),
    loss_metrics(minibatch=True, epoch=True, experience=True, stream=True),
    forgetting_metrics(experience=True, stream=True),
    timing_metrics(epoch=True),
    cpu_usage_metrics(experience=True),
    confusion_matrix_metrics(num_classes=scenario.n_classes, save_image=False, stream=True),
    disk_usage_metrics(minibatch=True, epoch=True, experience=True, stream=True),
    collect_all=True # this is default value anyway
)

# since no training and evaluation has been performed, this will return an empty dict.
metric_dict = eval_plugin.get_all_metrics()
```

Alternatively, the `train` and `eval` method of every `strategy` returns a dictionary storing, for each metric, the last value recorded for that metric. You can use these dictionaries to incrementally accumulate metrics.

This completes the "_Evaluation_" tutorial for the "_From Zero to Hero_" series. We hope you enjoyed it!

## 🤝 Run it on Google Colab

You can run _this chapter_ and play with it on Google Colaboratory: [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ContinualAI/colab/blob/master/notebooks/avalanche/4.-evaluation.ipynb)

