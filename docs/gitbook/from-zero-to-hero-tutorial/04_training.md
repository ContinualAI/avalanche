---
description: Continual Learning Algorithms Prototyping Made Easy
---
# Training

Welcome to the "_Training_" tutorial of the "_From Zero to Hero_" series. In this part we will present the functionalities offered by the `training` module.

First, let's install Avalanche. You can skip this step if you have installed it already.


```python
!pip install avalanche-lib
```

## 💪 The Training Module

The `training` module in _Avalanche_ is designed with modularity in mind. Its main goals are to:

1. provide a set of popular **continual learning baselines** that can be easily used to run experimental comparisons;
2. provide simple abstractions to **create and run your own strategy** as efficiently and easily as possible starting from a couple of basic building blocks we already prepared for you.

At the moment, the `training` module includes two main components:

* **Strategies**: these are popular baselines already implemented for you which you can use for comparisons or as base classes to define a custom strategy.
* **Plugins**: these are classes that allow to add some specific behaviour to your own strategy. The plugin system allows to define reusable components which can be easily combined together (e.g. a replay strategy, a regularization strategy). They are also used to automatically manage logging and evaluation.

Keep in mind that many Avalanche components are independent from Avalanche strategies. If you already have your own strategy which does not use Avalanche, you can use Avalanche's benchmarks, models, data loaders, and metrics without ever looking at Avalanche's strategies.

## 📈 How to Use Strategies & Plugins

If you want to compare your strategy with other classic continual learning algorithm or baselines, in _Avalanche_ you can instantiate a strategy with a couple lines of code.

### Strategy Instantiation
Most strategies require only 3 mandatory arguments:
- **model**: this must be a `torch.nn.Module`.
- **optimizer**: `torch.optim.Optimizer` already initialized on your `model`.
- **loss**: a loss function such as those in `torch.nn.functional`.

Additional arguments are optional and allow you to customize training (batch size, epochs, ...) or strategy-specific parameters (buffer size, regularization strength, ...).


```python
from torch.optim import SGD
from torch.nn import CrossEntropyLoss
from avalanche.models import SimpleMLP
from avalanche.training.strategies import Naive, CWRStar, Replay, GDumb, Cumulative, LwF, GEM, AGEM, EWC

model = SimpleMLP(num_classes=10)
optimizer = SGD(model.parameters(), lr=0.001, momentum=0.9)
criterion = CrossEntropyLoss()
cl_strategy = Naive(
    model, optimizer, criterion, 
    train_mb_size=100, train_epochs=4, eval_mb_size=100
)
```

### Training & Evaluation

Each strategy object offers two main methods: `train` and `eval`. Both of them, accept either a _single experience_(`Experience`) or a _list of them_, for maximum flexibility.

We can train the model continually by iterating over the `train_stream` provided by the scenario.


```python
from avalanche.benchmarks.classic import SplitMNIST

# scenario
benchmark = SplitMNIST(n_experiences=5, seed=1)

# TRAINING LOOP
print('Starting experiment...')
results = []
for experience in benchmark.train_stream:
    print("Start of experience: ", experience.current_experience)
    print("Current Classes: ", experience.classes_in_this_experience)

    cl_strategy.train(experience)
    print('Training completed')

    print('Computing accuracy on the whole test set')
    results.append(cl_strategy.eval(benchmark.test_stream))
```

### Adding Plugins

Most continual learning strategies follow roughly the same training/evaluation loops, i.e. a simple naive strategy (a.k.a. finetuning) augmented with additional behavior to counteract catastrophic forgetting. The plugin systems in Avalanche is designed to easily augment continual learning strategies with custom behavior, without having to rewrite the training loop from scratch. Avalanche strategies accept an optional list of `plugins` that will be executed during the training/evaluation loops.

For example, early stopping is implemented as a plugin:


```python
from avalanche.training.plugins import EarlyStoppingPlugin

strategy = Naive(
    model, optimizer, criterion,
    plugins=[EarlyStoppingPlugin(patience=10, val_stream_name='train')])
```

In Avalanche, most continual learning strategies are implemented using plugins, which makes it easy to combine them together. For example, it is extremely easy to create a hybrid strategy that combines replay and EWC together by passing the appropriate `plugins` list to the `BaseStrategy`:


```python
from avalanche.training.strategies import BaseStrategy
from avalanche.training.plugins import ReplayPlugin, EWCPlugin

replay = ReplayPlugin(mem_size=100)
ewc = EWCPlugin(ewc_lambda=0.001)
strategy = BaseStrategy(
    model, optimizer, criterion,
    plugins=[replay, ewc])
```

Beware that most strategy plugins modify the internal state. As a result, not all the strategy plugins can be combined together. For example, it does not make sense to use multiple replay plugins since they will try to modify the same strategy variables (mini-batches, dataloaders), and therefore they will be in conflict.

## 📝 A Look Inside Avalanche Strategies

If you arrived at this point you already know how to use Avalanche strategies and are ready to use it. However, before making your own strategies you need to understand a little bit the internal implementation of the training and evaluation loops.

In _Avalanche_ you can customize a strategy in 2 ways:

1. **Plugins**: Most strategies can be implemented as additional code that runs on top of the basic training and evaluation loops (`Naive` strategy, or `BaseStrategy`). Therefore, the easiest way to define a custom strategy such as a regularization or replay strategy, is to define it as a custom plugin. The advantage of plugins is that they can be combined together, as long as they are compatible, i.e. they do not modify the same part of the state. The disadvantage is that in order to do so you need to understand the `BaseStrategy` loop, which can be a bit complex at first.
2. **Subclassing**: In _Avalanche_, continual learning strategies inherit from the `BaseStrategy`, which provides generic training and evaluation loops. Most `BaseStrategy` methods can be safely overridden (with some caveats that we will see later).

Keep in mind that if you already have a working continual learning strategy that does not use _Avalanche_, you can use most Avalanche components such as `benchmarks`, `evaluation`, and `models` without using _Avalanche_'s strategies!

### Training and Evaluation Loops

As we already mentioned, _Avalanche_ strategies inherit from `BaseStrategy`. This strategy provides:

1. Basic _Training_ and _Evaluation_ loops which define a naive (finetuning) strategy.
2. _Callback_ points, which are used to call the plugins at a specific moments during the loop's execution.
3. A set of variables representing the state of the loops (current model, data, mini-batch, predictions, ...) which allows plugins and child classes to easily manipulate the state of the training loop.

The training loop has the following structure:
```text
train
    before_training

    before_train_dataset_adaptation
    train_dataset_adaptation
    after_train_dataset_adaptation
    make_train_dataloader
    model_adaptation
    make_optimizer
    before_training_exp  # for each exp
        before_training_epoch  # for each epoch
            before_training_iteration  # for each iteration
                before_forward
                after_forward
                before_backward
                after_backward
            after_training_iteration
            before_update
            after_update
        after_training_epoch
    after_training_exp
    after_training
```

The evaluation loop is similar:
```text
eval
    before_eval
    before_eval_dataset_adaptation
    eval_dataset_adaptation
    after_eval_dataset_adaptation
    make_eval_dataloader
    model_adaptation
    before_eval_exp  # for each exp
        eval_epoch  # we have a single epoch in evaluation mode
            before_eval_iteration  # for each iteration
                before_eval_forward
                after_eval_forward
            after_eval_iteration
    after_eval_exp
    after_eval
```

Methods starting with `before/after` are the methods responsible for calling the plugins.
Notice that before the start of each experience during training we have several phases:
- *dataset adaptation*: This is the phase where the training data can be modified by the strategy, for example by adding other samples from a separate buffer.
- *dataloader initialization*: Initialize the data loader. Many strategies (e.g. replay) use custom dataloaders to balance the data.
- *model adaptation*: Here, the dynamic models (see the `models` tutorial) are updated by calling their `adaptation` method.
- *optimizer initialization*: After the model has been updated, the optimizer should also be updated to ensure that the new parameters are optimized.

### Strategy State
The strategy state is accessible via several attributes. Most of these can be modified by plugins and subclasses:
- `self.clock`: keeps track of several event counters.
- `self.experience`: the current experience.
- `self.adapted_dataset`: the data modified by the dataset adaptation phase.
- `self.dataloader`: the current dataloader.
- `self.mbatch`: the current mini-batch. For classification problems, mini-batches have the form `<x, y, t>`, where `x` is the input, `y` is the target class, and `t` is the task label.
- `self.mb_output`: the current model's output.
- `self.loss`: the current loss.
- `self.is_training`: `True` if the strategy is in training mode.

## How to Write a Plugin
Plugins provide a simple solution to define a new strategy by augmenting the behavior of another strategy (typically a naive strategy). This approach reduces the overhead and code duplication, **improving code readability and prototyping speed**.

Creating a plugin is straightforward. You create a class which inherits from `StrategyPlugin` and implements the callbacks that you need. The exact callback to use depend on your strategy. You can use the loop shown above to understand what callbacks you need to use. For example, we show below a simple replay plugin that uses `after_training_exp` to update the buffer after each training experience, and the `before_training_exp` to customize the dataloader. Notice that `before_training_exp` is executed after `make_train_dataloader`, which means that the `BaseStrategy` already updated the dataloader. If we used another callback, such as `before_train_dataset_adaptation`, our dataloader would have been overwritten by the `BaseStrategy`. Plugin methods always receive the `strategy` as an argument, so they can access and modify the strategy's state.


```python
from avalanche.benchmarks.utils.data_loader import ReplayDataLoader
from avalanche.training.plugins import StrategyPlugin
from avalanche.training.storage_policy import ReservoirSamplingBuffer


class ReplayP(StrategyPlugin):

    def __init__(self, mem_size):
        """ A simple replay plugin with reservoir sampling. """
        super().__init__()
        self.buffer = ReservoirSamplingBuffer(max_size=mem_size)

    def before_training_exp(self, strategy: "BaseStrategy",
                            num_workers: int = 0, shuffle: bool = True,
                            **kwargs):
        """ Use a custom dataloader to combine samples from the current data and memory buffer. """
        if len(self.buffer.buffer) == 0:
            # first experience. We don't use the buffer, no need to change
            # the dataloader.
            return
        strategy.dataloader = ReplayDataLoader(
            strategy.adapted_dataset,
            self.buffer.buffer,
            oversample_small_tasks=True,
            num_workers=num_workers,
            batch_size=strategy.train_mb_size,
            shuffle=shuffle)

    def after_training_exp(self, strategy: "BaseStrategy", **kwargs):
        """ Update the buffer. """
        self.buffer.update(strategy, **kwargs)


benchmark = SplitMNIST(n_experiences=5, seed=1)
model = SimpleMLP(num_classes=10)
optimizer = SGD(model.parameters(), lr=0.01, momentum=0.9)
criterion = CrossEntropyLoss()
strategy = Naive(model=model, optimizer=optimizer, criterion=criterion, train_mb_size=128,
                 plugins=[ReplayP(mem_size=2000)])
strategy.train(benchmark.train_stream)
strategy.eval(benchmark.test_stream)
```

Check `StrategyPlugin`'s documentation for a complete list of the available callbacks.

## How to Write a Custom Strategy

You can always define a custom strategy by overriding `BaseStrategy` methods.
However, There is an important caveat to keep in mind. If you override a method, you must remember to call all the callback's handlers (the methods starting with `before/after`) at the appropriate points. For example, `train` calls `before_training` and `after_training` before and after the training loops, respectively. The easiest way to avoid mistakes is to start from the `BaseStrategy` method that you want to override and modify it to your own needs without removing the callbacks handling.

Notice that even though you don't use plugins, `BaseStrategy` implements some internal components as plugins. Also, the `EvaluationPlugin` (see `evaluation` tutorial) uses the strategy callbacks.

`BaseStrategy` provides the global state of the loop in the strategy's attributes, which you can safely use when you override a method. As an example, the `Cumulative` strategy trains a model continually on the union of all the experiences encountered so far. To achieve this, the cumulative strategy overrides `adapt_train_dataset` and updates `self.adapted_dataset' by concatenating all the previous experiences with the current one.


```python
from avalanche.benchmarks.utils import AvalancheConcatDataset
from avalanche.training import BaseStrategy


class Cumulative(BaseStrategy):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.dataset = None  # cumulative dataset

    def train_dataset_adaptation(self, **kwargs):
        super().train_dataset_adaptation(**kwargs)
        curr_data = self.experience.dataset
        if self.dataset is None:
            self.dataset = curr_data
        else:
            self.dataset = AvalancheConcatDataset([self.dataset, curr_data])
        self.adapted_dataset = self.dataset.train()

strategy = Cumulative(model=model, optimizer=optimizer, criterion=criterion, train_mb_size=128)
strategy.train(benchmark.train_stream)
```

Easy, isn't it? :-\)

In general, we recommend to _implement a Strategy via plugins_, if possible. This approach is the easiest to use and requires a minimal knowledge of the `BaseStrategy`. It also allows other people to use your plugin and facilitates interoperability among different strategies.

For example, replay strategies can be implemented as a custom strategy of the `BaseStrategy` or as plugins. However, creating a plugin is allows to use the replay strategy in conjunction with other strategies.

This completes the "_Training_" chapter for the "_From Zero to Hero_" series. We hope you enjoyed it!

## 🤝 Run it on Google Colab

You can run _this chapter_ and play with it on Google Colaboratory: [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ContinualAI/avalanche/blob/master/notebooks/from-zero-to-hero-tutorial/04_training.ipynb)
