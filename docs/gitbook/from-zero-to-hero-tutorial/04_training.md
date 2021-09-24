---
description: Continual Learning Algorithms Prototyping Made Easy
---
# Training

Welcome to the "_Training_" tutorial of the "_From Zero to Hero_" series. In this part we will present the functionalities offered by the `training` module.


```python
!pip install git+https://github.com/ContinualAI/avalanche.git
```

## 💪 The Training Module

The `training` module in _Avalanche_ is designed with modularity in mind. Its main goals are to:

1. provide a set of popular **continual learning baselines** that can be easily used to run experimental comparisons;
2. provide simple abstractions to **create and run your own strategy** as efficiently and easy as possible starting from a couple of basic building blocks we already prepared for you.

At the moment, the `training` module includes two main components:

* **Strategies**: these are popular baselines already implemented for you which you can use for comparisons or as base classes to define a custom strategy.
* **Plugins**: these are classes that allow to add some specific behaviour to your own strategy. The plugin system allows to define reusable components which can be easily combined together (e.g. a replay strategy, a regularization strategy). They are also used to automatically manage logging and evaluation.

Keep in mind that Avalanche's components are mostly independent from each other. If you already have your own strategy which does not use Avalanche, you can use benchmarks and metrics without ever looking at Avalanche's strategies.

## 📈 How to Use Strategies & Plugins

If you want to compare your strategy with other classic continual learning algorithm or baselines, in _Avalanche_ you can instantiate a strategy with a couple lines of code.

### Strategy Instantiation
Most strategies require only 3 mandatory arguments:
- **model**: this must be a `torch.nn.Module`.
- **optimizer**: `torch.optim.Optimizer` already initialized on your `model`.
- **loss**: a loss function such as those in `torch.nn.functional`.

Additional arguments are optional and allow you to customize training (batch size, epochs, ...) or strategy-specific parameters (buffer size, regularization strenght, ...).


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
scenario = SplitMNIST(n_experiences=5, seed=1)

# TRAINING LOOP
print('Starting experiment...')
results = []
for experience in scenario.train_stream:
    print("Start of experience: ", experience.current_experience)
    print("Current Classes: ", experience.classes_in_this_experience)

    cl_strategy.train(experience)
    print('Training completed')

    print('Computing accuracy on the whole test set')
    results.append(cl_strategy.eval(scenario.test_stream))
```

### Adding Plugins

We noticed that many continual learning strategies follow roughly the same training/evaluation loops and always implement the same boilerplate code. So, it seems natural to define most strategies by specializing the few methods that need to be changed. Most strategies only _augment_ the naive strategy with additional behavior, without changing the basic training and evlaution loops. These strategies can be easily implemented with a couple of methods.

_Avalanche_'s plugins allow to augment a strategy with additional behavior. Currently, most continual learning strategies are also implemented as plugins, which makes them easy to combine together. For example, it is extremely easy to create a hybrid strategy that combines replay and EWC together by passing the appropriate `plugins` list to the `BaseStrategy`:


```python
from avalanche.training.strategies import BaseStrategy
from avalanche.training.plugins import ReplayPlugin, EWCPlugin

replay = ReplayPlugin(mem_size=100)
ewc = EWCPlugin(ewc_lambda=0.001)
strategy = BaseStrategy(
    model, optimizer, criterion, 
    plugins=[replay, ewc])
```

## 📝Create your Strategy

In _Avalanche_ you can build your own strategy in 2 main ways:

1. **Plugins**: Most strategies can be defined by _augmenting_ the basic training and evaluation loops. The easiest way to define a custom strategy such as a regularization or replay strategy, is to define it as a custom plugin. This is the suggested approach as it is the easiest way to define custom strategies.
2. **Subclassing**: In _Avalanche_, continual learning strategies inherit from the `BaseStrategy`, which provides generic training and evaluation loops. You can safely override most methods to customize your strategy. However, there are some caveats to discuss (see later) and in general this approach is more difficult than plugins.

Keep in mind that if you already have a continual learning strategy that does not use _Avalanche_, you can always use `benchmarks` and `evaluation` without using _Avalanche_'s strategies!

### Avalanche Strategies - Under the Hood

As we already mentioned, _Avalanche_ strategies inherit from `BaseStrategy`. This strategy provides:

1. Basic _Training_ and _Evaluation_ loops which define a naive strategy.
2. _Callback_ points, which can be used to augment the strategy's behavior.
3. A set of variables representing the state of the loops (current batch, predictions, ...) which allows plugins and child classes to easily manipulate the state of the training loop.

The training loop has the following structure:
```text
train
    before_training
    before_training_exp
    adapt_train_dataset
    make_train_dataloader
    before_training_epoch
        before_training_iteration
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
    adapt_eval_dataset
    make_eval_dataloader
    before_eval_exp
        eval_epoch
            before_eval_iteration
            before_eval_forward
            after_eval_forward
            after_eval_iteration
    after_eval_exp
    after_eval
```

### Custom Plugin
Plugins provide a simple solution to define a new strategy by augmenting the behavior of another strategy (typically a naive strategy). This approach reduces the overhead and code duplication, **improving code readability and prototyping speed**.

Creating a plugin is straightforward. You create a class which inherits from `StrategyPlugin` and implements the callbacks that you need. The exact callback to use depend on your strategy. For example, the following replay plugin uses `after_training_exp` to update the buffer after each training experience, and the `adapt_training_dataset` to concatenate the buffer's data with the current experience:

```python
from avalanche.training.plugins import StrategyPlugin


class ReplayPlugin(StrategyPlugin):
    """
    Experience replay plugin.

    Handles an external memory filled with randomly selected
    patterns and implements the "adapt_train_dataset" callback to add them to
    the training set.

    The :mem_size: attribute controls the number of patterns to be stored in
    the external memory. In multitask scenarios, mem_size is the memory size
    for each task. We assume the training set contains at least :mem_size: data
    points.
    """

    def __init__(self, mem_size=200):
        super().__init__()

        self.mem_size = mem_size
        self.ext_mem = {}  # a Dict<task_id, Dataset>
        self.rm_add = None

    def adapt_train_dataset(self, strategy, **kwargs):
        """
        Expands the current training set with datapoint from
        the external memory before training.
        """
        curr_data = strategy.experience.dataset

        # Additional set of the current batch to be concatenated to the ext.
        # memory at the end of the training
        self.rm_add = None

        # how many patterns to save for next iter
        h = min(self.mem_size // (strategy.train_exp_counter + 1),
                len(curr_data))

        # We recover it using the random_split method and getting rid of the
        # second split.
        self.rm_add, _ = random_split(
            curr_data, [h, len(curr_data) - h]
        )

        if strategy.train_exp_counter > 0:
            # We update the train_dataset concatenating the external memory.
            # We assume the user will shuffle the data when creating the data
            # loader.
            for mem_task_id in self.ext_mem.keys():
                mem_data = self.ext_mem[mem_task_id]
                if mem_task_id in strategy.adapted_dataset:
                    cat_data = ConcatDataset([curr_data, mem_data])
                    strategy.adapted_dataset[mem_task_id] = cat_data
                else:
                    strategy.adapted_dataset[mem_task_id] = mem_data

    def after_training_exp(self, strategy, **kwargs):
        """ After training we update the external memory with the patterns of
         the current training batch/task. """
        curr_task_id = strategy.experience.task_label

        # replace patterns in random memory
        ext_mem = self.ext_mem
        if curr_task_id not in ext_mem:
            ext_mem[curr_task_id] = self.rm_add
        else:
            rem_len = len(ext_mem[curr_task_id]) - len(self.rm_add)
            _, saved_part = random_split(ext_mem[curr_task_id],
                                         [len(self.rm_add), rem_len]
                                         )
            ext_mem[curr_task_id] = ConcatDataset([saved_part, self.rm_add])
        self.ext_mem = ext_mem
```

Check `StrategyPlugin`'s documentation for a complete list of the available callbacks.

### Custom Strategy

You can always define a custom strategy by overriding `BaseStrategy` methods.
However, There is an important caveat to keep in mind. If you override a method, you must remember to call all the callback's handlers at the appropriate points. For example, `train` calls `before_training` and `after_training` before and after the training loops, respectively. If your strategy strategy does not call them, plugins may not work as expected. The easiest way to avoid mistakes is to start from the `BaseStrategy` method that you want to override and modify it to your own needs without removing the callbacks handling.

There is only a single plugin that is always used by default, the `EvaluationPlugin` (see `evaluation` tutorial). This means that if you break callbacks you must log metrics by yourself. This is totally possible but requires some manual work to update, log, and reset each metric, which is done automatically for you by the `BaseStrategy`.

`BaseStrategy` provides the global state of the loop in the strategy's attributes, which you can safely use when you override a method. As an example, the `Cumulative` strategy trains a model continually on the union of all the experiences encountered so far. To achieve this, the cumulative strategy overrides `adapt_train_dataset` and updates `self.adapted_dataset' by concatenating all the previous experiences with the current one.


```python
class Cumulative(BaseStrategy):
    def __init__(*args, **kwargs):
        super().__init__(*args, **kwargs)
        self.dataset = {}  # cumulative dataset

    def adapt_train_dataset(self, **kwargs):
        super().adapt_train_dataset(**kwargs)
        curr_task_id = self.experience.task_label
        curr_data = self.experience.dataset
        if curr_task_id in self.dataset:
            cat_data = ConcatDataset([self.dataset[curr_task_id],
                                      curr_data])
            self.dataset[curr_task_id] = cat_data
        else:
            self.dataset[curr_task_id] = curr_data
        self.adapted_dataset = self.dataset
```

Easy, isn't it? :-\)

In general, we recommend to _implement a Strategy via plugins_, if possible. This approach is the easiest to use and requires a minimal knowledge of the `BaseStrategy`. It also allows other people to use your plugin and facilitates interoperability among different strategies.

For example, replay strategies can be implemented as a custom strategy of the `BaseStrategy` or as plugins. However, creating a plugin is better because it allows to use our replay strategy in conjunction with other strategies.

This completes the "_Training_" chapter for the "_From Zero to Hero_" series. We hope you enjoyed it!

## 🤝 Run it on Google Colab

You can run _this chapter_ and play with it on Google Colaboratory: [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ContinualAI/colab/blob/master/notebooks/avalanche/3.-training.ipynb)
