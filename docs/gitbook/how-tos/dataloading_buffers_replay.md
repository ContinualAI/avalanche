---
description: How to implement replay and data loading
---
# Dataloading, Memory Buffers, and Replay

Avalanche provides several components that help you to balance data loading and implement rehearsal strategies.

**Dataloaders** are used to provide balancing between groups (e.g. tasks/classes/experiences). This is especially useful when you have unbalanced data.

**Buffers** are used to store data from the previous experiences. They are dynamic datasets with a fixed maximum size, and they can be updated with new data continuously.

Finally, **Replay** strategies implement rehearsal by using Avalanche's plugin system. Most rehearsal strategies use a custom dataloader to balance the buffer with the current experience and a buffer that is updated for each experience.

First, let's install Avalanche. You can skip this step if you have installed it already.


```python
!pip install avalanche-lib
```

## Dataloaders
Avalanche dataloaders are simple iterators, located under `avalanche.benchmarks.utils.data_loader`. Their interface is equivalent to pytorch's dataloaders. For example, `GroupBalancedDataLoader` takes a sequence of datasets and iterates over them by providing balanced mini-batches, where the number of samples is split equally among groups. Internally, it instantiate a `DataLoader` for each separate group. More specialized dataloaders exist such as `TaskBalancedDataLoader`.

All the dataloaders accept keyword arguments (`**kwargs`) that are passed directly to the dataloaders for each group.


```python
from avalanche.benchmarks import SplitMNIST
from avalanche.benchmarks.utils.data_loader import GroupBalancedDataLoader
benchmark = SplitMNIST(5, return_task_id=True)

dl = GroupBalancedDataLoader([exp.dataset for exp in benchmark.train_stream], batch_size=4)
for x, y, t in dl:
    print(t.tolist())
    break
```

## Memory Buffers
Memory buffers store data up to a maximum capacity, and they implement policies to select which data to store and which the to remove when the buffer is full. They are available in the module `avalanche.training.storage_policy`. The base class is the `ExemplarsBuffer`, which implements two methods:
- `update(strategy)` - given the strategy's state it updates the buffer (using the data in `strategy.experience.dataset`).
- `resize(strategy, new_size)` - updates the maximum size and updates the buffer accordingly.

The data can be access using the attribute `buffer`.


```python
from avalanche.training.storage_policy import ReservoirSamplingBuffer
from types import SimpleNamespace

benchmark = SplitMNIST(5, return_task_id=False)
storage_p = ReservoirSamplingBuffer(max_size=30)

print(f"Max buffer size: {storage_p.max_size}, current size: {len(storage_p.buffer)}")
```

At first, the buffer is empty. We can update it with data from a new experience.

Notice that we use a `SimpleNamespace` because we want to use the buffer standalone, without instantiating an Avalanche strategy. Reservoir sampling requires only the `experience` from the strategy's state.


```python
for i in range(5):
    strategy_state = SimpleNamespace(experience=benchmark.train_stream[i])
    storage_p.update(strategy_state)
    print(f"Max buffer size: {storage_p.max_size}, current size: {len(storage_p.buffer)}")
    print(f"class targets: {storage_p.buffer.targets}\n")
```

Notice after each update some samples are substituted with new data. Reservoir sampling select these samples randomly.

Avalanche offers many more storage policies. For example, `ParametricBuffer` is a buffer split into several groups according to the `groupby` parameters (`None`, 'class', 'task', 'experience'), and according to an optional `ExemplarsSelectionStrategy` (random selection is the default choice).


```python
from avalanche.training.storage_policy import ParametricBuffer, RandomExemplarsSelectionStrategy
storage_p = ParametricBuffer(
    max_size=30,
    groupby='class',
    selection_strategy=RandomExemplarsSelectionStrategy()
)

print(f"Max buffer size: {storage_p.max_size}, current size: {len(storage_p.buffer)}")
for i in range(5):
    strategy_state = SimpleNamespace(experience=benchmark.train_stream[i])
    storage_p.update(strategy_state)
    print(f"Max buffer size: {storage_p.max_size}, current size: {len(storage_p.buffer)}")
    print(f"class targets: {storage_p.buffer.targets}\n")
```

The advantage of using grouping buffers is that you get a balanced rehearsal buffer. You can even access the groups separately with the `buffer_groups` attribute. Combined with balanced dataloaders, you can ensure that the mini-batches stay balanced during training.


```python
for k, v in storage_p.buffer_groups.items():
    print(f"(group {k}) -> size {len(v.buffer)}")
```


```python
datas = [v.buffer for v in storage_p.buffer_groups.values()]
dl = GroupBalancedDataLoader(datas)

for x, y, t in dl:
    print(y.tolist())
    break
```

## Replay Plugins

Avalanche's strategy plugins can be used to update the rehearsal buffer and set the dataloader. This allows to easily implement replay strategies:


```python
from avalanche.benchmarks.utils.data_loader import ReplayDataLoader
from avalanche.training.plugins import StrategyPlugin

class CustomReplay(StrategyPlugin):
    def __init__(self, storage_policy):
        super().__init__()
        self.storage_policy = storage_policy

    def before_training_exp(self, strategy,
                            num_workers: int = 0, shuffle: bool = True,
                            **kwargs):
        """ Here we set the dataloader. """
        if len(self.storage_policy.buffer) == 0:
            # first experience. We don't use the buffer, no need to change
            # the dataloader.
            return

        # replay dataloader samples mini-batches from the memory and current
        # data separately and combines them together.
        print("Override the dataloader.")
        strategy.dataloader = ReplayDataLoader(
            strategy.adapted_dataset,
            self.storage_policy.buffer,
            oversample_small_tasks=True,
            num_workers=num_workers,
            batch_size=strategy.train_mb_size,
            shuffle=shuffle)

    def after_training_exp(self, strategy: "BaseStrategy", **kwargs):
        """ We update the buffer after the experience.
            You can use a different callback to update the buffer in a different place
        """
        print("Buffer update.")
        self.storage_policy.update(strategy, **kwargs)

```

And of course, we can use the plugin to train our continual model


```python
from torch.nn import CrossEntropyLoss
from avalanche.training import Naive
from avalanche.evaluation.metrics import accuracy_metrics
from avalanche.training.plugins import EvaluationPlugin
from avalanche.logging import InteractiveLogger
from avalanche.models import SimpleMLP
import torch

scenario = SplitMNIST(5)
model = SimpleMLP(num_classes=scenario.n_classes)
storage_p = ParametricBuffer(
    max_size=500,
    groupby='class',
    selection_strategy=RandomExemplarsSelectionStrategy()
)

# choose some metrics and evaluation method
interactive_logger = InteractiveLogger()

eval_plugin = EvaluationPlugin(
    accuracy_metrics(experience=True, stream=True),
    loggers=[interactive_logger])

# CREATE THE STRATEGY INSTANCE (NAIVE)
cl_strategy = Naive(model, torch.optim.Adam(model.parameters(), lr=0.001),
                    CrossEntropyLoss(),
                    train_mb_size=100, train_epochs=1, eval_mb_size=100,
                    plugins=[CustomReplay(storage_p)],
                    evaluator=eval_plugin
                    )

# TRAINING LOOP
print('Starting experiment...')
results = []
for experience in scenario.train_stream:
    print("Start of experience ", experience.current_experience)
    cl_strategy.train(experience)
    print('Training completed')

    print('Computing accuracy on the whole test set')
    results.append(cl_strategy.eval(scenario.test_stream))
```
