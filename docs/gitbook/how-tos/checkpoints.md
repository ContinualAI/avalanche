---
description: Save and load checkpoints
---
# Save and load checkpoints

The ability to **save and resume experiments** may be very useful when running long experiments. Avalanche offers a checkpointing functionality that can be used to save and restore your strategy including plugins, metrics, and loggers.

This guide will show how to plug the checkpointing functionality into the usual Avalanche main script. This only requires minor changes in the main: no changes on the strategy/plugins/... code is required! Also, make sure to check the [checkpointing.py](https://github.com/ContinualAI/avalanche/blob/master/examples/checkpointing.py) example in the repository for a ready-to-go template.

## Continual learning vs classic deep learning
**Resuming a continual learning experiment is not the same as resuming a classic deep learning training session.** In classic training setups, the elements needed to resume an experiment are limited to i) the model weights, ii) the optimizer state, and iii) additional info such as the number of epochs/iterations so far. On the contrary, continual learning experiments need far more info to be correctly resumed:

- The state of **plugins**, such as:
    - the examples saved in the replay buffer
    - the importance of model weights (EwC, Synaptic Intelligence)
    - a copy of the model (LwF)
    - ... any many others, which are *specific to each technique*!
- The state of **metrics**, as some are computed on the performance measured on previous experiences:
    - AMCA (Average Mean Class Accuracy) metric
    - Forgetting metric


## Resuming experiments in Avalanche
To handle all these elements, we opted to provide an easy-to-use plugin: the *CheckpointPlugin*. It will take care of loading:

- Strategy, including the model
- Plugins
- Metrics
- Loggers: this includes re-opening the logs for TensoBoard, Weights & Biases, ...
- State of all random number generators
    - In continual learning experiments, this affects the choice of replay examples and other critical elements. This is usually not needed in classic deep learning, but here may be useful!


Here, in a couple of cells, we'll show you how to use it. Remember that you can follow this guide by running it as a notebook (see below for a direct link to load it on Colab).

Let's install Avalanche:


```python
!pip install avalanche-lib
```

And let us import the needed elements:


```python
import sys
sys.path.append('/home/lorenzo/Desktop/ProjectsVCS/avalanche/')

import os
from typing import Sequence

import torch
from torch.nn import CrossEntropyLoss
from torch.optim import SGD

from avalanche.benchmarks import CLExperience, SplitMNIST
from avalanche.evaluation.metrics import accuracy_metrics, loss_metrics, \
    class_accuracy_metrics
from avalanche.logging import InteractiveLogger, TensorboardLogger, \
    WandBLogger, TextLogger
from avalanche.models import SimpleMLP, as_multitask
from avalanche.training.determinism.rng_manager import RNGManager
from avalanche.training.plugins import EvaluationPlugin, ReplayPlugin
from avalanche.training.plugins.checkpoint import CheckpointPlugin, \
    FileSystemCheckpointStorage
from avalanche.training.supervised import Naive
```

Let's proceed by defining a very vanilla Avalanche main script. Simply put, this usually comes down to defining:

0. Load any configuration, set seeds, etcetera
1. The benchmark
2. The model, optimizer, and loss function
3. Evaluation components
    - The list of metrics to track
    - The loggers
    - The evaluation plugin (that glues the metrics and loggers together)
4. The training plugins
5. The strategy
6. The train-eval loop

They do not have to be in this particular order, but this is the order followed in this guide.

To enable checkpointing, the following changes are needed:
1. In the very first part of the code, fix the seeds for reproducibility
    - The **RNGManager** class is used, which may be useful even in experiments in which checkpointing is not needed ;)
2. Instantiate the checkpointing plugin
3. Check if a checkpoint exists and load it
    - Only if not resuming from a checkpoint: create the Evaluation components, the plugins, and the strategy
5. Change the train/eval loop to start from the experience

Note that those changes are all properly annotated in the [checkpointing.py](https://github.com/ContinualAI/avalanche/blob/master/examples/checkpointing.py) example, which is the recommended template to follow when enabling checkpoint in a training script.

### Step by step
Let's start with the first change: defining a fixed seed. This is needed to correctly re-create the benchmark object and should be the same seed used to create the checkpoint.

The **RNGManager** takes care of setting the seed for the following generators: Python *random*, NumPy, and PyTorch (both cpu and device-specific generators). In this way, you can be sure that any randomness-dependent elements in the benchmark creation procedure are identical across save/resume operations.


```python
# Set a fixed seed: must be kept the same across save/resume operations
RNGManager.set_random_seeds(1234)
```

Let's then proceed with the usual Avalanche code. Note: nothing to change here to enable checkpointing. Here we create a SplitMNIST benchmark and instantiate a multi-task MLP model. Notice that checkpointing works fine with multi-task models wrapped using `as_multitask`.


```python

# Nothing new here...
device = torch.device(
    f"cuda:0"
    if torch.cuda.is_available()
    else "cpu"
)
print('Using device', device)

# CL Benchmark Creation
n_experiences = 5
benchmark = SplitMNIST(n_experiences=n_experiences,
                        return_task_id=True)
input_size = 28*28*1

# Define the model (and load initial weights if necessary)
# Again, not checkpoint-related
model = SimpleMLP(input_size=input_size,
                    num_classes=benchmark.n_classes // n_experiences)
model = as_multitask(model, 'classifier')

# Prepare for training & testing: not checkpoint-related
optimizer = SGD(model.parameters(), lr=0.01, momentum=0.9)
criterion = CrossEntropyLoss()
```

It's now time to instantiate the checkpointing plugin and load the checkpoint.


```python
checkpoint_plugin = CheckpointPlugin(
    FileSystemCheckpointStorage(
        directory='./checkpoints/task_incremental',
    ),
    map_location=device
)

# Load checkpoint (if exists in the given storage)
# If it does not exist, strategy will be None and initial_exp will be 0
strategy, initial_exp = checkpoint_plugin.load_checkpoint_if_exists()
```

Please notice the arguments passed to the *CheckpointPlugin* constructor:

1. The first parameter is a **storage object**. We decided to allow the checkpointing plugin to load checkpoints from arbitrary storages. The simpler storage, `FileSystemCheckpointStorage`, will use a given directory to store the file for the current experiment (**do not point multiple experiments/runs to the same directory!**). However, we plan to expand this in the future to support network/cloud storages. Contributions on this are welcome :-)! Remember that the `CheckpointStorage` interface is quite simple to implement in a way that best fits your needs.
2. The device used for training. This functionality may be particularly useful in some cases: the plugin will take care of *loading the checkpoint on the correct device, even if the checkpoint was created on a cuda device with a different id*. This means that it can also be used to resume a CUDA checkpoint on CPU. The only caveat is that it cannot be used to load a CPU checkpoint to CUDA. In brief: CUDA -> CPU (OK), CUDA:0 -> CUDA:1 (OK), CPU -> CUDA (NO!). This will also take care of updating the *device* field of the strategy (and plugins) to point to the current device.

The next change is in fact quite minimal. It only requires wrapping the creation of plugins, metrics, and loggers in an "if" that checks if a checkpoint was actually loaded. If a checkpoint is loaded, the resumed strategy already contains the properly restored plugins, metrics, and loggers: *it would be an error to create them*.


```python

# Create the CL strategy (if not already loaded from checkpoint)
if strategy is None:
    plugins = [
        checkpoint_plugin,  # Add the checkpoint plugin to the list!
        ReplayPlugin(mem_size=500),  # Other plugins you want to use
        # ...
    ]

    # Create loggers (as usual)
    # Note that the checkpoint plugin will automatically correctly
    # resume loggers!
    os.makedirs(f'./logs/checkpointing_example',
                exist_ok=True)
    loggers = [
        TextLogger(
            open(f'./logs/checkpointing_example/log.txt', 'w')),
        InteractiveLogger(),
        TensorboardLogger(f'./logs/checkpointing_example')
    ]

    # The W&B logger is correctly resumed without resulting in
    # duplicated runs!
    use_wandb = False
    if use_wandb:
        loggers.append(WandBLogger(
            project_name='AvalancheCheckpointing',
            run_name=f'checkpointing_example'
        ))

    # Create the evaluation plugin (as usual)
    evaluation_plugin = EvaluationPlugin(
        accuracy_metrics(minibatch=False, epoch=True,
                            experience=True, stream=True),
        loss_metrics(minibatch=False, epoch=True,
                        experience=True, stream=True),
        class_accuracy_metrics(
            stream=True
        ),
        loggers=loggers
    )

    # Create the strategy (as usual)
    strategy = Naive(
        model=model,
        optimizer=optimizer,
        criterion=criterion,
        train_mb_size=128,
        train_epochs=2,
        eval_mb_size=128,
        device=device,
        plugins=plugins,
        evaluator=evaluation_plugin
    )
```

Final change: adapt the for loop so that the training stream is iterated starting from `initial_exp`. This variable was created when loading the checkpoint and it tells the next training experience to run. If no checkpoint was found, then its value will be 0.


```python
exit_early = False

for train_task in benchmark.train_stream[initial_exp:]:
    strategy.train(train_task)
    strategy.eval(benchmark.test_stream)

    if exit_early:
        exit(0)
```

A new checkpoint is stored *at the end of each eval phase*! If the program is interrupted before, all progress from the previous eval phase is lost.

Here `exit_early` is a simple placeholder that you can use to experiment a bit. You may obtain a similar effect by stopping this notebook manually, restarting the kernel, and re-running all cells. You will notice that the last checkpoint will be loaded and training will resume as expected.

Usually, `exit_early` should be implemented as a mechanism able to gracefully stop the process. When using SLURM or other schedulers (or even when terminating processes using Ctrl-C), you can catch termination signals and manage them properly so that the process exits after the next eval phase. However, don't worry if the process is killed abruptly: the last checkpoint will be loaded correctly once the experiment is restarted by the scheduler.

That's it for the checkpointing functionality! This is relatively new mechanism and feedbacks on this are warmly welcomed in our [discussions section](https://github.com/ContinualAI/avalanche/discussions) in the repository!

## 🤝 Run it on Google Colab

You can run _this guide_ and play with it on Google Colaboratory by clicking here: [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ContinualAI/avalanche/blob/master/notebooks/how-tos/checkpoints.ipynb)
