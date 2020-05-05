# Avalanche: a Comprehensive Framework for Continual Learning Research

**Avalanche** is meant to provide a set of tools and resources for easily 
prototype new continual learning algorithms and assess them in a comprehensive
way without effort. This can also help standardize training and evaluation 
protocols in continual learning. 

In order to achieve this goal the *avalanche* framework should be 
general enough to quickly incorporate new CL strategies as well as new 
benchmarks and metrics. While it would be great to be DL framework independent, 
for simplicity I believe we should stick to Pytorch which today is becoming 
the standard de-facto for machine learning research.

The framework is than split in three main modules:

- [Benchmarks](avalanche/benchmarks): This module should maintain a uniform
 API for processing data in  a stream and contain all the major CL datasets
 / environments (similar to what has been done for Pytorch-vision).
- [Training](avalanche/training): This module should provide all the
 utilities as well as a standard interface to implement and add a new
  continual learning strategy. All major CL baselines should be provided here.
- [Evaluation](avalanche/evaluation): This modules should provide all the
 utilities and metrics that can help evaluate a CL strategy with respect to
  all the factors we think are important for CL.
  
Project Structure
-----------------

- [avalanche](avalanche)
    - [Benchmarks](avalanche/benchmarks)
        -  [cdata_loaders](avalanche/benchmarks/cdata_loaders)
        - [datasets_envs](avalanche/benchmarks/datasets_envs)
        - [utils.py](avalanche/benchmarks/utils.py)
    - [Training](avalanche/training)
    - [Evaluation](avalanche/evaluation)
    - [Extra](avalanche/extras)
- [Examples](examples)
- [Tests](tests)
- [docs](docs)


Getting Started
----------------

To start using avalanche you have to setup the conda environment first:

```bash
conda env create -f environment.yml
conda activate avalanche-env
```

Then you can use it as follows:

```python
from avalanche.benchmarks import CMNIST
from avalanche.evaluation.metrics import ACC, CF, RAMU, CM
from avalanche.extras.models import SimpleMLP
from avalanche.training.strategies import Naive
from avalanche.evaluation import EvalProtocol

from torch.utils.tensorboard import SummaryWriter

# Tensorboard setup
exp_name = "mnist_test"
log_dir = '../logs/' + exp_name
writer = SummaryWriter(log_dir)

# load the model with PyTorch for example
model = SimpleMLP()

# load the benchmark as a python iterator object
cdata = CMNIST()

# Eval Protocol
evalp = EvalProtocol(metrics=[ACC, CF, RAMU, CM], tb_writer=writer)

# adding the CL strategy
clmodel = Naive(model, eval_protocol=evalp)

# getting full test set beforehand
test_full = cdata.get_full_testset()

results = []

# loop over the training incremental batches
for i, (x, y, t) in enumerate(cdata):

    # training over the batch
    print("Batch {0}, task {1}".format(i, t))
    clmodel.train(x, y, t)

    # here we could get the growing test set too
    # test_grow = cdata.get_growing_testset()

    # testing
    results.append(clmodel.test(test_full))

writer.close()
```
  
  

