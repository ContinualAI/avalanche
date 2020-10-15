# Avalanche: a Comprehensive Framework for Continual Learning Research

<p align="center">
<img src="https://www.continualai.org/images/continualai_research_logo.png" width="200"/>
</p>

**Avalanche** is meant to provide a set of tools and resources for easily 
prototype new continual learning algorithms and assess them in a comprehensive
way without effort. This can also help standardize training and evaluation 
protocols in continual learning. 

In order to achieve this goal the *avalanche* framework should be 
general enough to quickly incorporate new CL strategies as well as new 
benchmarks and metrics. 

The framework, based on Pytorch, is than split in three main modules:

- [Benchmarks](avalanche/benchmarks): This module should maintain a uniform
 API for processing data in  a stream and contain all the major CL datasets
 / environments (similar to what has been done for Pytorch-vision).
- [Training](avalanche/training): This module should provide all the
 utilities as well as a standard interface to implement and add a new
  continual learning strategy. All major CL baselines should be provided here.
- [Evaluation](avalanche/evaluation): This modules should provide all the
 utilities and metrics that can help evaluate a CL strategy with respect to
  all the factors we think are important for CL.


Getting Started
----------------

To start using avalanche you have to setup the conda environment first:

```bash
git clone https://github.com/vlomonaco/avalanche.git
cd avalanche
conda env create -f environment.yml
conda activate avalanche-env
```

Then you can use it as follows:

```python

# --- CONFIG
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
n_batches = 5
# ---------

# --- TRANSFORMATIONS
train_transform = transforms.Compose([
    RandomCrop(28, padding=4),
    ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])
test_transform = transforms.Compose([
    ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])
# ---------

# --- SCENARIO CREATION
mnist_train = MNIST('./data/mnist', train=True,
                    download=True, transform=train_transform)
mnist_test = MNIST('./data/mnist', train=False,
                    download=True, transform=test_transform)
nc_scenario = create_nc_single_dataset_sit_scenario(
    mnist_train, mnist_test, n_batches, shuffle=True, seed=1234)
# ---------

# MODEL CREATION
model = SimpleMLP(num_classes=nc_scenario.n_classes)

# DEFINE THE EVALUATION PROTOCOL
evaluation_protocol = EvalProtocol(
    metrics=[ACC(num_class=nc_scenario.n_classes),  # Accuracy metric
             CF(num_class=nc_scenario.n_classes),  # Catastrophic forgetting
             RAMU(),  # Ram usage
             CM()],  # Confusion matrix
    tb_logdir='../logs/mnist_test_sit')

# CREATE THE STRATEGY INSTANCE (NAIVE)
cl_strategy = Naive(
    model, 'classifier', SGD(model.parameters(), lr=0.001, momentum=0.9),
    CrossEntropyLoss(), train_mb_size=100, train_epochs=4, test_mb_size=100,
    evaluation_protocol=evaluation_protocol, device=device)

# TRAINING LOOP
print('Starting experiment...')
results = []
batch_info: NCBatchInfo
for batch_info in nc_scenario:
    print("Start of step ", batch_info.current_step)

    cl_strategy.train(batch_info, num_workers=4)
    print('Training completed')

    print('Computing accuracy on the whole test set')
    results.append(cl_strategy.test(batch_info, DatasetPart.COMPLETE,
                                        num_workers=4))
```


How to contribute
----------------

Check the [CONTRIBUTE.md](CONTRIBUTE.md) file.
