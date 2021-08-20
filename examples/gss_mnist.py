import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss
from torch.optim import SGD

from avalanche.benchmarks import Experience
from avalanche.benchmarks.classic import SplitMNIST
from avalanche.benchmarks.generators.benchmark_generators import \
    data_incremental_benchmark
from avalanche.benchmarks.utils import AvalancheSubset
from avalanche.evaluation.metrics import accuracy_metrics, loss_metrics
from avalanche.logging import InteractiveLogger
from avalanche.training.plugins import EvaluationPlugin
from avalanche.training.strategies import GSS_greedy


class FlattenP(nn.Module):
    '''A nn-module to flatten a multi-dimensional tensor to 2-dim tensor.'''

    def forward(self, x):
        batch_size = x.size(0)   # first dimenstion should be batch-dimension.
        return x.view(batch_size, -1)

    def __repr__(self):
        tmpstr = self.__class__.__name__ + '()'
        return tmpstr


class MLP(nn.Module):
    def __init__(self, sizes, bias=True):
        super(MLP, self).__init__()
        layers = []

        for i in range(0, len(sizes) - 1):

            if i < (len(sizes)-2):
                layers.append(nn.Linear(sizes[i], sizes[i + 1]))

                layers.append(nn.ReLU())
            else:
                layers.append(nn.Linear(sizes[i], sizes[i + 1], bias=bias))

        self.net = nn.Sequential(FlattenP(), *layers)

    def forward(self, x):
        return self.net(x)


def shrinking_experience_size_split_strategy(
        experience: Experience):

    experience_size = 1000 

    exp_dataset = experience.dataset
    exp_indices = list(range(len(exp_dataset)))

    result_datasets = []

    exp_indices = \
        torch.as_tensor(exp_indices)[
            torch.randperm(len(exp_indices))
        ].tolist()

    result_datasets.append(AvalancheSubset(
        exp_dataset, indices=exp_indices[0:experience_size]))

    return result_datasets


def setup_mnist():

    scenario = data_incremental_benchmark(SplitMNIST(
        n_experiences=5, seed=1), experience_size=0, custom_split_strategy=shrinking_experience_size_split_strategy)
    n_inputs = 784
    nh = 100
    nl = 2
    n_outputs = 10
    model = MLP([n_inputs] + [nh] * nl + [n_outputs])

    return model, scenario


if __name__ == "__main__":

    dev = "cuda:0"

    device = torch.device(dev) 

    #_______________________________________Model and scenario
    model, scenario = setup_mnist()

    eval_plugin = EvaluationPlugin(
        accuracy_metrics(epoch=True, experience=True, stream=True),
        loss_metrics(stream=True), loggers=[InteractiveLogger()])

    # _____________________________Strategy

    optimizer = SGD(model.parameters(), lr=0.05)
    strategy = GSS_greedy(model, optimizer, criterion=CrossEntropyLoss(), train_mb_size=10, mem_strength=10, input_size=[
                          1, 28, 28], train_epochs=3, eval_mb_size=10, mem_size=300, evaluator=eval_plugin)

    # ___________________________________________train
    for experience in scenario.train_stream:
        print(">Experience ", experience.current_experience)

        res = strategy.train(experience)

        res = strategy.eval(scenario.test_stream)
