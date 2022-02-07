from torch.nn import CrossEntropyLoss
from torch.optim import SGD

from avalanche.benchmarks import SplitMNIST
from avalanche.benchmarks.utils.data_loader import TaskBalancedDataLoader
from avalanche.models import SimpleMLP
from avalanche.training.supervised import Naive, Cumulative


class MyCumulativeStrategy(Cumulative):
    def make_train_dataloader(self, shuffle=True, **kwargs):
        # you can override make_train_dataloader to change the
        # strategy's dataloader
        # remember to iterate over self.adapted_dataset
        self.dataloader = TaskBalancedDataLoader(
            self.adapted_dataset, batch_size=self.train_mb_size
        )


if __name__ == "__main__":
    benchmark = SplitMNIST(n_experiences=5)

    model = SimpleMLP(input_size=784, hidden_size=10)
    opt = SGD(model.parameters(), lr=0.001, momentum=0.9, weight_decay=0.001)

    # we use our custom strategy to change the dataloading policy.
    cl_strategy = MyCumulativeStrategy(
        model,
        opt,
        CrossEntropyLoss(),
        train_epochs=1,
        train_mb_size=512,
        eval_mb_size=512,
    )

    for step in benchmark.train_stream:
        cl_strategy.train(step)
        cl_strategy.eval(step)

    # If you don't use avalanche's strategies you can also use the dataloader
    # directly to iterate the data
    data = step.dataset
    dl = TaskBalancedDataLoader(data)
    for x, y, t in dl:
        # by default minibatches in Avalanche have the form <x, y, ..., t>
        # with arbitrary additional tensors between y and t.
        print(x, y, t)
        break
