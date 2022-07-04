import torch
from torch.utils.data import DataLoader
from torch.utils.data import TensorDataset
from torchvision import transforms

from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

from avalanche.models import (
    ExpertGate
)
from avalanche.benchmarks.classic import SplitMNIST
from avalanche.training.supervised import ExpertGateStrategy
from avalanche.models.utils import avalanche_model_adaptation
from avalanche.benchmarks import nc_benchmark


def test_expertgate():
    # only on mulit-task scenarios
    # eval on future tasks is not allowed

    # mnist shape is (1,28,28)
    # fake benchmark is (1,1,6)
    # scenario = SplitMNIST(n_experiences=5, seed=1234, return_task_id=True)

    AlexTransform = transforms.Compose([
        transforms.Resize((227, 227)),
        transforms.ToTensor(), 
        transforms.Lambda(lambda x: x.repeat(3, 1, 1)),      
    ])

    # scenario = get_custom_benchmark(
    #     use_task_labels=True, train_transform=AlexTransform, eval_transform=AlexTransform)

    scenario = SplitMNIST(n_experiences=10, seed=1,
                          return_task_id=True,
                          train_transform=AlexTransform, eval_transform=AlexTransform)

    # # 227 227
    model = ExpertGate(num_classes=scenario.n_classes, shape=(3, 227, 227)) 
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-1)
    strategy = ExpertGateStrategy(
        model,
        optimizer,
        train_mb_size=50,
        device="cpu",
        train_epochs=50,
        eval_every=-1, 
        ae_train_mb_size=500,
        ae_train_epochs=1, 
        ae_lr=2e-2,
    )

    # train and test loop
    for experience in (scenario.train_stream)[:1]:
        t = experience.task_label
        exp_id = experience.current_experience
        training_dataset = experience.dataset
        print('Task {} batch {} -> train'.format(t, exp_id))
        print('This batch contains', len(training_dataset), 'patterns')
        print(model.parameters())
        print(model.expert.parameters())
        # print("Targets: ", experience.dataset.targets)
        strategy.train(experience)
    # print((scenario.test_stream)[:1].task_label)
    # strategy.eval(scenario.test_stream[:1])


def get_custom_benchmark(use_task_labels=False, shuffle=False, n_samples_per_class=100, train_transform=None, eval_transform=None):

    dataset = make_classification(
        n_samples=10 * n_samples_per_class,
        n_classes=10,
        n_features=6,
        n_informative=6,
        n_redundant=0,
    )

    X = torch.from_numpy(dataset[0]).float()
    y = torch.from_numpy(dataset[1]).long()

    train_X, test_X, train_y, test_y = train_test_split(
        X, y, train_size=0.6, shuffle=True, stratify=y
    )

    train_dataset = TensorDataset(train_X, train_y)
    test_dataset = TensorDataset(test_X, test_y)
    my_nc_benchmark = nc_benchmark(
        train_dataset,
        test_dataset,
        5,
        task_labels=use_task_labels,
        shuffle=shuffle,
        train_transform=train_transform,
        eval_transform=eval_transform,
    )
    return my_nc_benchmark


test_expertgate()
