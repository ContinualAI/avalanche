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
    # Fake benchmark is (1,1,6)
    # Data needs to be transformed for AlexNet
    # Repeat the "channel" as AlexNet expects 3 channel input
    # Resize as the AlexNet convolution will reduce the data shape 
    AlexTransform = transforms.Compose([
        transforms.Lambda(lambda x: x.repeat(3, 1, 1)),      
        transforms.Resize((227, 227)),
    ])

    # Set up dummy data scenario
    scenario = get_custom_benchmark(
        use_task_labels=True, train_transform=AlexTransform, eval_transform=AlexTransform, shuffle=True)

    # Initialize model and specify shape
    model = ExpertGate(num_classes=scenario.n_classes, shape=(3, 227, 227)) 

    # Vanilla optimization
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)

    # Set up strategy
    strategy = ExpertGateStrategy(
        model,
        optimizer,
        device="cpu",
        train_mb_size=200,
        train_epochs=3,
        eval_every=-1, 
        ae_train_mb_size=10,
        ae_train_epochs=5, 
        ae_lr=2e-2,
    )

    # Train loop
    for experience in (scenario.train_stream):
        t = experience.task_label
        exp_id = experience.current_experience
        training_dataset = experience.dataset
        print('Task {} batch {} -> train'.format(t, exp_id))
        print('This batch contains', len(training_dataset), 'patterns')
        strategy.train(experience)

    # Evaluation loop
    print("\nEVALUATION")
    for experience in (scenario.train_stream):
        strategy.eval(experience)


def get_custom_benchmark(use_task_labels=False, shuffle=False, n_samples_per_class=100, train_transform=None, eval_transform=None):

    dataset = make_classification(
        n_samples=10 * n_samples_per_class,
        n_classes=5,
        n_features=6,
        n_informative=6,
        n_redundant=0,
    )

    X = torch.from_numpy(dataset[0]).float()
    y = torch.from_numpy(dataset[1]).long()

    train_X, test_X, train_y, test_y = train_test_split(
        X, y, train_size=0.8, shuffle=True, stratify=y
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
