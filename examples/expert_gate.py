import torch
from torch.utils.data import DataLoader
from torch.utils.data import TensorDataset
from torchvision import transforms
from torch.optim import SGD

from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

from avalanche.models import (
    ExpertGate
)
from avalanche.benchmarks.classic import SplitMNIST
from avalanche.training.supervised import ExpertGateStrategy
from avalanche.models.utils import avalanche_model_adaptation
from avalanche.benchmarks import nc_benchmark


def main():

    # Set device
    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

    # # Uncomment for simpler data
    # # Fake benchmark is (1,1,6)
    # # Data needs to be transformed for AlexNet
    # # Repeat the "channel" as AlexNet expects 3 channel input
    # # Resize as the AlexNet convolution will reduce the data shape 

    # CustomDataAlexTransform = transforms.Compose([
    #     transforms.Lambda(lambda x: x.repeat(3, 1, 1)),      
    #     transforms.Resize((227, 227)),
    # ])

    # scenario = get_custom_benchmark(
    #     use_task_labels=True, train_transform=CustomDataAlexTransform, 
    #     eval_transform=CustomDataAlexTransform, shuffle=True)

    # More resource intensive example
    MNISTAlexTransform = transforms.Compose([
        transforms.Resize((227, 227)),
        transforms.ToTensor(), 
        transforms.Lambda(lambda x: x.repeat(3, 1, 1)),
    ])

    # Note: Must provide task ID for training
    scenario = SplitMNIST(n_experiences=5,
                          return_task_id=True,
                          train_transform=MNISTAlexTransform, 
                          eval_transform=MNISTAlexTransform)

    # Initialize model and specify shape
    model = ExpertGate(shape=(3, 227, 227), device=device) 

    # Vanilla optimization
    optimizer = SGD(model.expert.parameters(), lr=0.001,
                    momentum=0.9, weight_decay=0.0005)

    # Set up strategy
    strategy = ExpertGateStrategy(
        model,
        optimizer,
        device=device,
        train_mb_size=32,
        train_epochs=2,
        eval_every=-1, 
        ae_train_mb_size=32,
        ae_train_epochs=1, 
        ae_lr=1e-4,
    )

    # Train loop
    for experience in (scenario.train_stream):
        t = experience.task_label
        exp_id = experience.current_experience
        training_dataset = experience.dataset
        print()
        print('Task {} batch {} -> train'.format(t, exp_id))
        print('This batch contains', len(training_dataset), 'patterns')
        strategy.train(experience)

    # Evaluation loop
    print("\nEVALUATION")
    for experience in (scenario.train_stream):
        strategy.eval(experience)


def get_custom_benchmark(use_task_labels=False, 
                         shuffle=False, 
                         n_samples_per_class=100, 
                         train_transform=None, 
                         eval_transform=None
                         ):

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


main()
