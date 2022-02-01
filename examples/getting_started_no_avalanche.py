################################################################################
# Copyright (c) 2021 ContinualAI.                                              #
# Copyrights licensed under the MIT License.                                   #
# See the accompanying LICENSE file for terms.                                 #
#                                                                              #
# Date: 20-11-2020                                                             #
# Author(s): Vincenzo Lomonaco                                                 #
# E-mail: contact@continualai.org                                              #
# Website: avalanche.continualai.org                                           #
################################################################################

"""
This is the getting started example, without the use of the Avalanche framework.
This is useful to understand why Avalanche is such a wonderful tool.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from os.path import expanduser

import argparse
import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss
from torch.optim import SGD
from torchvision import transforms
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader
import numpy as np


def main(args):

    # Config
    device = torch.device(
        f"cuda:{args.cuda}"
        if torch.cuda.is_available() and args.cuda >= 0
        else "cpu"
    )

    # model
    class SimpleMLP(nn.Module):
        def __init__(self, num_classes=10, input_size=28 * 28):
            super(SimpleMLP, self).__init__()

            self.features = nn.Sequential(
                nn.Linear(input_size, 512),
                nn.ReLU(inplace=True),
                nn.Dropout(),
            )
            self.classifier = nn.Linear(512, num_classes)
            self._input_size = input_size

        def forward(self, x):
            x = x.contiguous()
            x = x.view(x.size(0), self._input_size)
            x = self.features(x)
            x = self.classifier(x)
            return x

    model = SimpleMLP(num_classes=10)

    # CL Benchmark Creation
    print("Creating the benchmark...")
    list_train_dataset = []
    list_test_dataset = []
    rng_permute = np.random.RandomState(0)
    train_transform = transforms.Compose(
        [ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
    )
    test_transform = transforms.Compose(
        [ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
    )

    # for every incremental experience
    idx_permutations = []
    for i in range(2):
        idx_permutations.append(
            torch.from_numpy(rng_permute.permutation(784)).type(torch.int64)
        )

        # add the permutation to the default dataset transformation
        train_transform_list = train_transform.transforms.copy()
        train_transform_list.append(
            transforms.Lambda(
                lambda x, i=i: x.view(-1)[idx_permutations[i]].view(1, 28, 28)
            )
        )
        new_train_transform = transforms.Compose(train_transform_list)

        test_transform_list = test_transform.transforms.copy()
        test_transform_list.append(
            transforms.Lambda(
                lambda x, i=i: x.view(-1)[idx_permutations[i]].view(1, 28, 28)
            )
        )
        new_test_transform = transforms.Compose(test_transform_list)

        # get the datasets with the constructed transformation
        permuted_train = MNIST(
            root=expanduser("~") + "/.avalanche/data/mnist/",
            train=True,
            download=True,
            transform=new_train_transform,
        )
        permuted_test = MNIST(
            root=expanduser("~") + "/.avalanche/data/mnist/",
            train=False,
            download=True,
            transform=new_test_transform,
        )

        list_train_dataset.append(permuted_train)
        list_test_dataset.append(permuted_test)

    # Train
    optimizer = SGD(model.parameters(), lr=0.001, momentum=0.9)
    criterion = CrossEntropyLoss()

    model.to(device)
    print("Starting training...")
    for task_id, train_dataset in enumerate(list_train_dataset):
        print("Starting task:", task_id)
        train_data_loader = DataLoader(
            train_dataset, batch_size=32, shuffle=False
        )

        for ep in range(1):
            print("Epoch: ", ep)
            for iteration, (train_mb_x, train_mb_y) in enumerate(
                train_data_loader
            ):
                optimizer.zero_grad()
                train_mb_x = train_mb_x.to(device)
                train_mb_y = train_mb_y.to(device)

                # Forward
                logits = model(train_mb_x)
                # Loss
                loss = criterion(logits, train_mb_y)
                if iteration % 100 == 0:
                    print("Iter: {}, Loss: {}".format(iteration, loss.item()))
                # Backward
                loss.backward()
                # Update
                optimizer.step()

        # Test
        acc_results = []
        print("Starting testing...")
        for task_id, test_dataset in enumerate(list_test_dataset):

            test_data_loader = DataLoader(test_dataset, batch_size=32)

            correct = 0
            for iteration, (test_mb_x, test_mb_y) in enumerate(
                test_data_loader
            ):

                # Move mini-batch data to device
                test_mb_x = test_mb_x.to(device)
                test_mb_y = test_mb_y.to(device)

                # Forward
                test_logits = model(test_mb_x)
                preds = torch.argmax(test_logits.long(), dim=1)

                # compute acc
                correct += (test_mb_y.eq(preds)).sum().item()

            print("Task:", task_id)
            acc = (correct / len(test_dataset)) * 100
            print("Accuracy results: ", acc)
            acc_results.append(acc)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--cuda",
        type=int,
        default=0,
        help="Select zero-indexed cuda device. -1 to use CPU.",
    )
    args = parser.parse_args()
    main(args)
