#!/usr/bin/env python
# -*- coding: utf-8 -*-

################################################################################
# Copyright (c) 2020 ContinualAI Research                                      #
# Copyrights licensed under the CC BY 4.0 License.                             #
# See the accompanying LICENSE file for terms.                                 #
#                                                                              #
# Date: 24-05-2020                                                             #
# Author(s): Lorenzo Pellegrini                                                #
# E-mail: contact@continualai.org                                              #
# Website: clair.continualai.org                                               #
################################################################################

""" Avalanche usage examples """

# Python 2-3 compatible
from __future__ import print_function
from __future__ import division
from __future__ import absolute_import

from torchvision import transforms
from torchvision.datasets import CIFAR100
from torchvision.transforms import ToTensor

from avalanche.evaluation.metrics import ACC, CF, RAMU, CM
from avalanche.extras.models import SimpleMLP
from avalanche.training.strategies import Naive
from avalanche.evaluation import EvalProtocol
from avalanche.benchmarks.scenarios import NCBatchInfo, \
    create_nc_single_dataset_sit_scenario
from mnist_example_utils import get_default_device


def main():
    device = get_default_device()

    # In this example we can see how we can handle a more complex Single
    # Incremental Task scenario with the CIFAR100 dataset. Consider having a
    # look at the simpler MNIST example if you haven't already done it yet.
    #
    # We want to create a scenario in which the first batch contains half of the
    # classes. We'll refer to this batch as the "pretrain" batch while next
    # batches will be the "incremental" ones.
    #
    # First, let's define the number of incremental batches in our scenario.
    # The first 50 classes will be in the pretrain batch, so this means that we
    # can only use 1, 2, 5 or 10, 25 and 50 as the number of incremental
    # batches.

    n_incremental_batches = 10

    # This means that the overall number of batches will be
    # n_incremental_batches + 1
    n_batches = n_incremental_batches + 1

    # Define the transformations. For the training patterns we run a RandomCrop
    # and a horizontal flip which will act as our data augmentation strategy.
    # The crop is followed by a simple normalization, which is also applied to
    # test patterns.
    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465),
                             (0.2023, 0.1994, 0.2010))
    ])

    test_transform = transforms.Compose([
        ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465),
                             (0.2023, 0.1994, 0.2010))
    ])

    # Let's create instances of our train and test sets. We can use the CIFAR100
    # Dataset from the official torchvision package.
    cifar_train = CIFAR100('./data/cifar100', train=True,
                           download=True, transform=train_transform)
    cifar_test = CIFAR100('./data/cifar100', train=False,
                          download=True, transform=test_transform)

    # We can create our "New Classes" scenario by using the provided
    # create_nc_single_dataset_sit_scenario function.
    #
    # We can use the per_batch_classes parameter to create a scenario with an
    # arbitrary number of classes per batch. per_batch_classes must be a
    # dictionary whose keys are batch IDs and the respective value has to be the
    # number of classes to assign to that batch. Other batches will contain an
    # equal number of classes among the remaining ones.
    nc_scenario = create_nc_single_dataset_sit_scenario(
        cifar_train, cifar_test, n_batches, shuffle=True, seed=1234,
        per_batch_classes={0: 50})

    # Here we create an instance of the network architecture we want to use.
    # For this example we use a "SimpleMLP", which is a simple net with one
    # hidden layer.
    model = SimpleMLP(num_classes=nc_scenario.n_classes, input_size=32*32*3)

    # The Evaluation Protocol will keep track of the performance metrics of the
    # Continual Learning strategy. You can specify the metrics to use. For
    # instance, in this example we keep track of the accuracy, catastrophic
    # forgetting, RAM usage values. You can also keep track of complex
    # performance metrics, like the Confusion Matrix! Those will be logged to
    # TensorBoard.
    evalp = EvalProtocol(
        metrics=[ACC(num_class=nc_scenario.n_classes),  # Accuracy metric
                 CF(num_class=nc_scenario.n_classes),  # Catastrophic forgetting
                 RAMU(),  # Ram usage
                 CM()],  # Confusion matrix
        tb_logdir='../logs/cifar100_test'
    )

    # Here we create an instance of our CL strategy.
    clmodel = Naive(model, eval_protocol=evalp, device=device)

    # The remaining part is very similar to the one you saw in the MNIST example
    print('Batch order:')
    for batch_idx, batch_classes in enumerate(nc_scenario.classes_in_batch):
        print('Batch {}, classes = {}'.format(batch_idx, batch_classes))

    print('Starting experiment...')

    results = []  # Results will contain the metrics values for each batch
    batch_info: NCBatchInfo  # Define the batch_info as an NCBatchInfo instance

    for batch_info in nc_scenario:
        print("Start of batch ", batch_info.current_batch)
        # Note that batch 0 contains 50 classes!
        print('Classes in this batch:', batch_info.classes_in_this_batch)

        # Let's get the current training set
        current_training_set = batch_info.current_training_set()
        training_dataset, t = current_training_set

        # Execute a training step
        print('Task {} batch {} -> train'.format(t, batch_info.current_batch))
        print('This batch contains', len(training_dataset), 'patterns')

        clmodel.train_using_dataset(current_training_set, num_workers=4)
        print('Training completed')

        # Test on the complete test set
        complete_test_set = batch_info.complete_test_sets()

        print('Computing accuracy on the whole test set')
        results.append(clmodel.test(complete_test_set, num_workers=4))


if __name__ == '__main__':
    main()
