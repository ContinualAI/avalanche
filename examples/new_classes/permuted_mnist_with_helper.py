#!/usr/bin/env python
# -*- coding: utf-8 -*-

################################################################################
# Copyright (c) 2020 ContinualAI Research                                      #
# Copyrights licensed under the CC BY 4.0 License.                             #
# See the accompanying LICENSE file for terms.                                 #
#                                                                              #
# Date: 24-05-2020                                                             #
# Author(s): Gabriele Graffieti                                                #
# E-mail: contact@continualai.org                                              #
# Website: clair.continualai.org                                               #
################################################################################

""" Avalanche usage examples """

# Python 2-3 compatible
from __future__ import print_function
from __future__ import division
from __future__ import absolute_import

import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from avalanche.benchmarks.scenarios import NCTaskInfo
from avalanche.benchmarks.scenarios.new_classes.nc_scenario import \
    NCMultiTaskScenario
from avalanche.evaluation.metrics import ACC, CF, RAMU, CM
from avalanche.extras.models import SimpleMLP
from avalanche.training.strategies import Naive
from avalanche.evaluation import EvalProtocol
from avalanche.benchmarks.classic.cmnist import \
    create_permuted_mnist_benchmark
from example_utils import get_default_device


def main():
    device = get_default_device()

    # In this example we can see how we can handle the Multi Task
    # scenario with the permuted MNIST dataset using the helper function to
    # create the training scenario without having to specify the dataset and the
    # transformation that we have to use. Consider having a
    # look at the simpler MNIST example and at the cifar 100 complete example
    # if you haven't already done it yet.

    # First, let's define the number of incremental tasks in our scenario.
    # The number of incremental tasks is also the number of the different
    # permutations that we want to use to build the benchmark. In this case
    # we choose 5

    n_incremental_tasks = 5

    # We can create our Permuted MNIST scenario with 5 incremental tasks
    # by using the provided create_permuted_mnist_benchmark helper function.
    #
    # Using the function is really straightforward: we have to pass the
    # number of incremental tasks, and the seed used for randomly permute the
    # data. If the seed is not provided the default seed of the python random
    # number generator is used.
    # The helper function below download the MNIST dataset (if not present
    # in the machine) and apply a standard transformation to both test and
    # training data before permute the pixels. It can also be possible to define
    # a custom transformation for both train and test data, and pass them
    # directly to the helper using the fields train_transform and
    # test_transform. In these fields the permutation should not be specified,
    # since it will be added directly by the helper.
    nc_scenario: NCMultiTaskScenario
    nc_scenario = create_permuted_mnist_benchmark(
        incremental_steps=n_incremental_tasks,
        seed=1234,
    )

    # Here we create an instance of the network architecture we want to use.
    # For this example we use a "SimpleMLP", which is a simple net with one
    # hidden layer.
    model = SimpleMLP(num_classes=nc_scenario.n_classes, input_size=28*28)

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
        tb_logdir='../logs/permuted_mnist_test'
    )

    # Here we create an instance of our CL strategy.
    clmodel = Naive(model, eval_protocol=evalp, device=device)

    # The remaining part is very similar to the one you saw in the MNIST example
    print('Batch order:')
    for task_idx, task_classes in enumerate(nc_scenario.classes_in_task):
        print('Batch {}, classes = {}'.format(task_idx, task_classes))

    print('Starting experiment...')

    results = []  # Results will contain the metrics values for each batch
    task_info: NCTaskInfo  # Define the batch_info as an NCTaskInfo instance

    for task_info in nc_scenario:
        print("Start of batch ", task_info.current_task)
        print('Classes in this batch:', task_info.classes_in_this_task)

        # Let's get the current training set
        current_training_set = task_info.current_training_set()
        training_dataset, t = current_training_set

        # Execute a training step
        print('Task {} batch {} -> train'.format(t, task_info.current_task))
        print('This batch contains', len(training_dataset), 'patterns')

        dl = DataLoader(dataset=training_dataset, batch_size=1, shuffle=True)

        for _, (d, _) in enumerate(dl):
            plt.figure()
            plt.imshow(d.squeeze().numpy())
            plt.savefig("mygraph.png")

        clmodel.train_using_dataset(current_training_set, num_workers=4)
        print('Training completed')

        # Test on the complete test set
        complete_test_set = task_info.complete_test_sets()

        print('Computing accuracy on the whole test set')
        results.append(clmodel.test(complete_test_set, num_workers=4))


if __name__ == '__main__':
    main()
