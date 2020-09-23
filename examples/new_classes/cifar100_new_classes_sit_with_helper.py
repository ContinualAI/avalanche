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

from avalanche.benchmarks.scenarios import NCBatchInfo
from avalanche.benchmarks.scenarios.new_classes.nc_scenario import \
    NCSingleTaskScenario
from avalanche.evaluation.metrics import ACC, CF, RAMU, CM
from avalanche.extras.models import SimpleMLP
from avalanche.training.strategies import Naive
from avalanche.evaluation import EvalProtocol
from avalanche.benchmarks.classic.ccifar100 import \
    create_cifar100_benchmark
from example_utils import get_default_device


def main():
    device = get_default_device()

    # In this example we can see how we can handle a more complex Single
    # Incremental Task scenario with the CIFAR100 dataset using the helper
    # function to create the training scenario without having to specify
    # the dataset and the transformation that we have to use. Consider having a
    # look at the simpler MNIST example and at the cifar 100 complete example
    # if you haven't already done it yet.

    # We want to create a scenario in which the first batch contains half of the
    # classes. We'll refer to this batch as the "pretrain" batch while next
    # batches will be the "incremental" ones.
    #
    # First, let's define the number of incremental batches in our scenario.
    # The first 50 classes will be in the pretrain batch, so this means that we
    # can only use 1, 2, 5 or 10, 25 and 50 as the number of incremental
    # batches.

    n_incremental_batches = 25

    # We can create our CIFAR100 "New Classes" scenario with 10 incremental
    # batches and one pretrain batch with half of the classes by using the
    # provided create_single_task_cifar100 helper function.
    #
    # Using the function is really straightforward: we have to pass the
    # number of incremental batches, if we want a "pretrain" batch 0 before the
    # incremental batches that contains half of the classes, if we want to
    # the task ID (so we want to create a MT scenario) or we don't want the
    # tasks ids, so we are in a SIT scenario, and a seed to initialize the
    # random number generator (used for example to shuffle the classes.
    # Another paramenter not used in this example is fixed_class_order,
    # which can be used when we want to enforce a particular order of the
    # classes and not shuffle them randomly. The fixed_class_order paramenter
    # is a list on class labels (integers).
    # The helper function below download the cifar100 dataset (if not present
    # in the machine) and apply a standard transformation to both test and
    # training data.
    nc_scenario: NCSingleTaskScenario
    nc_scenario = create_cifar100_benchmark(
        incremental_steps=n_incremental_batches,
        first_batch_with_half_classes=True,
        return_task_id=False,
        seed=1234
    )

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
        s = set()
        s.update(current_training_set[0].targets)
        print(s)

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
