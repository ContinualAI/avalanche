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
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor, RandomCrop

from avalanche.evaluation.metrics import ACC, CF, RAMU, CM
from avalanche.extras.models import SimpleMLP
from avalanche.training.strategies import Naive
from avalanche.evaluation import EvalProtocol
from avalanche.benchmarks.scenarios import NCBatchInfo, \
    create_nc_single_dataset_sit_scenario
from example_utils import get_default_device


def main():
    device = get_default_device()

    # In this example we can see how we can handle a simple
    # Single Incremental Task (a.k.a. Task-Free) scenario. For this example we
    # use the MNIST Dataset without permutations. MNIST is a very popular
    # dataset made of images each containing a single handwritten digit. This
    # means that our dataset contains 10 classes. Here the goal is to learn to
    # correctly classify new classes of digits without forgetting about
    # previously encountered ones.
    #
    # In this example the Naive strategy will be used, which doesn't help in
    # mitigating the catastrophic forgetting that usually occurs in Continual
    # Learning scenarios. So don't worry about the final accuracy metrics!

    # First, let's define the number of batches in our scenario. The number of
    # classes in our dataset must be divisible without reminder by this value.
    # This means that for MNIST we can only use 1, 2, 5 or 10 as the number of
    # batches.

    n_batches = 5  # Note: can only be "1", "2", "5", or "10" for MNIST

    # Define the transformations. For the training patterns we run a RandomCrop
    # which will act as our data augmentation strategy. The crop is followed by
    # a simple normalization, which is also applied to test patterns.
    train_transform = transforms.Compose([
        RandomCrop(28, padding=4),
        ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    test_transform = transforms.Compose([
        ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    # Let's create instances of our train and test sets. We can use the MNIST
    # Dataset from the official torchvision package. Fortunately, the
    # constructor will download and store the dataset for us.
    mnist_train = MNIST('./data/mnist', train=True,
                        download=True, transform=train_transform)
    mnist_test = MNIST('./data/mnist', train=False,
                       download=True, transform=test_transform)

    # We can create our "New Classes" scenario by using the provided
    # create_nc_single_dataset_sit_scenario. This function only requires:
    # - The training set
    # - The test set
    # - The number of batches
    # We can also randomly shuffle the class order by providing a seed.
    # Note that one can also define a fixed class order by using the
    # fixed_class_order argument!
    nc_scenario = create_nc_single_dataset_sit_scenario(
        mnist_train, mnist_test, n_batches, shuffle=True, seed=1234)

    # Here we create an instance of the network architecture we want to use.
    # For this example we use a "SimpleMLP", which is a simple net with one
    # hidden layer.
    #
    # When creating the model, we must pass the number of classes contained in
    # our dataset. Instead of passing "10" directly, we can use the n_classes
    # field of our nc_scenario to get the overall number of classes. This  will
    # allow us to change MNIST with any other another dataset with a different
    # number of classes!
    model = SimpleMLP(num_classes=nc_scenario.n_classes)

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
        tb_logdir='../logs/mnist_test_sit'
    )

    # Here we create an instance of our CL strategy. Naive is a very simple
    # strategy that doesn't really try to mitigate Catastrophic Forgetting.
    # However, it's excellent when toying around with Avalanche for the first
    # time, as it is very fast.
    clmodel = Naive(model, eval_protocol=evalp, device=device)

    # Let's start looking at the nc_scenario API. First, print the classes
    # contained in each batch. We can use classes_in_batch to obtain the list of
    # classes in each batch. You can obtain a different class order by changing
    # the seed passed to create_nc_single_dataset_sit_scenario.
    print('Batch order:')
    for batch_idx, batch_classes in enumerate(nc_scenario.classes_in_batch):
        print('Batch {}, classes = {}'.format(batch_idx, batch_classes))

    print('Starting experiment...')

    results = []  # Results will contain the metrics values for each batch
    batch_info: NCBatchInfo  # Define the batch_info as an NCBatchInfo instance

    # Loop over the training incremental batches
    # For each batch, an instance of NCBatchInfo is obtained.
    # This instance exposes a lot of useful information about the current batch
    # as well as methods to obtain current / past / future batch datasets!

    # batch_info: IStepInfo
    # In fact, NCBatchInfo instances are also instances of IStepInfo, which is
    # the more general interface that is shared by both "New Classes" and
    # "New Instances" scenarios (no matter if multi-task or single-task).
    # Being more general, this interface defines only fields and methods shared
    # by all the scenario types. When in need of defining a custom strategy
    # targeting multiple types of scenarios, consider using the IStepInfo
    # interface.

    # Let's see how it looks in practice:
    for batch_info in nc_scenario:
        # We can use current_batch to obtain the batch ID.
        # Note: you can also keep track of the current batch ID by using
        # enumerate in the for loop!
        print("Start of batch ", batch_info.current_batch)

        # One can also use the more generic current_step field, which is a more
        # generic alias also found in the other scenario types.
        # print("Start of batch ", batch_info.current_step)

        # classes_in_this_batch contains the list of classes in this batch
        # This is way easier than using
        # nc_scenario.classes_in_batch[batch_info.current_batch]
        print('Classes in this batch:', batch_info.classes_in_this_batch)

        # batch_info can be used to access the list of previously
        # encountered / next classes, too!
        print('Previous classes:', batch_info.previous_classes)
        print('Past + current classes', batch_info.classes_seen_so_far)
        print('Next classes:', batch_info.future_classes)

        # batch_info exposes useful training set related functions.
        # For instance, the following methods return a list of tuples.
        # Each tuple has 2 elements: (Dataset, task_label)
        #
        # cumulative_training_sets = batch_info.cumulative_training_sets()
        #
        # past_training_sets = batch_info.cumulative_training_sets(
        #     include_current_batch=False)
        #
        # future_training_sets = batch_info.future_training_sets()
        #
        # complete_training_sets = batch_info.complete_training_sets()
        #
        # While the following methods return a single tuple
        # (Dataset, task_labels):
        #
        # second_batch_training_set = batch_info.batch_specific_training_set(1)
        #
        # second_batch_training_set = batch_info.step_specific_training_set(1)
        #
        # current_training_set = batch_info.current_training_set()

        # Let's ge the current training set
        current_training_set = batch_info.current_training_set()

        # Note that the first element of the current_training_set tuple is a
        # PyTorch Dataset that can be used to feed a DataLoader and any other
        # PyTorch library functions.
        #
        # The second element of the tuple is the task label.
        #
        # Remember: we are running a Single Incremental Task scenario, so the
        # task label will always be 0.
        training_dataset, t = current_training_set

        # Executing a training step is as simple as calling train_using_dataset
        # on our strategy.
        print('Task {} batch {} -> train'.format(t, batch_info.current_batch))
        print('This batch contains', len(training_dataset), 'patterns')

        # We pass current_training_set tuple here, not just training_dataset!
        clmodel.train_using_dataset(current_training_set, num_workers=4)
        print('Training completed')

        # batch_info exposes useful test set related functions, too. As with
        # their training counterparts, the methods used to retrieve the
        # cumulative, past, complete and future test sets will return a list of
        # tuple, with each tuple containing (Dataset, task_label). Methods used
        # to retrieve the current or a specific batch test set will just return
        # a tuple with the same format.
        #
        # The following methods will return a list of tuples
        # (Dataset, task_label):
        #
        # cumulative_test_sets = batch_info.cumulative_test_sets()
        #
        # past_test_sets = batch_info.cumulative_test_sets(
        #    include_current_batch=False)
        #
        # future_test_sets = batch_info.future_test_sets()
        #
        # complete_test_sets = batch_info.complete_test_sets()
        #
        # The following methods will return a tuple (Dataset, task_label):
        #
        # third_batch_test_set = batch_info.batch_specific_test_set(2)
        #
        # third_batch_test_set = batch_info.step_specific_test_set(2)
        #
        # current_test_set = batch_info.current_test_set()

        # Let's test on the complete test set!
        # The test method of our clmodel expects a list of tuples
        # (Dataset, task_label). """Fortunately""",
        # batch_info.complete_test_sets() is exactly what are we looking for.
        # We can just feed the test function with its return value and wait for
        # results.
        #
        # Note: Beware that, if you'll ever want to test on a single batch,
        # you'll have to wrap the tuple returned by
        # batch_info.current_test_set() or batch_info.batch_specific_test_set(x)
        # inside a list by creating a [(Dataset, task_label)] object.
        complete_test_set = batch_info.complete_test_sets()

        print('Computing accuracy on the whole test set')
        results.append(clmodel.test(complete_test_set, num_workers=4))


if __name__ == '__main__':
    main()
