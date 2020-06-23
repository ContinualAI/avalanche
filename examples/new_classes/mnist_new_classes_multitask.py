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
from avalanche.benchmarks.scenarios import NCTaskInfo, \
    create_nc_single_dataset_multi_task_scenario
from example_utils import get_default_device


def main():
    device = get_default_device()

    # In this example we can see how we can handle a simple Multi Task scenario.
    # For this example we use the MNIST Dataset without permutations.
    # MNIST is a very popular dataset made of images each containing a single
    # handwritten digit. This means that our dataset contains 10 classes.
    # Here the goal is to learn to correctly classify new classes of digits
    # without forgetting about previously encountered ones.
    #
    # In this example the Naive strategy will be used, which doesn't help in
    # mitigating the catastrophic forgetting that usually occurs in Continual
    # Learning scenarios. So don't worry about the final accuracy metrics!

    # First, let's define the number of tasks in our scenario. The number of
    # classes in our dataset must be divisible without reminder by this value.
    # This means that for MNIST we can only use 1, 2, 5 or 10 as the number of
    # tasks.

    n_tasks = 2  # Note: can only be "1", "2", "5", or "10" for MNIST

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
    # create_nc_single_dataset_multi_task_scenario. This function only requires:
    # - The training set
    # - The test set
    # - The number of tasks
    # We can also randomly shuffle the class order by providing a seed.
    # Note that one can also define a fixed class order by using the
    # fixed_class_order argument!
    nc_scenario = create_nc_single_dataset_multi_task_scenario(
        mnist_train, mnist_test, n_tasks, shuffle=True, seed=1234)

    # Here we create an instance of the network architecture we want to use.
    # For this example we use a "SimpleMLP", which is a simple net with one
    # hidden layer.
    #
    # Our strategy will use a single head approach by saving and loading weights
    # and biases depending on the task label. This means that our MLP must have
    # "dataset classes"/"tasks" output units, which is the number of classes
    # contained in each task. Instead of passing "10/n_tasks" directly, we can
    # use the n_classes and n_tasks fields of our nc_scenario to get the overall
    # number of classes and tasks. This will allow us to change MNIST with any
    # another dataset with a different number of classes!
    output_units = nc_scenario.n_classes//nc_scenario.n_tasks
    model = SimpleMLP(num_classes=output_units)

    # The Evaluation Protocol will keep track of the performance metrics of the
    # Continual Learning strategy. You can specify the metrics to use. For
    # instance, in this example we keep track of the accuracy, catastrophic
    # forgetting, RAM usage values. You can also keep track of complex
    # performance metrics, like the Confusion Matrix! Those will be logged to
    # TensorBoard.
    evalp = EvalProtocol(
        metrics=[ACC(num_class=output_units),  # Accuracy metric
                 CF(num_class=output_units),  # Catastrophic forgetting
                 RAMU(),  # Ram usage
                 CM()],  # Confusion matrix
        tb_logdir='../logs/mnist_test_mt'
    )

    # Here we create an instance of our CL strategy. Naive is a very simple
    # strategy that doesn't really try to mitigate Catastrophic Forgetting.
    # However, it's excellent when toying around with Avalanche for the first
    # time, as it is very fast.
    # We pass multi_head=True to Naive so that, depending on the task label,
    # a different set of weights and biases will be used.
    clmodel = Naive(model, eval_protocol=evalp, device=device, multi_head=True)

    # Let's start looking at the nc_scenario API. First, print the classes
    # contained in each task. We can use original_classes_in_task to obtain the
    # list of classes in each task. You can obtain a different class order by
    # changing the seed passed to create_nc_single_dataset_multi_task_scenario.
    #
    # Beware that "original"_classes_in_task will contain the class IDs from
    # MNIST dataset. Those IDs will be different from the ones used while
    # trainig and testing our Continual Learning strategy: we use an head for
    # each task which means that, for each task, class IDs must always be in
    # range [0, output_units). """Luckly""", nc_scenario already takes care of
    # this. With this in mind, note the difference between the contents
    # original_classes_in_task and classes_in_task.
    print('Task order:')
    for task_idx in range(nc_scenario.n_tasks):
        print('Task', task_idx)
        print('\toriginal classes:',
              nc_scenario.original_classes_in_task[task_idx])
        print('\tmapped classes:',
              nc_scenario.classes_in_task[task_idx])

    print('Starting experiment...')

    results = []  # Results will contain the metrics values for each task
    task_info: NCTaskInfo  # Define the task_info an NCTaskInfo instance

    # Loop over the training tasks
    # For each task, an instance of NCTaskInfo is obtained.
    # This instance exposes a lot of useful information about the current task
    # as well as methods to obtain current / past / future tasks datasets!

    # batch_info: IStepInfo
    # In fact, NCTaskInfo instances are also instances of IStepInfo, which is
    # the more general interface that is shared by both "New Classes" and
    # "New Instances" scenarios (no matter if multi-task or single-task).
    # Being more general, this interface defines only fields and methods shared
    # by all the scenario types. When in need of defining a custom strategy
    # targeting multiple types of scenarios, consider using the IStepInfo
    # interface.

    # Let's see how it looks in practice:
    for task_info in nc_scenario:
        # We can use current_task obtain the task label.
        # Note: you can also keep track of the current task ID by using
        # enumerate in the for loop!
        print("Start of task ", task_info.current_task)

        # One can also use the more generic current_step field, which is a more
        # generic alias also found in the other scenario types.
        # print("Start of batch ", batch_info.current_step)

        # classes_in_this_task contains the list of classes in this task
        # This is way easier than using
        # nc_scenario.original_classes_in_task[task_info.current_task]
        # Note that the original class IDs will be returned!
        print('Classes in this task:', task_info.classes_in_this_task)

        # task_info can be used to access the list of previously
        # encountered / next classes, too!
        # Note that the original class IDs will be returned!
        print('Previous classes:', task_info.previous_classes)
        print('Past + current classes', task_info.classes_seen_so_far)
        print('Next classes:', task_info.future_classes)

        # task_info exposes useful training set related functions.
        # For instance, the following methods return a list of tuples.
        # Each tuple has 2 elements: (Dataset, task_label)
        #
        # cumulative_training_sets = task_info.cumulative_training_sets()
        #
        # past_training_sets = task_info.cumulative_training_sets(
        #     include_current_task=False)
        #
        # future_training_sets = task_info.future_training_sets()
        #
        # complete_training_sets = task_info.complete_training_sets()
        #
        # While the following methods return a single tuple
        # (Dataset, task_labels):
        #
        # second_task_training_set = task_info.task_specific_training_set(1)
        #
        # second_task_training_set = task_info.step_specific_training_set(1)
        #
        # current_training_set = task_info.current_training_set()

        # Let's ge the current training set
        current_training_set = task_info.current_training_set()

        # Note that the first element of the current_training_set tuple is a
        # PyTorch Dataset that can be used to feed a DataLoader and any other
        # PyTorch library functions.
        #
        # The second element of the tuple is the task label.
        training_dataset, t = current_training_set

        # Executing a training step is as simple as calling train_using_dataset
        # on our strategy.
        print('Task {} -> training'.format(t))
        print('This task contains', len(training_dataset), 'patterns')

        # We pass current_training_set tuple here, not just training_dataset!
        clmodel.train_using_dataset(current_training_set, num_workers=4)
        print('Training completed')

        # task_info exposes useful test set related functions, too. As with
        # their training counterparts, the methods used to retrieve the
        # cumulative, past, complete and future test sets will return a list of
        # tuple, with each tuple containing (Dataset, task_label). Methods used
        # to retrieve the current or a specific task test set will just return
        # a tuple with the same format.
        #
        # The following methods will return a list of tuples
        # (Dataset, task_label):
        #
        # cumulative_test_sets = task_info.cumulative_test_sets()
        #
        # past_test_sets = task_info.cumulative_test_sets(
        #     include_current_task=False)
        #
        # future_test_sets = task_info.future_test_sets()
        #
        # complete_test_sets = task_info.complete_test_sets()
        #
        # The following methods will return a tuple (Dataset, task_label):
        #
        # third_task_test_set = task_info.task_specific_test_set(2)
        #
        # third_task_test_set = task_info.step_specific_test_set(2)
        #
        # current_test_set = task_info.current_test_set()

        # Let's test on the complete test set!
        # The test method of our clmodel expects a list of tuples
        # (Dataset, task_label). """Fortunately""",
        # task_info.complete_test_sets() is exactly what are we looking for. We
        # can just feed the test function with its return value and wait for
        # results.
        #
        # Note: Beware that, if you'll ever want to test on a single task,
        # you'll have to wrap the tuple returned by
        # task_info.current_test_set() or task_info.task_specific_test_set(x)
        # inside a list by creating a [(Dataset, task_label)] object.
        complete_test_set = task_info.complete_test_sets()

        print('Computing accuracy on the whole test set')
        results.append(clmodel.test(complete_test_set, num_workers=4))


if __name__ == '__main__':
    main()
