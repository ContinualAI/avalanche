################################################################################
# Copyright (c) 2020 ContinualAI                                               #
# Copyrights licensed under the MIT License.                                   #
# See the accompanying LICENSE file for terms.                                 #
#                                                                              #
# Date: 19-02-2021                                                             #
# Author(s): Tyler L. Hayes                                                    #
# E-mail: contact@continualai.org                                              #
# Website: www.continualai.org                                                 #
################################################################################

# from avalanche.benchmarks.datasets.stream51.stream51_data import STREAM51_DATA
from avalanche.benchmarks.datasets import Stream51
from avalanche.benchmarks.scenarios.generic_scenario_creation import \
    create_generic_scenario_from_lists_of_files
from torchvision import transforms
import math
import os

normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
_default_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    normalize])


def CLStream51(root, scenario="class_instance", transform=_default_transform,
               seed=10, eval_num=None, download=False):
    """ Stream-51 continual scenario generator

        root (string): Root directory path of dataset.
        scenario (string): Stream-51 main scenario. Can be chosen between
        'instance', or 'class_instance.'
        (default: 'class_instance')
        transform: A function/transform that takes in
            a sample and returns a transformed version.
            E.g, ``transforms.RandomCrop`` for images.
        bbox_crop: crop images to object bounding box (default: True)
        ratio: padding for bbox crop (default: 1.10)
        seed: random seed for shuffling classes or instances (default=10)
        eval_num: how many samples to see before evaluating the network
        (default=None)
        download: automatically download the dataset (default=False)

    :returns: it returns a :class:`GenericCLScenario` instance that can be
        iterated.
    """

    train_set = Stream51(root, train=True, download=download)
    test_set = Stream51(root, train=False, download=download)
    train_set._make_dataset(train_set.samples, ordering=scenario, seed=seed)

    # compute number of tasks
    if eval_num is None and scenario == 'instance':
        eval_num = 30000
        num_tasks = math.ceil(
            len(train_set) / eval_num)  # evaluate every 30000 samples
    elif eval_num is None and scenario == 'class_instance':
        eval_num = 10
        num_tasks = math.ceil(
            51 / eval_num)  # evaluate every 10 classes # todo: support
        # num_samples or num_classes for eval_num
    else:
        num_tasks = math.ceil(
            len(train_set) / eval_num)  # evaluate every eval_num samples

    if scenario == 'instance':
        # break files into task lists based on eval_num samples
        train_filelists_paths = []
        start = 0
        for i in range(num_tasks):
            end = min(start + eval_num, len(train_set))
            train_filelists_paths.append(
                [(os.path.join(root, train_set.samples[j][-1]),
                  train_set.samples[j][0]) for j in
                 range(start, end)])
            start = end

        # use all test data for instance ordering
        test_filelists_paths = [(os.path.join(root, test_set.samples[j][-1]),
                                 test_set.samples[j][0]) for
                                j in range(len(test_set))]
        test_ood_filelists_paths = None  # no ood testing for instance ordering
    elif scenario == 'class_instance':
        # break files into task lists based on classes
        train_filelists_paths = []
        test_filelists_paths = []
        test_ood_filelists_paths = []
        class_change = [i for i in range(1, len(train_set.targets)) if
                        train_set.targets[i] != train_set.targets[i - 1]]
        unique_so_far = []
        start = 0
        for i in range(num_tasks):
            if i == num_tasks - 1:
                end = len(train_set)
            else:
                end = class_change[
                    min(eval_num + eval_num * i - 1, len(class_change) - 1)]
            unique_labels = [train_set.targets[k] for k in range(start, end)]
            unique_labels = list(set(unique_labels))
            unique_so_far += unique_labels
            test_files = []
            test_ood_files = []
            for ix, test_label in enumerate(test_set.targets):
                if test_label in unique_so_far:
                    test_files.append(ix)
                else:
                    test_ood_files.append(ix)
            test_filelists_paths.append(
                [(os.path.join(root, test_set.samples[j][-1]),
                  test_set.samples[j][0]) for j in
                 test_files])
            test_ood_filelists_paths.append(
                [(os.path.join(root, test_set.samples[j][-1]),
                  test_set.samples[j][0]) for j in
                 test_ood_files])
            train_filelists_paths.append(
                [(os.path.join(root, train_set.samples[j][-1]),
                  train_set.samples[j][0]) for j in
                 range(start, end)])
            start = end
    else:
        raise NotImplementedError

    scenario_obj = create_generic_scenario_from_lists_of_files(
        train_list_of_files=train_filelists_paths,
        test_list_of_files=test_filelists_paths,
        task_labels=[0 for _ in range(num_tasks)],
        complete_test_set_only=scenario == 'instance',
        train_transform=transform,
        test_transform=transform)

    return scenario_obj


__all__ = [
    'Stream51'
]

if __name__ == "__main__":

    # this below can be taken as a usage example or a simple test script
    import sys
    from torch.utils.data.dataloader import DataLoader
    from torchvision import transforms

    root_dir = '/home/tyler/codes/avalanche/avalanche/data/stream51'
    scenario = CLStream51(root=root_dir, scenario="class_instance")
    for i, batch in enumerate(scenario.train_stream):
        print(i, batch)
        dataset, t = batch.dataset, batch.task_label
        dl = DataLoader(dataset, batch_size=300)

        for mb in dl:
            x, y = mb
            print(x.shape)
            print(y.shape)
        sys.exit(0)
