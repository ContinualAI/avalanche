################################################################################
# Copyright (c) 2021 ContinualAI.                                              #
# Copyrights licensed under the MIT License.                                   #
# See the accompanying LICENSE file for terms.                                 #
#                                                                              #
# Date: 05-17-2022                                                             #
# Author: Zhiqiu Lin, Jia Shi                                                  #
# E-mail: zl279@cornell.edu, jiashi@andrew.cmu.edu                             #
# Website: https://clear-benchmark.github.io                                   #
################################################################################

""" This module contains the high-level CLEAR benchmark/factor generator.
In the original CLEAR benchmark paper (https://arxiv.org/abs/2201.06289),
a novel Streaming evaluation protocol is proposed in contrast to traditional
IID evaluation protocol for CL. The major difference lies in that: 

IID Protocol: Sample a test set from current task, which requires splitting
    the data into 7:3 train:test set.
Streaming Protocol: Use the data of next task as the test set for current task,
    which is arguably more realistic since real-world model training and 
    deployment usually takes considerable amount of time. By the time the 
    model is applied, the task has already drifted.
    
We support both evaluation protocols for benchmark construction."""

from pathlib import Path
from typing import Sequence, Union, Any, Optional

from avalanche.benchmarks.datasets.clear import (
    _CLEARImage,
    _CLEARFeature,
    SEED_LIST,
    CLEAR_FEATURE_TYPES,
    _CLEAR_DATA_SPLITS,
)
from avalanche.benchmarks.scenarios.deprecated.generic_benchmark_creation import (
    create_generic_benchmark_from_paths,
    create_generic_benchmark_from_tensor_lists,
)

EVALUATION_PROTOCOLS = ["iid", "streaming"]


def CLEAR(
    *,
    data_name: str = "clear10",
    evaluation_protocol: str = "streaming",
    feature_type: Optional[str] = None,
    seed: Optional[int] = None,
    train_transform: Optional[Any] = None,
    eval_transform: Optional[Any] = None,
    dataset_root: Optional[Union[str, Path]] = None,
):
    """
    Creates a Domain-Incremental benchmark for CLEAR 10 & 100
    with 10 & 100 illustrative classes and an n+1 th background class.

    If the dataset is not present in the computer, **this method will be
    able to automatically download** and store it.

    This generator supports benchmark construction of both 'iid' and 'streaming'
    evaluation_protocol. The main difference is:

    'iid': Always sample testset from current task, which requires
        splitting the data into 7:3 train:test with a given random seed.
    'streaming': Use all data of next task as the testset for current task,
        which does not split the data and does not require random seed.


    The generator supports both Image and Feature (Tensor) datasets.
    If feature_type == None, then images will be used.
    If feature_type is specified, then feature tensors will be used.

    The benchmark instance returned by this method will have two fields,
    `train_stream` and `test_stream`, which can be iterated to obtain
    training and test :class:`Experience`. Each Experience contains the
    `dataset` and the associated task label.

    Note that the train/test streams will still be data of current task,
    regardless of whether evaluation protocol is 'iid' or 'streaming'.
    For 'iid' protocol, train stream is 70% of current task data,
    and test stream is 30% of current task data.
    For 'streaming' protocol, train stream is 100% of current task data,
    and test stream is just a duplicate of train stream.

    The task label 0 will be assigned to each experience.

    :param evaluation_protocol: Choose from ['iid', 'streaming']
        if chosen 'iid', then must specify a seed between [0,1,2,3,4];
        if chosen 'streaming', then the seed will be ignored.
    :param feature_type: Whether to return raw RGB images or feature tensors
        extracted by pre-trained models. Can choose between
        [None, 'moco_b0', 'moco_imagenet', 'byol_imagenet', 'imagenet'].
        If feature_type is None, then images will be returned.
        Otherwise feature tensors will be returned.
    :param seed: If evaluation_protocol is iid, then must specify a seed value
        for train:test split. Choose between [0,1,2,3,4].
    :param train_transform: The transformation to apply to the training data,
        e.g. a random crop, a normalization or a concatenation of different
        transformations (see torchvision.transform documentation for a
        comprehensive list of possible transformations). Defaults to None.
    :param eval_transform: The transformation to apply to the test data,
        e.g. a random crop, a normalization or a concatenation of different
        transformations (see torchvision.transform documentation for a
        comprehensive list of possible transformations). Defaults to None.
    :param dataset_root: The root path of the dataset.
        Defaults to None, which means that the default location for
        str(data_name) will be used.

    :returns: a properly initialized :class:`GenericCLScenario` instance.
    """
    assert data_name in _CLEAR_DATA_SPLITS

    assert evaluation_protocol in EVALUATION_PROTOCOLS, (
        "Must specify a evaluation protocol from " f"{EVALUATION_PROTOCOLS}"
    )

    if evaluation_protocol == "streaming":
        assert seed is None, (
            "Seed for train/test split is not required " "under streaming protocol"
        )
        train_split = "all"
        test_split = "all"
    elif evaluation_protocol == "iid":
        assert seed in SEED_LIST, "No seed for train/test split"
        train_split = "train"
        test_split = "test"
    else:
        raise NotImplementedError()

    if feature_type is None:
        clear_dataset_train = _CLEARImage(
            root=dataset_root,
            data_name=data_name,
            download=True,
            split=train_split,
            seed=seed,
            transform=train_transform,
        )
        clear_dataset_test = _CLEARImage(
            root=dataset_root,
            data_name=data_name,
            download=True,
            split=test_split,
            seed=seed,
            transform=eval_transform,
        )
        train_samples_paths = clear_dataset_train.get_paths_and_targets(
            root_appended=True
        )
        test_samples_paths = clear_dataset_test.get_paths_and_targets(
            root_appended=True
        )
        benchmark_obj = create_generic_benchmark_from_paths(
            train_samples_paths,
            test_samples_paths,
            task_labels=list(range(len(train_samples_paths))),
            complete_test_set_only=False,
            train_transform=train_transform,
            eval_transform=eval_transform,
        )
    else:
        clear_dataset_train = _CLEARFeature(
            root=dataset_root,
            data_name=data_name,
            download=True,
            feature_type=feature_type,
            split=train_split,
            seed=seed,
        )
        clear_dataset_test = _CLEARFeature(
            root=dataset_root,
            data_name=data_name,
            download=True,
            feature_type=feature_type,
            split=test_split,
            seed=seed,
        )
        train_samples = clear_dataset_train.tensors_and_targets
        test_samples = clear_dataset_test.tensors_and_targets

        benchmark_obj = create_generic_benchmark_from_tensor_lists(
            train_samples,
            test_samples,
            task_labels=list(range(len(train_samples))),
            complete_test_set_only=False,
            train_transform=train_transform,
            eval_transform=eval_transform,
        )

    return benchmark_obj


class CLEARMetric:
    """All metrics used in CLEAR paper.
    More information can be found at:
    https://clear-benchmark.github.io/
    """

    def __init__(self):
        super(CLEARMetric, self).__init__()

    def get_metrics(self, matrix):
        """Given an accuracy matrix, returns the 5 metrics used in CLEAR paper

        These are:
            'in_domain' : In-domain accuracy (avg of diagonal)
            'next_domain' : In-domain accuracy (avg of superdiagonal)
            'accuracy' : Accuracy (avg of diagonal + lower triangular)
            'backward_transfer' : BwT (avg of lower triangular)
            'forward_transfer' : FwT (avg of upper triangular)

        :param matrix: Accuracy matrix,
            e.g., matrix[5][0] is the test accuracy on 0-th-task at timestamp 5
        :return: A dictionary containing these 5 metrics
        """
        assert matrix.shape[0] == matrix.shape[1]
        metrics_dict = {
            "in_domain": self.in_domain(matrix),
            "next_domain": self.next_domain(matrix),
            "accuracy": self.accuracy(matrix),
            "forward_transfer": self.forward_transfer(matrix),
            "backward_transfer": self.backward_transfer(matrix),
        }
        return metrics_dict

    def accuracy(self, matrix):
        """
        Average of lower triangle + diagonal
        Evaluate accuracy on seen tasks
        """
        r, _ = matrix.shape
        res = [matrix[i, j] for i in range(r) for j in range(i + 1)]
        return sum(res) / len(res)

    def in_domain(self, matrix):
        """
        Diagonal average
        Evaluate accuracy on the current task only
        """
        r, _ = matrix.shape
        res = [matrix[i, i] for i in range(r)]
        return sum(res) / r

    def next_domain(self, matrix):
        """
        Superdiagonal average
        Evaluate on the immediate next timestamp
        """
        r, _ = matrix.shape
        res = [matrix[i, i + 1] for i in range(r - 1)]
        return sum(res) / (r - 1)

    def forward_transfer(self, matrix):
        """
        Upper trianglar average
        Evaluate generalization to all future task
        """
        r, _ = matrix.shape
        res = [matrix[i, j] for i in range(r) for j in range(i + 1, r)]
        return sum(res) / len(res)

    def backward_transfer(self, matrix):
        """
        Lower triangular average
        Evaluate learning without forgetting
        """
        r, _ = matrix.shape
        res = [matrix[i, j] for i in range(r) for j in range(i)]
        return sum(res) / len(res)


__all__ = ["CLEAR", "CLEARMetric"]

if __name__ == "__main__":
    import sys
    from torchvision import transforms

    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
    )
    transform = transforms.Compose(
        [
            transforms.Resize(224),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ]
    )

    data_name = "clear10"
    root = f"../avalanche_datasets/{data_name}"

    for p in EVALUATION_PROTOCOLS:
        seed_list: Sequence[Optional[int]]
        if p == "streaming":
            seed_list = [None]
        else:
            seed_list = SEED_LIST

        for f in [None] + CLEAR_FEATURE_TYPES[data_name]:
            t = transform if f is None else None
            for seed in seed_list:
                benchmark_instance = CLEAR(
                    evaluation_protocol=p,
                    feature_type=f,
                    seed=seed,
                    train_transform=t,
                    eval_transform=t,
                    dataset_root=root,
                )
                benchmark_instance.train_stream[0]
                # check_vision_benchmark(benchmark_instance)
                print(
                    f"Check pass for {p} protocol, and "
                    f"feature type {f} and seed {seed}"
                )
    sys.exit(0)
