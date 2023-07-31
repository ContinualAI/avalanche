################################################################################
# Copyright (c) 2021 ContinualAI.                                              #
# Copyrights licensed under the MIT License.                                   #
# See the accompanying LICENSE file for terms.                                 #
#                                                                              #
# Date: 01-12-2020                                                             #
# Author(s): Antonio Carta, Lorenzo Pellegrini                                 #
# E-mail: contact@continualai.org                                              #
# Website: avalanche.continualai.org                                           #
################################################################################
"""
    Avalanche supports data loading using pytorch's dataloaders.
    This module provides custom dataloaders for continual learning such as
    support for balanced dataloading between different tasks or balancing
    between the current data and the replay memory.
"""
from typing import (
    Any,
    Dict,
    Iterator,
    List,
    Optional,
    Sequence,
    Sized,
    Union,
)
import numpy as np

from torch.utils.data import DistributedSampler, Dataset
from torch.utils.data.dataloader import DataLoader

from avalanche.benchmarks.utils.data import AvalancheDataset
from avalanche.benchmarks.utils.data_attribute import DataAttribute
from avalanche.benchmarks.utils.ffcv_support.ffcv_components import (
    HybridFfcvLoader,
    has_ffcv_support,
)
from avalanche.distributed.distributed_helper import DistributedHelper

from torch.utils.data.sampler import Sampler, BatchSampler
from torch.utils.data import ConcatDataset


def return_identity(x):
    """
    The identity function. Can be wrapped in 'partial'
    to act as a getter function.
    Used to avoid lambda functions that cannot be pickled.
    """
    return x


def collate_from_data_or_kwargs(data, kwargs):
    if "collate_fn" in kwargs:
        return
    elif hasattr(data, "collate_fn"):
        kwargs["collate_fn"] = data.collate_fn


class MultiDatasetDataLoader:
    """Custom data loader for multiple datasets."""

    def __init__(
        self,
        datasets: Sequence[AvalancheDataset],
        batch_sizes: Sequence[int],
        termination_dataset: int = -1,
        oversample_small_datasets: bool = False,
        distributed_sampling: bool = True,
        never_ending: bool = False,
        **kwargs
    ):
        """Custom data loader for loading batches from multiple datasets.

        This dataloader iterates in parallel multiple datasets which are used
        to create mini-batches by concatenating their data together.
        The number of examples from each dataset in each mini-batch
        is defined by the `batch_sizes` parameter.

        The length of the loader (that is, the number of iterations in an
        epoch) is determined by the `termination_dataset`.

        The `oversample_small_datasets` parameter can be used to control what
        to do when smaller datasets are consumed and the epoch is not ended
        yet.

        :param datasets: A list of :class:`AvalancheDataset`.
        :param batch_sizes: A list of int, describing the number of examples
            per minibatch to take from each dataset.
        :param termination_dataset: The index of the dataset used to determine
            the number of iterations per epoch. Defaults to -1, which means
            that the highest number of iterations among all datasets will be
            used.
        :param oversample_small_datasets: If `True`, smaller datasets will be
            cycled again until the epoch is ended. If it is `False`, smaller
            datasets will not be cycled again, which means that some datasets
            will not contribute to the minibatch composition near the end of
            the epoch.
        :param distributed_sampling: If True, apply the PyTorch
            :class:`DistributedSampler`. Defaults to True.
            Note: the distributed sampler is not applied if not running
            a distributed training, even when True is passed.
        :param never_ending: If True, this data loader will cycle indefinitely
            by iterating over all datasets again and again and the epoch will
            never end. In this case, the `termination_dataset` and
            `oversample_small_datasets` parameters are ignored. Defaults to
            False.
        :param kwargs: data loader arguments used to instantiate the loader for
            each dataset. See PyTorch :class:`DataLoader`.
        """
        if "collate_mbatches" in kwargs:
            raise ValueError(
                "collate_mbatches is not needed anymore and it has been "
                "deprecated. Data loaders will use the collate function"
                "`data.collate_fn`."
            )

        if termination_dataset >= len(datasets):
            raise ValueError(
                "termination_dataset must be the index of the "
                "dataset used to determine the termination of an epoch"
            )

        distributed_sampling = distributed_sampling and DistributedHelper.is_distributed

        self.datasets = list(datasets)
        self.oversample_small_datasets: bool = oversample_small_datasets
        self.batch_sizes: List[int] = list(batch_sizes)
        self.distributed_sampling: bool = distributed_sampling
        self.loader_kwargs: Dict[str, Any] = kwargs
        self.termination_dataset: int = termination_dataset
        self.never_ending: bool = never_ending

        self.loader_kwargs, self.ffcv_args = self._extract_ffcv_args(self.loader_kwargs)

        # Only used if persistent_workers == True in loader kwargs
        self._persistent_loader: Optional[DataLoader] = None

        if "collate_fn" not in self.loader_kwargs:
            self.loader_kwargs["collate_fn"] = self.datasets[0].collate_fn

        if self.never_ending:
            # Infinite data loader
            self.termination_dataset = -1
            self.n_iterations = 10**10
            self.oversample_small_datasets = True
        else:
            # Estimate number of iterations per epoch
            loaders_len = np.full(
                (
                    len(
                        self.datasets,
                    )
                ),
                -1,
            )

            if self.termination_dataset < 0:
                for i, (data_subset, subset_mb_size) in enumerate(
                    zip(self.datasets, self.batch_sizes)
                ):
                    loaders_len[i] = len(
                        _make_data_loader(
                            data_subset,
                            distributed_sampling,
                            self.loader_kwargs,
                            subset_mb_size,
                            force_no_workers=True,
                        )[0]
                    )
            else:
                loaders_len[self.termination_dataset] = len(
                    _make_data_loader(
                        self.datasets[self.termination_dataset],
                        distributed_sampling,
                        self.loader_kwargs,
                        self.batch_sizes[self.termination_dataset],
                        force_no_workers=True,
                    )[0]
                )

            self.termination_dataset = loaders_len.argmax().item()
            self.n_iterations = loaders_len.max().item()

    def __iter__(self):
        # Adapted from the __iter__ of PyTorch DataLoader:
        # https://pytorch.org/docs/stable/_modules/torch/utils/data/dataloader.html#DataLoader
        # Needed to support 'persistent_workers'

        use_persistent_workers = self.loader_kwargs.get("persistent_workers", False)
        num_workers = self.loader_kwargs.get("num_workers", 0)

        if use_persistent_workers and num_workers > 0:
            if self._persistent_loader is None:
                self._persistent_loader = self._get_loader()

            yield from self._persistent_loader
        else:
            yield from self._get_loader()

    def _get_loader(self):
        samplers = self._create_samplers(
            self.datasets,
            self.batch_sizes,
            self.distributed_sampling,
            self.loader_kwargs,
        )

        multi_dataset_batch_sampler = MultiDatasetSampler(
            self.datasets,
            samplers,
            termination_dataset_idx=self.termination_dataset,
            oversample_small_datasets=self.oversample_small_datasets,
            never_ending=self.never_ending,
        )

        if has_ffcv_support(self.datasets):
            loader = self._make_ffcv_loader(
                self.datasets,
                multi_dataset_batch_sampler,
            )
        else:
            loader = self._make_pytorch_loader(
                self.datasets,
                multi_dataset_batch_sampler,
            )

        return loader

    def _make_pytorch_loader(
        self, datasets: List[AvalancheDataset], batch_sampler: Sampler[List[int]]
    ):
        return _make_data_loader_with_batched_sampler(
            ConcatDataset(datasets),
            batch_sampler=batch_sampler,
            data_loader_args=self.loader_kwargs,
        )

    def _make_ffcv_loader(
        self, datasets: List[AvalancheDataset], batch_sampler: Sampler[List[int]]
    ):
        ffcv_args = dict(self.ffcv_args)
        device = ffcv_args.pop("device")
        print_ffcv_summary = ffcv_args.pop("print_ffcv_summary")

        persistent_workers = self.loader_kwargs.get("persistent_workers", False)

        return HybridFfcvLoader(
            dataset=AvalancheDataset(datasets),
            batch_sampler=batch_sampler,
            ffcv_loader_parameters=ffcv_args,
            device=device,
            persistent_workers=persistent_workers,
            print_ffcv_summary=print_ffcv_summary,
        )

    def _extract_ffcv_args(self, loader_args):
        loader_args = dict(loader_args)
        ffcv_args: Dict[str, Any] = loader_args.pop("ffcv_args", dict())
        ffcv_args.setdefault("device", None)
        ffcv_args.setdefault("print_ffcv_summary", False)

        for arg_name, arg_value in loader_args.items():
            if arg_name in ffcv_args:
                # Already specified in ffcv_args -> discard
                continue

            if arg_name in HybridFfcvLoader.VALID_FFCV_PARAMS:
                ffcv_args[arg_name] = arg_value
        return loader_args, ffcv_args

    def __len__(self):
        return self.n_iterations

    @staticmethod
    def _create_samplers(
        datasets: List[AvalancheDataset],
        batch_sizes: List[int],
        distributed_sampling: bool,
        loader_kwargs: Dict[str, Any],
    ):
        samplers = []

        for dataset, dataset_mb_size in zip(datasets, batch_sizes):
            sampler = _make_sampler(
                dataset, distributed_sampling, loader_kwargs, dataset_mb_size
            )

            samplers.append(sampler)

        return samplers


class SingleDatasetDataLoader(MultiDatasetDataLoader):
    """
    Replacement of PyTorch DataLoader that also supports
    the additional loading mechanisms implemented in
    :class:`MultiDatasetDataLoader`.
    """

    def __init__(self, datasets: AvalancheDataset, batch_size: int = 1, **kwargs):
        super().__init__([datasets], [batch_size], **kwargs)


class GroupBalancedDataLoader(MultiDatasetDataLoader):
    """Data loader that balances data from multiple datasets."""

    def __init__(
        self,
        datasets: Sequence[AvalancheDataset],
        oversample_small_groups: bool = False,
        batch_size: int = 32,
        distributed_sampling: bool = True,
        **kwargs
    ):
        """Data loader that balances data from multiple datasets.

        Mini-batches emitted by this dataloader are created by collating
        together mini-batches from each group. It may be used to balance data
        among classes, experiences, tasks, and so on.

        If `oversample_small_groups == True` smaller groups are oversampled to
        match the largest group. Otherwise, once data from a group is
        completely iterated, the group will be skipped.

        :param datasets: an instance of `AvalancheDataset`.
        :param oversample_small_groups: whether smaller groups should be
            oversampled to match the largest one.
        :param batch_size: the size of the batch. It must be greater than or
            equal to the number of groups.
        :param distributed_sampling: If True, apply the PyTorch
            :class:`DistributedSampler`. Defaults to True.
            Note: the distributed sampler is not applied if not running
            a distributed training, even when True is passed.
        :param kwargs: data loader arguments used to instantiate the loader for
            each group separately. See pytorch :class:`DataLoader`.
        """

        # check if batch_size is larger than or equal to the number of datasets
        assert batch_size >= len(datasets)

        # divide the batch between all datasets in the group
        ds_batch_size = batch_size // len(datasets)
        remaining = batch_size % len(datasets)

        batch_sizes = []
        for _ in datasets:
            bs = ds_batch_size
            if remaining > 0:
                bs += 1
                remaining -= 1
            batch_sizes.append(bs)

        super().__init__(
            datasets,
            batch_sizes,
            oversample_small_datasets=oversample_small_groups,
            distributed_sampling=distributed_sampling,
            **kwargs
        )


class TaskBalancedDataLoader(GroupBalancedDataLoader):
    """Task-balanced data loader for Avalanche's datasets."""

    def __init__(
        self,
        data: AvalancheDataset,
        batch_size: int = 32,
        oversample_small_groups: bool = False,
        distributed_sampling: bool = True,
        **kwargs
    ):
        """Task-balanced data loader for Avalanche's datasets.

        The iterator returns a mini-batch balanced across each task, which
        makes it useful when training in multi-task scenarios whenever data is
        highly unbalanced.

        If `oversample_small_tasks == True` smaller tasks are
        oversampled to match the largest task. Otherwise, once the data for a
        specific task is terminated, that task will not be present in the
        subsequent mini-batches.

        :param data: an instance of `AvalancheDataset`.
        :param oversample_small_groups: whether smaller tasks should be
            oversampled to match the largest one.
        :param distributed_sampling: If True, apply the PyTorch
            :class:`DistributedSampler`. Defaults to True.
            Note: the distributed sampler is not applied if not running
            a distributed training, even when True is passed.
        :param kwargs: data loader arguments used to instantiate the loader for
            each task separately. See pytorch :class:`DataLoader`.
        """

        if "oversample_small_tasks" in kwargs:
            raise ValueError(
                "oversample_small_tasks is deprecated in favor of "
                "oversample_small_groups"
            )

        # Split data by task
        task_datasets = []
        task_labels_field = getattr(data, "targets_task_labels")
        assert isinstance(task_labels_field, DataAttribute)
        for task_label in task_labels_field.uniques:
            tidxs = task_labels_field.val_to_idx[task_label]
            tdata = data.subset(tidxs)
            task_datasets.append(tdata)

        super().__init__(
            task_datasets,
            oversample_small_groups=oversample_small_groups,
            batch_size=batch_size,
            distributed_sampling=distributed_sampling,
            **kwargs
        )


class GroupBalancedInfiniteDataLoader(MultiDatasetDataLoader):
    """Data loader that balances data from multiple datasets emitting an
    infinite stream."""

    def __init__(
        self,
        datasets: Sequence[AvalancheDataset],
        batch_size=32,
        distributed_sampling: bool = True,
        **kwargs
    ):
        """Data loader that balances data from multiple datasets emitting an
        infinite stream.

        Mini-batches emitted by this dataloader are created by collating
        together mini-batches from each group. It may be used to balance data
        among classes, experiences, tasks, and so on.

        :param datasets: an instance of `AvalancheDataset`.
        :param batch_size: the size of the batch to take from each dataset.
            Please note that, differently from other Avalanche multi dataset
            loaders, this value is the per-dataset contribution to the
            final mini-batch, NOT the final mini-batch size. The final
            mini-batches will be of size `len(datasets) * batch_size`.
        :param distributed_sampling: If True, apply the PyTorch
            :class:`DistributedSampler`. Defaults to True.
            Note: the distributed sampler is not applied if not running
            a distributed training, even when True is passed.
        :param kwargs: data loader arguments used to instantiate the loader for
            each group separately. See pytorch :class:`DataLoader`.
        """

        batch_sizes = [batch_size] * len(datasets)

        super().__init__(
            datasets,
            batch_sizes,
            termination_dataset=-1,
            oversample_small_datasets=True,
            distributed_sampling=distributed_sampling,
            never_ending=True,
            **kwargs
        )


class ReplayDataLoader(MultiDatasetDataLoader):
    """Custom data loader for rehearsal/replay strategies."""

    def __init__(
        self,
        data: AvalancheDataset,
        memory: Optional[AvalancheDataset] = None,
        oversample_small_tasks: bool = False,
        batch_size: int = 32,
        batch_size_mem: int = 32,
        task_balanced_dataloader: bool = False,
        distributed_sampling: bool = True,
        **kwargs
    ):
        """Custom data loader for rehearsal strategies.

        This dataloader iterates in parallel two datasets, the current `data`
        and the rehearsal `memory`, which are used to create mini-batches by
        concatenating their data together. Mini-batches from both of them are
        balanced using the task label (i.e. each mini-batch contains a balanced
        number of examples from all the tasks in the `data` and `memory`).

        The length of the loader is determined only by the current
        task data and is the same than what it would be when creating a
        data loader for this dataset.

        If `oversample_small_tasks == True` smaller tasks are oversampled to
        match the largest task.

        :param data: AvalancheDataset.
        :param memory: AvalancheDataset.
        :param oversample_small_tasks: whether smaller tasks should be
            oversampled to match the largest one.
        :param batch_size: the size of the data batch. It must be greater
            than or equal to the number of tasks.
        :param batch_size_mem: the size of the memory batch. If
            `task_balanced_dataloader` is set to True, it must be greater than
            or equal to the number of tasks.
        :param task_balanced_dataloader: if true, buffer data loaders will be
            task-balanced, otherwise it creates a single data loader for the
            buffer samples.
        :param distributed_sampling: If True, apply the PyTorch
            :class:`DistributedSampler`. Defaults to True.
            Note: the distributed sampler is not applied if not running
            a distributed training, even when True is passed.
        :param kwargs: data loader arguments used to instantiate the loader for
            each task separately. See pytorch :class:`DataLoader`.
        """

        if "collate_fn" not in kwargs:
            kwargs["collate_fn"] = data.collate_fn

        # Create dataloader for memory items
        if task_balanced_dataloader:
            memory_task_labels = getattr(memory, "targets_task_labels")
            assert isinstance(memory_task_labels, DataAttribute)
            num_keys = len(memory_task_labels.uniques)

            # Ensure that the per-task batch size will end up > 0
            assert batch_size_mem >= num_keys, (
                "Batch size must be greator or equal "
                "to the number of tasks in the memory "
                "and current data."
            )

            # Make the batch size balanced between tasks
            # The remainder (remaining_example) will be distributed
            # across tasks by "self._get_datasets_and_batch_sizes(...)"
            single_group_batch_size = batch_size_mem // num_keys
            remaining_example = batch_size_mem % num_keys
        else:
            single_group_batch_size = batch_size_mem
            remaining_example = 0

        # For current data, use the batch_size from the input "batch_size".
        # batch_size can be an int (do not split by task)
        # or a dictionary task_id -> mb_size
        # In both cases, remaining_examples=0
        data_batch_sizes, data_subsets = self._get_datasets_and_batch_sizes(
            data, batch_size, 0, False
        )

        memory_batch_sizes, memory_subsets = self._get_datasets_and_batch_sizes(
            memory,
            single_group_batch_size,
            remaining_example,
            task_balanced_dataloader,
        )

        # Obtain the subset with the highest number of iterations
        # This is the one that defines when an epoch ends
        # Note: this is aligned with the behavior of the legacy
        # multi-loader version of ReplayDataLoader
        loaders_for_len_estimation = []

        for data_subset, subset_mb_size in zip(data_subsets, data_batch_sizes):
            loaders_for_len_estimation.append(
                _make_data_loader(
                    data_subset,
                    distributed_sampling,
                    kwargs,
                    subset_mb_size,
                    force_no_workers=True,
                )[0]
            )

        longest_data_subset_idx = (
            np.array(len(d) for d in loaders_for_len_estimation).argmax().item()
        )

        super().__init__(
            data_subsets + memory_subsets,
            data_batch_sizes + memory_batch_sizes,
            termination_dataset=longest_data_subset_idx,
            oversample_small_datasets=oversample_small_tasks,
            distributed_sampling=distributed_sampling,
            **kwargs
        )

    @staticmethod
    def _get_datasets_and_batch_sizes(
        data: AvalancheDataset,
        batch_sizes_def: Union[int, Dict[int, int]],
        remaining_examples: int,
        task_balanced_dataloader: bool,
    ):
        datasets: List[AvalancheDataset] = []
        batch_sizes: List[int] = []
        batch_size_per_task = not isinstance(batch_sizes_def, int)

        if task_balanced_dataloader or batch_size_per_task:
            for task_id in data.task_set:
                dataset = data.task_set[task_id]

                if batch_size_per_task:
                    current_batch_size = batch_sizes_def[task_id]
                else:
                    current_batch_size = batch_sizes_def

                if remaining_examples > 0:
                    current_batch_size += 1
                    remaining_examples -= 1

                datasets.append(dataset)
                batch_sizes.append(current_batch_size)
        else:
            # Current data is loaded without task balancing
            datasets.append(data)
            batch_sizes.append(batch_sizes_def)
        return batch_sizes, datasets


class MultiDatasetSampler(Sampler[List[int]]):
    """
    Iterate over datasets and provide a batch per dataset in each mini-batch.
    """

    def __init__(
        self,
        datasets: Sequence[Sized],
        samplers: Sequence[BatchSampler],
        termination_dataset_idx: int = 0,
        oversample_small_datasets: bool = False,
        never_ending: bool = False,
    ):
        assert len(datasets) == len(samplers)
        assert never_ending or (
            termination_dataset_idx >= 0 and termination_dataset_idx < len(datasets)
        )

        self.datasets = list(datasets)
        self.samplers = list(samplers)
        self.cumulative_sizes = ConcatDataset.cumsum(self.datasets)
        self.never_ending = never_ending

        if self.never_ending:
            self.termination_dataset_idx = -1
            self.termination_dataset_iterations = 10**10
            self.oversample_small_datasets = True

            if sum(len(x) for x in self.samplers) == 0:
                raise RuntimeError(
                    "The never ending sampler must able to create a mini-batch"
                )
        else:
            # termination_dataset_idx => dataset used to determine the epoch end
            self.termination_dataset_idx = termination_dataset_idx
            self.termination_dataset_iterations = len(
                self.samplers[self.termination_dataset_idx]
            )

            self.oversample_small_datasets = oversample_small_datasets

    def __len__(self):
        return self.termination_dataset_iterations

    def __iter__(self):
        number_of_datasets = len(self.datasets)
        samplers_list = []
        sampler_iterators = []

        for dataset_idx in range(number_of_datasets):
            sampler = self.samplers[dataset_idx]
            samplers_list.append(sampler)
            cur_sampler_iterator = sampler.__iter__()
            sampler_iterators.append(cur_sampler_iterator)

        index_offsets = np.array([0] + self.cumulative_sizes[:-1])

        while True:
            per_dataset_indices: List[Optional[np.ndarray]] = [
                None
            ] * number_of_datasets

            if self.never_ending:
                sampling_dataset_order = list(range(number_of_datasets))
                is_termination_dataset = [False] * number_of_datasets
            else:
                # Obtain the indices for the "main" dataset first
                sampling_dataset_order = [self.termination_dataset_idx] + list(
                    x
                    for x in range(number_of_datasets)
                    if x != self.termination_dataset_idx
                )
                is_termination_dataset = [True] + ([False] * (number_of_datasets - 1))

            for dataset_idx, is_term_dataset in zip(
                sampling_dataset_order, is_termination_dataset
            ):
                sampler = samplers_list[dataset_idx]
                sampler_iterator = sampler_iterators[dataset_idx]

                if sampler is None:
                    continue

                if len(sampler) == 0:
                    if is_term_dataset and (not self.never_ending):
                        return

                    samplers_list[dataset_idx] = None
                    sampler_iterators[dataset_idx] = None
                    continue

                should_stop_if_ended = (
                    is_term_dataset or not self.oversample_small_datasets
                ) and (not self.never_ending)

                continue_epoch, updated_iterator, next_batch_indices = self._next_batch(
                    sampler, sampler_iterator, stop_on_last_batch=should_stop_if_ended
                )

                if not continue_epoch:
                    if is_term_dataset:
                        # The main dataset terminated -> exit
                        return
                    else:
                        # Not the main dataset
                        # Happens if oversample_small_tasks is False
                        # Remove the dataset and sampler from the list
                        samplers_list[dataset_idx] = None
                        sampler_iterators[dataset_idx] = None
                        continue

                assert next_batch_indices is not None
                next_batch_indices = np.array(next_batch_indices)

                # Shift indices according to the position of the
                # dataset in the list
                next_batch_indices += index_offsets[dataset_idx]

                sampler_iterators[dataset_idx] = updated_iterator
                per_dataset_indices[dataset_idx] = next_batch_indices
            per_dataset_indices = [x for x in per_dataset_indices if x is not None]
            yield np.concatenate(per_dataset_indices).tolist()

    @staticmethod
    def _next_batch(
        sampler: Sampler,
        sampler_iterator: Iterator[Sequence[int]],
        stop_on_last_batch: bool,
    ):
        try:
            next_batch_indices = next(sampler_iterator)
            return True, sampler_iterator, next_batch_indices
        except StopIteration:
            if stop_on_last_batch:
                return False, None, None

        # Re-create the iterator
        # This time, do not catch StopIteration
        if isinstance(sampler, BatchSampler):
            if isinstance(sampler.sampler, DistributedSampler):
                sampler.sampler.set_epoch(sampler.sampler.epoch + 1)
        elif isinstance(sampler, DistributedSampler):
            # Manage shuffling in DistributedSampler
            sampler.set_epoch(sampler.epoch + 1)

        sampler_iterator = iter(sampler)
        next_batch_indices = next(sampler_iterator)
        return True, sampler_iterator, next_batch_indices


def _make_data_loader(
    dataset: Dataset,
    distributed_sampling: bool,
    data_loader_args: Dict[str, Any],
    batch_size: int,
    force_no_workers: bool = False,
):
    data_loader_args = data_loader_args.copy()
    data_loader_args.pop("ffcv_args", None)

    collate_from_data_or_kwargs(dataset, data_loader_args)

    if force_no_workers:
        data_loader_args["num_workers"] = 0
        if "persistent_workers" in data_loader_args:
            data_loader_args["persistent_workers"] = False
        if "prefetch_factor" in data_loader_args:
            data_loader_args["prefetch_factor"] = 2

    if DistributedHelper.is_distributed and distributed_sampling:
        # Note: shuffle only goes in the sampler, while
        # drop_last must be passed to both the sampler
        # and the DataLoader
        drop_last = data_loader_args.pop("drop_last", False)
        sampler = DistributedSampler(
            dataset,
            shuffle=data_loader_args.pop("shuffle", True),
            drop_last=drop_last,
        )
        data_loader = DataLoader(
            dataset,
            sampler=sampler,
            batch_size=batch_size,
            drop_last=drop_last,
            **data_loader_args
        )
    else:
        sampler = None
        data_loader = DataLoader(dataset, batch_size=batch_size, **data_loader_args)

    return data_loader, sampler


def _make_data_loader_with_batched_sampler(
    dataset: Dataset, batch_sampler: Any, data_loader_args: Dict[str, Any]
):
    data_loader_args = data_loader_args.copy()

    # See documentation of batch_sampler:
    # https://pytorch.org/docs/stable/data.html#torch.utils.data.DataLoader
    # In fact, "generator" could be dropped too
    data_loader_args.pop("batch_size", False)
    data_loader_args.pop("shuffle", False)
    data_loader_args.pop("sampler", False)
    data_loader_args.pop("drop_last", False)

    data_loader_args.pop("ffcv_args", None)

    return DataLoader(dataset, batch_sampler=batch_sampler, **data_loader_args)


def _make_sampler(
    dataset: Any,
    distributed_sampling: bool,
    data_loader_args: Dict[str, Any],
    batch_size: int,
):
    loader, _ = _make_data_loader(
        dataset,
        distributed_sampling,
        data_loader_args,
        batch_size,
        force_no_workers=True,
    )

    sampler = loader.batch_sampler
    return sampler


__all__ = [
    "collate_from_data_or_kwargs",
    "TaskBalancedDataLoader",
    "GroupBalancedDataLoader",
    "ReplayDataLoader",
    "GroupBalancedInfiniteDataLoader",
]
