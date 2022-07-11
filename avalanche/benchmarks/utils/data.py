################################################################################
# Copyright (c) 2021 ContinualAI.                                              #
# Copyrights licensed under the MIT License.                                   #
# See the accompanying LICENSE file for terms.                                 #
#                                                                              #
# Date: 12-05-2020                                                             #
# Author(s): Lorenzo Pellegrini, Antonio Carta                                 #
# E-mail: contact@continualai.org                                              #
# Website: avalanche.continualai.org                                           #
################################################################################

"""
This module contains the implementation of the Avalanche Dataset,
Avalanche dataset class which extends PyTorch's dataset.
AvalancheDataset (and its derivatives) offers additional features like the
management of preprocessing pipelines and task/class labels.
"""
import copy

from torch.utils.data.dataloader import default_collate
from torch.utils.data.dataset import Dataset, Subset, ConcatDataset

from avalanche.benchmarks.utils.dataset_definitions import IDataset
from .data_attribute import DataAttribute
from .dataset_utils import (
    find_list_from_index,
)
from .dataset_definitions import (
    ClassificationDataset,
)

from typing import (
    List,
    Any,
    Sequence,
    Union,
    TypeVar,
    Callable,
    Collection,
)

from .transforms import TransformGroups, EmptyTransformGroups

T_co = TypeVar("T_co", covariant=True)
TAvalancheDataset = TypeVar("TAvalancheDataset", bound="AvalancheDataset")


class AvalancheDataset(Dataset[T_co]):
    """Avalanche Dataset.

    This class extends pytorch Datasets with some additional functionality:
    - management of transformation groups via :class:`AvalancheTransform`
    - support for sample attributes such as class targets and task labels

    This dataset can also be used to apply several advanced operations involving
    transformations. For instance, it allows the user to add and replace
    transformations, freeze them so that they can't be changed, etc.

    This dataset also allows the user to keep distinct transformations groups.
    Simply put, a transformation group is a pair of transform+target_transform
    (exactly as in torchvision datasets). This dataset natively supports keeping
    two transformation groups: the first, 'train', contains transformations
    applied to training patterns. Those transformations usually involve some
    kind of data augmentation. The second one is 'eval', that will contain
    transformations applied to test patterns. Having both groups can be
    useful when, for instance, in need to test on the training data (as this
    process usually involves removing data augmentation operations). Switching
    between transformations can be easily achieved by using the
    :func:`train` and :func:`eval` methods.

    Moreover, arbitrary transformation groups can be added and used. For more
    info see the constructor and the :func:`with_transforms` method.
    AvalancheDataset can switch between the 'train' and
    'eval' groups by calling the ``train()`` and ``eval()`` methods. When
    using custom groups one can use the ``with_transforms(group_name)``
    method instead.

    This dataset will try to inherit the task labels from the input
    dataset. If none are available and none are given via the `task_labels`
    parameter, each pattern will be assigned a default task label "0".
    See the constructor for more details.
    """

    def __init__(
        self,
        dataset: IDataset,
        *,
        data_attributes: List[DataAttribute] = None,
        transform_groups: TransformGroups = None,
        collate_fn: Callable[[List], Any] = None
    ):
        """Creates a ``AvalancheDataset`` instance.

        :param dataset: Original dataset. Beware that
            AvalancheDataset will not overwrite transformations already
            applied by this dataset.
        :param transform_groups: Avalanche transform groups.
        """
        self._dataset = dataset  # original dataset

        if isinstance(dataset, AvalancheDataset):
            # inherit data attributes from original dataset
            self._data_attributes = {**dataset._data_attributes}
        else:
            self._data_attributes = {}

        self._mbatch_dattributes = []  # attributes to append to mbatch
        if data_attributes is not None:
            da_dict = {da.name: da for da in data_attributes}
            self._data_attributes.update(da_dict)

        for el in self._data_attributes.values():
            if el.append_to_mbatch:
                self._mbatch_dattributes.append(el)

        for el in self._data_attributes.values():
            setattr(self, el.name, el)

        if transform_groups is None:
            transform_groups = EmptyTransformGroups()
        self.transform_groups = transform_groups
        self.collate_fn = collate_fn

        self.collate_fn = self._init_collate_fn(
            dataset, collate_fn
        )
        """
        The collate function to use when creating mini-batches from this
        dataset.
        """

    def __add__(self, other: Dataset) -> "AvalancheDataset":
        return AvalancheConcatDataset([self, other])

    def __radd__(self, other: Dataset) -> "AvalancheDataset":
        return AvalancheConcatDataset([other, self])

    def __getitem__(self, idx) -> Union[T_co, Sequence[T_co]]:
        element = self._dataset[idx]
        if self.transform_groups is not None:
            element = self.transform_groups(element)
        atrs = [at[idx] for at in self._mbatch_dattributes]
        return *element, *atrs

    def __len__(self):
        return len(self._dataset)

    def train(self):
        """Returns a new dataset with the transformations of the 'train' group
        loaded.

        The current dataset will not be affected.

        :return: A new dataset with the training transformations loaded.
        """
        return self.with_transforms("train")

    def eval(self):
        """
        Returns a new dataset with the transformations of the 'eval' group
        loaded.

        Eval transformations usually don't contain augmentation procedures.
        This function may be useful when in need to test on training data
        (for instance, in order to run a validation pass).

        The current dataset will not be affected.

        :return: A new dataset with the eval transformations loaded.
        """
        return self.with_transforms("eval")

    def with_transforms(
        self: TAvalancheDataset, group_name: str
    ) -> TAvalancheDataset:
        """
        Returns a new dataset with the transformations of a different group
        loaded.

        The current dataset will not be affected.

        :param group_name: The name of the transformations group to use.
        :return: A new dataset with the new transformations.
        """
        datacopy = self._clone_dataset()
        datacopy.transform_groups.with_transform(group_name)
        if isinstance(self._dataset, AvalancheDataset):
            datacopy._dataset = datacopy._dataset.with_transform_group(group_name)
        return datacopy

    def _clone_dataset(self: TAvalancheDataset) -> TAvalancheDataset:
        dataset_copy = copy.copy(self)
        dataset_copy.transform_groups = copy.copy(dataset_copy.transform_groups)
        return dataset_copy

    def _init_collate_fn(self, dataset, collate_fn):
        if collate_fn is not None:
            return collate_fn

        if hasattr(dataset, "collate_fn"):
            return getattr(dataset, "collate_fn")

        return default_collate


class AvalancheSubset(AvalancheDataset[T_co]):
    """Avalanche Dataset that behaves like a PyTorch
    :class:`torch.utils.data.Subset`.

    See :class:`AvalancheDataset` for more details.
    """
    def __init__(
        self,
        dataset: IDataset,
        indices: Sequence[int],
        *,
        data_attributes: List[DataAttribute] = None,
        transform_groups: TransformGroups = None,
        collate_fn: Callable[[List], Any] = None
    ):
        """Creates an ``AvalancheSubset`` instance.

        :param dataset: The whole dataset.
        :param indices: Indices in the whole set selected for subset. Can
            be None, which means that the whole dataset will be returned.
        """
        self._indices = indices

        super().__init__(
            dataset,
            data_attributes=data_attributes,
            transform_groups=transform_groups,
            collate_fn=collate_fn)

        self._flatten_dataset()
        for da in self._data_attributes:  # subset for attributes
            setattr(self, da.name, da.subset(self._indices))

    def __getitem__(self, idx):
        if isinstance(idx, list):
            return super().__getitem__([[self._indices[i] for i in idx]])
        return super().__getitem__(self._indices[idx])

    def __len__(self):
        return len(self._indices)

    def _flatten_dataset(self):
        """Flattens this subset by borrowing indices from the original dataset
        (if it's an AvalancheSubset or PyTorch Subset)"""

        if isinstance(self._dataset, AvalancheSubset):
            # we need to take trasnforms and indices from parent
            self.transform_groups = copy.copy(self._dataset.transform_groups)
            # TODO: merge transform groups.

            parent_idxs = self._dataset._indices
            self._indices = [parent_idxs[x] for x in self._indices]


class AvalancheConcatDataset(AvalancheDataset[T_co]):
    """A Dataset that behaves like a PyTorch
    :class:`torch.utils.data.ConcatDataset`.

    However, this Dataset also supports
    transformations, slicing the targets field and all
    the other goodies listed in :class:`AvalancheDataset`.

    This dataset guarantees that the operations involving the transformations
    and transformations groups are consistent across the concatenated dataset
    (if they are subclasses of :class:`AvalancheDataset`).
    """

    def __init__(
        self,
        datasets: Collection[IDataset],
        *,
        data_attributes: List[DataAttribute] = None,
        transform_groups: TransformGroups = None,
        collate_fn: Callable[[List], Any] = None
    ):
        """Creates a ``AvalancheConcatDataset`` instance.

        :param datasets: A collection of datasets.
        """
        dataset_list = list(datasets)
        self.datasets = dataset_list
        self._flatten_dataset()

        self._datasets_lengths = [len(dataset) for dataset in dataset_list]
        self.cumulative_sizes = ConcatDataset.cumsum(dataset_list)
        self._total_length = sum(self._datasets_lengths)

        self._data_attributes = {}
        for dd in self.datasets:
            if isinstance(dd, AvalancheDataset):
                # inherit data attributes from original dataset
                self._data_attributes.update(dd._data_attributes)

        self._mbatch_dattributes = []  # attributes to append to mbatch
        if data_attributes is not None:
            da_dict = {da.name: da for da in data_attributes}
            self._data_attributes.update(da_dict)

        for el in self._data_attributes.values():
            if el.append_to_mbatch:
                self._mbatch_dattributes.append(el)

        for el in self._data_attributes.values():
            setattr(self, el.name, el)

        if transform_groups is None:
            transform_groups = EmptyTransformGroups()
        self.transform_groups = transform_groups
        self.collate_fn = collate_fn

        self.collate_fn = self._init_collate_fn(dataset_list[0], collate_fn)
        """
        The collate function to use when creating mini-batches from this
        dataset.
        """

    def __len__(self) -> int:
        return self._total_length

    def __getitem__(self, idx: int):
        # same logic as pytorch's ConcatDataset to get item's index
        element = ConcatDataset.__getitem__(self, idx)
        if self.transform_groups is not None:
            element = self.transform_groups(element)
        return element

    def _clone_dataset(self: TAvalancheDataset) -> TAvalancheDataset:
        dataset_copy = super()._clone_dataset()
        dataset_copy.datasets = list(dataset_copy.datasets)
        return dataset_copy

    def _flatten_dataset(self):
        # Flattens this subset by borrowing the list of concatenated datasets
        # from the original datasets (if they're 'AvalancheConcatSubset's or
        # PyTorch 'ConcatDataset's)

        flattened_list = []
        for dataset in self.datasets:
            if isinstance(dataset, AvalancheConcatDataset):
                if isinstance(dataset.transform_groups, EmptyTransformGroups):
                    flattened_list.extend(dataset.datasets)
                else:
                    # Can't flatten as the dataset has custom transformations
                    flattened_list.append(dataset)
            elif isinstance(dataset, AvalancheSubset):
                flattened_list += self._flatten_subset_concat_branch(dataset)
            else:
                flattened_list.append(dataset)
        self.datasets = flattened_list

    def _flatten_subset_concat_branch(
        self, dataset: AvalancheSubset
    ) -> List[Dataset]:
        """Optimizes the dataset hierarchy in the corner case:

        self -> [Subset, Subset, ] -> ConcatDataset -> [Dataset]

        :param dataset: The dataset. This function returns [dataset] if the
            dataset is not a subset containing a concat dataset (or if other
            corner cases are encountered).
        :return: The flattened list of datasets to be concatenated.
        """
        concat_dataset: AvalancheConcatDataset = dataset._original_dataset

        if not isinstance(concat_dataset, AvalancheConcatDataset):
            return [dataset]
        if not isinstance(concat_dataset.transform_groups, EmptyTransformGroups):
            return [dataset]

        result: List[AvalancheSubset] = []
        last_c_dataset = None
        last_c_idxs = []
        last_c_targets = []
        last_c_tasks = []

        for subset_idx, idx in enumerate(dataset._indices):
            dataset_idx, internal_idx = find_list_from_index(
                idx,
                concat_dataset._datasets_lengths,
                concat_dataset._total_length,
                cumulative_sizes=concat_dataset.cumulative_sizes,
            )

            if last_c_dataset is None:
                last_c_dataset = dataset_idx
            elif last_c_dataset != dataset_idx:
                # Consolidate current subset
                result.append(
                    AvalancheConcatDataset._make_similar_subset(
                        dataset,
                        concat_dataset.datasets[last_c_dataset],
                        last_c_idxs,
                        last_c_targets,
                        last_c_tasks,
                    )
                )

                # Switch to next dataset
                last_c_dataset = dataset_idx
                last_c_idxs = []
                last_c_targets = []
                last_c_tasks = []

            last_c_idxs.append(internal_idx)
            # TODO: merge transforms
            # TODO: merge data attributes
            last_c_targets.append(dataset.targets[subset_idx])
            last_c_tasks.append(dataset.targets_task_labels[subset_idx])

        if last_c_dataset is not None:
            result.append(
                AvalancheConcatDataset._make_similar_subset(
                    dataset,
                    concat_dataset.datasets[last_c_dataset],
                    last_c_idxs,
                    last_c_targets,
                    last_c_tasks,
                )
            )

        return result

    @staticmethod
    def _make_similar_subset(subset, ref_dataset, indices, targets, tasks):
        t_groups = dict()
        f_groups = dict()
        AvalancheDataset._borrow_transformations(subset, t_groups, f_groups)

        collate_fn = None
        if hasattr(subset, "collate_fn"):
            collate_fn = subset.collate_fn

        result = AvalancheSubset(
            ref_dataset,
            indices=indices,
            class_mapping=subset._class_mapping,
            transform_groups=t_groups,
            initial_transform_group=subset.current_transform_group,
            task_labels=tasks,
            targets=targets,
            collate_fn=collate_fn,
        )

        result._frozen_transforms = f_groups
        return result


__all__ = [
    "AvalancheDataset",
    "AvalancheSubset",
    "AvalancheConcatDataset",
]
