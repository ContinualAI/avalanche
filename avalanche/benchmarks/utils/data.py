################################################################################
# Copyright (c) 2022 ContinualAI.                                              #
# Copyrights licensed under the MIT License.                                   #
# See the accompanying LICENSE file for terms.                                 #
#                                                                              #
# Date: 19-07-2022                                                             #
# Author(s): Antonio Carta                                                     #
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

from typing import (
    List,
    Any,
    Sequence,
    Union,
    TypeVar,
    Callable,
    Collection, Tuple,
)

from .transforms import TransformGroups, EmptyTransformGroups, FrozenTransformGroups

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
        self._data_attributes = data_attributes

        if isinstance(dataset, AvalancheDataset):
            # inherit data attributes from original dataset
            self._data_attributes = {**dataset._data_attributes}
        else:
            self._data_attributes = {}

        if data_attributes is not None:
            da_dict = {da.name: da for da in data_attributes}
            self._data_attributes.update(da_dict)

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

        self.is_frozen_transforms = True

    def __add__(self, other: Dataset) -> "AvalancheDataset":
        return AvalancheConcatDataset([self, other])

    def __radd__(self, other: Dataset) -> "AvalancheDataset":
        return AvalancheConcatDataset([other, self])

    def __eq__(self, other: "AvalancheDataset"):
        return self._dataset == other._dataset and \
            self.transform_groups == other.transform_groups and \
            self._data_attributes == other._data_attributes and \
            self.collate_fn == other.collate_fn

    def _getitem_recursive_call(self, idx):
        """We need this recursive call to avoid appending task
        label multiple times inside the __getitem__."""
        if isinstance(self._dataset, AvalancheDataset):
            element = self._dataset._getitem_recursive_call(idx)
        else:
            element = self._dataset[idx]

        if self.transform_groups is not None:
            element = self.transform_groups(element)
        return element

    def __getitem__(self, idx) -> Union[T_co, Sequence[T_co]]:
        element = self._dataset[idx]
        if self.transform_groups is not None:
            element = self.transform_groups(element)
        return element

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
        if isinstance(datacopy._dataset, AvalancheDataset):
            datacopy._dataset = datacopy._dataset.with_transforms(group_name)
        return datacopy

    def get_transform_groups(self):
        """Recursively collects transform groups across the entire data tree.
        Warning: assumes transforms are the same across all concat
        branches."""
        if isinstance(self._dataset, AvalancheDataset):
            return self._dataset.get_transform_groups() + self.transform_groups
        else:
            return self.transform_groups

    def freeze_transforms(self):
        tgroups = self.get_transform_groups()
        dataset_copy = self.remove_transform_groups()
        return FrozenTransformDataset(
            dataset_copy,
            frozen_transforms=tgroups
        )

    def remove_transform_groups(self):
        """Recursively remove transformation groups from dataset tree."""
        dataset_copy = self._clone_dataset()
        if isinstance(dataset_copy._dataset, AvalancheDataset):
            dataset_copy._dataset = dataset_copy._dataset.remove_transform_groups()
        dataset_copy.transform_groups = None
        return dataset_copy

    def replace_transforms(self, transform, target_transform):
        """Recursively remove transformation groups from dataset tree."""
        dataset_copy = self.remove_transform_groups()
        dataset_copy.transform_groups = \
            TransformGroups(transform_groups={
                'train': (transform, target_transform),
                'eval': (transform, target_transform)})
        return dataset_copy

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


class FrozenTransformDataset(AvalancheDataset[T_co]):
    def __init__(self, data: AvalancheDataset,
                 frozen_transforms: Union[TransformGroups, Tuple[Callable]]):
        if isinstance(frozen_transforms, TransformGroups):
            self.frozen_transforms = frozen_transforms
        else:
            self.frozen_transforms = FrozenTransformGroups(frozen_transforms)
        super().__init__(dataset=data)

    def _getitem_recursive_call(self, idx):
        """We need this recursive call to avoid appending task
        label multiple times inside the __getitem__."""
        elem = super()._getitem_recursive_call(idx)
        return self.frozen_transforms(elem)

    def __getitem__(self, idx) -> Union[T_co, Sequence[T_co]]:
        elem = super().__getitem__(idx)
        return self.frozen_transforms(elem)


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

        ll = []
        for da in data_attributes:  # subset attributes if needed
            if len(da) != len(indices):
                ll.append(da.subset(self._indices))
            else:
                ll.append(da)

        super().__init__(
            dataset,
            data_attributes=ll,
            transform_groups=transform_groups,
            collate_fn=collate_fn)
        self._flatten_dataset()

    def _getitem_recursive_call(self, idx):
        """We need this recursive call to avoid appending task
        label multiple times inside the __getitem__."""
        if isinstance(self._dataset, AvalancheDataset):
            element = self._dataset._getitem_recursive_call(self._indices[idx])
        else:
            element = self._dataset[self._indices[idx]]

        if self.transform_groups is not None:
            element = self.transform_groups(element)
        return element

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
            # we need to take transforms and indices from parent
            old_tgroups = copy.copy(self._dataset.transform_groups)
            new_tgroups = self.transform_groups
            self.transform_groups = old_tgroups + new_tgroups

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
        transform_groups: TransformGroups = None,
        collate_fn: Callable[[List], Any] = None
    ):
        """Creates a ``AvalancheConcatDataset`` instance.

        :param datasets: A collection of datasets.
        """
        self._dataset = None

        dataset_list = list(datasets)
        self.datasets = dataset_list
        self._flatten_datasets_list()

        self._datasets_lengths = [len(dataset) for dataset in dataset_list]
        self.cumulative_sizes = ConcatDataset.cumsum(dataset_list)
        self._total_length = sum(self._datasets_lengths)

        self._data_attributes = {}
        for dd in self.datasets:
            if isinstance(dd, AvalancheDataset):
                # inherit data attributes from all the original datasets
                # and concatenate them
                for k, v in dd._data_attributes.items():
                    if k not in self._data_attributes:
                        self._data_attributes[k] = v
                    else:
                        attr_cat = self._data_attributes[k].concat(v)
                        self._data_attributes[k] = attr_cat

        for dd in self._data_attributes.values():
            if len(dd) != len(self):
                raise ValueError(f"Wrong size for attribute {dd.name}")

        for el in self._data_attributes.values():
            setattr(self, el.name, el)

        if transform_groups is None:
            transform_groups = EmptyTransformGroups()
        self.transform_groups = transform_groups

        self.collate_fn = self._init_collate_fn(dataset_list[0], collate_fn)
        """
        The collate function to use when creating mini-batches from this
        dataset.
        """
        self._flatten_datasets_list()

    def __len__(self) -> int:
        return self._total_length

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
        dds = []
        for d in self.datasets:
            if isinstance(d, AvalancheDataset):
                dds.append(d.with_transforms(group_name))
            else:
                dds.append(d)
        datacopy.datasets = dds
        return datacopy

    def get_transform_groups(self):
        """Recursively collects transform groups across the entire data tree.
        Warning: assumes transforms are the same across all concat
        branches."""
        if isinstance(self.datasets[0], AvalancheDataset):
            return self.datasets[0].get_transform_groups() + self.transform_groups
        else:
            return self.transform_groups

    def remove_transform_groups(self):
        """Recursively remove transformation groups from dataset tree."""
        if self.is_frozen_transforms:
            return
        dataset_copy = self._clone_dataset()
        dds = []
        for el in dataset_copy.datasets:
            if isinstance(el, AvalancheDataset):
                dds.append(el.remove_transform_groups())
            else:
                dds.append(el)
        dataset_copy.datasets = dds
        dataset_copy.transform_groups = None
        return dataset_copy

    def _getitem_recursive_call(self, idx):
        """We need this recursive call to avoid appending task
        label multiple times inside the __getitem__."""
        return self.__getitem__(idx)[:-1]

    def __getitem__(self, idx: int):
        # same logic as pytorch's ConcatDataset to get item's index
        element = ConcatDataset.__getitem__(self, idx)
        if self.transform_groups is not None:
            element = self.transform_groups(element)
        return element

    def _flatten_datasets_list(self):
        # Flattens this subset by borrowing the list of concatenated datasets
        # from the original datasets.

        flattened_list = []
        for dataset in self.datasets:
            if isinstance(dataset, AvalancheConcatDataset):
                if isinstance(dataset.transform_groups,
                              EmptyTransformGroups) or \
                        dataset.transform_groups is None:
                    flattened_list.extend(dataset.datasets)
                else:
                    # Can't flatten as the dataset has custom transformations
                    flattened_list.append(dataset)
            else:
                flattened_list.append(dataset)
        self.datasets = flattened_list


__all__ = [
    "AvalancheDataset",
    "AvalancheSubset",
    "AvalancheConcatDataset",
]
