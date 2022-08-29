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
import bisect
import copy

from torch.utils.data.dataloader import default_collate
from torch.utils.data.dataset import Dataset, ConcatDataset

from avalanche.benchmarks.utils.dataset_definitions import IDataset
from .data_attribute import DataAttribute

from typing import (
    List,
    Any,
    Sequence,
    Union,
    TypeVar,
    Callable,
    Collection
)

from .transform_groups import TransformGroups, EmptyTransformGroups

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
        frozen_transform_groups: TransformGroups = None,
        collate_fn: Callable[[List], Any] = None
    ):
        """Creates a ``AvalancheDataset`` instance.

        :param dataset: Original dataset. Beware that
            AvalancheDataset will not overwrite transformations already
            applied by this dataset.
        :param transform_groups: Avalanche transform groups.
        """
        # NOTES on implementation:
        # - raw datasets operations are implemented by _FlatData
        # - data attributes are implemented by DataAttribute
        # - transformations are implemented by TransformGroups
        # AvalancheDataset just takes care to manage all of these attributes
        # together and decides how the information propagates through
        # operations (e.g. how to pass attributes after concat/subset
        # operations).

        # original dataset. Don't use this attribute directly because some subclasses
        # (e.g. AvalancheConcatDataset) don't have it.
        # This attribute is a list to make it compatible with AvalancheConcatDataset
        self._dataset = dataset
        self._data_attributes = {}

        ####################################
        # Init data attributes
        ####################################
        if isinstance(dataset, AvalancheDataset):
            # inherit data attributes from original dataset
            self._data_attributes = {**dataset._data_attributes}

        if data_attributes is not None:
            da_dict = {da.name: da for da in data_attributes}
            self._data_attributes.update(da_dict)

        for el in self._data_attributes.values():
            setattr(self, el.name, el)

        ####################################
        # Init transformations
        ####################################
        if frozen_transform_groups is None:
            frozen_transform_groups = EmptyTransformGroups()
        self._frozen_transform_groups = frozen_transform_groups
        if transform_groups is None:
            transform_groups = EmptyTransformGroups()
        self._transform_groups: TransformGroups = transform_groups

        if isinstance(dataset, AvalancheDataset):
            # inherit transformations from original dataset
            cgroup = dataset._transform_groups.current_group
            self._frozen_transform_groups.current_group = cgroup
            self._transform_groups.current_group = cgroup

        self.collate_fn = collate_fn
        self.collate_fn = self._init_collate_fn(
            dataset, collate_fn
        )
        """
        The collate function to use when creating mini-batches from this
        dataset.
        """

    @property
    def data_list(self):
        return [self._dataset]

    @data_list.setter
    def data_list(self, value):
        assert len(value) == 1
        self._dataset = value[0]

    def __add__(self, other: Dataset) -> "AvalancheDataset":
        return AvalancheConcatDataset([self, other])

    def __radd__(self, other: Dataset) -> "AvalancheDataset":
        return AvalancheConcatDataset([other, self])

    def __eq__(self, other: "AvalancheDataset"):
        if not hasattr(other, '_dataset'):
            return False
        return self._dataset == other._dataset and \
            self._transform_groups == other._transform_groups and \
            self._data_attributes == other._data_attributes and \
            self.collate_fn == other.collate_fn

    def _getitem_recursive_call(self, idx):
        """Private method only for internal use.

        We need this recursive call to avoid appending task
        label multiple times inside the __getitem__.
        """
        if isinstance(self._dataset, AvalancheDataset):
            element = self._dataset._getitem_recursive_call(idx)
        else:
            element = self._dataset[idx]

        element = self._frozen_transform_groups(element)
        element = self._transform_groups(element)
        return element

    def __getitem__(self, idx) -> Union[T_co, Sequence[T_co]]:
        return self._getitem_recursive_call(idx)

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
        datacopy = self._shallow_clone_dataset()
        datacopy._frozen_transform_groups.with_transform(group_name)
        datacopy._transform_groups.with_transform(group_name)

        dds = []
        for dd in datacopy.data_list:
            if isinstance(dd, AvalancheDataset):
                dds.append(dd.with_transforms(group_name))
            else:
                dds.append(dd)
        datacopy.data_list = dds
        return datacopy

    def freeze_transforms(self):
        """Returns a new dataset with the transformation groups frozen."""
        tgroups = copy.copy(self._transform_groups)
        datacopy = self._shallow_clone_dataset()
        datacopy._frozen_transform_groups = tgroups
        datacopy._transform_groups = EmptyTransformGroups()
        dds = []
        for dd in datacopy.data_list:
            if isinstance(dd, AvalancheDataset):
                dds.append(dd.freeze_transforms())
            else:
                dds.append(dd)
        datacopy.data_list = dds
        return datacopy

    def remove_current_transform_group(self):
        """Recursively remove transformation groups from dataset tree."""
        dataset_copy = self._shallow_clone_dataset()
        cgroup = dataset_copy._transform_groups.current_group
        dataset_copy._transform_groups[cgroup] = None
        dds = []
        for dd in dataset_copy.data_list:
            if isinstance(dd, AvalancheDataset):
                dds.append(dd.remove_current_transform_group())
            else:
                dds.append(dd)
        dataset_copy.data_list = dds
        return dataset_copy

    def replace_current_transform_group(self, transform):
        """Recursively remove the current transformation group from the dataset tree and replaces
        it."""
        dataset_copy = self.remove_current_transform_group()
        cgroup = dataset_copy._transform_groups.current_group
        dataset_copy._transform_groups[cgroup] = transform
        dds = []
        for dd in dataset_copy.data_list:
            if isinstance(dd, AvalancheDataset):
                dds.append(dd.remove_current_transform_group())
            else:
                dds.append(dd)
        dataset_copy.data_list = dds
        return dataset_copy

    def _shallow_clone_dataset(self: TAvalancheDataset) -> TAvalancheDataset:
        """Clone dataset.
        This is a shallow copy, i.e. the data attributes are not copied.
        """
        dataset_copy = copy.copy(self)
        dataset_copy._transform_groups = copy.copy(dataset_copy._transform_groups)
        dataset_copy._frozen_transform_groups = copy.copy(dataset_copy._frozen_transform_groups)
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

        ll = []
        for da in data_attributes:  # subset attributes if needed
            if len(da) != len(indices):
                assert len(da) == len(dataset)
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

        element = self._frozen_transform_groups(element)
        element = self._transform_groups(element)
        return element

    def __len__(self):
        return len(self._indices)

    def _flatten_dataset(self):
        """Flattens this subset by borrowing indices from the original dataset
        (if it's an AvalancheSubset or PyTorch Subset)"""

        if isinstance(self._dataset, AvalancheSubset):
            # combine self and self.dataset transforms
            old_tgroups = copy.copy(self._dataset._transform_groups)
            new_tgroups = self._transform_groups
            self.transform_groups = old_tgroups + new_tgroups

            # update indices
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
        self._datasets = list(datasets)
        self._flatten_datasets_list()
        self._cumulative_sizes = ConcatDataset.cumsum(self.datasets)

        # update rule for each attribute:
        # 1. if all datasets have the attribute, it is propagated
        # 2. if not, the attribute is ignored
        self._data_attributes = {}
        if len(self.datasets) > 0 and \
                isinstance(self.datasets[0], AvalancheDataset):
            for attr in self.datasets[0]._data_attributes.values():
                acat = attr
                found_all = True
                for d2 in self.datasets[1:]:
                    if hasattr(d2, attr.name):
                        acat = acat.concat(getattr(d2, attr.name))
                    else:
                        found_all = False
                        break
                if found_all:
                    self._data_attributes[attr.name] = acat
                    assert len(acat) == len(self), \
                        f"BUG: Wrong size for attribute {acat.name}"

        for el in self._data_attributes.values():
            setattr(self, el.name, el)

        ################################
        # Init transform groups
        ################################
        if transform_groups is None:
            transform_groups = EmptyTransformGroups()
        self._transform_groups = transform_groups
        self._frozen_transform_groups = EmptyTransformGroups()

        dds = []
        cgroup = None
        for dd in self.data_list:
            if isinstance(dd, AvalancheDataset):
                if cgroup is None:  # inherit transformations from original dataset
                    cgroup = dd._transform_groups.current_group
                # all datasets must have the same transformation group
                dds.append(dd.with_transforms(cgroup))

        if cgroup is not None:
            self._frozen_transform_groups.current_group = cgroup
            self._transform_groups.current_group = cgroup

        self.collate_fn = self._init_collate_fn(self.datasets[0], collate_fn)
        """
        The collate function to use when creating mini-batches from this
        dataset.
        """

    @property
    def data_list(self):
        """Internal list of datasets."""
        return self._datasets

    @data_list.setter
    def data_list(self, value):
        """setter for internal list of datasets."""
        assert len(value) == len(self._datasets)
        assert all([len(d1) == len(d2) for d1, d2 in zip(value, self._datasets)])
        self._datasets = value

    def __getattr__(self, item):
        if item.startswith('__') and item.endswith('__'):  # check needed for copy/pickling
            raise AttributeError(f"Attribute {item} not found")
        if item in self._data_attributes:
            return self._data_attributes[item]
        else:
            raise AttributeError(f"Attribute {item} not found")

    def _getitem_recursive_call(self, idx):
        """We need this recursive call to avoid appending task
        label multiple times inside the __getitem__."""
        if idx < 0:
            if -idx > len(self):
                raise ValueError("absolute value of index should not exceed dataset length")
            idx = len(self) + idx
        dataset_idx = bisect.bisect_right(self._cumulative_sizes, idx)
        if dataset_idx == 0:
            sample_idx = idx
        else:
            sample_idx = idx - self._cumulative_sizes[dataset_idx - 1]

        dd = self._datasets[dataset_idx]
        if isinstance(dd, AvalancheDataset):
            element = dd._getitem_recursive_call(sample_idx)
        else:
            element = dd[sample_idx]

        element = self._frozen_transform_groups(element)
        element = self._transform_groups(element)
        return element

    def __len__(self) -> int:
        return self._cumulative_sizes[-1]

    def get_transform_groups(self):
        """Recursively collects transform groups across the entire data tree.
        Warning: assumes transforms are the same across all concat
        branches."""
        if isinstance(self.datasets[0], AvalancheDataset):
            return self.datasets[0].get_transform_groups() + self._transform_groups
        else:
            return self._transform_groups

    def _flatten_datasets_list(self):
        # Flattens this subset by borrowing the list of concatenated datasets
        # from the original datasets.

        flattened_list = []
        for dataset in self._datasets:
            if isinstance(dataset, AvalancheConcatDataset):
                if isinstance(dataset._transform_groups, EmptyTransformGroups):
                    flattened_list.extend(dataset._datasets)
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
