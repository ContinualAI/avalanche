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
AvalancheDataset offers additional features like the
management of preprocessing pipelines and task/class labels.
"""
import copy
import warnings

from torch.utils.data.dataloader import default_collate

from avalanche.benchmarks.utils.dataset_definitions import IDataset
from .data_attribute import DataAttribute

from typing import List, Any, Sequence, Union, TypeVar, Callable

from .flat_data import FlatData
from .transform_groups import TransformGroups, EmptyTransformGroups
from torch.utils.data import Dataset as TorchDataset


T_co = TypeVar("T_co", covariant=True)
TAvalancheDataset = TypeVar("TAvalancheDataset", bound="AvalancheDataset")


class AvalancheDataset(FlatData):
    """Avalanche Dataset.

    Avlanche dataset are pytorch-compatible Datasets with some additional
    functionality such as:
    - management of transformation groups via :class:`AvalancheTransform`
    - support for sample attributes such as class targets and task labels

    Data Attributes
    ---------------

    Avalanche datasets manage sample-wise information such as class or task
    labels via :class:`DataAttribute`.

    Transformation Groups
    ---------------------

    Avalanche datasets manage transformation via transformation groups.
    Simply put, a transformation group is a named preprocessing function
    (as in torchvision datasets). By default, Avalanche expects
    two transformation groups:
    - 'train', which contains transformations applied to training patterns.
    - 'eval', that contain transformations applied to test patterns.

    Having both groups allows to use different transformations during training
    and evaluation and to seamlessly switch between them by using the
    :func:`train` and :func:`eval` methods. Arbitrary transformation groups
    can be added and used.  If you define custom groups, you can use them by
    calling the `:func:with_transforms` method.

    switching to a different transformation group by calling the ``train()``,
    ``eval()`` or ``with_transforms` methods always returns a new dataset,
    levaing the original one unchanged.

    Ttransformation groups can be manipulated by removing, freezing, or
    replacing transformations. Each operation returns a new dataset, leaving
    the original one unchanged.
    """

    def __init__(
        self,
        datasets: List[IDataset],
        *,
        indices: List[int] = None,
        data_attributes: List[DataAttribute] = None,
        transform_groups: TransformGroups = None,
        frozen_transform_groups: TransformGroups = None,
        collate_fn: Callable[[List], Any] = None,
    ):
        """Creates a ``AvalancheDataset`` instance.

        :param dataset: Original dataset. Beware that
            AvalancheDataset will not overwrite transformations already
            applied by this dataset.
        :param transform_groups: Avalanche transform groups.
        """
        if isinstance(datasets, TorchDataset) or isinstance(
            datasets, AvalancheDataset
        ):
            warnings.warn(
                "AvalancheDataset constructor has been changed. "
                "Please check the documentation for the correct usage. You can"
                " use `avalanche.benchmarks.utils.make_classification_dataset"
                "if you need the old behavior.",
                DeprecationWarning,
            )

        if issubclass(type(datasets), TorchDataset) or  \
                issubclass(type(datasets), AvalancheDataset):
            datasets = [datasets]

        # NOTES on implementation:
        # - raw datasets operations are implemented by _FlatData
        # - data attributes are implemented by DataAttribute
        # - transformations are implemented by TransformGroups
        # AvalancheDataset just takes care to manage all of these attributes
        # together and decides how the information propagates through
        # operations (e.g. how to pass attributes after concat/subset
        # operations).
        can_flatten = (
            (transform_groups is None)
            and (frozen_transform_groups is None)
            and data_attributes is None
            and collate_fn is None
        )
        super().__init__(datasets, indices, can_flatten)

        if data_attributes is None:
            self._data_attributes = {}
        else:
            self._data_attributes = {da.name: da for da in data_attributes}
            for da in data_attributes:
                ld = sum(len(d) for d in self._datasets)
                if len(da) != ld:
                    raise ValueError(
                        "Data attribute {} has length {} but the dataset "
                        "has length {}".format(da.name, len(da), ld)
                    )
        if isinstance(transform_groups, dict):
            transform_groups = TransformGroups(transform_groups)
        if isinstance(frozen_transform_groups, dict):
            frozen_transform_groups = TransformGroups(frozen_transform_groups)
        self._transform_groups = transform_groups
        self._frozen_transform_groups = frozen_transform_groups
        self.collate_fn = collate_fn

        ####################################
        # Init transformations
        ####################################
        cgroup = None
        # inherit transformation group from original dataset
        for dd in self._datasets:
            if isinstance(dd, AvalancheDataset):
                if cgroup is None and dd._transform_groups is not None:
                    cgroup = dd._transform_groups.current_group
                elif (
                    dd._transform_groups is not None
                    and dd._transform_groups.current_group != cgroup
                ):
                    # all datasets must have the same transformation group
                    warnings.warn(
                        f"Concatenated datasets have different transformation "
                        f"groups. Using group={cgroup}."
                    )
        if self._frozen_transform_groups is None:
            self._frozen_transform_groups = EmptyTransformGroups()
        if self._transform_groups is None:
            self._transform_groups = EmptyTransformGroups()

        if cgroup is None:
            cgroup = "train"
        self._frozen_transform_groups.current_group = cgroup
        self._transform_groups.current_group = cgroup

        ####################################
        # Init collate_fn
        ####################################
        if len(datasets) > 0:
            self.collate_fn = self._init_collate_fn(datasets[0], collate_fn)
        else:
            self.collate_fn = default_collate
        """
        The collate function to use when creating mini-batches from this
        dataset.
        """

        ####################################
        # Init data attributes
        ####################################
        # concat attributes from child datasets
        if len(self._datasets) > 0 and isinstance(
            self._datasets[0], AvalancheDataset
        ):
            for attr in self._datasets[0]._data_attributes.values():
                if attr.name in self._data_attributes:
                    continue  # don't touch overridden attributes
                acat = attr
                found_all = True
                for d2 in self._datasets[1:]:
                    if hasattr(d2, attr.name):
                        acat = acat.concat(getattr(d2, attr.name))
                    else:
                        found_all = False
                        break
                if found_all:
                    self._data_attributes[attr.name] = acat

        if self._indices is not None:  # subset operation for attributes
            for da in self._data_attributes.values():
                # TODO: this was the old behavior. How do we know what to do if
                # we permute the entire dataset?
                # DEPRECATED! always subset attributes
                # we keep this behavior only for `classification_subset`
                # if len(da) != sum([len(d) for d in datasets]):
                #     self._data_attributes[da.name] = da
                # else:
                #     self._data_attributes[da.name] = da.subset(self._indices)
                #
                #     dasub = da.subset(indices)
                #     self._data_attributes[da.name] = dasub
                dasub = da.subset(self._indices)
                self._data_attributes[da.name] = dasub

        # set attributes dynamically
        for el in self._data_attributes.values():
            assert len(el) == len(
                self
            ), f"BUG: Wrong size for attribute {el.name}"

            if hasattr(self, el.name):
                raise ValueError(
                    f"Trying to add DataAttribute `{el.name}` to "
                    f"AvalancheDataset but the attribute name is already used."
                )
            setattr(self, el.name, el)

    @property
    def transform(self):
        raise AttributeError(
            "Cannot access or modify transform directly. Use transform_groups "
            "methods such as `replace_current_transform_group`. "
            "See the documentation for more info."
        )

    def __eq__(self, other: "make_avalanche_dataset"):
        if not hasattr(other, "_datasets"):
            return False
        eq_datasets = len(self._datasets) == len(other._datasets)
        eq_datasets = eq_datasets and all(
            d1 == d2 for d1, d2 in zip(self._datasets, other._datasets)
        )
        return (
            eq_datasets
            and self._transform_groups == other._transform_groups
            and self._data_attributes == other._data_attributes
            and self.collate_fn == other.collate_fn
        )

    def _getitem_recursive_call(self, idx, group_name):
        """Private method only for internal use.

        We need this recursive call to avoid appending task
        label multiple times inside the __getitem__.
        """
        dataset_idx, idx = self._get_idx(idx)

        dd = self._datasets[dataset_idx]
        if isinstance(dd, AvalancheDataset):
            element = dd._getitem_recursive_call(idx, group_name=group_name)
        else:
            element = dd[idx]

        if self._frozen_transform_groups is not None:
            element = self._frozen_transform_groups(
                element, group_name=group_name
            )
        if self._transform_groups is not None:
            element = self._transform_groups(element, group_name=group_name)
        return element

    def __getitem__(self, idx) -> Union[T_co, Sequence[T_co]]:
        elem = self._getitem_recursive_call(
            idx, self._transform_groups.current_group
        )
        for da in self._data_attributes.values():
            if da.use_in_getitem:
                if isinstance(elem, dict):
                    elem[da.name] = da[idx]
                elif isinstance(elem, tuple):
                    elem = list(elem)
                    elem.append(da[idx])
                else:
                    elem.append(da[idx])
        return elem

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
        return datacopy

    def freeze_transforms(self):
        """Returns a new dataset with the transformation groups frozen."""
        tgroups = copy.copy(self._transform_groups)
        frozen_tgroups = copy.copy(self._frozen_transform_groups)
        datacopy = self._shallow_clone_dataset()
        datacopy._frozen_transform_groups = frozen_tgroups + tgroups
        datacopy._transform_groups = EmptyTransformGroups()
        dds = []
        for dd in datacopy._datasets:
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
        for dd in dataset_copy._datasets:
            if isinstance(dd, AvalancheDataset):
                dds.append(dd.remove_current_transform_group())
            else:
                dds.append(dd)
        dataset_copy._datasets = dds
        return dataset_copy

    def replace_current_transform_group(self, transform):
        """Recursively remove the current transformation group from the
        dataset tree and replaces it."""
        dataset_copy = self.remove_current_transform_group()
        cgroup = dataset_copy._transform_groups.current_group
        dataset_copy._transform_groups[cgroup] = transform
        dds = []
        for dd in dataset_copy._datasets:
            if isinstance(dd, AvalancheDataset):
                dds.append(dd.remove_current_transform_group())
            else:
                dds.append(dd)
        dataset_copy._datasets = dds
        return dataset_copy

    def _shallow_clone_dataset(self: TAvalancheDataset) -> TAvalancheDataset:
        """Clone dataset.
        This is a shallow copy, i.e. the data attributes are not copied.
        """
        dataset_copy = copy.copy(self)
        dataset_copy._transform_groups = copy.copy(
            dataset_copy._transform_groups
        )
        dataset_copy._frozen_transform_groups = copy.copy(
            dataset_copy._frozen_transform_groups
        )
        return dataset_copy

    def _init_collate_fn(self, dataset, collate_fn):
        if collate_fn is not None:
            return collate_fn

        if hasattr(dataset, "collate_fn"):
            return getattr(dataset, "collate_fn")

        return default_collate


def make_avalanche_dataset(
    dataset: IDataset,
    *,
    data_attributes: List[DataAttribute] = None,
    transform_groups: TransformGroups = None,
    frozen_transform_groups: TransformGroups = None,
    collate_fn: Callable[[List], Any] = None,
):
    """Avalanche Dataset.

    Creates a ``AvalancheDataset`` instance.
    See ``AvalancheDataset`` for more details.

    :param dataset: Original dataset. Beware that
        AvalancheDataset will not overwrite transformations already
        applied by this dataset.
    :param transform_groups: Avalanche transform groups.
    """
    return AvalancheDataset(
        [dataset],
        data_attributes=data_attributes,
        transform_groups=transform_groups,
        frozen_transform_groups=frozen_transform_groups,
        collate_fn=collate_fn,
    )


def _print_frozen_transforms(self):
    """Internal debugging method. Do not use it.
    Prints the current frozen transformations."""
    print("FROZEN TRANSFORMS:\n" + str(self._frozen_transform_groups))
    for dd in self._datasets:
        if isinstance(dd, AvalancheDataset):
            print("PARENT FROZEN:\n")
            _print_frozen_transforms(dd)


def _print_nonfrozen_transforms(self):
    """Internal debugging method. Do not use it.
    Prints the current non-frozen transformations."""
    print("TRANSFORMS:\n" + str(self._transform_groups))
    for dd in self._datasets:
        if isinstance(dd, AvalancheDataset):
            print("PARENT TRANSFORMS:\n")
            _print_nonfrozen_transforms(dd)


def _print_transforms(self):
    """Internal debugging method. Do not use it.
    Prints the current transformations."""
    self._print_frozen_transforms()
    self._print_nonfrozen_transforms()


__all__ = ["AvalancheDataset", "make_avalanche_dataset"]
