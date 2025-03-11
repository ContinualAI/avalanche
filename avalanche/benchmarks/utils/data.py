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
import numpy as np

from torch.utils.data.dataloader import default_collate

from avalanche.benchmarks.utils.dataset_definitions import IDataset
from .data_attribute import DataAttribute

from typing import (
    Dict,
    List,
    Any,
    Optional,
    Sequence,
    TypeVar,
    Callable,
    Union,
    overload,
)

from .flat_data import FlatData
from .transform_groups import TransformGroups, EmptyTransformGroups
from torch.utils.data import Dataset as TorchDataset
from collections import OrderedDict


T_co = TypeVar("T_co", covariant=True)
TAvalancheDataset = TypeVar("TAvalancheDataset", bound="AvalancheDataset")
TDataWTransform = TypeVar("TDataWTransform", bound="_FlatDataWithTransform")


class AvalancheDataset(IDataset[T_co]):
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
        datasets: Sequence[IDataset[T_co]],
        *,
        indices: Optional[List[int]] = None,
        data_attributes: Optional[List[DataAttribute]] = None,
        transform_groups: Optional[TransformGroups] = None,
        frozen_transform_groups: Optional[TransformGroups] = None,
        collate_fn: Optional[Callable[[List], Any]] = None,
    ):
        """Creates a ``AvalancheDataset`` instance.

        :param dataset: Original dataset. Beware that
            AvalancheDataset will not overwrite transformations already
            applied by this dataset.
        :param transform_groups: Avalanche transform groups.
        """
        if issubclass(type(datasets), TorchDataset) or issubclass(
            type(datasets), AvalancheDataset
        ):
            datasets = [datasets]  # type: ignore

        # NOTES on implementation:
        # - raw datasets operations are implemented by _FlatData
        # - data attributes are implemented by DataAttribute
        # - transformations are implemented by TransformGroups
        # AvalancheDataset just takes care to manage all of these attributes
        # together and decides how the information propagates through
        # operations (e.g. how to pass attributes after concat/subset
        # operations).
        flat_datas = []
        for d in datasets:
            if len(d) > 0:
                if isinstance(d, AvalancheDataset):
                    flat_datas.append(d._flat_data)
                elif not isinstance(d, _FlatDataWithTransform):
                    flat_datas.append(_FlatDataWithTransform([d]))
                else:
                    flat_datas.append(d)
        if (
            transform_groups is None
            and frozen_transform_groups is None
            and indices is not None
            and len(flat_datas) == 1
        ):
            # TODO: remove. shouldn't be needed but helps with flattening
            assert len(flat_datas) == 1
            self._flat_data = flat_datas[0].subset(indices)
        elif (
            transform_groups is None
            and frozen_transform_groups is None
            and indices is None
            and len(flat_datas) >= 1
        ):
            # TODO: remove. shouldn't be needed but helps with flattening
            if len(flat_datas) == 0:
                self._flat_data = _FlatDataWithTransform([])
            self._flat_data = flat_datas[0]
            if not isinstance(self._flat_data, _FlatDataWithTransform):
                self._flat_data = _FlatDataWithTransform([self._flat_data])

            for d in flat_datas[1:]:
                if not isinstance(d, _FlatDataWithTransform):
                    d = _FlatDataWithTransform([d])
                self._flat_data = self._flat_data.concat(d)
        else:
            self._flat_data: _FlatDataWithTransform[T_co] = _FlatDataWithTransform(
                flat_datas,
                indices=indices,
                transform_groups=transform_groups,
                frozen_transform_groups=frozen_transform_groups,
            )
        self.collate_fn = collate_fn

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
        new_data_attributes: Dict[str, DataAttribute] = dict()
        if data_attributes is not None:
            new_data_attributes = {da.name: da for da in data_attributes}
            ld = sum(len(d) for d in datasets)
            for da in data_attributes:
                if len(da) != ld:
                    raise ValueError(
                        "Data attribute {} has length {} but the dataset "
                        "has length {}".format(da.name, len(da), ld)
                    )

        self._data_attributes: Dict[str, DataAttribute] = OrderedDict()
        first_dataset = datasets[0] if len(datasets) > 0 else None
        if isinstance(first_dataset, AvalancheDataset):
            for attr in first_dataset._data_attributes.values():
                if attr.name in new_data_attributes:
                    # Keep overridden attributes in their previous position
                    self._data_attributes[attr.name] = new_data_attributes.pop(
                        attr.name
                    )
                    continue

                acat = attr
                found_all = True
                for d2 in datasets[1:]:
                    if hasattr(d2, attr.name):
                        acat = acat.concat(getattr(d2, attr.name))
                    elif len(d2) > 0:  # if empty we allow missing attributes
                        found_all = False
                        break
                if found_all:
                    self._data_attributes[attr.name] = acat

        # Insert new data attributes after inherited ones
        for da in new_data_attributes.values():
            self._data_attributes[da.name] = da

        if indices is not None:  # subset operation for attributes
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
                dasub = da.subset(indices)
                self._data_attributes[da.name] = dasub

        # set attributes dynamically
        for el in self._data_attributes.values():
            assert len(el) == len(self), f"BUG: Wrong size for attribute {el.name}"

            is_property = False
            if hasattr(self, el.name):
                is_property = True
                # Do not raise an error if a property.
                # Any check related to the property will be done
                # in the property setter method.
                if not isinstance(getattr(type(self), el.name, None), property):
                    raise ValueError(
                        f"Trying to add DataAttribute `{el.name}` to "
                        f"AvalancheDataset but the attribute name is "
                        f"already used."
                    )
            if not is_property:
                setattr(self, el.name, el)

    def __len__(self) -> int:
        return len(self._flat_data)

    def __add__(self: TAvalancheDataset, other: TAvalancheDataset) -> TAvalancheDataset:
        return self.concat(other)

    def __radd__(
        self: TAvalancheDataset, other: TAvalancheDataset
    ) -> TAvalancheDataset:
        return other.concat(self)

    @property
    def _datasets(self):
        """Only for backward compatibility of old unit tests. Do not use."""
        return self._flat_data._datasets

    def concat(self: TAvalancheDataset, other: TAvalancheDataset) -> TAvalancheDataset:
        """Concatenate this dataset with other.

        :param other: Other dataset to concatenate.
        :return: A new dataset.
        """
        return self.__class__([self, other])

    def subset(self: TAvalancheDataset, indices: Sequence[int]) -> TAvalancheDataset:
        """Subset this dataset.

        :param indices: The indices to keep.
        :return: A new dataset.
        """
        return self.__class__([self], indices=indices)

    @property
    def transform(self):
        raise AttributeError(
            "Cannot access or modify transform directly. Use transform_groups "
            "methods such as `replace_current_transform_group`. "
            "See the documentation for more info."
        )

    def update_data_attribute(
        self: TAvalancheDataset, name: str, new_value
    ) -> TAvalancheDataset:
        """
        Return a new dataset with the added or replaced data attribute.

        If a object of type :class:`DataAttribute` is passed, then the data
        attribute is setted as is.

        Otherwise, if a raw value is passed, a new DataAttribute is created.
        If a DataAttribute with the same already exists, the use_in_getitem
        flag is inherited, otherwise it is set to False.

        :param name: The name of the data attribute to add/replace.
        :param new_value: Either a :class:`DataAttribute` or a sequence
            containing as many elements as the datasets.
        :returns: A copy of this dataset with the given data attribute set.
        """
        assert len(new_value) == len(
            self
        ), f"Size mismatch when updating data attribute {name}"

        datacopy = self._shallow_clone_dataset()
        datacopy._data_attributes = copy.copy(datacopy._data_attributes)

        if isinstance(new_value, DataAttribute):
            assert name == new_value.name
            datacopy._data_attributes[name] = new_value
        else:
            use_in_getitem = False
            prev_attr = datacopy._data_attributes.get(name, None)
            if prev_attr is not None:
                use_in_getitem = prev_attr.use_in_getitem

            datacopy._data_attributes[name] = DataAttribute(
                new_value, name=name, use_in_getitem=use_in_getitem
            )

        if not hasattr(datacopy, name):
            # Creates the field if it does not exist
            setattr(datacopy, name, datacopy._data_attributes[name])

        return datacopy

    def __eq__(self, other: object):
        for required_attr in ["_flat_data", "_data_attributes", "collate_fn"]:
            if not hasattr(other, required_attr):
                return False

        return (
            other._flat_data == self._flat_data
            and self._data_attributes == other._data_attributes  # type: ignore
            and self.collate_fn == other.collate_fn  # type: ignore
        )

    @overload
    def __getitem__(self, exp_id: int) -> T_co: ...

    @overload
    def __getitem__(self: TAvalancheDataset, exp_id: slice) -> TAvalancheDataset: ...

    def __getitem__(
        self: TAvalancheDataset, idx: Union[int, slice]
    ) -> Union[T_co, TAvalancheDataset]:
        elem = self._flat_data[idx]
        for da in self._data_attributes.values():
            if da.use_in_getitem:
                if isinstance(elem, dict):
                    elem[da.name] = da[idx]
                elif isinstance(elem, tuple):
                    elem = list(elem)  # type: ignore
                    elem.append(da[idx])  # type: ignore
                else:
                    elem.append(da[idx])  # type: ignore
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

    def with_transforms(self: TAvalancheDataset, group_name: str) -> TAvalancheDataset:
        """
        Returns a new dataset with the transformations of a different group
        loaded.

        The current dataset will not be affected.

        :param group_name: The name of the transformations group to use.
        :return: A new dataset with the new transformations.
        """
        datacopy = self._shallow_clone_dataset()
        datacopy._flat_data = datacopy._flat_data.with_transforms(group_name)
        return datacopy

    def freeze_transforms(self: TAvalancheDataset) -> TAvalancheDataset:
        """Returns a new dataset with the transformation groups frozen."""
        datacopy = self._shallow_clone_dataset()
        datacopy._flat_data = datacopy._flat_data.freeze_transforms()
        return datacopy

    def remove_current_transform_group(self):
        """Recursively remove transformation groups from dataset tree."""
        datacopy = self._shallow_clone_dataset()
        fdata = datacopy._flat_data
        datacopy._flat_data = fdata.remove_current_transform_group()
        return datacopy

    def replace_current_transform_group(self, transform):
        """Recursively remove the current transformation group from the
        dataset tree and replaces it."""
        datacopy = self._shallow_clone_dataset()
        fdata = datacopy._flat_data
        datacopy._flat_data = fdata.replace_current_transform_group(transform)
        return datacopy

    def _shallow_clone_dataset(self: TAvalancheDataset) -> TAvalancheDataset:
        """Clone dataset.
        This is a shallow copy, i.e. the data attributes are not copied.
        """
        dataset_copy = copy.copy(self)
        dataset_copy._flat_data = self._flat_data._shallow_clone_dataset()
        return dataset_copy

    def _init_collate_fn(self, dataset, collate_fn):
        if collate_fn is not None:
            return collate_fn

        if hasattr(dataset, "collate_fn"):
            return getattr(dataset, "collate_fn")

        return default_collate

    def __repr__(self):
        return repr(self._flat_data)

    def _tree_depth(self):
        """Return the depth of the tree of datasets.
        Use only to debug performance issues.
        """
        return self._flat_data._tree_depth()


class _FlatDataWithTransform(FlatData[T_co]):
    """Private class used to wrap a dataset with a transformation group.

    Do not use outside of this file.
    """

    def __init__(
        self,
        datasets: Sequence[IDataset[T_co]],
        *,
        indices: Optional[List[int]] = None,
        transform_groups: Optional[TransformGroups] = None,
        frozen_transform_groups: Optional[TransformGroups] = None,
        discard_elements_not_in_indices: bool = False,
    ):
        can_flatten = (transform_groups is None) and (frozen_transform_groups is None)
        super().__init__(
            datasets,
            indices=indices,
            can_flatten=can_flatten,
            discard_elements_not_in_indices=discard_elements_not_in_indices,
        )
        if isinstance(transform_groups, dict):
            transform_groups = TransformGroups(transform_groups)
        if isinstance(frozen_transform_groups, dict):
            frozen_transform_groups = TransformGroups(frozen_transform_groups)

        if transform_groups is None:
            transform_groups = EmptyTransformGroups()

        if frozen_transform_groups is None:
            frozen_transform_groups = EmptyTransformGroups()

        self._transform_groups: TransformGroups = transform_groups
        self._frozen_transform_groups: TransformGroups = frozen_transform_groups

        ####################################
        # Init transformations
        ####################################
        cgroup = None
        # inherit transformation group from original dataset
        for dd in datasets:
            if isinstance(dd, _FlatDataWithTransform):
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

        if cgroup is None:
            cgroup = "train"
        self._frozen_transform_groups.current_group = cgroup
        self._transform_groups.current_group = cgroup

    def __eq__(self, other):
        for required_attr in [
            "_datasets",
            "_transform_groups",
            "_frozen_transform_groups",
        ]:
            if not hasattr(other, required_attr):
                return False

        eq_datasets = len(self._datasets) == len(other._datasets)  # type: ignore
        eq_datasets = eq_datasets and all(
            d1 == d2 for d1, d2 in zip(self._datasets, other._datasets)  # type: ignore
        )
        ftg = other._frozen_transform_groups  # type: ignore
        return (
            eq_datasets
            and self._transform_groups == other._transform_groups  # type: ignore
            and self._frozen_transform_groups == ftg  # type: ignore
        )

    def _getitem_recursive_call(self, idx, group_name) -> T_co:
        """Private method only for internal use.

        We need this recursive call to avoid appending task
        label multiple times inside the __getitem__.
        """
        dataset_idx, idx = self._get_idx(idx)

        dd = self._datasets[dataset_idx]
        if isinstance(dd, _FlatDataWithTransform):
            element = dd._getitem_recursive_call(idx, group_name=group_name)
        else:
            element = dd[idx]

        if self._frozen_transform_groups is not None:
            element = self._frozen_transform_groups(element, group_name=group_name)
        if self._transform_groups is not None:
            element = self._transform_groups(element, group_name=group_name)
        return element

    def __getitem__(
        self: TDataWTransform, idx: Union[int, slice]
    ) -> Union[T_co, TDataWTransform]:
        if isinstance(idx, (int, np.integer)):
            elem = self._getitem_recursive_call(
                idx, self._transform_groups.current_group
            )
            return elem  # type: ignore
        else:
            return super().__getitem__(idx)

    def with_transforms(self: TDataWTransform, group_name: str) -> TDataWTransform:
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

    def freeze_transforms(self: TDataWTransform) -> TDataWTransform:
        """Returns a new dataset with the transformation groups frozen."""
        tgroups = copy.copy(self._transform_groups)
        frozen_tgroups = copy.copy(self._frozen_transform_groups)
        datacopy = self._shallow_clone_dataset()
        datacopy._frozen_transform_groups = frozen_tgroups + tgroups
        datacopy._transform_groups = EmptyTransformGroups()
        dds: List[IDataset] = []
        for dd in datacopy._datasets:
            if isinstance(dd, _FlatDataWithTransform):
                dds.append(dd.freeze_transforms())
            else:
                dds.append(dd)
        datacopy._datasets = dds
        return datacopy

    def remove_current_transform_group(self):
        """Recursively remove transformation groups from dataset tree."""
        dataset_copy = self._shallow_clone_dataset()
        cgroup = dataset_copy._transform_groups.current_group
        dataset_copy._transform_groups[cgroup] = None
        dds = []
        for dd in dataset_copy._datasets:
            if isinstance(dd, _FlatDataWithTransform):
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
            if isinstance(dd, _FlatDataWithTransform):
                dds.append(dd.remove_current_transform_group())
            else:
                dds.append(dd)
        dataset_copy._datasets = dds
        return dataset_copy

    def _shallow_clone_dataset(self: TDataWTransform) -> TDataWTransform:
        """Clone dataset.
        This is a shallow copy, i.e. the data attributes are not copied.
        """
        dataset_copy = copy.copy(self)
        dataset_copy._transform_groups = copy.copy(dataset_copy._transform_groups)
        dataset_copy._frozen_transform_groups = copy.copy(
            dataset_copy._frozen_transform_groups
        )
        return dataset_copy


def make_avalanche_dataset(
    dataset: IDataset[T_co],
    *,
    data_attributes: Optional[List[DataAttribute]] = None,
    transform_groups: Optional[TransformGroups] = None,
    frozen_transform_groups: Optional[TransformGroups] = None,
    collate_fn: Optional[Callable[[List], Any]] = None,
) -> AvalancheDataset[T_co]:
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
