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
    Transformation groups manage transformations that are in different
    phases of the optimization process, such as different train/eval
    transformations.

    They support multi-argument transforms as defined in
    `avalanche.benchmarks.utils.transforms`.
"""
from collections import defaultdict
from functools import partial
from typing import (
    Any,
    Dict,
    List,
    Mapping,
    Optional,
    Tuple,
    Union,
    Callable,
    Sequence,
    Protocol,
)

from avalanche.benchmarks.utils.transforms import (
    MultiParamCompose,
    TupleTransform,
    MultiParamTransform,
)


# Info: https://mypy.readthedocs.io/en/stable/protocols.html#callback-protocols
class XComposedTransformDef(Protocol):
    def __call__(self, *input_values: Any) -> Any:
        pass


class XTransformDef(Protocol):
    def __call__(self, input_value: Any) -> Any:
        pass


class YTransformDef(Protocol):
    def __call__(self, input_value: Any) -> Any:
        pass


XTransform = Optional[Union[XTransformDef, XComposedTransformDef]]
YTransform = Optional[YTransformDef]
TransformGroupDef = Union[None, XTransform, Tuple[XTransform, YTransform]]


def identity(x):
    """
    this is used together with partial to replace a lambda function
    that causes pickle to fail
    """
    return x


class TransformGroups:
    """Transformation groups for Avalanche datasets.

    TransformGroups supports preprocessing and augmentation pipelines for
    Avalanche datasets. Transfomations are separated into groups (e.g. `train`
    transforms and `test` transforms), that can be easily switched using the
    `with_transform` method.
    """

    def __init__(
        self,
        transform_groups: Mapping[
            str,
            Union[None, Callable, Sequence[Union[Callable, XTransform, YTransform]]],
        ],
        current_group="train",
    ):
        """Constructor.

        :param transform_groups: A dictionary with group names (string) as keys
            and transformations (pytorch transformations) as values.
        :param current_group: the currently active group.
        """
        self.transform_groups: Dict[
            str, Union[TupleTransform, MultiParamTransform, None]
        ] = dict()
        for group, transform in transform_groups.items():
            norm_transform = _normalize_transform(transform)
            self.transform_groups[group] = norm_transform

        self.current_group = current_group

        if "train" in self.transform_groups:
            if "eval" not in self.transform_groups:
                self.transform_groups["eval"] = self.transform_groups["train"]

        if "train" not in self.transform_groups:
            self.transform_groups["train"] = None

        if "eval" not in self.transform_groups:
            self.transform_groups["eval"] = None

    def __getitem__(self, item):
        return self.transform_groups[item]

    def __setitem__(self, key, value):
        self.transform_groups[key] = _normalize_transform(value)

    def __call__(self, *args, group_name=None):
        """Apply current transformation group to element."""
        element: List[Any] = list(*args)

        if group_name is None:
            curr_t = self.transform_groups[self.current_group]
        else:
            curr_t = self.transform_groups[group_name]
        if curr_t is None:  # empty group
            return element
        elif not isinstance(curr_t, MultiParamTransform):  #
            element[0] = curr_t(element[0])
        else:
            element = curr_t(*element)
        return element

    def __add__(self, other: "TransformGroups"):
        tgroups = {**self.transform_groups}
        for gname, gtrans in other.transform_groups.items():
            if gname not in tgroups:
                tgroups[gname] = gtrans
            elif gtrans is not None:
                composed_transforms = []
                self_group = tgroups[gname]
                other_group = gtrans

                to_expand_group: Union[TupleTransform, MultiParamTransform, None]
                for to_expand_group in [self_group, other_group]:
                    if to_expand_group is None:
                        pass
                    else:
                        assert callable(to_expand_group)
                        composed_transforms.append(to_expand_group)

                tgroups[gname] = MultiParamCompose(composed_transforms)
        return TransformGroups(tgroups, self.current_group)

    def __eq__(self, other: object):
        if not isinstance(other, TransformGroups):
            return NotImplemented
        return (
            self.transform_groups == other.transform_groups
            and self.current_group == other.current_group
        )

    def with_transform(self, group_name):
        assert group_name in self.transform_groups
        self.current_group = group_name

    def __str__(self):
        res = ""
        for k, v in self.transform_groups.items():
            if len(res) > 0:
                res += "\n"
            res += f"- {k}: {v}"
        res = f"current_group: '{self.current_group}'\n" + res
        return res

    def __copy__(self):
        # copy of TransformGroups should copy the dictionary
        # to avoid side effects
        cls = self.__class__
        result = cls.__new__(cls)
        result.__dict__.update(self.__dict__)
        result.transform_groups = self.transform_groups.copy()
        return result


class DefaultTransformGroups(TransformGroups):
    """A transformation groups that is equal for all groups."""

    def __init__(self, transform):
        super().__init__({})
        transform = _normalize_transform(transform)
        self.transform_groups = defaultdict(partial(identity, transform))

    def with_transform(self, group_name):
        self.current_group = group_name


class EmptyTransformGroups(DefaultTransformGroups):
    def __init__(self):
        super().__init__({})
        self.transform_groups = defaultdict(partial(identity, None))

    def __call__(self, elem, group_name=None):
        """Apply current transformation group to element."""
        if self.transform_groups[group_name] is None:
            return elem
        else:
            return super().__call__(elem, group_name=group_name)


def _normalize_transform(transforms):
    """Normalize transform to MultiParamTransform."""
    if transforms is None:
        return None
    if not isinstance(transforms, MultiParamTransform):
        if isinstance(transforms, Sequence):
            return TupleTransform(transforms)
        else:
            return TupleTransform([transforms])
    return transforms


__all__ = [
    "XComposedTransformDef",
    "XTransformDef",
    "YTransformDef",
    "XTransform",
    "YTransform",
    "TransformGroupDef",
    "TransformGroups",
    "DefaultTransformGroups",
    "EmptyTransformGroups",
]
