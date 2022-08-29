from collections import defaultdict
from typing import Dict, Union, Callable, Tuple, Sequence

from avalanche.benchmarks.utils import MultiParamTransform, Compose


class TransformGroups:
    """Transformation groups for Avalanche datasets.

    AvalancheTransform supports preprocessing and augmentation pipelines for
    Avalanche datasets. Transfomations are separated into groups (e.g. `train`
    transforms and `test` transforms), that can be easily switched using the
    `with_transform` method.
    """

    def __init__(self,
                 transform_groups: Dict[str, Union[Callable, Tuple[Callable, Callable]]],
                 current_group="train"):
        self.transform_groups = transform_groups
        self.current_group = current_group

        if "train" in transform_groups:
            if "eval" not in transform_groups:
                transform_groups["eval"] = transform_groups["train"]

        if "train" not in transform_groups:
            transform_groups["train"] = None

        if "eval" not in transform_groups:
            transform_groups["eval"] = None

    def __getitem__(self, item):
        return self.transform_groups[item]

    def __setitem__(self, key, value):
        self.transform_groups[key] = value

    def __call__(self, *args, **kwargs):
        """Apply current transformation group to element."""
        element = list(*args)

        curr_t = self.transform_groups[self.current_group]
        if curr_t is None:
            return element
        if not isinstance(curr_t, Sequence):
            element[0] = curr_t(element[0])
        else:
            if curr_t[0] is not None:
                element = MultiParamTransform(curr_t[0])(*element)
            if curr_t[1] is not None:
                element[1] = curr_t[1](element[1])
        return element

    def __add__(self, other: "TransformGroups"):
        tgroups = {**self.transform_groups}
        for gname, gtrans in other.transform_groups.items():
            if gname not in tgroups:
                tgroups[gname] = gtrans
            elif gtrans is not None:
                tgroups[gname] = Compose([gtrans, tgroups[gname]])
        return TransformGroups(tgroups, self.current_group)

    def __eq__(self, other: "TransformGroups"):
        return self.transform_groups == other.transform_groups and \
            self.current_group == other.current_group

    def with_transform(self, group_name):
        assert group_name in self.transform_groups
        self.current_group = group_name


class DefaultTransformGroups(TransformGroups):
    """A transformation groups that is equal for all groups."""

    def __init__(self, transforms):
        super().__init__({})
        self.transform_groups = defaultdict(lambda: transforms)

    def with_transform(self, group_name):
        pass


class EmptyTransformGroups(DefaultTransformGroups):
    def __init__(self):
        super().__init__({})
        self.transform_groups = defaultdict(lambda: None)
