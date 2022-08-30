from collections import defaultdict
from typing import Dict, Union, Callable, Tuple, Sequence

from avalanche.benchmarks.utils.transforms import \
    MultiParamTransformCallable, MultiParamCompose, \
    TupleTransform, MultiParamTranform


class TransformGroups:
    """Transformation groups for Avalanche datasets.

    AvalancheTransform supports preprocessing and augmentation pipelines for
    Avalanche datasets. Transfomations are separated into groups (e.g. `train`
    transforms and `test` transforms), that can be easily switched using the
    `with_transform` method.
    """

    def __init__(self,
                 transform_groups: Dict[str, Union[Callable, Sequence[Callable]]],
                 current_group="train"):
        for group, transform in transform_groups.items():
            if not isinstance(transform, MultiParamTranform):
                if isinstance(transform, Sequence):
                    transform_groups[group] = TupleTransform(transform)
                else:
                    transform_groups[group] = TupleTransform([transform])
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
        if value is not None:
            if isinstance(value, Sequence):
                value = TupleTransform(value)
        self.transform_groups[key] = value

    def __call__(self, *args, **kwargs):
        """Apply current transformation group to element."""
        element = list(*args)

        curr_t = self.transform_groups[self.current_group]
        if curr_t is None:  # empty group
            return element
        elif not isinstance(curr_t, MultiParamTranform):  #
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
                tgroups[gname] = MultiParamCompose([tgroups[gname], gtrans])
        return TransformGroups(tgroups, self.current_group)

    def __eq__(self, other: "TransformGroups"):
        return self.transform_groups == other.transform_groups and \
            self.current_group == other.current_group

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

    def __init__(self, transforms):
        super().__init__({})
        self.transform_groups = defaultdict(lambda: transforms)

    def with_transform(self, group_name):
        pass


class EmptyTransformGroups(DefaultTransformGroups):
    def __init__(self):
        super().__init__({})
        self.transform_groups = defaultdict(lambda: None)
