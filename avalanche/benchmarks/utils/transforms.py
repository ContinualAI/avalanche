from collections import defaultdict

from benchmarks.utils import MultiParamTransform


class AvalancheTransform:
    """Transformation groups for Avalanche datasets.

    AvalancheTransform supports preprocessing and augmentation pipelines for
    Avalanche datasets. Transfomations are separated into groups (e.g. `train`
    transforms and `test` transforms), that can be easily switched using the
    `with_transform` method.
    """
    def __init__(self,
                 transform_groups,
                 current_group="eval"):
        self.transform_groups = transform_groups
        self.current_group = current_group

    def __call__(self, *args, **kwargs):
        """Apply current transformation group to element."""
        element = list(*args)

        Xt, Yt = self.transform_groups[self.current_group]
        if Xt is not None:
            element = MultiParamTransform(Xt)(*element)
        if Yt is not None:
            element[1] = Yt(element[1])
        return element

    def with_transform(self, group_name):
        assert group_name in self.transform_groups
        self.current_group = group_name


class EmptyTransform(AvalancheTransform):
    def __init__(self):
        transform_groups = defaultdict(lambda: None, None)
        super().__init__(transform_groups)


class FrozenTransform(AvalancheTransform):
    def __init__(self, transforms):
        transform_groups = defaultdict(lambda: transforms)
        super().__init__(transform_groups)
