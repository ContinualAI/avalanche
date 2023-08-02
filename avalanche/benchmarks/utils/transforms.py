################################################################################
# Copyright (c) 2022 ContinualAI.                                              #
# Copyrights licensed under the MIT License.                                   #
# See the accompanying LICENSE file for terms.                                 #
#                                                                              #
# Date: 19-07-2022                                                             #
# Author(s): Lorenzo Pellegrini, Antonio Carta                                 #
# E-mail: contact@continualai.org                                              #
# Website: avalanche.continualai.org                                           #
################################################################################
"""
    Avalanche transformations are multi-argument.
    This module contains a bunch of utility classes to help define
    multi-argument transformations.
"""
from abc import ABC, abstractmethod
import warnings
from typing import Any, Callable, Iterable, List, Sequence, Tuple, Union
from inspect import signature, Parameter
from torchvision.transforms import Compose


class MultiParamTransform(ABC):
    """We need this class to be able to distinguish between a single argument
    transformation and multi-argument ones.

    Transformations are callable objects.
    """

    @abstractmethod
    def __call__(self, *args, **kwargs):
        """
        Applies this transformations to the given inputs.
        """
        pass

    @abstractmethod
    def flat_transforms(self, position: int) -> List[Any]:
        """
        Returns a flat list of transformations.

        A flat list of transformations is a list in which
        all intermediate wrappers (such as torchvision Compose,
        Avalanche MultiParamCompose, ...) are removed.

        The position parameter is used to control which transformations
        are to be returned based on the position of the tranformed element.
        Position 0 means transformations on the "x" value,
        1 means "target" (or y) transformations, and so on.

        Please note that transformations acting on multiple parameters
        may be returned when appropriate. This is common for object
        detection augmentations that transform x (image) and y (bounding boxes)
        inputs at the same time.

        :position: The position of the tranformed element.
        :return: A list of transformations for the given position.
        """
        pass


class MultiParamCompose(MultiParamTransform):
    """Compose transformation for multi-argument transformations.

    Differently from torchvision Compose, this transformation can handle both
    single-element and multi-elements transformations.

    For instance, single-element transformations are commonly used in
    classification tasks where there is no need to transform the class label.
    Multi-element transformations are used to transform the image and
    bounding box annotations at the same timein object detection tasks. This
    is needed as applying augmentations (such as flipping) may change the
    position of objects in the image.

    This class automatically detects the type of augmentation by inspecting
    its signature. Keyword-only arguments are never filled.
    """

    def __init__(self, transforms: Sequence[Callable]):
        # skip empty transforms
        transforms = list(filter(lambda x: x is not None, transforms))
        self.transforms = list(transforms)
        self.param_def: List[Tuple[int, int]] = []

        self.max_params = -1
        self.min_params = -1

        if len(transforms) > 0:
            for tr in transforms:
                self.param_def.append(
                    MultiParamTransformCallable._detect_parameters(tr)
                )
            all_maxes = set([max_p for _, max_p in self.param_def])
            if len(all_maxes) > 1:
                warnings.warn(
                    "Transformations define a different number of parameters. "
                    "This may lead to errors. This warning will only appear"
                    "once.",
                    ComposeMaxParamsWarning,
                )

            if -1 in all_maxes:
                self.max_param = -1  # At least one transform has an *args param
            else:
                self.max_params = max(all_maxes)
            self.min_params = min([min_p for min_p, _ in self.param_def])

    def __eq__(self, other):
        if self is other:
            return True

        if not isinstance(other, MultiParamCompose):
            return False

        return (
            self.transforms == other.transforms
            and self.param_def == other.param_def
            and self.min_params == other.min_params
            and self.max_params == other.max_params
        )

    def __call__(self, *args, force_tuple_output=False):
        if len(self.transforms) > 0:
            for transform, (min_par, max_par) in zip(self.transforms, self.param_def):
                args = MultiParamTransformCallable._call_transform(
                    transform, min_par, max_par, *args
                )

        if len(args) == 1 and not force_tuple_output:
            return args[0]  # Single return value (as an unwrapped value)
        return args  # Multiple return values (as a tuple)

    def __repr__(self):
        format_string = self.__class__.__name__ + "("
        for t in self.transforms:
            format_string += "\n"
            format_string += "    {0}".format(t)
        format_string += "\n)"
        return format_string

    def __str__(self):
        return self.__repr__()

    def flat_transforms(self, position: int):
        all_transforms = []

        for transform, par_def in zip(self.transforms, self.param_def):
            max_params = par_def[1]

            if position < max_params or max_params == -1:
                all_transforms.append(transform)

        return flat_transforms_recursive(all_transforms, position)


class MultiParamTransformCallable(MultiParamTransform):
    """Generic multi-argument transformation."""

    def __init__(self, transform: Callable):
        self.transform = transform

        (
            self.min_params,
            self.max_params,
        ) = MultiParamTransformCallable._detect_parameters(transform)

    def __call__(self, *args, force_tuple_output=False):
        args = MultiParamTransformCallable._call_transform(
            self.transform, self.min_params, self.max_params, *args
        )

        if len(args) == 1 and not force_tuple_output:
            return args[0]  # Single return value (as an unwrapped value)
        return args  # Multiple return values (as a tuple)

    def __repr__(self):
        format_string = self.__class__.__name__ + "("
        format_string += "\n"
        format_string += "    {0}".format(self.transform)
        format_string += "\n)"
        return format_string

    @staticmethod
    def _call_transform(transform_callable, _, max_par, *params):
        # Here we ignore the min_param
        if max_par == -1:  # The transform accepts *args
            n_params = len(params)
        else:
            n_params = min(max_par, len(params))
        params_list = list(params)

        transform_result = transform_callable(*params_list[:n_params])
        if not isinstance(transform_result, Sequence):
            transform_result = (transform_result,)

        # In this way the transform is free to return more or less elements
        # than the amount of input parameters. May be useful in the future.
        params_list[:n_params] = transform_result

        return params_list

    @staticmethod
    def _detect_parameters(transform_callable) -> Tuple[int, int]:
        min_params = 0
        max_params = 0

        if hasattr(transform_callable, "min_params") and hasattr(
            transform_callable, "max_params"
        ):
            min_params = transform_callable.min_params
            max_params = transform_callable.max_params
        elif MultiParamTransformCallable._is_torchvision_transform(transform_callable):
            min_params = 1
            max_params = 1
        else:
            t_sig = signature(transform_callable)
            for param_name in t_sig.parameters:
                param = t_sig.parameters[param_name]
                if param.kind == Parameter.KEYWORD_ONLY:
                    raise ValueError(
                        f"Invalid transformation {transform_callable}: "
                        f"keyword-only parameters (such as {param_name}) are "
                        "not supported."
                    )
                elif param.kind == Parameter.POSITIONAL_ONLY:
                    # Positional-only (not much used)
                    min_params += 1
                    max_params += 1
                elif param.kind == Parameter.POSITIONAL_OR_KEYWORD:
                    # Standard function parameter
                    if param.default == Parameter.empty:
                        # Not optional
                        min_params += 1
                        max_params += 1
                    else:
                        # Has a default value -> optional
                        max_params += 1
                elif param.kind == Parameter.VAR_POSITIONAL:  # *args
                    max_params = -1  # As for "infinite"
                # elif param.kind == Parameter.VAR_KEYWORD  # **kwargs
                # **kwargs can be safely ignored (they will be empty)
        return min_params, max_params

    @staticmethod
    def _is_torchvision_transform(transform_callable):
        tc_class = transform_callable.__class__
        tc_module = tc_class.__module__
        return "torchvision.transforms" in tc_module

    def flat_transforms(self, position: int):
        if position < self.max_params or self.max_params == -1:
            return flat_transforms_recursive(self.transform, position)
        return []

    def __eq__(self, other):
        if self is other:
            return True

        if not isinstance(other, MultiParamTransformCallable):
            return False

        return (
            self.transform == other.transform
            and self.min_params == other.min_params
            and self.max_params == other.max_params
        )


class TupleTransform(MultiParamTransform):
    """Multi-argument transformation represented as tuples."""

    def __init__(self, transforms: Sequence[Callable]):
        self.transforms = list(transforms)

    def __call__(self, *args):
        args_list = list(args)
        for idx, transform in enumerate(self.transforms):
            if transform is not None:
                args_list[idx] = transform(args_list[idx])
        return args_list

    def __str__(self):
        return "TupleTransform({})".format(self.transforms)

    def __repr__(self):
        return "TupleTransform({})".format(self.transforms)

    def __eq__(self, other):
        if self is other:
            return True

        if not isinstance(other, TupleTransform):
            return False

        return self.transforms == other.transforms

    def flat_transforms(self, position: int):
        if position < len(self.transforms):
            return flat_transforms_recursive(self.transforms[position], position)
        return []


def flat_transforms_recursive(transforms: Union[List, Any], position: int) -> List[Any]:
    """
    Flattens a list of transformations.

    :param transforms: The list of transformations to flatten.
    :param position: The position of the transformed element.
    :return: A flat list of transformations.
    """
    if not isinstance(transforms, Iterable):
        transforms = [transforms]

    must_flat = True
    while must_flat:
        must_flat = False
        flattened_list = []

        for transform in transforms:
            flat_strat = getattr(transform, "flat_transforms", None)
            if callable(flat_strat):
                flattened_list.extend(flat_strat(position))
                must_flat = True
            elif isinstance(transform, Compose):
                flattened_list.extend(transform.transforms)
                must_flat = True
            elif isinstance(transform, Sequence):
                flattened_list.extend(transform)
                must_flat = True
            elif transform is None:
                pass
            else:
                flattened_list.append(transform)

        transforms = flattened_list

    return transforms


class ComposeMaxParamsWarning(Warning):
    def __init__(self, message):
        self.message = message


warnings.simplefilter("once", ComposeMaxParamsWarning)


__all__ = [
    "MultiParamTransform",
    "MultiParamCompose",
    "MultiParamTransformCallable",
    "ComposeMaxParamsWarning",
    "TupleTransform",
    "flat_transforms_recursive",
]
