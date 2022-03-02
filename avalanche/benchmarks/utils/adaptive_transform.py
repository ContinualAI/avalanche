import warnings
from typing import Callable, Sequence
from inspect import signature, Parameter


class Compose:
    """
    A replacement for torchvision's Compose transformation.

    Differently from the original Compose, this transformation can handle both
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
        self.transforms = transforms
        self.param_def = []

        self.max_params = -1
        self.min_params = -1

        if len(transforms) > 0:
            for tr in transforms:
                self.param_def.append(
                    MultiParamTransform._detect_parameters(tr))
            all_maxes = set([max_p for _, max_p in self.param_def])
            if len(all_maxes) > 1:
                warnings.warn(
                    'Transformations define a different amount of parameters. '
                    'This may lead to errors. This warning will only appear'
                    'once.', ComposeMaxParamsWarning)

            if -1 in all_maxes:
                self.max_param = -1  # At least one transform has an *args param
            else:
                self.max_params = max(all_maxes)
            self.min_params = min([min_p for min_p, _ in self.param_def])

    def __call__(self, *args, force_tuple_output=False):
        if len(self.transforms) > 0:
            for transform, (min_par, max_par) in zip(self.transforms,
                                                     self.param_def):
                args = MultiParamTransform._call_transform(
                    transform, min_par, max_par, *args)

        if len(args) == 1 and not force_tuple_output:
            return args[0]  # Single return value (as an unwrapped value)
        return args  # Multiple return values (as a tuple)

    def __repr__(self):
        format_string = self.__class__.__name__ + '('
        for t in self.transforms:
            format_string += '\n'
            format_string += '    {0}'.format(t)
        format_string += '\n)'
        return format_string


class MultiParamTransform:
    def __init__(self, transform: Callable):
        self.transform = transform

        self.min_params, self.max_params = \
            MultiParamTransform._detect_parameters(transform)

    def __call__(self, *args, force_tuple_output=False):
        args = MultiParamTransform._call_transform(
            self.transform, self.min_params, self.max_params, *args)

        if len(args) == 1 and not force_tuple_output:
            return args[0]  # Single return value (as an unwrapped value)
        return args  # Multiple return values (as a tuple)

    def __repr__(self):
        format_string = self.__class__.__name__ + '('
        format_string += '\n'
        format_string += '    {0}'.format(self.transform)
        format_string += '\n)'
        return format_string

    @staticmethod
    def _call_transform(transform_callable, _, max_par, *params):
        # Here we ignore the min_param
        if max_par == -1:  # The transform accepts *args
            n_params = len(params)
        else:
            n_params = min(max_par, len(params))
        params = list(params)

        transform_result = transform_callable(*params[:n_params])
        if not isinstance(transform_result, tuple):
            transform_result = (transform_result,)

        # In this way the transform is free to return more or less elements
        # than the amount of input parameters. May be useful in the future.
        params[:n_params] = transform_result

        return params

    @staticmethod
    def _detect_parameters(transform_callable):
        min_params = 0
        max_params = 0

        if hasattr(transform_callable, 'min_params') and \
                hasattr(transform_callable, 'max_params'):
            min_params = transform_callable.min_params
            max_params = transform_callable.max_params
        elif MultiParamTransform._is_torchvision_transform(transform_callable):
            min_params = 1
            max_params = 1
        else:
            t_sig = signature(transform_callable)
            for param_name in t_sig.parameters:
                param = t_sig.parameters[param_name]
                if param.kind == Parameter.KEYWORD_ONLY:
                    raise ValueError(
                        f'Invalid transformation {transform_callable}: '
                        f'keyword-only parameters (such as {param_name}) are '
                        'not supported.')
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
        return 'torchvision.transforms' in tc_module


class ComposeMaxParamsWarning(Warning):
    def __init__(self, message):
        self.message = message


warnings.simplefilter("once", ComposeMaxParamsWarning)


__all__ = [
    'Compose',
    'MultiParamTransform',
    'ComposeMaxParamsWarning'
]
