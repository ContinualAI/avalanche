"""
API annotations like provided by ray
https://docs.ray.io/en/latest/ray-contribute/stability.html

only for internal use in the library.
"""

import inspect
import warnings
import functools
from typing import Optional


def experimental(reason: Optional[str] = None):
    """Decorator for experimental API.

    Experimental APIs are newer functionality.
    They are well tested, but due to their novelty there may be minor bugs
    or their intergace may still change in the next releases.

    It can be used to decorate methods.

    .. code-block:: python
        from avalanche._annotations import ExperimentalAPI
        @DeveloperAPI
        def shining_new_method():
            print("Hello, world!")

    :param reason: a message to append to the documentation explaining
        the motivation for the experimental tag.
    :return:
    """
    if reason is None:
        reason = ""

    def decorator(func):
        if func.__doc__ is None:
            func.__doc__ = ""
        else:
            func.__doc__ += "\n\n"

        func.__doc__ += "Warning: Experimental API. " + reason

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            return func(*args, **kwargs)

        return wrapper

    return decorator


def deprecated(version: float, reason: str):
    """Decorator to mark functions as deprecated.

    Emits a warning when the function is used.

    :param version: when it will be removed
    :param reason: motivation for deprecation, possibly with suggested
        alternative
    :return:
    """

    def decorator(func):
        if inspect.isclass(func):
            msg_prefix = "Call to deprecated class {name}"
        else:
            msg_prefix = "Call to deprecated function {name}"

        msg_suffix = " (removal in version {version}: {reason})"
        msg = msg_prefix + msg_suffix

        if func.__doc__ is None:
            func.__doc__ = ""
        else:
            func.__doc__ += "\n\n"

        func.__doc__ += "Warning: Deprecated" + msg_suffix.format(
            name=func.__name__, version=version, reason=reason
        )

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            warnings.simplefilter("once", DeprecationWarning)
            warnings.warn(
                msg.format(name=func.__name__, version=version, reason=reason),
                category=DeprecationWarning,
                stacklevel=2,
            )
            return func(*args, **kwargs)

        return wrapper

    return decorator
