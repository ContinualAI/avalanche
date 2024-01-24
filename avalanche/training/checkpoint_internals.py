import warnings
from typing import Any, Optional, Sequence, Tuple, Type, TypeVar, BinaryIO

import dill
import torch
import io

from avalanche.training.plugins.checkpoint_common_recipes import (
    _set_checkpoint_device_map,
)

CHECKPOINT_MECHANISM_VERSION = "1"

UNIQUE_OBJECTS_CONTAINER = None
REGISTERING_OBJECTS = False
DEDUPLICATING_OBJECTS = False

T = TypeVar("T")


def _constructor_based_unpickle(
    cls: Type[T], checkpointing_version: int, deduplicate: bool, args, kwargs
) -> T:
    """
    This function is used to unpickle a object by reconstructing it from the constructor parameters.

    This function is called directly by dill/pickle when loading a checkpoint that contains an object of class `cls`.
    """

    if checkpointing_version != CHECKPOINT_MECHANISM_VERSION:
        warnings.warn(
            f"Checkpointing mechanism version mismatch: "
            f"expected {CHECKPOINT_MECHANISM_VERSION}, "
            f"got {checkpointing_version}. "
            f"Checkpointing may fail."
        )

    if args is None:
        args = []
    if kwargs is None:
        kwargs = {}

    if deduplicate and _is_deduplicating_objects():
        obj, success = _unpickle_unique(cls, args, kwargs)
        if success:
            return obj

    return cls(*args, **kwargs)


class _NullBinaryIO(BinaryIO):
    """
    An implementation of BinaryIO that just discards the written bytes.
    """

    def write(self, b):
        # Discard the bytes by doing nothing
        return len(b)  # write() should return the number of bytes written


def _register_unique_object(obj, cls, args, kwargs):
    global UNIQUE_OBJECTS_CONTAINER
    if UNIQUE_OBJECTS_CONTAINER is None:
        UNIQUE_OBJECTS_CONTAINER = dict()

    args_key = _object_make_key((args, kwargs))
    if (
        cls not in UNIQUE_OBJECTS_CONTAINER
        or args_key not in UNIQUE_OBJECTS_CONTAINER[cls]
    ):
        # print("Object of class", cls, "NOT already registered")
        if cls not in UNIQUE_OBJECTS_CONTAINER:
            UNIQUE_OBJECTS_CONTAINER[cls] = dict()
        UNIQUE_OBJECTS_CONTAINER[cls][args_key] = obj
    # else:
    #     print("Object of class", cls, "already registered")


def _unpickle_unique(cls: Type[T], args, kwargs) -> Tuple[Optional[T], bool]:
    global UNIQUE_OBJECTS_CONTAINER
    if UNIQUE_OBJECTS_CONTAINER is None:
        return None, False

    if cls not in UNIQUE_OBJECTS_CONTAINER:
        return None, False

    args_key = _object_make_key((args, kwargs))
    if args_key not in UNIQUE_OBJECTS_CONTAINER[cls]:
        return None, False
    else:
        # print("Deduplicated object of class", cls)
        return UNIQUE_OBJECTS_CONTAINER[cls][args_key], True


def _object_make_key(obj):
    # Pickle an object to a byte buffer
    # use torch.save to pickle
    io_buffer = io.BytesIO()
    torch.save(obj, io_buffer, pickle_module=dill)

    # Reset the buffer position to the start of the buffer
    io_buffer.seek(0)

    # Read the buffer as bytes
    return io_buffer.read()


class _CheckpointLoadingContext:
    """
    The context manager used to load a checkpoint.

    This ensures that some optional functionality work as expected.

    This is an internal utility, use at your own risk.

    Current functionalities supported by this context are:
    - device mapping, to ensure that the tensors are loaded on the correct
        device. Supports loading on a different device than the one used
        during the serialization.
    - object de-duplication, to ensure that unique objects are not
        created This is useful to avoid duplicating the memory usage when
        loading a checkpoint with a large number of datasets
        (or an experiment that was already checkpointed are re-loaded multiple times).
        Objects that need de-duplication must be registered as such using helpers
        such as :func:`constructor_based_serialization`.
        Standard use: dataset objects used to create the benchmark.
    """

    def __init__(self, map_location, unique_objects: Optional[Sequence[Any]]):
        self._map_location = map_location
        self._unique_objects = unique_objects

    def __enter__(self):
        global UNIQUE_OBJECTS_CONTAINER

        _set_checkpoint_device_map(self._map_location)

        if self._unique_objects is not None:
            _set_registering_objects(True)
            _set_deduplicating_objects(False)
            UNIQUE_OBJECTS_CONTAINER = dict()
            null_io_buffer = _NullBinaryIO()
            torch.save(self._unique_objects, null_io_buffer, pickle_module=dill)
            _set_registering_objects(False)
            _set_deduplicating_objects(True)

    def __exit__(self, exc_type, exc_val, exc_tb):
        global UNIQUE_OBJECTS_CONTAINER

        _set_checkpoint_device_map(None)
        _set_registering_objects(False)
        _set_deduplicating_objects(False)
        UNIQUE_OBJECTS_CONTAINER = None


def _is_deduplicating_objects():
    global DEDUPLICATING_OBJECTS
    return DEDUPLICATING_OBJECTS


def _set_deduplicating_objects(deduplicating: bool):
    global DEDUPLICATING_OBJECTS
    DEDUPLICATING_OBJECTS = deduplicating


def _is_registering_objects():
    global REGISTERING_OBJECTS
    return REGISTERING_OBJECTS


def _set_registering_objects(registering: bool):
    global REGISTERING_OBJECTS
    REGISTERING_OBJECTS = registering
