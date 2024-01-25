import warnings
from typing import (
    Any,
    Callable,
    Dict,
    Optional,
    Sequence,
    Tuple,
    Type,
    TypeVar,
    BinaryIO,
    IO,
    Union,
)
from typing_extensions import TypeAlias

import dill
import torch
import io
import os

FILE_LIKE: TypeAlias = Union[str, os.PathLike, BinaryIO, IO[bytes]]
MAP_LOCATION: TypeAlias = Optional[
    Union[
        Callable[[torch.Tensor, str], torch.Tensor], torch.device, str, Dict[str, str]
    ]
]

T = TypeVar("T")

CHECKPOINT_MECHANISM_VERSION = "1"

UNIQUE_OBJECTS_CONTAINER = None
REGISTERING_OBJECTS = False
DEDUPLICATING_OBJECTS = False

CHECKPOINT_DEVICE_MAP: MAP_LOCATION = None


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


def _set_checkpoint_device_map(device_map: MAP_LOCATION):
    global CHECKPOINT_DEVICE_MAP
    CHECKPOINT_DEVICE_MAP = device_map


def _get_checkpoint_device_map():
    global CHECKPOINT_DEVICE_MAP
    return CHECKPOINT_DEVICE_MAP


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
        such as :func:`avalanche.checkpointing.constructor_based_serialization`.
        Standard use: dataset objects used to create the benchmark.
    """

    def __init__(
        self, map_location: MAP_LOCATION, unique_objects: Optional[Sequence[Any]]
    ):
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


def fixed_pytorch_1_13_load(
    f: FILE_LIKE,
    map_location: MAP_LOCATION = None,
    pickle_module: Any = None,
    *,
    weights_only: bool = False,
    **pickle_load_args: Any,
) -> Any:
    """
    A patched version of torch.load for PyTorch versions 1.13.0 and 1.13.1.
    """
    import pickle
    import torch._weights_only_unpickler as _weights_only_unpickler
    from torch.serialization import (
        _check_dill_version,
        _open_file_like,
        _open_zipfile_reader,
        _is_zipfile,
        _is_torchscript_zip,
        _legacy_load,
        _load,
    )

    UNSAFE_MESSAGE = (
        "Weights only load failed. Re-running `torch.load` with `weights_only` set to `False`"
        " will likely succeed, but it can result in arbitrary code execution."
        "Do it only if you get the file from a trusted source. WeightsUnpickler error: "
    )
    # Add ability to force safe only weight loads via environment variable
    if os.getenv("TORCH_FORCE_WEIGHTS_ONLY_LOAD", "0").lower() in [
        "1",
        "y",
        "yes",
        "true",
    ]:
        weights_only = True

    if weights_only:
        if pickle_module is not None:
            raise RuntimeError(
                "Can not safely load weights when expiclit picke_module is specified"
            )
    else:
        if pickle_module is None:
            pickle_module = pickle

    _check_dill_version(pickle_module)

    if "encoding" not in pickle_load_args.keys():
        pickle_load_args["encoding"] = "utf-8"

    with _open_file_like(f, "rb") as opened_file:
        if _is_zipfile(opened_file):
            # The zipfile reader is going to advance the current file position.
            # If we want to actually tail call to torch.jit.load, we need to
            # reset back to the original position.
            orig_position = opened_file.tell()
            with _open_zipfile_reader(opened_file) as opened_zipfile:
                if _is_torchscript_zip(opened_zipfile):
                    warnings.warn(
                        "'torch.load' received a zip file that looks like a TorchScript archive"
                        " dispatching to 'torch.jit.load' (call 'torch.jit.load' directly to"
                        " silence this warning)",
                        UserWarning,
                    )
                    opened_file.seek(orig_position)
                    return torch.jit.load(opened_file, map_location=map_location)
                if weights_only:
                    try:
                        return _load(
                            opened_zipfile,
                            map_location,
                            _weights_only_unpickler,
                            **pickle_load_args,
                        )
                    except RuntimeError as e:
                        raise pickle.UnpicklingError(UNSAFE_MESSAGE + str(e)) from None
                return _load(
                    opened_zipfile, map_location, pickle_module, **pickle_load_args
                )
        if weights_only:
            try:
                return _legacy_load(
                    opened_file,
                    map_location,
                    _weights_only_unpickler,
                    **pickle_load_args,
                )
            except RuntimeError as e:
                raise pickle.UnpicklingError(UNSAFE_MESSAGE + str(e)) from None
        return _legacy_load(
            opened_file, map_location, pickle_module, **pickle_load_args
        )


def _recreate_pytorch_device(*args):
    device_map = globals().get("CHECKPOINT_DEVICE_MAP", None)
    device_object = torch.device(*args)
    mapped_object = device_object

    if device_map is not None:
        mapped_object = torch.device(
            device_map.get(str(device_object), str(device_object))
        )
    print("Mapping", device_object, "to", mapped_object)
    return mapped_object


@dill.register(torch.device)
def _save_pytorch_device(pickler, obj: torch.device):
    has_index = obj.index is not None
    reduction: Union[Tuple[str, int], Tuple[str]]
    if has_index:
        reduction = (obj.type, obj.index)
    else:
        reduction = (obj.type,)

    pickler.save_reduce(_recreate_pytorch_device, reduction, obj=obj)
