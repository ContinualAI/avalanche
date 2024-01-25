import os.path
from pathlib import Path
from copy import copy
from typing import (
    Any,
    Callable,
    Dict,
    Optional,
    Tuple,
    Type,
    TypeVar,
    BinaryIO,
    IO,
    Union,
    Collection,
)
from typing_extensions import TypeAlias

import dill
import torch
from functools import partial
from packaging.version import parse
from .checkpoint_internals import (
    CHECKPOINT_MECHANISM_VERSION,
    _CheckpointLoadingContext,
    _constructor_based_unpickle,
    _is_registering_objects,
    _register_unique_object,
    fixed_pytorch_1_13_load,
)

FILE_LIKE: TypeAlias = Union[str, os.PathLike, BinaryIO, IO[bytes]]
MAP_LOCATION: TypeAlias = Optional[
    Union[
        Callable[[torch.Tensor, str], torch.Tensor], torch.device, str, Dict[str, str]
    ]
]

T = TypeVar("T")


def constructor_based_serialization(
    pickler, obj: T, cls: Type[T], deduplicate: bool = False, args=None, kwargs=None
):
    """
    This utility is used manage the pickling of an object by only storing its constructor parameters.

    This will also register the function that will be used to unpickle the object.

    Classes whose objects can be serialized by only providing the constructor parameters
    can be registered using this utility.

    The standard way to register a class is to put the following function in the same script
    where the class is defined (in this example, the class is `CIFAR100`):

    ```python
    @dill.register(CIFAR100)
    def checkpoint_CIFAR100(pickler, obj: CIFAR100):
        constructor_based_serialization(
            pickler,
            obj,
            CIFAR100,
            deduplicate=True,  # check `constructor_based_serialization` for details on de-duplication
            kwargs=dict(
                root=obj.root,
                train=obj.train,
                transform=obj.transform,
                target_transform=obj.target_transform,
            )
        )
    ```

    Consider that alternative mechanisms exists, such as implementing custom `__getstate__` and `__setstate__`
    methods or by manually providing a custom `@dill.register` function.
    For the last option, see:
    https://stackoverflow.com/questions/27351980/how-to-add-a-custom-type-to-dills-pickleable-types

    This mechanism also supports de-duplicating unique objects, such as datasets.
    This is useful to avoid duplicating the memory usage when
    loading a checkpoint with a large number of datasets
    (or an experiment that was already checkpointed are re-loaded multiple times).

    If `deduplicate` is True, then the object is marked as elegible for de-duplication.
    It will be de-duplicated (not loaded from the checkpoint) if the user provides,
    in the `unique_objects` parameter of `maybe_load_checkpoint`, an object with identical
    constructor parameters. Deduplication should be activated for dataset objects.
    """

    unpickler = partial(_constructor_based_unpickle, cls)

    pickler.save_reduce(
        unpickler,
        (
            CHECKPOINT_MECHANISM_VERSION,
            deduplicate,
            args,
            kwargs,
        ),
        obj=obj,
    )

    if deduplicate and _is_registering_objects():
        if args is None:
            args = []
        if kwargs is None:
            kwargs = {}

        _register_unique_object(obj, cls, args, kwargs)


def maybe_load_checkpoint(
    strategy: T,
    fname: FILE_LIKE,
    map_location: MAP_LOCATION = None,
    unique_objects: Any = None,
) -> Tuple[T, int]:
    """Load the strategy state from a checkpoint file.

    The method returns the strategy with the state deserialized from the file
    and the index of the training experience to resume training.

    If the file does not exists, the method returns the strategy unmodified
    and the index 0. As a result, the method can be safely called even if no
    checkpoint has been previously created (e.g. during the first run).

    Example:

    ```
    strategy = Naive(model, opt, train_mb_size=128)
    strategy, initial_exp = maybe_load_checkpoint(strategy, fname)

    for exp in benchmark.train_stream[initial_exp:]:
        strategy.train(exp)
        save_checkpoint(strat, fname)
    ```

    This also supports de-duplicating unique objects, such as datasets.
    This is useful to avoid duplicating the memory usage when
    loading a checkpoint with a large number of datasets
    (or an experiment that was already checkpointed are re-loaded multiple times).

    Consider passing the benchmark object as `unique_objects` to avoid
    duplicating the memory associated with dataset(s).

    In practice, de-duplication works by taking `unique_objects` and listing objects of
    classes serialized by using :func:`constructor_based_serialization` and comparing their constructor
    arguments. If an object found in `unique_objects` is already present in the
    checkpoint, it is re-used instead of being re-loaded from the checkpoint.
    This prevents the checkpoint size from exploding when checkpointing and re-loading
    frequently (such as when running on a SLURM cluster that frequently preempts you job).

    :param strategy: strategy to load. It must be already initialized.
    :param fname: file name
    :param map_location: sets the location of the tensors after serialization.
        Same as `map_location` of `torch.load`, except that you can also pass
        a device object or a string (a proper
        map will be created accordingly). The recommended way to use this
        parameter is to pass the used reference device.
        In addition, all `torch.device` objects will be un-pickled using
        that map (this is not usually done by `torch.load`,
        but it is needed to properly manage things in Avalanche).
        Defaults to None, which means that no mapping will take place.
    :param unique_objects: list of (or a single) unique object(s) that do not need to be unpickled.
        This is useful to avoid duplicating the memory associated with a dataset
        (or an experiment that was already checkpointed are re-loaded multiple times).
        Classes of objects that need de-duplication must be registered as such using helpers
        such as :func:`constructor_based_serialization`. Defaults to None.
        Recommended: at least pass the benchmark object.
    :return: tuple <strategy, exp_counter>
        strategy after deserialization,
        index of the current experience to resume training.
    """
    if isinstance(fname, (str, Path)) and not os.path.exists(fname):
        return strategy, 0

    if isinstance(map_location, (str, torch.device)):
        device = torch.device(map_location)
        map_location = dict()

        map_location["cpu"] = "cpu"
        for cuda_idx in range(100):
            map_location[f"cuda:{cuda_idx}"] = str(device)

    with _CheckpointLoadingContext(map_location, unique_objects):
        # The load function in PyTorch 1.13.* is broken, so we use
        # fixed_pytorch_1_13_load instead.
        pytorch_version = parse(torch.__version__)
        if pytorch_version >= parse("1.13.0") and pytorch_version < parse("1.14.0"):
            ckp = fixed_pytorch_1_13_load(
                fname, pickle_module=dill, map_location=map_location
            )
        else:
            ckp = torch.load(fname, pickle_module=dill, map_location=map_location)

    strategy.__dict__.update(ckp["strategy"].__dict__)
    exp_counter = ckp["exp_counter"]
    return strategy, exp_counter


def save_checkpoint(
    strategy, fname: FILE_LIKE, exclude: Optional[Collection[str]] = None
):
    """Save the strategy state into a file.

    The file can be loaded using `maybe_load_checkpoint`.

    For efficiency, the user can specify some attributes to `exclude`.
    For example, if the optimizer is static and doesn't change during training,
    it can be safely excluded. These helps to speed up the serialization.

    WARNING: the method cannot be used inside the training and evaluation
    loops of the strategy.

    :param strategy: strategy to serialize.
    :param fname: name of the file.
    :param exclude: List[string] list of attributes to remove before the
        serialization.
    """
    from avalanche.training.determinism.rng_manager import RNGManager

    if exclude is None:
        exclude = []
    ended_experience_counter = strategy.clock.train_exp_counter

    strategy = copy(strategy)
    for attr in exclude:
        delattr(strategy, attr)

    checkpoint_data = {
        "strategy": strategy,
        "rng_manager": RNGManager,
        "exp_counter": ended_experience_counter,
    }
    torch.save(checkpoint_data, fname, pickle_module=dill)


__all__ = [
    "constructor_based_serialization",
    "maybe_load_checkpoint",
    "save_checkpoint",
]
