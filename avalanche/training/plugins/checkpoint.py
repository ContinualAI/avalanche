from abc import ABC, abstractmethod
from typing import Union, Callable, IO, Any, Dict, Optional, Iterable, BinaryIO

import torch

from avalanche._annotations import deprecated
from avalanche.core import BaseSGDPlugin
from avalanche.training.templates import BaseSGDTemplate


@deprecated(0.5, "Please use `save_checkpoint` and `maybe_load_checkpoint` " "instead.")
class CheckpointStorage(ABC):
    """
    Abstract class for the checkpoint storage component.

    A checkpoint storage implementations must provide mechanisms to store, list,
    and load checkpoints from a persistent storage. Instances of this class are
    used by the :class:`CheckpointPlugin` strategy plugin.
    """

    def __init__(self):
        """
        Initializes the checkpoint storage.
        """
        super(CheckpointStorage, self).__init__()

    @abstractmethod
    def store_checkpoint(
        self,
        checkpoint_name: str,
        checkpoint_writer: Callable[[Union[BinaryIO, IO[bytes]]], None],
    ) -> None:
        """
        Stores a checkpoint.

        This method expects a checkpoint name and a callable.

        The callable must accept a file-like object as input. The file-like
        object is created by the checkpoint storage (this object) and it will
        accept binary write operations to store the byte representation of the
        checkpoint.

        :param checkpoint_name: The name of the checkpoint.
        :param checkpoint_writer: A callable that accepts a writable file-like
            object. The callable must write the checkpoint to the provided file
            object.
        :return: None. It will raise an exception if the checkpoint cannot be
            loaded, depending on the specific implementation.
        """
        pass

    @abstractmethod
    def list_checkpoints(self) -> Iterable[str]:
        """
        Retrieves a list of available checkpoints

        :return: The names of available checkpoints.
        """
        pass

    @abstractmethod
    def checkpoint_exists(self, checkpoint_name: str) -> bool:
        """
        Checks if a checkpoint exists

        :param checkpoint_name: The name of the checkpoint to check.
        :return: True if it exists.
        """
        pass

    @abstractmethod
    def load_checkpoint(
        self,
        checkpoint_name: str,
        checkpoint_loader: Callable[[Union[BinaryIO, IO[bytes]]], Any],
    ) -> Any:
        """
        Loads a checkpoint.

        The `checkpoint_loader` parameter works similarly to the
        `checkpoint_writer` parameter of :meth:`store_checkpoint`,
        with the difference that it should read the checkpoint.

        :param checkpoint_name: The name of the checkpoint to load.
        :param checkpoint_loader: A callable that accepts a readable file-like
            object. The callable must read the checkpoint from the provided file
            object and return it.
        :return: The loaded checkpoint, as returned from `checkpoint_loader. It
            will raise an exception if the checkpoint cannot be loaded,
            depending on the specific implementation.
        """
        pass


@deprecated(0.5, "Please use `save_checkpoint` and `maybe_load_checkpoint` " "instead.")
class CheckpointPlugin(BaseSGDPlugin[BaseSGDTemplate]):
    """
    A checkpointing facility that can be used to persist the entire state of the
    strategy (including its model, plugins, metrics, loggers, etcetera).

    This strategy plugin will store a checkpoint after each evaluation phase.

    Using this plugin required minor changes to the usual Avalanche main script.
    Please refer to the `task_incremental_with_checkpointing.py` example for a
    simple guide on how to use this plugin.

    The checkpoint is stored using the :class:`CheckpointStorage` object
    provided in the input. The :class:`FileSystemCheckpointStorage` is the
    simpler option, but more sophisticated storage mechanisms can be used
    (based on W&B, S3, etcetera).

    This implementation sends a pickled version of the strategy object to the
    checkpoint storage. The strategy object is pickled using `dill`, which
    is a powerful drop-in replacement of pickle supported by PyTorch.

    This class also adds the ability to load a checkpoint by using a different
    target device than the one used when producing the checkpoint. In other
    words, this plugin adds the ability to load a cuda checkpoint to CPU,
    or to load a checkpoint created on cuda:1 to cuda:0. See the `map_location`
    parameter of the constructor for more details.

    Critical objects such as loggers are pickled as well, but the whole process
    is managed carefully to avoid errors. Datasets are pickled as well, but
    for datasets like torchvision ones, that load the whole content of the
    dataset in-memory, a custom pickling method is provided by Avalanche out
    of the box, so there is no need to worry about the checkpoint size. The
    replay plugin already uses the subset operation to manage the replay,
    which means that no dataset elements are actually saved to checkpoints.

    However, this will also mean that the main script must correctly re-create
    the benchmark object when loading a checkpoint. This is usually not a big
    deal as random number generators state are saved and loaded as well.
    """

    def __init__(
        self,
        storage: CheckpointStorage,
        map_location: Optional[Union[str, torch.device, Dict[str, str]]] = None,
    ):
        """
        Creates an instance of the checkpoint plugin.

        :param storage: The checkpoint storage to use. The most common one is
            :class:`FileSystemCheckpointStorage`.
        :param map_location: This parameter can change where the cuda tensors
            (including the model ones) are put when loading checkpoints.
            This works similar to the `map_location` parameter of `torch.load`,
            except that you can also pass a device object or a string (a proper
            map will be created accordingly). The recommended way to use this
            parameter is to pass the used reference device.
            In addition, all `torch.device` objects will be un-pickled using
            that map (this is not usually done by `torch.load`,
            but it is needed to properly manage things in Avalanche).
            Defaults to None, which means that no mapping will take place.
        """
        raise ValueError(
            "Please use `save_checkpoint` and " "`maybe_load_checkpoint` " "instead."
        )


__all__ = [
    "CheckpointStorage",
    "CheckpointPlugin",
]
