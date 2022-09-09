from abc import ABC, abstractmethod
from functools import partial
from pathlib import Path
from typing import Union, Callable, IO, Any, Dict, Optional, Iterable, \
    List, BinaryIO

import dill
import torch

from avalanche.core import BaseSGDPlugin
from avalanche.training.determinism.rng_manager import RNGManager
from avalanche.training.plugins.checkpoint_common_recipes import *
from avalanche.training.templates import BaseSGDTemplate, BaseTemplate


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
            checkpoint_writer: Callable[[Union[BinaryIO, IO[bytes]]], None]) \
            -> None:
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
            checkpoint_loader: Callable[[Union[BinaryIO, IO[bytes]]], Any]) \
            -> Any:
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
            map_location: Optional[Union[str,
                                         torch.device,
                                         Dict[str, str]]] = None):
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
        super(CheckpointPlugin, self).__init__()
        self.map_location = CheckpointPlugin._make_map(map_location)
        self.storage = storage
        self._training = False

    def load_checkpoint_if_exists(
            self, update_checkpoint_plugin=True):
        """
        Loads the latest checkpoint if it exists.

        This will load the strategy (including the model weights, all the
        plugins, metrics, and loggers), load and set the state of the
        global random number generators (torch, torch cuda, numpy, and Python's
        random), and the number of training experiences so far.

        The loaded checkpoint refers to the last successful evaluation.

        :param update_checkpoint_plugin: Defaults to True, which means that the
            CheckpointPlugin in the un-pickled strategy will be replaced with
            self (this plugin instance).
        :return: The loaded strategy and the number experiences so far (this
            number can also be interpreted as the index of the next training
            experience).
        """
        existing_checkpoints = list(self.storage.list_checkpoints())
        if len(existing_checkpoints) == 0:
            # No checkpoints exist
            return None, 0

        last_exp = max(
            [int(checkpoint_name) for checkpoint_name in existing_checkpoints]
        )

        loaded_checkpoint = self.storage.load_checkpoint(
            str(last_exp), self.load_checkpoint)

        strategy: BaseTemplate = loaded_checkpoint['strategy']
        exp_counter = loaded_checkpoint['exp_counter']

        if update_checkpoint_plugin:
            # Replace the previous CheckpointPlugin with "self" in strategy.
            # Useful if the save/load path (or even the storage object class)
            # has changed.
            checkpoint_plugin_indices = set(
                idx for idx, plugin in enumerate(strategy.plugins)
                if isinstance(plugin, CheckpointPlugin)
            )

            if len(checkpoint_plugin_indices) > 1:
                raise RuntimeError(
                    'Cannot update the strategy CheckpointPlugin: more than '
                    'one found.')
            elif len(checkpoint_plugin_indices) == 1:
                to_replace_idx = list(checkpoint_plugin_indices)[0]
                strategy.plugins[to_replace_idx] = self

        return strategy, exp_counter

    def after_eval(self, strategy: BaseSGDTemplate, *args, **kwargs):
        if self._training:
            # Do not checkpoint on periodic evaluation
            return

        ended_experience_counter = strategy.clock.train_exp_counter

        checkpoint_data = {
            'strategy': strategy,
            'rng_manager': RNGManager,
            'exp_counter': ended_experience_counter
        }

        self.storage.store_checkpoint(
            str(ended_experience_counter),
            partial(CheckpointPlugin.save_checkpoint,
                    checkpoint_data))
        print('Checkpoint', ended_experience_counter, 'saved!')

    def before_training(self, strategy, *args, **kwargs):
        # Used to track periodic evaluation
        self._training = True

    def after_training(self, strategy, *args, **kwargs):
        # Used to track periodic evaluation
        self._training = False

    @staticmethod
    def save_checkpoint(checkpoint_data, fobj: Union[BinaryIO, IO[bytes]]):
        # import dill.detect
        # dill.detect.trace(True)
        torch.save(checkpoint_data, fobj, pickle_module=dill)

    def load_checkpoint(self, fobj: Union[BinaryIO, IO[bytes]]):
        """
        Loads the checkpoint given the file-like object coming from the storage.

        This function is mostly an internal mechanism. Do not use it if you are
        not 100% sure of what it does (and does not).

        :param fobj: A file-like object, usually provided by the
            :class:`CheckpointStorage` object.
        :return: The loaded checkpoint.
        """
        try:
            _set_checkpoint_device_map(self.map_location)

            return torch.load(fobj, pickle_module=dill,
                              map_location=self.map_location)
        finally:
            _set_checkpoint_device_map(None)

    @staticmethod
    def _make_map(device_or_map) -> Optional[Dict[str, str]]:
        if not isinstance(device_or_map, (torch.device, str)):
            return device_or_map

        device = torch.device(device_or_map)
        map_location = dict()

        map_location['cpu'] = 'cpu'
        for cuda_idx in range(100):
            map_location[f'cuda:{cuda_idx}'] = str(device)
        return map_location


class FileSystemCheckpointStorage(CheckpointStorage):
    """
    A checkpoint storage that stores the checkpoint of an experiments in the
    given directory.
    """
    def __init__(
            self,
            directory: Union[str, Path]):
        """
        Creates an instance of the filesystem checkpoint storage.

        :param directory: The directory in which to save the checkpoints.
            This should be an experiment/run specific directory. Do not use
            the same directory for more than one experiment.
        """
        super(FileSystemCheckpointStorage, self).__init__()
        self.directory = Path(directory)

    def store_checkpoint(
            self,
            checkpoint_name: str,
            checkpoint_writer: Callable[[IO[bytes]], None]):
        checkpoint_file = self._make_checkpoint_file_path(checkpoint_name)
        if checkpoint_file.exists():
            raise RuntimeError(
                f'Checkpoint file {str(checkpoint_file)} already exists.')

        checkpoint_file.parent.mkdir(exist_ok=True, parents=True)

        try:
            with open(checkpoint_file, 'wb') as f:
                checkpoint_writer(f)
        except BaseException:
            try:
                checkpoint_file.unlink()
            except OSError:
                pass
            raise

    def list_checkpoints(self) -> List[str]:
        if not self.directory.exists():
            return []

        return [
            x.name
            for x in self.directory.iterdir()
            if self.checkpoint_exists(x.name)
        ]

    def checkpoint_exists(self, checkpoint_name: str) -> bool:
        return (self.directory / checkpoint_name / 'checkpoint.pth').exists()

    def load_checkpoint(
            self,
            checkpoint_name: str,
            checkpoint_loader: Callable[[IO[bytes]], Any]) -> Any:
        checkpoint_file = self._make_checkpoint_file_path(checkpoint_name)

        with open(checkpoint_file, 'rb') as f:
            return checkpoint_loader(f)

    def _make_checkpoint_dir(self, checkpoint_name: str) -> Path:
        return self.directory / checkpoint_name

    def _make_checkpoint_file_path(self, checkpoint_name: str) -> Path:
        return self._make_checkpoint_dir(checkpoint_name) / 'checkpoint.pth'

    def __str__(self):
        available_checkpoints = ','.join(self.list_checkpoints())
        return f'FileSystemCheckpointStorage[\n' \
               f'   path={str(self.directory)}\n' \
               f'   checkpoints={available_checkpoints}\n' \
               f']'


__all__ = [
    'CheckpointStorage',
    'CheckpointPlugin',
    'FileSystemCheckpointStorage'
]
