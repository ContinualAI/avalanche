import random
from abc import ABC, abstractmethod
from functools import partial
from pathlib import Path
from typing import Union, Callable, IO, Any
import dill
import numpy as np
import torch

from avalanche.core import BaseSGDPlugin
from avalanche.training.determinism.rng_manager import RNGManager
from avalanche.training.templates import BaseSGDTemplate, BaseTemplate

from avalanche.training.plugins.checkpoint_common_recipes import *


class CheckpointStorage(ABC):
    def __init__(self):
        super(CheckpointStorage, self).__init__()

    @abstractmethod
    def store_checkpoint(
            self,
            checkpoint_name: str,
            checkpoint_writer: Callable[[IO[bytes]], None]):
        pass

    @abstractmethod
    def list_checkpoints(self):
        pass

    @abstractmethod
    def checkpoint_exists(self, checkpoint_name: str):
        pass

    @abstractmethod
    def load_checkpoint(
            self,
            checkpoint_name: str,
            checkpoint_loader: Callable[[IO[bytes]], Any]):
        pass


class CheckpointPlugin(BaseSGDPlugin[BaseSGDTemplate]):

    def __init__(
            self,
            storage: CheckpointStorage,
            map_location=None):
        super(CheckpointPlugin, self).__init__()
        self.map_location = map_location
        self.storage = storage

    def load_checkpoint_if_exists(self,
                                  update_checkpoint_plugin=True):
        existing_checkpoints = self.storage.list_checkpoints()
        if len(existing_checkpoints) == 0:
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
        ended_experience_counter = strategy.clock.train_exp_counter

        checkpoint_data = {
            'strategy': strategy,
            'rng_manager': RNGManager,
            'exp_counter': ended_experience_counter
        }

        self.storage.store_checkpoint(
            str(ended_experience_counter),
            partial(self.save_checkpoint, checkpoint_data))
        print('Checkpoint', ended_experience_counter, 'saved!')

    def save_checkpoint(self, checkpoint_data, fobj):
        # import dill.detect
        # dill.detect.trace(True)
        torch.save(checkpoint_data, fobj, pickle_module=dill)

    def load_checkpoint(self, fobj):
        try:
            _set_checkpoint_device_map(self.map_location)

            return torch.load(fobj, pickle_module=dill,
                              map_location=self.map_location)
        finally:
            _set_checkpoint_device_map(None)


class FileSystemCheckpointStorage(CheckpointStorage):
    def __init__(
            self,
            directory: Union[str, Path]):
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
        except:
            try:
                checkpoint_file.unlink()
            except OSError:
                pass
            raise

        return True

    def list_checkpoints(self):
        if not self.directory.exists():
            return []

        return [
            x.name
            for x in self.directory.iterdir()
            if self.checkpoint_exists(x.name)
        ]

    def checkpoint_exists(self, checkpoint_name: str):
        return (self.directory / checkpoint_name / 'checkpoint.pth').exists()

    def load_checkpoint(
            self,
            checkpoint_name: str,
            checkpoint_loader: Callable[[IO[bytes]], Any]):
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
