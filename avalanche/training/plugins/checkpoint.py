from abc import ABC, abstractmethod
from pathlib import Path
from typing import Union
import dill

from avalanche.core import BaseSGDPlugin
from avalanche.training.templates import BaseSGDTemplate


class CheckpointStorage(ABC):
    def __init__(self):
        super(CheckpointStorage, self).__init__()

    @abstractmethod
    def store_checkpoint(self, checkpoint_name: str, checkpoint):
        pass

    @abstractmethod
    def list_checkpoints(self):
        pass

    @abstractmethod
    def checkpoint_exists(self, checkpoint_name: str):
        pass

    @abstractmethod
    def load_checkpoint(self, checkpoint_name: str):
        pass


class CheckpointPlugin(BaseSGDPlugin[BaseSGDTemplate]):

    def __init__(
            self,
            storage: CheckpointStorage):
        super(CheckpointPlugin, self).__init__()
        self.storage = storage

    def load_checkpoint_if_exists(self):
        existing_checkpoints = self.storage.list_checkpoints()
        if len(existing_checkpoints) == 0:
            return None, 0

        last_exp = max(
            [int(checkpoint_name) for checkpoint_name in existing_checkpoints]
        )

        loaded_checkpoint = self.storage.load_checkpoint(str(last_exp))

        return loaded_checkpoint, last_exp

    def after_training(self, strategy: BaseSGDTemplate, *args, **kwargs):
        """Called after `train` by the `BaseTemplate`."""
        ended_experience_counter = strategy.clock.train_exp_counter

        self.storage.store_checkpoint(str(ended_experience_counter), strategy)
        print('Checkpoint', ended_experience_counter, 'saved!')


class FileSystemCheckpointStorage(CheckpointStorage):
    def __init__(
            self,
            directory: Union[str, Path],
            device=None):
        super(FileSystemCheckpointStorage, self).__init__()
        self.directory = Path(directory)
        self.device = device

    def store_checkpoint(self, checkpoint_name: str, checkpoint):
        checkpoint_file = self._make_checkpoint_file_path(checkpoint_name)
        if checkpoint_file.exists():
            raise RuntimeError(
                f'Checkpoint file {str(checkpoint_file)} already exists.')

        checkpoint_file.parent.mkdir(exist_ok=True, parents=True)

        try:
            # import dill.detect
            # dill.detect.trace(True)
            with open(checkpoint_file, 'wb') as f:
                dill.dump(checkpoint, f)
        except:
            try:
                checkpoint_file.unlink()
            except:
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

    def load_checkpoint(self, checkpoint_name: str):
        checkpoint_file = self._make_checkpoint_file_path(checkpoint_name)

        with open(checkpoint_file, 'rb') as f:
            loaded_checkpoint = dill.load(f)

        return loaded_checkpoint

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
    'CheckpointPlugin'
]
