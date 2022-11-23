import hashlib
import io

from typing import Tuple, TYPE_CHECKING

import torch
from torch import Tensor
from torch.nn import Module
from torch.utils.data import Dataset, DataLoader

if TYPE_CHECKING:
    from avalanche.benchmarks import GenericCLScenario


def hash_benchmark(benchmark: 'GenericCLScenario') -> str:
    hash_engine = hashlib.sha256()
    for stream_name, stream in benchmark.streams.items():
        hash_engine.update(stream_name.encode())
        for experience in stream:
            exp_dataset = experience.dataset
            hash_dataset(exp_dataset, hash_engine=hash_engine)
    return hash_engine.hexdigest()


def hash_dataset(dataset: 'Dataset', *, hash_engine=None, num_workers=0) -> str:
    from avalanche.distributed import DistributedHelper
    if hash_engine is None:
        hash_engine = hashlib.sha256()

    data_loader = DataLoader(
        dataset,
        collate_fn=lambda batch: tuple(zip(*batch)),
        num_workers=num_workers
    )
    for loaded_elem in data_loader:
        example = tuple(tuple_element[0] for tuple_element in loaded_elem)

        # https://stackoverflow.com/a/63880190
        buff = io.BytesIO()
        torch.save(example, buff)
        buff.seek(0)
        hash_engine.update(buff.read())
    return hash_engine.hexdigest()


def hash_minibatch(minibatch: Tuple[Tensor]) -> str:
    hash_engine = hashlib.sha256()
    for tuple_elem in minibatch:
        buff = io.BytesIO()
        torch.save(tuple_elem, buff)
        buff.seek(0)
        hash_engine.update(buff.read())
    return hash_engine.hexdigest()


def hash_tensor(tensor: Tensor) -> str:
    hash_engine = hashlib.sha256()
    buff = io.BytesIO()
    torch.save(tensor, buff)
    buff.seek(0)
    hash_engine.update(buff.read())
    return hash_engine.hexdigest()


def hash_model(model: Module) -> str:
    hash_engine = hashlib.sha256()
    for name, param in model.named_parameters():
        hash_engine.update(name.encode())
        buff = io.BytesIO()
        torch.save(param, buff)
        buff.seek(0)
        hash_engine.update(buff.read())
    return hash_engine.hexdigest()


__all__ = [
    'hash_benchmark',
    'hash_dataset',
    'hash_minibatch',
    'hash_tensor',
    'hash_model'
]
