from typing import Tuple, TYPE_CHECKING

import torch
from torch import Tensor
from torch.nn import Module

if TYPE_CHECKING:
    from avalanche.benchmarks import GenericCLScenario


def hash_benchmark(benchmark: 'GenericCLScenario') -> str:
    import hashlib
    import io

    hash_engine = hashlib.sha256()
    for stream_name, stream in benchmark.streams.items():
        hash_engine.update(stream_name.encode())
        for experience in stream:
            exp_dataset = experience.dataset
            dataset_content = exp_dataset[:]
            for tuple_elem in dataset_content:
                # https://stackoverflow.com/a/63880190
                buff = io.BytesIO()
                torch.save(tuple_elem, buff)
                buff.seek(0)
                hash_engine.update(buff.read())
    return hash_engine.hexdigest()


def hash_minibatch(minibatch: Tuple[Tensor]) -> str:
    import hashlib
    import io

    hash_engine = hashlib.sha256()
    for tuple_elem in minibatch:
        buff = io.BytesIO()
        torch.save(tuple_elem, buff)
        buff.seek(0)
        hash_engine.update(buff.read())
    return hash_engine.hexdigest()


def hash_tensor(tensor: Tensor) -> str:
    import hashlib
    import io

    hash_engine = hashlib.sha256()
    buff = io.BytesIO()
    torch.save(tensor, buff)
    buff.seek(0)
    hash_engine.update(buff.read())
    return hash_engine.hexdigest()


def hash_model(model: Module) -> str:
    import hashlib
    import io

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
    'hash_minibatch',
    'hash_tensor',
    'hash_model'
]
