import os
import random
import warnings
from collections import OrderedDict
from typing import Optional, List, Tuple

import numpy as np
import torch
from torch import Tensor
from torch.distributed import init_process_group
from torch.nn.modules import Module
from torch.nn.parallel import DistributedDataParallel
from typing_extensions import Literal

from avalanche.benchmarks import GenericCLScenario


class _Singleton(type):
    _instances = {}

    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super(_Singleton, cls).__call__(
                *args, **kwargs)
        return cls._instances[cls]


class _RollingSeedContext(object):
    """
    Implement seed alignment by storing random number generators state.

    Doesn't require a distributed communication (even broadcast), which makes
    this the best choices when wrapping sections that (may) both:
      - behave differently depending on the rank
      - change the global state of random number generators
    """
    def __init__(self):
        self.generators_state = None

    def save_generators_state(self):
        self.generators_state = dict()
        for gen_name, gen_def in DistributedHelper.random_generators.items():
            self.generators_state[gen_name] = gen_def['save_state']()

    def load_generators_state(self):
        for gen_name, gen_def in DistributedHelper.random_generators.items():
            gen_def['load_state'](self.generators_state[gen_name])

    def step_random_generators(self):
        for gen_name, gen_def in DistributedHelper.random_generators.items():
            gen_def['step']()

    def __enter__(self):
        self.save_generators_state()

    def __exit__(self, *_):
        self.load_generators_state()
        self.step_random_generators()


class _BroadcastSeedContext(object):
    """
    Implement seed alignment by broadcasting a new seed from the main process.

    This is usually slower than using :class:`_RollingSeedContext`.
    """
    def __init__(self):
        pass

    def __enter__(self):
        pass

    def __exit__(self, *_):
        DistributedHelper.align_seeds()


class _MainProcessFirstContext(object):
    """
    A context in which the main process must enter and exit the section before
    other processes.

    For instance, can be used to wrap the dataset download procedure.
    """

    def __init__(
            self,
            seed_alignment: Literal["rolling", "broadcast"] = 'rolling',
            final_barrier: bool = False):
        if seed_alignment == 'rolling':
            self._seed_aligner = _RollingSeedContext()
        else:
            self._seed_aligner = _BroadcastSeedContext()

        self._final_barrier = final_barrier

    def __enter__(self):
        self._seed_aligner.__enter__()

        if not DistributedHelper.is_main_process:
            # Wait for the main process
            DistributedHelper.barrier()

    def __exit__(self, exc_type, exc_val, exc_tb):
        if DistributedHelper.is_main_process:
            # Let other process enter the section
            DistributedHelper.barrier()

        self._seed_aligner.__exit__()
        if self._final_barrier:
            DistributedHelper.barrier()


class _DistributedHelperCls(object):
    __metaclass__ = _Singleton

    def __init__(self):
        self.use_cuda = True

        self.random_generators = OrderedDict()

        self.register_random_generator('torch', {
            'seed': torch.random.manual_seed,
            'save_state': torch.random.get_rng_state,
            'load_state': torch.random.set_rng_state,
            'step': lambda: torch.rand(1)
        })

        self.register_random_generator('numpy', {
            'seed': np.random.seed,
            'save_state': np.random.get_state,
            'load_state': np.random.set_state,
            'step': lambda: np.random.rand(1)
        })

        self.register_random_generator('random', {
            'seed': random.seed,
            'save_state': random.getstate,
            'load_state': random.setstate,
            'step': random.random
        })

    def init_distributed(self, random_seed, backend=None, use_cuda=True):
        if self.is_distributed:
            raise RuntimeError('Distributed API already initialized')

        if backend is None:
            if use_cuda:
                backend = 'nccl'
            else:
                backend = 'gloo'

        if backend == 'nccl' and not use_cuda:
            warnings.warn(
                'Bad configuration: using NCCL, but you set use_cuda=False!')

        if os.environ.get('LOCAL_RANK', None) is None:
            warnings.warn(
                'Torch distributed could not be initialized '
                '(missing environment configuration)')
        else:
            init_process_group(backend=backend)

        self.set_random_seeds(random_seed)
        self.use_cuda = use_cuda

        if use_cuda or backend == 'nccl':
            # https://github.com/pytorch/pytorch/issues/6351
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
        return True

    def get_device_id(self):
        if self.is_distributed:
            device_id = self.rank
        else:
            device_id = 0

        if self.use_cuda and torch.cuda.is_available():
            return device_id

        return -1

    def make_device(self):
        if self.is_distributed:
            device_id = self.rank
        else:
            device_id = 0

        if self.use_cuda and torch.cuda.is_available() and device_id >= 0:
            torch.cuda.set_device(device_id)
            ref_device = torch.device(f'cuda:{device_id}')
        else:
            ref_device = torch.device('cpu')
        return ref_device

    def wrap_model(self, model: Module) -> Module:
        if self.is_distributed:
            if self.forced_cuda_comm or self.use_cuda:
                # forced_cuda_comm is True if using NCCL; use_cuda may be true
                # even when not using NCCL.
                # User already warned if using NCCL with use_cuda==False.
                # device_ids must be a single device id
                # (an int, a device object or a str)
                # If not set, output_device defaults to device_ids[0]
                return DistributedDataParallel(
                    model, device_ids=[self.make_device()])
            else:
                return DistributedDataParallel(model)
        else:
            return model

    def unwrap_model(self, model: Module) -> Module:
        if isinstance(model, DistributedDataParallel):
            return model.module

        return model

    def register_random_generator(self, name: str, rng_def: dict):
        if 'save_state' not in rng_def or \
                'load_state' not in rng_def or 'step' not in rng_def:
            raise ValueError('Invalid random number generator definition')

        self.random_generators[name] = rng_def

    def set_random_seeds(self, random_seed):
        for gen_name, gen_dict in self.random_generators.items():
            gen_dict['seed'](random_seed)

    def align_seeds(self):
        if not self.is_distributed:
            return

        if self.is_main_process:
            reference_seed = torch.randint(0, 2**32-1, (1,), dtype=torch.int64)
        else:
            reference_seed = torch.empty((1,), dtype=torch.int64)

        self.broadcast(reference_seed)
        seed = int(reference_seed)
        self.set_random_seeds(seed)

    def main_process_first(self):
        return _MainProcessFirstContext()

    def barrier(self):
        if self.is_distributed:
            torch.distributed.barrier()

    def broadcast(self, tensor: Tensor, src=0):
        if not self.is_distributed:
            return tensor

        tensor_distrib, orig_data = self._prepare_for_distributed_comm(tensor)
        torch.distributed.broadcast(tensor_distrib, src=src)
        tensor = self._revert_to_original_device(tensor_distrib, orig_data)

        return tensor

    def cat_all(self, tensor: Tensor):
        if not self.is_distributed:
            return tensor

        gathered_tensors = self.gather_all(
            tensor, different_shape0=True, different_shape1_n=False)
        for i, t in enumerate(gathered_tensors):
            if len(t.shape) == 0:
                # Tensor with 0-length shape
                gathered_tensors[i] = torch.reshape(t, (1,))

        return torch.cat(gathered_tensors)

    def gather_all(
            self,
            tensor: Tensor,
            out_tensors: Optional[List[Tensor]] = None,
            different_shape0: bool = None,
            different_shape1_n: bool = None):
        if not self.is_distributed:
            return [tensor]

        if different_shape0 is None or different_shape1_n is None:
            warnings.warn('different_shape0 and different_shape1_n not set. '
                          'This may lead to inefficiencies.')

        if different_shape0 is None:
            different_shape0 = True

        if different_shape1_n is None:
            different_shape1_n = True

        # Based on:
        # https://discuss.pytorch.org/t/how-to-concatenate-different-size-tensors-from-distributed-processes/44819/4

        if out_tensors is None:
            all_tensors_shape = None
            if different_shape1_n:
                # TODO: needs unit test (especially for 0-shaped tensors)
                # Tensor differ by whole shape (not very common case)
                tensor_size = torch.zeros(10, dtype=torch.int64)
                for i in range(len(tensor.shape)):
                    tensor_size[i] = tensor.shape[i]

            elif different_shape0:
                # Tensors differ by shape[0] (most common case)
                if len(tensor.shape) > 0:
                    # Usual case
                    tensor_size = torch.tensor([tensor.shape[0]],
                                               dtype=torch.int64)
                else:
                    # Some tensors, especially loss tensors, have 0-length shape
                    tensor_size = torch.tensor([0], dtype=torch.int64)
            else:
                # TODO: needs unit test (especially for 0-shaped tensors)
                # Same size for all tensors
                tensor_size = torch.tensor(tensor.shape, dtype=torch.int64)
                all_tensors_shape = \
                    [tensor_size for _ in range(self.world_size)]

            if all_tensors_shape is None:
                all_tensors_shape = [
                    self._prepare_for_distributed_comm(
                        torch.zeros_like(tensor_size))[0]
                    for _ in range(self.world_size)]
                tensor_size, _ = self._prepare_for_distributed_comm(tensor_size)

                torch.distributed.all_gather(all_tensors_shape, tensor_size)

                all_tensors_shape = [t.cpu() for t in all_tensors_shape]

                if different_shape1_n:
                    # TODO: needs unit test (especially for 0-shaped tensors)
                    # Trim shape
                    for i, t in enumerate(all_tensors_shape):
                        for x in range(len(t)):
                            if t[x] == 0:
                                if x == 0:
                                    # Tensor with 0-length shape
                                    all_tensors_shape[i] = t[:x+1]
                                else:
                                    all_tensors_shape[i] = t[:x]

                                break

                elif different_shape0:
                    if len(tensor.shape[1:]) == 0:
                        # To manage tensors with 0-length shape
                        pass
                    else:
                        all_tensors_shape = \
                            [torch.cat(
                                [t,
                                 torch.as_tensor(tensor.shape[1:],
                                                 dtype=torch.int64)])
                                for t in all_tensors_shape]

            all_tensors_shape = \
                [t_shape.tolist() for t_shape in all_tensors_shape]
            dtype = tensor.dtype

            out_tensors = []
            for t_shape in all_tensors_shape:
                if t_shape[0] == 0 and len(t_shape) == 1:
                    # Tensor with 0-length shape
                    out_tensors.append(torch.zeros(tuple(), dtype=dtype))
                else:
                    out_tensors.append(torch.zeros(*t_shape, dtype=dtype))

        orig_device = tensor.device
        tensor, _ = self._prepare_for_distributed_comm(tensor)
        out_tensors = [self._prepare_for_distributed_comm(t)[0]
                       for t in out_tensors]
        torch.distributed.all_gather(out_tensors, tensor)
        out_tensors = [t.to(orig_device) for t in out_tensors]
        return out_tensors

    def gather_all_objects(self, obj):
        out_list = [None for _ in range(self.world_size)]
        torch.distributed.all_gather_object(out_list, obj)
        return out_list

    def check_equal_tensors(self, tensor: Tensor):
        if not DistributedHelper.is_distributed:
            return

        all_tensors = self.gather_all(
            tensor,
            different_shape0=True,
            different_shape1_n=True)

        tensors_hashes = [hash_tensor(t) for t in all_tensors]

        if len(set(tensors_hashes)) != 1:
            # Equal tensors
            raise ValueError('Different tensors. Got hashes: {}'.format(
                tensors_hashes))

    def check_equal_objects(self, obj):
        if not DistributedHelper.is_distributed:
            return

        output = [None for _ in range(self.world_size)]
        torch.distributed.all_gather_object(output, obj)

        for i, o in enumerate(output):
            if obj != o:
                raise ValueError(
                    'Different object ranks this={}, remote={}. '
                    'Got this={}, remote={}'.format(
                        self.rank, i, obj, o))

    def _prepare_for_distributed_comm(self, tensor: Tensor):
        original_device = tensor.device
        copy_back = self.forced_cuda_comm and not tensor.is_cuda
        if self.forced_cuda_comm:
            tensor_distributed = tensor.cuda()
        else:
            tensor_distributed = tensor

        return tensor_distributed, (original_device, copy_back, tensor)

    def _revert_to_original_device(self, tensor_distributed, orig_data):
        original_device, copy_back, tensor = orig_data
        if copy_back:
            if tensor is None:
                tensor = tensor_distributed.to(original_device)
            else:
                tensor[:] = tensor_distributed

        return tensor

    @property
    def rank(self) -> int:
        if torch.distributed.is_initialized():
            return torch.distributed.get_rank()
        return 0

    @property
    def world_size(self) -> int:
        if torch.distributed.is_initialized():
            return torch.distributed.get_world_size()
        return 1

    @property
    def is_distributed(self) -> bool:
        return torch.distributed.is_initialized()

    @property
    def is_main_process(self) -> bool:
        return self.rank == 0

    @property
    def backend(self) -> str:
        return torch.distributed.get_backend()

    @property
    def forced_cuda_comm(self) -> bool:
        return self.backend == 'nccl'


def hash_benchmark(benchmark: GenericCLScenario) -> str:
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


DistributedHelper = _DistributedHelperCls()

__all__ = [
    'DistributedHelper',
    '_DistributedHelperCls'
]
