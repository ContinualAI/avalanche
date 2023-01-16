import os
import pickle
import warnings
from io import BytesIO
from typing import Optional, List, Any, Iterable, Dict, TypeVar

import torch
from torch import Tensor
from torch.nn.modules import Module
from torch.nn.parallel import DistributedDataParallel
from typing_extensions import Literal
from torch.distributed import (
    init_process_group,
    broadcast_object_list
)


BroadcastT = TypeVar('BroadcastT')


from avalanche.distributed.distributed_consistency_verification import \
    hash_tensor


class _Singleton(type):
    _instances = {}

    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super(_Singleton, cls).__call__(
                *args, **kwargs)
        return cls._instances[cls]


class RollingSeedContext(object):
    """
    Implement seed alignment by storing random number generators state.

    Doesn't require a distributed communication (even broadcast), which makes
    this the best choices when wrapping sections that (may) both:
      - behave differently depending on the rank
      - change the global state of random number generators
    """
    def __init__(self):
        self.rng_manager_state = None

    def save_generators_state(self):
        from avalanche.training.determinism.rng_manager import RNGManager
        self.rng_manager_state = RNGManager.__getstate__()

    def load_generators_state(self):
        from avalanche.training.determinism.rng_manager import RNGManager
        self.rng_manager_state = RNGManager.__setstate__(self.rng_manager_state)

    def step_random_generators(self):
        from avalanche.training.determinism.rng_manager import RNGManager
        RNGManager.step_generators()

    def __enter__(self):
        self.save_generators_state()

    def __exit__(self, *_):
        self.load_generators_state()
        self.step_random_generators()


class BroadcastSeedContext(object):
    """
    Implement seed alignment by broadcasting a new seed from the main process.

    This is usually slower than using :class:`RollingSeedContext`.
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
            self._seed_aligner = RollingSeedContext()
        else:
            self._seed_aligner = BroadcastSeedContext()

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
        self.use_cuda = False
        self._dev_map = _DistributedHelperCls._make_map('cpu')

    def init_distributed(self, random_seed, backend=None, use_cuda=True):
        if self.is_distributed:
            raise RuntimeError('Distributed API already initialized')

        use_cuda = use_cuda and torch.cuda.is_available()

        if backend is None:
            if use_cuda:
                backend = 'nccl'
            else:
                backend = 'gloo'

        if backend == 'nccl' and not use_cuda:
            warnings.warn(
                'Bad configuration: using NCCL, but you set use_cuda=False!')

        could_initialize_distributed = False
        if os.environ.get('LOCAL_RANK', None) is None:
            warnings.warn(
                'Torch distributed could not be initialized '
                '(missing environment configuration)')
        else:
            init_process_group(backend=backend)
            could_initialize_distributed = True

        self.set_random_seeds(random_seed)
        self.use_cuda = use_cuda

        if use_cuda or backend == 'nccl':  # TODO: remove in final release
            # https://github.com/pytorch/pytorch/issues/6351
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

        # Force-init the default CUDA device (if any)
        reference_device = self.make_device(set_cuda_device=True)

        # Create map for device placement of unpickled tensors
        self._dev_map = _DistributedHelperCls._make_map(reference_device)

        return could_initialize_distributed

    def get_device_id(self):
        if self.is_distributed:
            device_id = self.rank
        else:
            device_id = 0

        if self.use_cuda:
            return device_id

        return -1

    def make_device(self, set_cuda_device=False):
        if self.is_distributed:
            device_id = self.rank
        else:
            device_id = 0

        if self.use_cuda and device_id >= 0:
            ref_device = torch.device(f'cuda:{device_id}')
            if set_cuda_device:
                torch.cuda.set_device(ref_device)
        else:
            ref_device = torch.device('cpu')
        return ref_device

    def wrap_model(self, model: Module) -> Module:
        # Note: find_unused_parameters is needed for multi task models.
        if self.is_distributed:
            if self.forced_cuda_comm or self.use_cuda:
                # forced_cuda_comm is True if using NCCL; use_cuda may be true
                # even when not using NCCL.
                # User already warned if using NCCL with use_cuda==False.
                # device_ids must be a single device id
                # (an int, a device object or a str)
                # If not set, output_device defaults to device_ids[0]
                return DistributedDataParallel(
                    model, device_ids=[self.make_device()], 
                    find_unused_parameters=True)
            else:
                return DistributedDataParallel(
                    model,
                    find_unused_parameters=True)
        else:
            return model

    def unwrap_model(self, model: Module) -> Module:
        if isinstance(model, DistributedDataParallel):
            return model.module

        return model

    def set_random_seeds(self, random_seed):
        from avalanche.training.determinism.rng_manager import RNGManager
        RNGManager.set_random_seeds(random_seed)

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
    
    def broadcast_object(self, obj: BroadcastT, src=0) -> BroadcastT:
        if not self.is_distributed:
            return obj

        io_list = [obj]

        broadcast_object_list(io_list, src=src)
        return io_list[0]

    def cat_all(self, tensor: Tensor):
        # TODO: use all_gather_into_tensor (if available and
        # if NCCL and tensor.device == 'default device')

        if not self.is_distributed:
            return tensor

        gathered_tensors = self.gather_all(tensor)
        for i, t in enumerate(gathered_tensors):
            if len(t.shape) == 0:
                # Tensor with 0-length shape
                gathered_tensors[i] = torch.reshape(t, (1,))

        return torch.cat(gathered_tensors)

    def gather_tensor_shapes(self, tensor: Tensor, max_shape_len=10) \
            -> List[List[int]]:
        """
        Gathers the shapes of all the tensors.
        """
        # Tensor differ by whole shape
        tensor_size = torch.zeros(max_shape_len, dtype=torch.int64)
        for i in range(len(tensor.shape)):
            tensor_size[i] = tensor.shape[i]
        all_tensors_shape = [
            self._prepare_for_distributed_comm(
                torch.zeros_like(tensor_size))[0]
            for _ in range(self.world_size)]
        tensor_size, _ = self._prepare_for_distributed_comm(tensor_size)

        torch.distributed.all_gather(all_tensors_shape, tensor_size)

        all_tensors_shape = [t.cpu() for t in all_tensors_shape]
        
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
        
        return [t_shape.tolist() for t_shape in all_tensors_shape]

    def gather_all(
            self,
            tensor: Tensor,
            same_shape: bool = False,
            shapes: Optional[List[List[int]]] = None):
        """
        Gather all for tensors only.
        
        Note: differently from the original Pytorch function, which requires
        that input tensor is to be moved to the default device (forced to 
        CUDA if using NCCL), this function also manages input tensors
        residing on a different devics. The resulting list of tensors will
        be moved to the same device of the input tensor.

        This will also manage tensors of different shapes. If you
        are sure that the tensors will be of the same shape, consider
        passing same_shape to speed up the communication.

        Beware that, if you are in need of concatenating multiple tensors,
        method `cat_all` may be more suitable.
        """
        if not self.is_distributed:
            return [tensor]

        # Based on:
        # https://discuss.pytorch.org/t/how-to-concatenate-different-size-tensors-from-distributed-processes/44819/4

        if same_shape:
            # Same size for all tensors
            if len(tensor.shape) > 0:
                tensor_size = list(tensor.shape)
            else:
                tensor_size = [0]
            all_tensors_shape = \
                [tensor_size for _ in range(self.world_size)]
        elif shapes is not None:
            # Shapes given by the user
            # make sure it is a list of lists
            all_tensors_shape = [list(s) for s in shapes]
        else:
            # Tensor differ by whole shape
            all_tensors_shape = self.gather_tensor_shapes(tensor)
        
        same_shape = all(all_tensors_shape[0] == x for x in all_tensors_shape)
        orig_device = tensor.device

        if same_shape:
            # Same shape: create identical tensors and proceed with all_gather
            out_tensors = [torch.empty_like(tensor) for _ in all_tensors_shape]
        else:
            # Different shapes: create a tensors of the size of the bigger one
            all_tensors_numel = []
            dtype = tensor.dtype
            for t_shape in all_tensors_shape:
                if t_shape[0] == 0 and len(t_shape) == 1:
                    # Tensor with 0-length shape
                    curr_size = 1
                else:
                    curr_size = 1
                    for t_s in t_shape:
                        curr_size *= t_s
                all_tensors_numel.append(curr_size)

            max_numel = max(all_tensors_numel)
            out_tensors = [torch.empty((max_numel,), dtype=dtype) 
                           for _ in all_tensors_shape]
            
            tensor = tensor.flatten()
            n_padding = max_numel - tensor.numel()
            if n_padding > 0:
                padding = torch.zeros((n_padding,), 
                                      dtype=tensor.dtype,
                                      device=orig_device)
                tensor = torch.cat((tensor, padding), dim=0)

        tensor, _ = self._prepare_for_distributed_comm(tensor)
        out_tensors = [self._prepare_for_distributed_comm(t)[0]
                       for t in out_tensors]
                        
        torch.distributed.all_gather(out_tensors, tensor)

        if not same_shape:
            # The tensors are flat and of the wrong dimension: re-shape them
            for tensor_idx, (tensor_sz, tensor_numel, out_t) in \
                    enumerate(zip(all_tensors_shape, 
                                  all_tensors_numel,
                                  out_tensors)):
                if tensor_sz[0] == 0:
                    # Tensor with 0-length shape
                    out_tensors[tensor_idx] = \
                        out_t[:tensor_numel].reshape(tuple())
                else:
                    out_tensors[tensor_idx] = \
                        out_t[:tensor_numel].reshape(tensor_sz)

        out_tensors = [t.to(orig_device) for t in out_tensors]
        return out_tensors

    def gather_all_objects(self, obj: BroadcastT) -> List[BroadcastT]:
        """
        Gather all for objects. This will also take care of moving cuda tensors
        (even the ones nested inside objects) to the correct default device.
        """
        out_list = [None for _ in range(self.world_size)]
        torch.distributed.all_gather_object(out_list, obj)
        return out_list

    def check_equal_tensors(self, tensor: Tensor):
        if not DistributedHelper.is_distributed:
            return

        all_tensors = self.gather_all(tensor)

        tensors_hashes = [hash_tensor(t) for t in all_tensors]

        if len(set(tensors_hashes)) != 1:
            # Equal tensors
            raise ValueError('Different tensors. Got hashes: {}'.format(
                tensors_hashes))

    def check_equal_objects(self, obj):
        if not DistributedHelper.is_distributed:
            return

        output: List[Any] = [None for _ in range(self.world_size)]
        torch.distributed.all_gather_object(output, obj)

        obj_bt = base_typed(obj)

        for i, o in enumerate(output):
            o_bt = base_typed(o)
            if obj_bt != o_bt:
                raise ValueError(
                    'Different objects (ranks this={}, remote={}). '
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

    @property
    def device_map(self) -> Dict[str, str]:
        return self._dev_map

    @staticmethod
    def _make_map(device_or_map) -> Dict[str, str]:
        # TODO: borrowed from checkpointing plugins
        # it would be better to have a single function in a shared utils
        if not isinstance(device_or_map, (torch.device, str)):
            return device_or_map

        device = torch.device(device_or_map)
        map_location = dict()

        map_location['cpu'] = 'cpu'
        for cuda_idx in range(100):
            map_location[f'cuda:{cuda_idx}'] = str(device)
        return map_location


BASE_TYPES = [str, int, float, bool, type(None)]


def base_typed(obj):
    """
    Improved version of https://stackoverflow.com/a/62420097
    """
    T = type(obj)
    from_numpy = T.__module__ == 'numpy'
    from_pytorch = T.__module__ == 'torch'

    if from_numpy or from_pytorch:
        return obj.tolist()

    if T in BASE_TYPES or callable(obj) or ((from_numpy or from_pytorch)
                                            and not isinstance(T, Iterable)):
        return obj

    if isinstance(obj, Dict):
        return {base_typed(k): base_typed(v) for k, v in obj.items()}
    elif isinstance(obj, Iterable):
        base_items = [base_typed(item) for item in obj]
        return base_items if (from_numpy or from_pytorch) else T(base_items)

    d = obj if T is dict else obj.__dict__

    return {k: base_typed(v) for k, v in d.items()}


DistributedHelper = _DistributedHelperCls()


def fix():
    return lambda b: torch.load(BytesIO(b),
                                map_location=DistributedHelper.device_map)


class MappedUnpickler(pickle.Unpickler):
    # Based on:
    # https://github.com/pytorch/pytorch/issues/16797#issuecomment-777059657

    # In turn based on:
    # https://github.com/pytorch/pytorch/issues/16797#issuecomment-633423219

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def find_class(self, module, name):
        if module == 'torch.storage' and name == '_load_from_bytes':
            return fix()
        else:
            return super().find_class(module, name)


torch.distributed.distributed_c10d._unpickler = MappedUnpickler


__all__ = [
    'RollingSeedContext',
    'BroadcastSeedContext',
    'DistributedHelper',
    '_DistributedHelperCls'
]
