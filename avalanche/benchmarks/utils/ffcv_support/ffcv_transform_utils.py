    

from typing import Any, Callable, Dict, List, NamedTuple, Optional, Tuple, Type, Union
from typing_extensions import Literal
import warnings
import numpy as np

import torch

from avalanche.benchmarks.utils.transforms import flat_transforms_recursive

from torchvision.transforms import ToTensor as ToTensorTV
from torchvision.transforms import PILToTensor as PILToTensorTV
from torchvision.transforms import Normalize as NormalizeTV
from torchvision.transforms import ConvertImageDtype as ConvertTV
from torchvision.transforms import RandomResizedCrop as RandomResizedCropTV
from torchvision.transforms import RandomHorizontalFlip as RandomHorizontalFlipTV
from torchvision.transforms import RandomCrop as RandomCropTV
from torchvision.transforms import Lambda

from ffcv.transforms import ToTensor as ToTensorFFCV
from ffcv.transforms import ToDevice as ToDeviceFFCV
from ffcv.transforms import ToTorchImage as ToTorchImageFFCV
from ffcv.transforms import NormalizeImage as NormalizeFFCV
from ffcv.transforms import Convert as ConvertFFCV
from ffcv.transforms import View as ViewFFCV
from ffcv.transforms import Squeeze as SqueezeFFCV
from ffcv.transforms import RandomResizedCrop as RandomResizedCropFFCV
from ffcv.transforms import RandomHorizontalFlip as RandomHorizontalFlipFFCV
from ffcv.transforms import RandomTranslate as RandomTranslateFFCV
from ffcv.transforms import Cutout as CutoutFFCV
from ffcv.transforms import ImageMixup as ImageMixupFFCV
from ffcv.transforms import LabelMixup as LabelMixupFFCV
from ffcv.transforms import MixupToOneHot as MixupToOneHotFFCV
from ffcv.transforms import Poison as PoisonFFCV
from ffcv.transforms import ReplaceLabel as ReplaceLabelFFCV
from ffcv.transforms import RandomBrightness as RandomBrightnessFFCV
from ffcv.transforms import RandomContrast as RandomContrastFFCV
from ffcv.transforms import RandomSaturation as RandomSaturationFFCV
from ffcv.transforms import ModuleWrapper
from ffcv.pipeline.operation import Operation
from ffcv.pipeline.state import State
from ffcv.pipeline.allocation_query import AllocationQuery

from ffcv.fields.decoders import (
    SimpleRGBImageDecoder,
    RandomResizedCropRGBImageDecoder
)
from dataclasses import replace


class CallableAdapter:
    def __init__(self, callable_obj):
        self.callable_obj = callable_obj

    def __call__(self, batch):
        result = []
        for element in batch:
            result.append(
                self.callable_obj(element)
            )

        if isinstance(batch, np.ndarray):
            return np.array(result)
        elif isinstance(batch, torch.Tensor):
            return torch.asarray(result)
        else:
            return result
        

class ScaleFrom255To1(torch.nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, input):
        default_float_dtype = torch.get_default_dtype()

        return input.to(dtype=default_float_dtype).div(255)


class FFCVTransformRegistry(NamedTuple):
    numpy_cpu: bool
    pytorch_cpu: bool
    pytorch_gpu: bool
    

FFCV_TRANSFORMS_DEFS: Dict[Type, FFCVTransformRegistry] = {}


def make_transform_defs():
    """
    Fills a series of definition obtained by the FFCV documentation
    and source code.
    """
    global FFCV_TRANSFORMS_DEFS


    FFCV_TRANSFORMS_DEFS[ToDeviceFFCV] = FFCVTransformRegistry(
        numpy_cpu=False,
        pytorch_cpu=True,
        pytorch_gpu=True  # GPU -> CPU, probably unused
    )

    FFCV_TRANSFORMS_DEFS[ToTorchImageFFCV] = FFCVTransformRegistry(
        numpy_cpu=False,
        pytorch_cpu=True,
        pytorch_gpu=False
    )

    FFCV_TRANSFORMS_DEFS[NormalizeFFCV] = FFCVTransformRegistry(
        numpy_cpu=True,
        pytorch_cpu=False,
        pytorch_gpu=True
    )

    # TODO: test
    # FFCV_TRANSFORMS_DEFS[ConvertFFCV] = FFCVTransformRegistry(
    #     numpy_cpu=True,
    #     pytorch_cpu=False,
    #     pytorch_gpu=True
    # )

    FFCV_TRANSFORMS_DEFS[SqueezeFFCV] = FFCVTransformRegistry(
        numpy_cpu=False,
        pytorch_cpu=True,
        pytorch_gpu=True  # TODO: test
    )

    # TODO: test
    # FFCV_TRANSFORMS_DEFS[ViewFFCV] = FFCVTransformRegistry(
    #     numpy_cpu=False,
    #     pytorch_cpu=True,
    #     pytorch_gpu=True
    # )

    FFCV_TRANSFORMS_DEFS[MixupToOneHotFFCV] = FFCVTransformRegistry(
        numpy_cpu=False,
        pytorch_cpu=True,
        pytorch_gpu=True
    )

    FFCV_TRANSFORMS_DEFS[ModuleWrapper] = FFCVTransformRegistry(
        numpy_cpu=False,
        pytorch_cpu=True,
        pytorch_gpu=True
    )

    FFCV_TRANSFORMS_DEFS[SmartModuleWrapper] = FFCVTransformRegistry(
        numpy_cpu=True,
        pytorch_cpu=True,
        pytorch_gpu=True
    )

    numpy_only_types = [
        ToTensorFFCV,
        RandomResizedCropFFCV,
        RandomHorizontalFlipFFCV,
        RandomTranslateFFCV,
        CutoutFFCV,
        ImageMixupFFCV,
        LabelMixupFFCV,
        PoisonFFCV,
        ReplaceLabelFFCV,
        RandomBrightnessFFCV,
        RandomContrastFFCV,
        RandomSaturationFFCV
    ]

    for t_type in numpy_only_types:
        FFCV_TRANSFORMS_DEFS[t_type] = \
            FFCVTransformRegistry(
                numpy_cpu=True,
                pytorch_cpu=False,
                pytorch_gpu=False
            )


def adapt_transforms(
        transforms_list,
        ffcv_decoder_list,
        device: Optional[torch.device] = None
    ):

    result = []
    for field_idx, pipeline_head in enumerate(ffcv_decoder_list):
        transforms = flat_transforms_recursive(transforms_list, field_idx)
        transforms = pipeline_head + transforms
        transforms = apply_pre_optimization(transforms, device=device)

        field_transforms: List[Operation] = []
        for t in transforms:
            if isinstance(t, Operation):
                # Already an FFCV transform
                field_transforms.append(t)
            elif isinstance(t, PILToTensorTV):
                field_transforms.append(ToTensorFFCV())
                field_transforms.append(ToTorchImageFFCV())
            elif isinstance(t, ToTensorTV):
                field_transforms.append(ToTensorFFCV())
                field_transforms.append(ToTorchImageFFCV())
                field_transforms.append(ModuleWrapper(ScaleFrom255To1()))
            elif isinstance(t, ConvertTV):
                field_transforms.append(
                    ConvertFFCV(t.dtype)
                )
            elif isinstance(t, RandomResizedCropTV):
                field_transforms.append(
                    RandomResizedCropFFCV(t.scale, t.ratio, t.size)
                )
            elif isinstance(t, RandomHorizontalFlipTV):
                field_transforms.append(
                    RandomHorizontalFlipFFCV(t.p)
                )
            elif isinstance(t, RandomCropTV):
                field_transforms.append(
                    SmartModuleWrapper(
                        t,
                        expected_out_type='as_previous',
                        expected_shape=t.size
                    )
                )
            elif isinstance(t, torch.nn.Module):
                field_transforms.append(
                    SmartModuleWrapper(
                        t
                    )
                )
            else:
                # Last hope...
                field_transforms.append(
                    SmartModuleWrapper(CallableAdapter(t))
                )
        field_transforms = add_to_device_operation(
            field_transforms,
            device=device
        )
        result.append(field_transforms)
    return result


def apply_pre_optimization(  # TODO: support RandomCrop
    transformations: List[Any],
    device: Optional[torch.device] = None
):

    if len(transformations) < 2:
        # No optimizations to apply if there are less than 2 transformations
        return transformations

    result = [transformations[0]]

    for t in transformations[1:]:
        if isinstance(t, NormalizeTV) and \
                isinstance(result[-1], ToTensorTV) and \
                device is not None and \
                device.type == 'cuda':
            # Optimize ToTensor+Normalize combo

            # ToTensor from torchvision does the following:
            # 1. PIL/NDArray -> Tensor
            # 2. Shape (H x W x C) -> (C x H x W)
            # 3. [0, 255] -> [0.0, 1.0]
            # In FFCV, the fist two steps are implemented as separate
            # transformations. The range change is not available in a 
            # standalone way, but it is applied when normalizing.

            # Note: we apply this optimization only when running on CUDA
            # as the FFCV Normalize is currently bugged and
            # does not work on CPU with PyTorch Tensor inputs.
            # It *may* work with CPU+NDArray...

            result[-1] = ToTensorFFCV()
            # result.append(ToDeviceFFCV(device))  # TODO: re-add
            result.append(ToTorchImageFFCV())

            dtype = torch.zeros(
                0,
                dtype=torch.get_default_dtype()
            ).numpy().dtype

            mean = np.array(t.mean) * 255
            std = np.array(t.std) * 255
            result.append(
                NormalizeFFCV(
                    mean,
                    std,
                    dtype
                )
            )

        elif isinstance(t, RandomResizedCropTV) and \
                isinstance(result[-1], SimpleRGBImageDecoder):
            size = t.size
            if isinstance(size, int):
                size = [size, size]
            elif len(size) == 1:
                size = [size[0], size[0]]
            result[-1] = RandomResizedCropRGBImageDecoder(
                size,
                t.scale,
                t.ratio
            )
        else:
            result.append(t)

    return result


def add_to_device_operation(
    transformations,
    device: Optional[torch.device] = None
):
    if device is None:
        return transformations

    # Check if ToDevice is laready in the pipeline 
    for t in transformations:
        if isinstance(t, ToDeviceFFCV):
            # Already set
            return transformations
        
    # All decoders (first operation in the pipeline) return NumPy arrays
    is_numpy = True
    is_cpu = True
        
    transformations = list(transformations)
    inserted = False
    for i, t in enumerate(transformations):
        t_def = FFCV_TRANSFORMS_DEFS.get(type(t), None)
        if t_def is None:
            # Unknown operation
            continue

        if is_numpy and not t_def.numpy_cpu:
            # Unmanageable situation: the current input is a NumPy array
            # but the transformation only supports PyTorch Tensor.

            # A warning is already raised by check_transforms_consistency,
            # so it's not a big issue...
            # Anyway, the pipeline is probably doomed to fail
            break
        elif (not is_numpy):
            if not (t_def.pytorch_cpu or t_def.pytorch_gpu):
                # Unmanageable situation: the current input is a PyTorch Tensor
                # but the transformation only supports NumPy arrays.

                # A warning is already raised by check_transforms_consistency
                break

            if is_cpu and t_def.pytorch_gpu:
                transformations.insert(i, ToDeviceFFCV(device=device))
                inserted = True
                break

            elif (not is_cpu) and t_def.pytorch_cpu:
                # From GPU to CPU is currently unsupported
                # Maybe in the future we can try to manage this...
                break
        
        if isinstance(t, ToTensorFFCV):
            is_numpy = False
        elif isinstance(t, ToDeviceFFCV):
            is_cpu = t.device.type == 'cpu'

    if not inserted:
        transformations.append(ToDeviceFFCV(device))

    return transformations

def check_transforms_consistency(
        transformations,
        warn_gpu_to_cpu: bool = True
    ):

    had_issues = False

    # All decoders (first operation in the pipeline) return NumPy arrays
    is_numpy = True
    is_cpu = True

    for t in transformations:
        t_def = FFCV_TRANSFORMS_DEFS.get(type(t), None)
        if t_def is None:
            # Unknown operation
            continue

        bad_usage_type = None

        if is_numpy and not t_def.numpy_cpu:
            bad_usage_type = 'NumPy arrays'
        elif (not is_numpy):
            if is_cpu and not t_def.pytorch_cpu:
                bad_usage_type = 'CPU PyTorch Tensors'
            elif (not is_cpu) and not t_def.pytorch_gpu:
                bad_usage_type = 'GPU PyTorch Tensors'

        if bad_usage_type is not None:
            warnings.warn(
                f'Transformation {type(t)} cannot be used on {bad_usage_type}.\n'
                f'Its registered definition is: {t_def}.\n'
                f'This may lead to issues with Numba...'
            )
            had_issues = True

        if isinstance(t, ToTensorFFCV):
            is_numpy = False
        elif isinstance(t, ToDeviceFFCV):
            if (not is_cpu) and t.device.type == 'cpu':
                if warn_gpu_to_cpu:
                    warnings.warn(
                        f'Moving a Tensor from GPU to CPU is quite unusual...'
                    )
                    had_issues = True
            
            is_cpu = t.device.type == 'cpu'
    
    return not had_issues



class SmartModuleWrapper(Operation):
    """Transform using the given torch.nn.Module

    Parameters
    ----------
    module: torch.nn.Module
        The module for transformation
    """
    def __init__(
            self,
            module: torch.nn.Module,
            expected_out_type: Union[np.dtype, torch.dtype, Literal['as_previous']] = 'as_previous',
            expected_shape: Union[Tuple[int, ...], Literal['as_previous']] = 'as_previous',
            smart_reshape: bool = True
        ):
        super().__init__()
        self.module = module
        self.expected_out_type = expected_out_type
        self.expected_shape = expected_shape
        self.input_type = 'numpy'
        self.output_type = 'numpy'
        self.smart_reshape = smart_reshape

    def generate_code(self) -> Callable:

        def convert_apply_convert_reshape(inp, _):
            inp_as_tensor = torch.from_numpy(inp)
            # N, H, W, C -> N, C, H, W
            inp_as_tensor = inp_as_tensor.permute([0, 3, 1, 2])
            res = self.module(inp_as_tensor)

            # N, C, H, W -> N, H, W, C
            res_as_np: np.ndarray = res.numpy()
            return res_as_np.transpose((0, 2, 3, 1))
        
        def convert_apply_reshape(inp, _):
            inp_as_tensor = torch.from_numpy(inp)
            # N, H, W, C -> N, C, H, W
            inp_as_tensor = inp_as_tensor.permute([0, 3, 1, 2])
            
            res = self.module(inp_as_tensor)
            return res
        
        def apply_convert_reshape(inp, _):
            res = self.module(inp)

            # N, C, H, W -> N, H, W, C
            res_as_np: np.ndarray = res.numpy()
            return res_as_np.transpose((0, 2, 3, 1))
        
        def convert_apply_convert(inp, _):
            inp_as_tensor = torch.from_numpy(inp)
            res = self.module(inp_as_tensor)
            return res.numpy()
        
        def convert_apply(inp, _):
            inp_as_tensor = torch.from_numpy(inp)
            res = self.module(inp_as_tensor)
            return res
        
        def apply_convert(inp, _):
            res = self.module(inp)
            return res.numpy()
        
        def apply(inp, _):
            device = inp.device
            return self.module(inp).to(device, non_blocking=True)
        
        # (input_type, output_type) -> func
        func_table = {
            ('numpy', 'numpy', True): convert_apply_convert_reshape,
            ('numpy', 'torch', True): convert_apply_reshape,
            ('torch', 'numpy', True): apply_convert_reshape,
            ('numpy', 'numpy', False): convert_apply_convert,
            ('numpy', 'torch', False): convert_apply,
            ('torch', 'numpy', False): apply_convert,
            ('torch', 'torch', True): apply,
            ('torch', 'torch', False): apply
        }

        return func_table[(self.input_type, self.output_type, self.smart_reshape)]

    def declare_state_and_memory(self, previous_state: State) -> Tuple[State, Optional[AllocationQuery]]:
        if len(previous_state.shape) != 3:
            self.smart_reshape = False

        self._fill_types(previous_state)
        self._to_device(previous_state)
        self._compute_smart_shape(previous_state)

        state_changes = dict()
        if self.expected_out_type != 'as_previous':
            # Output type != input type
            state_changes['dtype'] = self.expected_out_type

        state_changes['shape'] = self.expected_shape

        return replace(previous_state, jit_mode=False, **state_changes), None
    
    def _fill_types(self, previous_state: State):
        if isinstance(previous_state.dtype, torch.dtype):
            self.input_type = 'torch'
        else:
            self.input_type = 'numpy'

        if self.expected_out_type == 'as_previous':
            self.output_type = self.input_type
        else:
            if isinstance(self.expected_out_type, torch.dtype):
                self.output_type = 'torch'
            else:
                self.output_type = 'numpy'

    def _to_device(self, previous_state: State):
        if previous_state.device.type != 'cpu':
            if hasattr(self.module, 'to'):
                self.module = self.module.to(previous_state.device)

    def _compute_smart_shape(self, previous_state: State):
        if self.smart_reshape:
            if self.input_type == 'numpy':
                h, w, c = previous_state.shape
            else:
                c, h, w = previous_state.shape

            patch_shape = True
            if self.expected_shape != 'as_previous':
                if isinstance(self.expected_shape, int) or len(self.expected_shape) == 1:
                    h = self.expected_shape
                    w = self.expected_shape
                elif len(self.expected_shape) == 2:
                    h, w = self.expected_shape
                else:
                    # Completely user-managed
                    patch_shape = False      
                
            if patch_shape:            
                if self.output_type == 'numpy':
                    self.expected_shape = (h, w, c)
                else:
                    self.expected_shape = (c, h, w)

        
make_transform_defs()
