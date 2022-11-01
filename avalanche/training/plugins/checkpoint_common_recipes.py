# Manage torch.device objects
# See:
# https://github.com/pytorch/pytorch/blob/30fb2c4abaaaa966999eab11674f25b18460e609/torch/csrc/Device.cpp#L144
import torch
import dill

CHECKPOINT_DEVICE_MAP = None


def _set_checkpoint_device_map(device_map):
    global CHECKPOINT_DEVICE_MAP
    CHECKPOINT_DEVICE_MAP = device_map


def _get_checkpoint_device_map():
    global CHECKPOINT_DEVICE_MAP
    return CHECKPOINT_DEVICE_MAP


def _recreate_pytorch_device(*args):
    device_map = globals().get('CHECKPOINT_DEVICE_MAP', None)
    device_object = torch.device(*args)
    mapped_object = device_object

    if device_map is not None:
        mapped_object = torch.device(
            device_map.get(str(device_object), str(device_object)))
    print('Mapping', device_object, 'to', mapped_object)
    return mapped_object


@dill.register(torch.device)
def _save_pytorch_device(pickler, obj: torch.device):
    has_index = obj.index is not None
    if has_index:
        reduction = (obj.type, obj.index)
    else:
        reduction = (obj.type,)

    pickler.save_reduce(
        _recreate_pytorch_device,
        reduction, obj=obj)


__all__ = [
    '_set_checkpoint_device_map',
    '_get_checkpoint_device_map',
    '_recreate_pytorch_device',
    '_save_pytorch_device'
]
