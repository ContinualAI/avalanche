from typing import (
    TYPE_CHECKING,
    Any, Callable,
    Dict,
    List,
    Optional,
    Sequence,
    Union,
)
from collections import OrderedDict
import numpy as np

from torch import Tensor

from PIL.Image import Image

from ffcv.fields import TorchTensorField
from ffcv.fields.decoders import (
    IntDecoder,
    FloatDecoder,
    NDArrayDecoder,
    SimpleRGBImageDecoder,
)


if TYPE_CHECKING:
    from ffcv.fields import Field
    from ffcv.pipeline.operation import Operation
    FFCVEncodeDef = OrderedDict[str, Field]
    FFCVDecodeDef = OrderedDict[str, List[Operation]]

    FFCVParameters = Dict[str, Any]
    EncoderDef = \
        Optional[
            Union[
                'FFCVEncodeDef',
                Callable[[FFCVParameters], 'FFCVEncodeDef']
            ]
        ]
    DecoderDef = \
        Optional[
            Union[
                'FFCVDecodeDef', 
                Callable[[FFCVParameters], 'FFCVDecodeDef']
            ]
        ]


def _image_encoder(ffcv_parameters: 'FFCVParameters'):
    from ffcv.fields import RGBImageField

    return RGBImageField(
        write_mode=ffcv_parameters.get('write_mode', 'raw'),
        max_resolution=ffcv_parameters.get('max_resolution', None),
        smart_threshold=ffcv_parameters.get('smart_threshold', None),
        jpeg_quality=ffcv_parameters.get('jpeg_quality', 90),
        compress_probability=ffcv_parameters.get(
            'compress_probability', 0.5
        ),
    )


def _ffcv_infer_encoder(
    value,
    ffcv_parameters: 'FFCVParameters'
) -> Optional['Field']:
    
    from ffcv.fields import (
        IntField,
        FloatField,
        NDArrayField,
        TorchTensorField,
    )

    if isinstance(value, int):
        return IntField()
    
    if isinstance(value, float):
        return FloatField()
    
    if isinstance(value, np.ndarray):
        return NDArrayField(
            value.dtype,
            shape=value.shape
        )
    
    if isinstance(value, Tensor):
        return TorchTensorField(
            value.dtype,
            shape=value.shape
        )
    
    if isinstance(value, Image):
        return _image_encoder(ffcv_parameters)
    
    return None


def _ffcv_infer_decoder(
    value,
    ffcv_parameters: 'FFCVParameters',
    encoder: Optional['Field'] = None,
    add_common_collate: bool = True
) -> Optional[List['Operation']]:
    from ffcv.transforms import ToTensor, Squeeze

    if encoder is not None:
        if isinstance(encoder, TorchTensorField):
            return [NDArrayDecoder(), ToTensor()]
        
        encoder_class = encoder.get_decoder_class()
        pipeline: List['Operation'] = [encoder_class()]
        if add_common_collate and encoder_class in [IntDecoder, FloatDecoder]:
            pipeline.extend((ToTensor(), Squeeze()))
        return pipeline
        
    if isinstance(value, int):
        pipeline: List['Operation'] = [IntDecoder()]

        if add_common_collate:
            pipeline.extend((ToTensor(), Squeeze()))
        return pipeline
    
    if isinstance(value, float):
        pipeline: List['Operation'] = [FloatDecoder()]
    
        if add_common_collate:
            pipeline.extend((ToTensor(), Squeeze()))
        return pipeline
    
    if isinstance(value, np.ndarray):
        return [NDArrayDecoder()]
    
    if isinstance(value, Tensor):
        return [NDArrayDecoder(), ToTensor()]
    
    if isinstance(value, Image):
        return [SimpleRGBImageDecoder()]
    
    return None


def _check_dataset_ffcv_encoder(dataset) -> 'EncoderDef':
    encoder_fn_or_def = getattr(dataset, '_ffcv_encoder', None)
    return encoder_fn_or_def


def _check_dataset_ffcv_decoder(dataset) -> 'DecoderDef':
    decoder_fn_or_def = getattr(dataset, '_ffcv_decoder', None)
    return decoder_fn_or_def


def _encoder_infer_all(
    dataset,
    ffcv_parameters: 'FFCVParameters'
) -> Optional['FFCVEncodeDef']:
    dataset_item = dataset[0]

    types = []

    # Try to infer the field type for each element
    for item in dataset_item:
        inferred_type = _ffcv_infer_encoder(
            item,
            ffcv_parameters
        )

        if inferred_type is None:
            return None
        
        types.append(inferred_type)

    # Type inferred for all fields
    # Let's apply a generic name and return the dictionary
    result = OrderedDict()
    for i, t in enumerate(types):
        result[f'field_{i}'] = t
    
    return result


def _decoder_infer_all(
    dataset,
    ffcv_parameters: 'FFCVParameters',
    encoder_dictionary: Optional['FFCVEncodeDef'] = None
) -> Optional['FFCVDecodeDef']:
    dataset_item: Sequence[Any] = dataset[0]

    types: List[List['Operation']] = []

    encoder_hints: List[Optional['Field']] = []
    field_names: List[str]

    if encoder_dictionary is None:
        encoder_hints = [None] * len(dataset_item)
        field_names = [f'field_{i}' for i in range(len(dataset_item))]
    else: 
        if len(encoder_dictionary) != len(dataset_item):
            raise ValueError('Wrong number of elements in encoder dictionary.')

        encoder_hints.extend(encoder_dictionary.values())
        field_names = list(encoder_dictionary.keys())

    # Try to infer the field type for each element
    for item, field_encoder in zip(dataset_item, encoder_hints):
        inferred_type = _ffcv_infer_decoder(
            item,
            ffcv_parameters,
            encoder=field_encoder
        )

        if inferred_type is None:
            return None
        
        types.append(inferred_type)

    # Type inferred for all fields
    # Let's apply the name and return the dictionary
    result = OrderedDict()
    for t, field_name in zip(types, field_names):
        result[field_name] = t
    
    return result


def _make_ffcv_encoder(
    dataset, 
    user_encoder_def: 'EncoderDef',
    ffcv_parameters: 'FFCVParameters'
) -> Optional['FFCVEncodeDef']:

    encoder_def = None

    # Use the user-provided pipeline / pipeline factory
    if user_encoder_def is not None:
        encoder_def = user_encoder_def
        if callable(encoder_def):
            encoder_def = encoder_def(ffcv_parameters)
    
    # Check if the dataset has an explicit field/method
    if encoder_def is None:
        encoder_def = _check_dataset_ffcv_encoder(dataset)
        if callable(encoder_def):
            encoder_def = encoder_def(ffcv_parameters)

    # Try to infer the pipeline from the dataset
    if encoder_def is None:
        encoder_def = _encoder_infer_all(
            dataset,
            ffcv_parameters
        )
    
    return encoder_def


def _make_ffcv_decoder(
    dataset, 
    user_decoder_def: 'DecoderDef',
    ffcv_parameters: 'FFCVParameters',
    encoder_dictionary: Optional['FFCVEncodeDef']
) -> Optional['FFCVDecodeDef']:
    
    decode_def = None

    # Use the user-provided pipeline / pipeline factory
    if user_decoder_def is not None:
        decode_def = user_decoder_def
        if callable(decode_def):
            decode_def = decode_def(ffcv_parameters)
    
    # Check if the dataset has an explicit field/method
    if decode_def is None:
        decode_def = _check_dataset_ffcv_decoder(dataset)
        if callable(decode_def):
            decode_def = decode_def(ffcv_parameters)
    
    # Try to infer the pipeline from the dataset
    if decode_def is None:
        decode_def = _decoder_infer_all(
            dataset,
            ffcv_parameters,
            encoder_dictionary=encoder_dictionary
        )
    
    return decode_def


__all__ = [
    '_make_ffcv_encoder',
    '_make_ffcv_decoder'
]
