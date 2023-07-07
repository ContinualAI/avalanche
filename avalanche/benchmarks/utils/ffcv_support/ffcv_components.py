from dataclasses import dataclass
from pathlib import Path
from typing import (
    TYPE_CHECKING,
    Any,
    Dict,
    Iterable,
    List,
    Optional,
    Union,
)
from collections import OrderedDict
import warnings

import torch
from avalanche.benchmarks.scenarios.generic_scenario import CLScenario
from avalanche.benchmarks.utils.data import AvalancheDataset
from avalanche.benchmarks.utils.dataset_traversal_utils import (
    flat_datasets_from_benchmark,
    single_flat_dataset,
)

from avalanche.benchmarks.utils.utils import concat_datasets

if TYPE_CHECKING:
    from avalanche.benchmarks.utils.ffcv_support.ffcv_support_internals \
        import (
            FFCVDecodeDef,
            EncoderDef,
            DecoderDef
        )
    

FFCV_EXPERIMENTAL_WARNED = False


@dataclass
class FFCVInfo:
    path: Path
    encoder_dictionary: 'EncoderDef'
    decoder_dictionary: 'DecoderDef'
    decoder_includes_transformations: bool
    device: torch.device


def prepare_ffcv_datasets(
    benchmark: CLScenario,
    write_dir: Union[str, Path],
    device: torch.device,
    ffcv_parameters: Dict[str, Any],
    force_overwrite: bool = False,
    encoder_def: 'EncoderDef' = None,
    decoder_def: 'DecoderDef' = None,
    decoder_includes_transformations: Optional[bool] = None,
    print_summary: bool = True
):
    global FFCV_EXPERIMENTAL_WARNED

    if not FFCV_EXPERIMENTAL_WARNED:
        warnings.warn(
            'The support for FFCV is experimental. Use at your own risk!'
        )
        FFCV_EXPERIMENTAL_WARNED = True

    from ffcv.writer import DatasetWriter
    from ffcv.fields import IntField
    from ffcv.fields.decoders import IntDecoder
    from avalanche.benchmarks.utils.ffcv_support.ffcv_support_internals \
        import (
            _make_ffcv_decoder,
            _make_ffcv_encoder
        )
    
    if decoder_def is not None:
        if decoder_includes_transformations is None:
            raise ValueError(
                'When defining the decoder pipeline, '
                'please specify `decoder_includes_transformations`'
            )
        assert isinstance(decoder_includes_transformations, bool)

    if decoder_includes_transformations is None:
        decoder_includes_transformations = False

    write_dir = Path(write_dir)
    write_dir.mkdir(exist_ok=True, parents=True)

    flattened_datasets = flat_datasets_from_benchmark(benchmark)

    if print_summary:
        print('FFCV will serialize', len(flattened_datasets), 'datasets')
    
    for idx, (dataset, _, _) in enumerate(flattened_datasets):
        if print_summary:
            print('-' * 25, 'Dataset', idx, '-' * 25)
        
        # Note: it is appropriate to serialize the dataset in its raw
        # version (without transformations). Transformations will be
        # applied at loading time.
        with SuppressTransformations(dataset):

            dataset_ffcv_path = write_dir / f'dataset{idx}.beton'

            # Obtain the encoder dictionary
            # The FFCV encoder is a ordered dictionary mapping each
            # field (by name) to the field encoder.
            #
            # Example:
            # {
            #   'image': RGBImageField(),
            #   'label: IntField()
            # }
            #
            # Some fields (especcially the RGBImageField) accept 
            # some parameters that are here contained in ffcv_parameters.
            encoder_dict = _make_ffcv_encoder(
                dataset,
                encoder_def,
                ffcv_parameters
            )

            if encoder_dict is None:
                raise RuntimeError(
                    'Could not create the encoder pipeline for '
                    'the given dataset'
                )
            
            # Add the `index` field, which is needed to keep the
            # mapping from the original dataset to the subsets
            encoder_dict_with_index = OrderedDict()
            encoder_dict_with_index['index'] = IntField()
            encoder_dict_with_index.update(encoder_dict)

            if print_summary:
                print('### Encoder ###')
                for field_name, encoder_pipeline in \
                        encoder_dict_with_index.items():
                    print(f'Field "{field_name}"')
                    print('\t', encoder_pipeline)

            # Obtain the decoder dictionary
            # The FFCV decoder is a ordered dictionary mapping each
            # field (by name) to the field pipeline.
            # A field pipeline is made of a decoder followed by 
            # transformations.
            #
            # Example:
            # {
            #   'image': [
            #       SimpleRGBImageDecoder(),
            #       RandomHorizontalFlip(),
            #       ToTensor(),
            #       ...
            #   ],
            #   'label: [IntDecoder(), ToTensor(), Squeeze(), ...]
            # }
            #
            # However, unless the user specified a full custom decoder 
            # pipeline, Avalanche will obtain only the decoder for each 
            # field. The transformations, which may vary, will be added by the
            # data loader.
            decoder_dict = _make_ffcv_decoder(
                dataset,
                decoder_def,
                ffcv_parameters,
                encoder_dictionary=encoder_dict
            )

            if decoder_dict is None:
                raise RuntimeError(
                    'Could not create the decoder pipeline '
                    'for the given dataset'
                )

            decoder_dict_with_index = OrderedDict()
            decoder_dict_with_index['index'] = [IntDecoder()]
            decoder_dict_with_index.update(decoder_dict)

            if print_summary:
                print('### Decoder ###')
                for field_name, decoder_pipeline in \
                        decoder_dict_with_index.items():
                    print(f'Field "{field_name}"')
                    for pipeline_element in decoder_pipeline:
                        print('\t', pipeline_element)
                
                if decoder_includes_transformations:
                    print('This pipeline already includes transformations')
                else:
                    print('This pipeline does not include transformations')

            if force_overwrite or not dataset_ffcv_path.exists():
                if print_summary:
                    print('Serializing dataset to:', str(dataset_ffcv_path))
                
                writer_kwarg_parameters = dict()
                if 'page_size' in ffcv_parameters:
                    writer_kwarg_parameters['page_size'] = \
                        ffcv_parameters['page_size']

                if 'num_workers' in ffcv_parameters:
                    writer_kwarg_parameters['num_workers'] = \
                        ffcv_parameters['num_workers']

                writer = DatasetWriter(
                    str(dataset_ffcv_path), 
                    OrderedDict(encoder_dict_with_index),
                    **writer_kwarg_parameters
                )
                writer.from_indexed_dataset(IndexDataset(dataset))

                if print_summary:
                    print('Dataset serialized successfully')
        
            # Set the FFCV file path and encoder/decoder dictionaries
            # Those will be used later in the data loading process and may
            # also be useful for debugging purposes
            dataset.ffcv_info = FFCVInfo(
                dataset_ffcv_path,
                encoder_dict_with_index,
                decoder_dict_with_index,
                decoder_includes_transformations,
                torch.device(device)
            )
    
    if print_summary:
        print('-' * 61)


class IndexDataset:
    """
    A dataset implementation that adds the index of the example as the
    first element in the tuple returned by `__getitem__`.
    """

    def __init__(self, dataset):
        self.dataset = dataset

    def __getitem__(self, index):
        return (index, *self.dataset[index])

    def __len__(self):
        return len(self.dataset)


class SuppressTransformations:
    """
    Suppress the transformations of a dataset.

    This will act on the transformation fields. 
    
    Note: there are no ways to suppress hard coded transformations
    or transformations held in fields with custom names.
    """

    SUPPRESS_FIELDS = ['transform', 'target_transform', 'transforms']

    def __init__(self, dataset):
        self.dataset = dataset
        self._held_out_transforms = dict()

    def __enter__(self):
        self._held_out_transforms = dict()
        for transform_field in SuppressTransformations.SUPPRESS_FIELDS:
            if hasattr(self.dataset, transform_field):
                field_content = getattr(self.dataset, transform_field)
                self._held_out_transforms[transform_field] = field_content
                setattr(self.dataset, transform_field, None)

    def __exit__(self, *_):
        for transform_field, field_content in self._held_out_transforms.items():
            setattr(self.dataset, transform_field, field_content)
        self._held_out_transforms.clear()


class GetItemDataset:

    def __init__(
            self,
            dataset: AvalancheDataset,
            reversed_indices: Dict[int, int],
            collate_fn=None
    ):
        self.dataset: AvalancheDataset = dataset
        self.reversed_indices: Dict[int, int] = reversed_indices

        all_data_attributes = self.dataset._data_attributes.values()
        self.get_item_data_attributes = list(
            filter(lambda x: x.use_in_getitem, all_data_attributes)
        )

        self.collate_fn = collate_fn if collate_fn is not None \
            else self.dataset.collate_fn
        
        if self.collate_fn is None:
            raise RuntimeError('Undefined collate function')

    def __getitem__(self, indices):
        elements_from_attributes = []
        for idx in indices:
            reversed_idx = self.reversed_indices[int(idx)]
            values = []
            for da in self.get_item_data_attributes:
                values.append(da[reversed_idx])
            elements_from_attributes.append(tuple(values))

        return tuple(self.collate_fn(elements_from_attributes))
    

def has_ffcv_support(datasets: List[AvalancheDataset]):
    try:
        flat_set = single_flat_dataset(
            concat_datasets(datasets)
        )
    except Exception:
        return False

    if flat_set is None:
        return False
    
    leaf_dataset = flat_set[0]
    
    return hasattr(leaf_dataset, 'ffcv_info')


class HybridFfcvLoader:

    ALREADY_COVERED_PARAMS = set((
        'fname',
        'batch_size',
        'order'
        'distributed',
        'seed',
        'indices',
        'pipelines',
    ))

    VALID_FFCV_PARAMS = set((
        'fname',
        'batch_size',
        'num_workers',
        'os_cache',
        'order',
        'distributed',
        'seed',
        'indices',
        'pipelines',
        'custom_fields',
        'drop_last',
        'batches_ahead',
        'recompile'
    ))

    def __init__(
        self,
        dataset: AvalancheDataset,
        batch_sampler: Iterable[List[int]],
        batch_size: int,
        ffcv_loader_parameters: Dict[str, Any],
        device: Optional[Union[str, torch.device]] = None,
        persistent_workers: bool = True,
        print_ffcv_summary: bool = True,
        start_immediately=False
    ):
        from ffcv.loader import Loader
        
        self.dataset: AvalancheDataset = dataset
        self.batch_sampler = batch_sampler
        self.batch_size: int = batch_size
        self.ffcv_loader_parameters = ffcv_loader_parameters
        self.persistent_workers: bool = persistent_workers

        for param_name in HybridFfcvLoader.ALREADY_COVERED_PARAMS:
            if param_name in self.ffcv_loader_parameters:
                warnings.warn(
                    f'`{param_name}` should not be passed to the ffcv loader!'
                )

        if print_ffcv_summary:
            print('-' * 15, 'HybridFfcvLoader summary', '-' * 15)

        ffcv_info = self._extract_ffcv_info(
            dataset=self.dataset,
            device=device,
            print_summary=print_ffcv_summary
        )

        if print_ffcv_summary:
            print('-' * 56)
        
        self.ffcv_dataset_path, self.ffcv_decoder_dictionary, \
            self.leaf_indices, self.get_item_dataset, self.device = ffcv_info
        
        self._persistent_loader: Optional['Loader'] = None

        if start_immediately:
            # If persistent_workers is False, this loader will be
            # used at first __iter__ and immediately set to None
            self._persistent_loader = self._make_loader()

    @staticmethod
    def _extract_ffcv_info(
        dataset: AvalancheDataset,
        device: Optional[Union[str, torch.device]] = None,
        print_summary: bool = True
    ):
        from avalanche.benchmarks.utils.ffcv_support.ffcv_transform_utils \
            import (
                adapt_transforms,
                check_transforms_consistency,
            )
                
        # Obtain the leaf dataset, the indices,
        # and the transformations to apply
        flat_set_def = single_flat_dataset(
            dataset
        )
        if flat_set_def is None:
            raise RuntimeError(
                'The dataset cannot be traversed to the leaf dataset.'
            )
        
        leaf_dataset, indices, transforms = flat_set_def
        if print_summary:
            print(
                'The input AvalancheDataset is a subset of the leaf dataset',
                leaf_dataset
            )
            print('The input dataset contains', len(indices), 'elements')
            print('The original chain of transformations is:')
            for t in transforms:
                print('\t', t)
            print('Will try to translate those transformations to FFCV')

        ffcv_info: FFCVInfo = leaf_dataset.ffcv_info

        ffcv_dataset_path = ffcv_info.path
        ffcv_decoder_dictionary = ffcv_info.decoder_dictionary
        decoder_includes_transformations = \
            ffcv_info.decoder_includes_transformations
        
        if device is None:
            device = ffcv_info.device
        device = torch.device(device)

        # Map the indices so that we know how leaf
        # dataset indices are mapped in the AvalancheDataset
        reversed_indices = dict()
        for avl_idx, leaf_idx in enumerate(indices):
            reversed_indices[leaf_idx] = avl_idx

        # We will use the GetItemDataset to get those Avalanche-specific
        # dynamic fields that are not loaded by FFCV, such as the task label
        get_item_dataset = GetItemDataset(
            dataset,
            reversed_indices=reversed_indices
        )

        if print_summary:
            if len(get_item_dataset.get_item_data_attributes) > 0:
                print(
                    'The following data attributes are returned in '
                    'the example tuple:'
                )
                for da in get_item_dataset.get_item_data_attributes:
                    print('\t', da.name)
            else:
                print('No data attributes are returned in the example tuple.')
        
        # Defensive copy
        # Alas, FFCV Loader internally modifies it, so this is also
        # needed when decoder_includes_transformations is True
        ffcv_decoder_dictionary = OrderedDict(ffcv_decoder_dictionary)

        if not decoder_includes_transformations:
            # Adapt the transformations (usually from torchvision) to FFCV.
            # Most torchvision transformations cannot be mapped to FFCV ones, 
            # but they still work.
            # num_fields is "|dictionary|-1" as there is an additional 'index' 
            # field that is internally managed by Avalanche and is not being
            # transformed.
            ffcv_decoder_dictionary_lst = \
                list(ffcv_decoder_dictionary.values())[1:]

            adapted_transforms = adapt_transforms(
                transforms,
                ffcv_decoder_dictionary_lst,
                device=device
            )
            
            for i, field_name in enumerate(ffcv_decoder_dictionary.keys()):
                if i == 0:
                    continue
                ffcv_decoder_dictionary[field_name] = adapted_transforms[i-1]

        for field_name, field_decoder in ffcv_decoder_dictionary.items():
            if print_summary:
                print(f'Checking pipeline for field "{field_name}"')
            no_issues = check_transforms_consistency(field_decoder)
            
            if print_summary and no_issues:
                print(f'No issues for this field')

        if print_summary:
            print('### The final chain of transformations is: ###')
            for field_name, field_transforms in ffcv_decoder_dictionary.items():
                print(f'Field "{field_name}":')
                for t in field_transforms:
                    print('\t', t)
            print('Note: "index" is an internal field managed by Avalanche')

        return (
            ffcv_dataset_path,
            ffcv_decoder_dictionary,
            indices,
            get_item_dataset,
            device
        )
    
    def _make_loader(self):
        from ffcv.loader import Loader, OrderOption
        
        ffcv_dataset_path = self.ffcv_dataset_path
        ffcv_decoder_dictionary = OrderedDict(self.ffcv_decoder_dictionary)
        leaf_indices = list(self.leaf_indices)

        # TODO: batch sampling
        return Loader(
            str(ffcv_dataset_path),
            self.batch_size,
            indices=leaf_indices,
            order=OrderOption.SEQUENTIAL,
            pipelines=ffcv_decoder_dictionary,
            **self.ffcv_loader_parameters
        )

    def __iter__(self):
        get_item_dataset = self.get_item_dataset

        # Instantiate the FFCV loader
        if self._persistent_loader is not None:
            ffcv_loader = self._persistent_loader

            if not self.persistent_workers:
                # Corner case:
                # This may happen if start_immediately is True
                # but persistent_workers is False
                self._persistent_loader = None
        else:
            ffcv_loader = self._make_loader()

            if self.persistent_workers:
                self._persistent_loader = ffcv_loader

        for batch in ffcv_loader:
            # Before returning the batch, obtain the custom Avalanche values
            # and add it to the batch.
            # Those are the values not found in the FFCV dataset
            # (and not stored on disk!).
            # 
            # A common element is the task label, which is usually returned
            # as the third element.
            #
            # In practice, those fields are "data attributes" 
            # of the input AvalancheDataset whose `use_in_getitem`
            # field is True.
            # 
            # This means in practice:
            # 1. obtain the `batch` from FFCV (usually is a tuple `x, y`).
            # 2. obtain the Avalanche values such as `t` (or others). 
            #   We do this through the `get_item_dataset`.
            # 3. create an overall tuple `x, y, t, ...`.

            indices = batch[0]

            elements_from_attributes = get_item_dataset[indices]

            elements_from_attributes_device = []

            for element in elements_from_attributes:
                if isinstance(element, torch.Tensor):
                    element = element.to(self.device, non_blocking=True)
                elements_from_attributes_device.append(element)

            overall_batch = tuple(batch[1:]) + \
                tuple(elements_from_attributes_device)
            
            yield overall_batch


__all__ = [
    'prepare_ffcv_datasets',
    'has_ffcv_support',
    'HybridFfcvLoader'
]
