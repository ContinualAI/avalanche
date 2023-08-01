"""
Components used to enable the FFCV dataloading mechanisms.

It is usually sufficient to call `enable_ffcv` on the given
benchmark to get started with the FFCV support.

Please refer to the examples for more details.
"""

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
import numpy as np

import torch
from torch.utils.data.sampler import Sampler
from avalanche.benchmarks.scenarios.generic_scenario import CLScenario
from avalanche.benchmarks.utils.data import AvalancheDataset
from avalanche.benchmarks.utils.dataset_traversal_utils import (
    flat_datasets_from_benchmark,
    single_flat_dataset,
)

from avalanche.benchmarks.utils.utils import concat_datasets

if TYPE_CHECKING:
    from avalanche.benchmarks.utils.ffcv_support.ffcv_support_internals import (
        EncoderDef,
        DecoderDef,
    )


FFCV_EXPERIMENTAL_WARNED = False


@dataclass
class FFCVInfo:
    path: Path
    encoder_dictionary: "EncoderDef"
    decoder_dictionary: "DecoderDef"
    decoder_includes_transformations: bool
    device: torch.device


def enable_ffcv(
    benchmark: CLScenario,
    write_dir: Union[str, Path],
    device: torch.device,
    ffcv_parameters: Dict[str, Any],
    force_overwrite: bool = False,
    encoder_def: "EncoderDef" = None,
    decoder_def: "DecoderDef" = None,
    decoder_includes_transformations: Optional[bool] = None,
    print_summary: bool = True,
) -> None:
    """
    Enables the support for FFCV data loading for the given benchmark.

    Once the support is added, the strategies will create FFCV-based dataloaders
    instead of the usual PyTorch-based ones.

    Please note that FFCV is an optional dependency whose installation process
    is described in the official FFCV website.

    This function supposes that the benchmark is based on a few base datasets
    (usually one for train and one for test). This is the case for Split-* benchmarks
    and is also the usual case for the vast majority of benchmarks. The support for
    "sparse" datasets such as CORe50 will be added in the near future.

    When this function is first called on a benchmark, the underlying datasets are
    serialized on disk. If a `encoder_def` is given, that definition is used. Otherwise,
    a definition is searched in the leaf dataset (`_ffcv_encoder` field, if available).
    If such a definition is not found, it is created automatically.
    Refer to the FFCV documentation for more details on the encoder pipeline.

    Please note that the serialized datasets are independent of the benchmark seed,
    number of experiences, presence of task labels, etcetera. This means that the
    same folder can be reused for the same benchmark type.

    The definition of the decoder pipeline is created later, if not
    given using `decoder_def`. However, creating the decoder pipeline is a complex
    task and not all field types and transformations are fully supported. Consider
    passing an explicit `decoder_def` in case of unexpected outputs. If a decoder
    definition is not passed explicitly, Avalanche will try to use the dataset
    `_ffcv_decoder` field if available before attempting to create one automatically.

    See the `ffcv` examples for more info on how to tune the decoder definitions and for
    examples of advanced use of the FFCV support.

    :param benchmark: The benchmark for which the support for FFCV loader should be enabled.
    :param write_dir: Where the datasets should be serialized in FFCV format.
    :param device: The device used for training.
    :param ffcv_parameters: Parameters to be passed to FFCV writer and RGB fields.
    :param force_overwrite: If True, serialized datasets already found in `write_dir` will be
        overwritten.
    :param encoder_def: The definition of the dataset fields. See the FFCV guide for more details.
    :param decoder_def: The definition of the decoder pipeline. If not None, then
        `decoder_includes_transformations` must be passed.
    :param decoder_includes_transformations: If True, then Avalanche will treat `decoder_def` as
        the complete pipeline, transformations included. If False, Avalanche will suppose that only
        the decoder is passed for each field and transformations will be translated by Avalanche
        from the torchvision ones.
    :param print_summary: If True (default), will print some verbose info to stdout regaring the
        datasets and pipelines. Once you have a complete working FFCV pipeline, you can consider
        setting this to False.
    """
    global FFCV_EXPERIMENTAL_WARNED

    if not FFCV_EXPERIMENTAL_WARNED:
        warnings.warn("The support for FFCV is experimental. Use at your own risk!")
        FFCV_EXPERIMENTAL_WARNED = True

    from ffcv.writer import DatasetWriter
    from avalanche.benchmarks.utils.ffcv_support.ffcv_support_internals import (
        _make_ffcv_decoder,
        _make_ffcv_encoder,
    )

    if decoder_def is not None:
        if decoder_includes_transformations is None:
            raise ValueError(
                "When defining the decoder pipeline, "
                "please specify `decoder_includes_transformations`"
            )
        assert isinstance(decoder_includes_transformations, bool)

    if decoder_includes_transformations is None:
        decoder_includes_transformations = False

    write_dir = Path(write_dir)
    write_dir.mkdir(exist_ok=True, parents=True)

    flattened_datasets = flat_datasets_from_benchmark(benchmark)

    if print_summary:
        print("FFCV will serialize", len(flattened_datasets), "datasets")

    for idx, (dataset, _, _) in enumerate(flattened_datasets):
        if print_summary:
            print("-" * 25, "Dataset", idx, "-" * 25)

        # Note: it is appropriate to serialize the dataset in its raw
        # version (without transformations). Transformations will be
        # applied at loading time.
        with _SuppressTransformations(dataset):
            dataset_ffcv_path = write_dir / f"dataset{idx}.beton"

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
            encoder_dict = _make_ffcv_encoder(dataset, encoder_def, ffcv_parameters)

            if encoder_dict is None:
                raise RuntimeError(
                    "Could not create the encoder pipeline for " "the given dataset"
                )

            if print_summary:
                print("### Encoder ###")
                for field_name, encoder_pipeline in encoder_dict.items():
                    print(f'Field "{field_name}"')
                    print("\t", encoder_pipeline)

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
                dataset, decoder_def, ffcv_parameters, encoder_dictionary=encoder_dict
            )

            if decoder_dict is None:
                raise RuntimeError(
                    "Could not create the decoder pipeline " "for the given dataset"
                )

            if print_summary:
                print("### Decoder ###")
                for field_name, decoder_pipeline in decoder_dict.items():
                    print(f'Field "{field_name}"')
                    for pipeline_element in decoder_pipeline:
                        print("\t", pipeline_element)

                if decoder_includes_transformations:
                    print("This pipeline already includes transformations")
                else:
                    print("This pipeline does not include transformations")

            if force_overwrite or not dataset_ffcv_path.exists():
                if print_summary:
                    print("Serializing dataset to:", str(dataset_ffcv_path))

                writer_kwarg_parameters = dict()
                if "page_size" in ffcv_parameters:
                    writer_kwarg_parameters["page_size"] = ffcv_parameters["page_size"]

                if "num_workers" in ffcv_parameters:
                    writer_kwarg_parameters["num_workers"] = ffcv_parameters[
                        "num_workers"
                    ]

                writer = DatasetWriter(
                    str(dataset_ffcv_path),
                    OrderedDict(encoder_dict),
                    **writer_kwarg_parameters,
                )
                writer.from_indexed_dataset(dataset)

                if print_summary:
                    print("Dataset serialized successfully")

            # Set the FFCV file path and encoder/decoder dictionaries
            # Those will be used later in the data loading process and may
            # also be useful for debugging purposes
            dataset.ffcv_info = FFCVInfo(
                dataset_ffcv_path,
                encoder_dict,
                decoder_dict,
                decoder_includes_transformations,
                torch.device(device),
            )

    if print_summary:
        print("-" * 61)


class _SuppressTransformations:
    """
    Suppress the transformations of a dataset.

    This will act on the transformation fields.

    Note: there are no ways to suppress hard coded transformations
    or transformations held in fields with custom names.
    """

    SUPPRESS_FIELDS = ["transform", "target_transform", "transforms"]

    def __init__(self, dataset):
        self.dataset = dataset
        self._held_out_transforms = dict()

    def __enter__(self):
        self._held_out_transforms = dict()
        for transform_field in _SuppressTransformations.SUPPRESS_FIELDS:
            if hasattr(self.dataset, transform_field):
                field_content = getattr(self.dataset, transform_field)
                self._held_out_transforms[transform_field] = field_content
                setattr(self.dataset, transform_field, None)

    def __exit__(self, *_):
        for transform_field, field_content in self._held_out_transforms.items():
            setattr(self.dataset, transform_field, field_content)
        self._held_out_transforms.clear()


class _GetItemDataset:
    def __init__(
        self,
        dataset: AvalancheDataset,
        reversed_indices: Dict[int, int],
        collate_fn=None,
    ):
        self.dataset: AvalancheDataset = dataset
        self.reversed_indices: Dict[int, int] = reversed_indices

        all_data_attributes = self.dataset._data_attributes.values()
        self.get_item_data_attributes = list(
            filter(lambda x: x.use_in_getitem, all_data_attributes)
        )

        self.collate_fn = (
            collate_fn if collate_fn is not None else self.dataset.collate_fn
        )

        if self.collate_fn is None:
            raise RuntimeError("Undefined collate function")

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
    """
    Checks if the support for FFCV was enabled for the given
    dataset list.

    This will 1) check if all the given :class:`AvalancheDataset`
    point to the same leaf dataset and 2) if the leaf dataset
    has the proper FFCV info setted by the :func:`enable_ffcv`
    function.

    :param dataset: The list of datasets.
    :return: True if FFCV can be used to load the given datasets,
        False otherwise.
    """
    try:
        flat_set = single_flat_dataset(concat_datasets(datasets))
    except Exception:
        return False

    if flat_set is None:
        return False

    leaf_dataset = flat_set[0]

    return hasattr(leaf_dataset, "ffcv_info")


class _MappedBatchsampler(Sampler[List[int]]):
    """
    Internal utility to better support the `set_epoch` method in FFCV.

    This is a wrapper of a batch sampler that may be based on a PyTorch
    :class:`DistributedSampler`. This allows passing the `set_epoch`
    call to the underlying sampler.
    """

    def __init__(self, batch_sampler: Sampler[List[int]], indices):
        self.batch_sampler = batch_sampler
        self.indices = indices

    def __iter__(self):
        for batch in self.batch_sampler:
            batch_mapped = [self.indices[int(x)] for x in batch]
            yield np.array(batch_mapped)

    def __len__(self):
        return len(self.batch_sampler)

    def set_epoch(self, epoch: int):
        if hasattr(self.batch_sampler, "set_epoch"):
            self.batch_sampler.set_epoch(epoch)
        else:
            if hasattr(self.batch_sampler, "sampler"):
                if hasattr(self.batch_sampler.sampler, "set_epoch"):
                    self.batch_sampler.sampler.set_epoch(epoch)


class HybridFfcvLoader:
    """
    A dataloader used to load :class:`AvalancheDataset`s for which
    the FFCV support was previously enabled by using :func:`enable_ffcv`.

    This is not a pure wrapper of a FFCV loader: this hybrid dataloader
    is in charge of both creating the FFCV loader and merging
    the Avalanche-specific info contained in the :class:`DataAttribute`
    fields of the datasets (such as task labels).
    """

    ALREADY_COVERED_PARAMS = set(
        (
            "fname",
            "batch_size",
            "order",
            "distributed",
            "seed",
            "indices",
            "pipelines",
        )
    )

    VALID_FFCV_PARAMS = set(
        (
            "fname",
            "batch_size",
            "num_workers",
            "os_cache",
            "order",
            "distributed",
            "seed",
            "indices",
            "pipelines",
            "custom_fields",
            "drop_last",
            "batches_ahead",
            "recompile",
        )
    )

    def __init__(
        self,
        dataset: AvalancheDataset,
        batch_sampler: Iterable[List[int]],
        ffcv_loader_parameters: Dict[str, Any],
        device: Optional[Union[str, torch.device]] = None,
        persistent_workers: bool = False,
        print_ffcv_summary: bool = True,
        start_immediately: bool = False,
    ):
        """
        Creates an instance of the Avalanche-FFCV hybrid dataloader.

        :param dataset: The dataset to be loaded.
        :param batch_sampler: The batch sampler to use.
        :param ffcv_loader_parameters: The FFCV-specific parameters to pass to
            the FFCV loader. Should not contain the elements such as `fname`,
            `batch_size`, `order`, and all the parameters listed in the
            `ALREADY_COVERED_PARAMS` class field, as they are already set by Avalanche.
        :param device: The target device.
        :param persistent_workers: If True, this loader will not re-create the FFCV loader
            between epochs. Defaults to False.
        :param print_ffcv_summary: If True, a summary of the decoder pipeline (and additional
            useful info) will be printed. Defaults to True.
        :param start_immediately: If True, the FFCV loader should be started immediately.
            Defaults to False.
        """
        from avalanche.benchmarks.utils.ffcv_support.ffcv_loader import _CustomLoader

        self.dataset: AvalancheDataset = dataset
        self.batch_sampler = batch_sampler
        self.ffcv_loader_parameters = ffcv_loader_parameters
        self.persistent_workers: bool = persistent_workers

        for param_name in HybridFfcvLoader.ALREADY_COVERED_PARAMS:
            if param_name in self.ffcv_loader_parameters:
                warnings.warn(
                    f"`{param_name}` should not be passed to the ffcv loader!"
                )

        if print_ffcv_summary:
            print("-" * 15, "HybridFfcvLoader summary", "-" * 15)

        ffcv_info = self._extract_ffcv_info(
            dataset=self.dataset, device=device, print_summary=print_ffcv_summary
        )

        if print_ffcv_summary:
            print("-" * 56)

        (
            self.ffcv_dataset_path,
            self.ffcv_decoder_dictionary,
            self.leaf_indices,
            self.get_item_dataset,
            self.device,
        ) = ffcv_info

        self._persistent_loader: Optional["_CustomLoader"] = None

        if start_immediately:
            # If persistent_workers is False, this loader will be
            # used at first __iter__ and immediately set to None
            self._persistent_loader = self._make_loader()

    @staticmethod
    def _extract_ffcv_info(
        dataset: AvalancheDataset,
        device: Optional[Union[str, torch.device]] = None,
        print_summary: bool = True,
    ):
        from avalanche.benchmarks.utils.ffcv_support.ffcv_transform_utils import (
            adapt_transforms,
            check_transforms_consistency,
        )

        # Obtain the leaf dataset, the indices,
        # and the transformations to apply
        flat_set_def = single_flat_dataset(dataset)
        if flat_set_def is None:
            raise RuntimeError("The dataset cannot be traversed to the leaf dataset.")

        leaf_dataset, indices, transforms = flat_set_def
        if print_summary:
            print(
                "The input AvalancheDataset is a subset of the leaf dataset",
                leaf_dataset,
            )
            print("The input dataset contains", len(indices), "elements")
            print("The original chain of transformations is:")
            for t in transforms:
                print("\t", t)
            print("Will try to translate those transformations to FFCV")

        ffcv_info: FFCVInfo = leaf_dataset.ffcv_info

        ffcv_dataset_path = ffcv_info.path
        ffcv_decoder_dictionary = ffcv_info.decoder_dictionary
        decoder_includes_transformations = ffcv_info.decoder_includes_transformations

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
        get_item_dataset = _GetItemDataset(dataset, reversed_indices=reversed_indices)

        if print_summary:
            if len(get_item_dataset.get_item_data_attributes) > 0:
                print(
                    "The following data attributes are returned in "
                    "the example tuple:"
                )
                for da in get_item_dataset.get_item_data_attributes:
                    print("\t", da.name)
            else:
                print("No data attributes are returned in the example tuple.")

        # Defensive copy
        # Alas, FFCV Loader internally modifies it, so this is also
        # needed when decoder_includes_transformations is True
        ffcv_decoder_dictionary = OrderedDict(ffcv_decoder_dictionary)

        if not decoder_includes_transformations:
            # Adapt the transformations (usually from torchvision) to FFCV.
            # Most torchvision transformations cannot be mapped to FFCV ones,
            # but they still work.
            ffcv_decoder_dictionary_lst = list(ffcv_decoder_dictionary.values())

            adapted_transforms = adapt_transforms(
                transforms, ffcv_decoder_dictionary_lst, device=device
            )

            for i, field_name in enumerate(ffcv_decoder_dictionary.keys()):
                ffcv_decoder_dictionary[field_name] = adapted_transforms[i]

        for field_name, field_decoder in ffcv_decoder_dictionary.items():
            if print_summary:
                print(f'Checking pipeline for field "{field_name}"')
            no_issues = check_transforms_consistency(field_decoder)

            if print_summary and no_issues:
                print(f"No issues for this field")

        if print_summary:
            print("### The final chain of transformations is: ###")
            for field_name, field_transforms in ffcv_decoder_dictionary.items():
                print(f'Field "{field_name}":')
                for t in field_transforms:
                    print("\t", t)

        return (
            ffcv_dataset_path,
            ffcv_decoder_dictionary,
            indices,
            get_item_dataset,
            device,
        )

    def _make_loader(self):
        from ffcv.loader import OrderOption
        from avalanche.benchmarks.utils.ffcv_support.ffcv_loader import _CustomLoader

        ffcv_dataset_path = self.ffcv_dataset_path
        ffcv_decoder_dictionary = OrderedDict(self.ffcv_decoder_dictionary)
        leaf_indices = list(self.leaf_indices)

        return _CustomLoader(
            str(ffcv_dataset_path),
            batch_size=len(leaf_indices) // len(self.batch_sampler),  # Not used
            indices=leaf_indices,
            order=OrderOption.SEQUENTIAL,
            pipelines=ffcv_decoder_dictionary,
            batch_sampler=_MappedBatchsampler(self.batch_sampler, leaf_indices),
            **self.ffcv_loader_parameters,
        )

    def __iter__(self):
        from avalanche.benchmarks.utils.ffcv_support.ffcv_epoch_iterator import (
            _CustomEpochIterator,
        )

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

        epoch_iterator: "_CustomEpochIterator" = iter(ffcv_loader)

        for indices, batch in epoch_iterator:
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

            elements_from_attributes = get_item_dataset[indices]

            elements_from_attributes_device = []

            for element in elements_from_attributes:
                if isinstance(element, torch.Tensor):
                    element = element.to(self.device, non_blocking=True)
                elements_from_attributes_device.append(element)

            overall_batch = tuple(batch) + tuple(elements_from_attributes_device)

            yield overall_batch

    def __len__(self):
        return len(self.batch_sampler)


__all__ = ["enable_ffcv", "has_ffcv_support", "HybridFfcvLoader"]
