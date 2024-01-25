import os
from pathlib import Path
import torch
import json
from typing import (
    Any,
    Generic,
    List,
    Union,
    TypeVar,
    Tuple,
    Dict,
    TYPE_CHECKING,
    Type,
    Callable,
    Sequence,
    Optional,
    Protocol,
)

from avalanche.benchmarks.utils.data import AvalancheDataset

try:
    from lvis import LVIS
    from pycocotools.coco import COCO
    from pycocotools import mask as coco_mask
except ImportError:
    import warnings

    warnings.warn(
        "LVIS or PyCocoTools not found, "
        "if you want to use detection "
        "please install avalanche with the "
        "detection dependencies: "
        "pip install avalanche-lib[detection]"
    )
    LVIS = object  # type: ignore
    COCO = object  # type: ignore
    coco_mask = object  # type: ignore

from torch import Tensor
from json import JSONEncoder

from torch.utils.data import Subset, ConcatDataset

from avalanche.evaluation import PluginMetric
from avalanche.evaluation.metric_results import MetricValue
from avalanche.evaluation.metric_utils import get_metric_name

if TYPE_CHECKING:
    from avalanche.training.supervised.naive_object_detection import (
        ObjectDetectionTemplate,
    )


TDetPredictions_co = TypeVar("TDetPredictions_co", covariant=True)
TDetModelOutput = TypeVar("TDetModelOutput", contravariant=True)

TCommonDetectionOutput = Dict[str, Dict[str, Tensor]]


class TensorEncoder(JSONEncoder):
    def __init__(self, **kwargs):
        super(TensorEncoder, self).__init__(**kwargs)

    def default(self, o: Any) -> Any:
        if isinstance(o, Tensor):
            o = o.detach().cpu().tolist()

        return o


def tensor_decoder(dct):
    for t_name in ["boxes", "mask", "scores", "keypoints", "labels"]:
        if t_name in dct:
            if t_name == "labels":
                dct[t_name] = torch.as_tensor(dct[t_name], dtype=torch.int64)
            else:
                dct[t_name] = torch.as_tensor(dct[t_name])

            if t_name == "boxes":
                dct[t_name] = torch.reshape(dct[t_name], shape=(-1, 4))
            # TODO: implement mask shape

    return dct


class DetectionEvaluator(Protocol[TDetPredictions_co, TDetModelOutput]):
    """
    Interface for object detection/segmentation evaluators.

    The evaluator should be able to accumulate the model outputs and compute
    the relevant metrics.
    """

    def update(self, model_output: TDetModelOutput):
        """
        Adds new predictions.

        The evaluator will internally accumulate these predictions so that
        they can be later evaluated using `evaluate()`.

        :param model_output: The predictions from the model.
        :return: None
        """
        pass

    def evaluate(
        self,
    ) -> Optional[Union[Dict[str, Any], Tuple[Dict[str, Any], TDetPredictions_co]]]:
        """
        Computes the performance metrics on the outputs previously obtained
        through `update()`.

        :return: If running in the main process, the predicted metrics. At
            least a dictionary of metric_name -> value is required. In
            addition, the evaluator may return a second value representing
            dataset/evaluator-specific additional info. If running in
            a non-main process, it should do nothing and return None.
        """
        pass

    def summarize(self):
        """
        Prints a summary of computed metrics to standard output.

        This should be called after `evaluate()`.

        :return: None
        """
        pass


SupportedDatasetApiDef = Tuple["str", Union[Tuple[Type], Type]]


DEFAULT_SUPPROTED_DETECTION_DATASETS: Sequence[SupportedDatasetApiDef] = (
    ("coco", COCO),  # CocoDetection from torchvision
    ("lvis_api", LVIS),  # LvisDataset from Avalanche
)


def coco_evaluator_factory(coco_gt: COCO, iou_types: List[str]):
    from avalanche.evaluation.metrics.detection_evaluators.coco_evaluator import (
        CocoEvaluator,
    )

    return CocoEvaluator(coco_gt=coco_gt, iou_types=iou_types)


class DetectionMetrics(
    PluginMetric[dict], Generic[TDetPredictions_co, TDetModelOutput]
):
    """
    Metric used to compute the detection and segmentation metrics using the
    dataset-specific API.

    Metrics are returned after each evaluation experience.

    This metric can also be used to serialize model outputs to JSON files,
    by producing one file for each evaluation experience. This can be useful
    if outputs have to been processed later (like in a competition).

    If no dataset-specific API is used, the COCO API (pycocotools) will be used.
    """

    def __init__(
        self,
        *,
        evaluator_factory: Callable[
            [Any, List[str]], DetectionEvaluator[TDetPredictions_co, TDetModelOutput]
        ] = coco_evaluator_factory,
        gt_api_def: Sequence[
            SupportedDatasetApiDef
        ] = DEFAULT_SUPPROTED_DETECTION_DATASETS,
        default_to_coco=False,
        save_folder=None,
        filename_prefix="model_output",
        save_stream="test",
        iou_types: Union[str, List[str]] = "bbox",
        summarize_to_stdout: bool = True,
    ):
        """
        Creates an instance of DetectionMetrics.

        :param evaluator_factory: The factory for the evaluator to use. By
            default, the COCO evaluator will be used. The factory should accept
            2 parameters: the API object containing the test annotations and
            the list of IOU types to consider. It must return an instance
            of a DetectionEvaluator.
        :param gt_api_def: The name and type of the API to search.
            The name must be the name of the field of the original dataset,
            while the Type must be the one the API object.
            For instance, for :class:`LvisDataset` is `('lvis_api', lvis.LVIS)`.
            Defaults to the datasets explicitly supported by Avalanche.
        :param default_to_coco: If True, it will try to convert the dataset
            to the COCO format.
        :param save_folder: path to the folder where to write model output
            files. Defaults to None, which means that the model output of
            test instances will not be stored.
        :param filename_prefix: prefix common to all model outputs files.
            Ignored if `save_folder` is None. Defaults to "model_output"
        :param iou_types: list of (or a single string) strings describing
            the iou types to use when computing metrics.
            Defaults to "bbox". Valid values are usually "bbox" and "segm",
            but this may vary depending on the dataset.
        :param summarize_to_stdout: if True, a summary of evaluation metrics
            will be printed to stdout (as a table) using the Lvis API.
            Defaults to True.
        """
        super().__init__()

        if save_folder is not None:
            os.makedirs(save_folder, exist_ok=True)

        if isinstance(iou_types, str):
            iou_types = [iou_types]

        self.save_folder = save_folder
        """
        The folder to use when storing the model outputs.
        """

        self.filename_prefix = filename_prefix
        """
        The file name prefix to use when storing the model outputs.
        """

        self.save_stream = save_stream
        """
        The stream for which the model outputs should be saved.
        """

        self.iou_types = iou_types
        """
        The IoU types for which metrics will be computed.
        """

        self.summarize_to_stdout = summarize_to_stdout
        """
        If True, a summary of evaluation metrics will be printed to stdout.
        """

        self.evaluator_factory = evaluator_factory
        """
        The factory of the evaluator object.
        """

        self.evaluator: Optional[
            DetectionEvaluator[TDetPredictions_co, TDetModelOutput]
        ] = None
        """
        Main evaluator object to compute metrics.
        """

        self.gt_api_def = gt_api_def
        """
        The name and type of the dataset API object containing the ground
        truth test annotations.
        """

        self.default_to_coco = default_to_coco
        """
        If True, it will try to convert the dataset to the COCO format.
        """

        self.current_filename: Optional[Union[str, Path]] = None
        """
        File containing the current model outputs.
        """

        self.current_outputs: List[TDetModelOutput] = []
        """
        List of dictionaries containing the current model outputs.
        """

        self.current_additional_metrics = None
        """
        The current additional metrics. Computed after each eval experience.
        May be None if the evaluator doesn't support additional metrics.
        """

        self.save = save_folder is not None
        """
        If True, model outputs will be written to file.
        """

    def reset(self) -> None:
        self.current_outputs = []
        self.current_filename = None
        self.evaluator = None
        self.current_additional_metrics = None

    def initialize_evaluator(self, dataset: Any):
        detection_api = get_detection_api_from_dataset(
            dataset,
            supported_types=self.gt_api_def,
            default_to_coco=self.default_to_coco,
        )
        self.evaluator = self.evaluator_factory(detection_api, self.iou_types)

    def update(self, res: TDetModelOutput):
        self._check_evaluator()
        if self.save:
            self.current_outputs.append(res)

        self.evaluator.update(res)  # type: ignore

    def result(self):
        self._check_evaluator()
        # result_dict may be None if not running in the main process
        result_dict = self.evaluator.evaluate()  # type: ignore
        if result_dict is not None and self.summarize_to_stdout:
            self.evaluator.summarize()  # type: ignore

        if isinstance(result_dict, tuple):
            result, self.current_additional_metrics = result_dict
        else:
            result = result_dict

        return result

    def before_eval_exp(self, strategy) -> None:
        assert strategy.experience is not None

        self.reset()
        self.initialize_evaluator(strategy.experience.dataset)
        if self.save:
            self.current_filename = self._get_filename(strategy)

    def after_eval_iteration(  # type: ignore[override]
        self, strategy: "ObjectDetectionTemplate"
    ):
        assert strategy.detection_predictions is not None
        self.update(strategy.detection_predictions)

    def after_eval_exp(  # type: ignore[override]
        self, strategy: "ObjectDetectionTemplate"
    ):
        assert strategy.experience is not None
        if self.save and strategy.experience.origin_stream.name == self.save_stream:
            assert self.current_filename is not None, (
                "The current_filename field is None, which may happen if the "
                "`before_eval_exp` was not properly invoked."
            )

            with open(self.current_filename, "w") as f:
                json.dump(self.current_outputs, f, cls=TensorEncoder)

        packaged_results = self._package_result(strategy)
        return packaged_results

    def _package_result(self, strategy):
        base_metric_name = get_metric_name(
            self, strategy, add_experience=True, add_task=False
        )
        plot_x_position = strategy.clock.train_iterations
        result_dict = self.result()

        if result_dict is None:
            return

        metric_values = []
        for iou, iou_dict in result_dict.items():
            for metric_key, metric_value in iou_dict.items():
                metric_name = base_metric_name + f"/{iou}/{metric_key}"
                metric_values.append(
                    MetricValue(self, metric_name, metric_value, plot_x_position)
                )

        return metric_values

    def _get_filename(self, strategy) -> Union[str, Path]:
        """e.g. prefix_eval_exp0.json"""
        middle = "_eval_exp"
        if self.filename_prefix == "":
            middle = middle[1:]
        return os.path.join(
            self.save_folder,
            f"{self.filename_prefix}{middle}"
            f"{strategy.experience.current_experience}.json",
        )

    def _check_evaluator(self):
        assert self.evaluator is not None, (
            "The evaluator was not initialized. This may happen if you try "
            "to update or obtain results for this metric before the "
            "`before_eval_exp` callback is invoked. If you are using this "
            "metric in a standalone way, you can initialize the evaluator "
            "by calling `initialize_evaluator` instead."
        )

    def __str__(self):
        return "DetectionMetrics"


def lvis_evaluator_factory(lvis_gt: LVIS, iou_types: List[str]):
    from avalanche.evaluation.metrics.detection_evaluators.lvis_evaluator import (
        LvisEvaluator,
    )

    return LvisEvaluator(lvis_gt=lvis_gt, iou_types=iou_types)


def make_lvis_metrics(
    save_folder=None,
    filename_prefix="model_output",
    iou_types: Union[str, List[str]] = "bbox",
    summarize_to_stdout: bool = True,
    evaluator_factory: Callable[
        [Any, List[str]], DetectionEvaluator
    ] = lvis_evaluator_factory,
    gt_api_def: Sequence[SupportedDatasetApiDef] = DEFAULT_SUPPROTED_DETECTION_DATASETS,
):
    """
    Returns an instance of :class:`DetectionMetrics` initialized for the LVIS
    dataset.

    :param save_folder: path to the folder where to write model output
            files. Defaults to None, which means that the model output of
            test instances will not be stored.
    :param filename_prefix: prefix common to all model outputs files.
        Ignored if `save_folder` is None. Defaults to "model_output"
    :param iou_types: list of (or a single string) strings describing
        the iou types to use when computing metrics.
        Defaults to "bbox". Valid values are "bbox" and "segm".
    :param summarize_to_stdout: if True, a summary of evaluation metrics
        will be printed to stdout (as a table) using the Lvis API.
        Defaults to True.
    :param evaluator_factory: Defaults to :class:`LvisEvaluator` constructor.
    :param gt_api_def: Defaults to the list of supported datasets (LVIS is
        supported in Avalanche through class:`LvisDataset`).
    :return: A metric plugin that can compute metrics on the LVIS dataset.
    """
    return DetectionMetrics(
        evaluator_factory=evaluator_factory,
        gt_api_def=gt_api_def,
        save_folder=save_folder,
        filename_prefix=filename_prefix,
        iou_types=iou_types,
        summarize_to_stdout=summarize_to_stdout,
    )


def get_detection_api_from_dataset(
    dataset,
    supported_types: Sequence[
        Tuple["str", Union[Type, Tuple[Type]]]
    ] = DEFAULT_SUPPROTED_DETECTION_DATASETS,
    default_to_coco: bool = True,
    none_if_not_found=False,
):
    """
    Adapted from:
    https://github.com/pytorch/vision/blob/main/references/detection/engine.py

    :param dataset: The test dataset.
    :param supported_types: The supported API types
    :param default_to_coco: If True, if no API object can be found, the dataset
        will be converted to COCO.
    :param none_if_not_found: If True, it will return None if no valid
        detection API object is found. Else, it will consider `default_to_coco`
        or will raise an error.

    :return: The detection object.
    """

    recursion_result = None
    if isinstance(dataset, Subset):
        recursion_result = get_detection_api_from_dataset(
            dataset.dataset, supported_types, none_if_not_found=True
        )
    elif isinstance(dataset, AvalancheDataset) and len(dataset._datasets) == 1:
        recursion_result = get_detection_api_from_dataset(
            dataset._datasets[0], supported_types, none_if_not_found=True
        )
    elif isinstance(dataset, (AvalancheDataset, ConcatDataset)):
        if isinstance(dataset, AvalancheDataset):
            datasets_list = dataset._datasets
        else:
            datasets_list = dataset.datasets

        for dataset in datasets_list:
            res = get_detection_api_from_dataset(
                dataset, supported_types, none_if_not_found=True
            )
            if res is not None:
                recursion_result = res
                break

    if recursion_result is not None:
        return recursion_result

    for supported_n, supported_t in supported_types:
        candidate_api = getattr(dataset, supported_n, None)
        if candidate_api is not None:
            if isinstance(candidate_api, supported_t):
                return candidate_api
    if none_if_not_found:
        return None
    elif default_to_coco:
        return convert_to_coco_api(dataset)
    else:
        raise ValueError("Could not find a valid dataset API object")


def convert_to_coco_api(ds):
    """
    Adapted from:
    https://github.com/pytorch/vision/blob/main/references/detection/coco_utils.py
    """
    coco_ds = COCO()
    # annotation IDs need to start at 1, not 0, see torchvision issue #1530
    ann_id = 1
    dataset: Dict[str, List[Any]] = {"images": [], "categories": [], "annotations": []}
    categories = set()
    for img_idx in range(len(ds)):
        img_dict = {}

        # find better way to get target
        # targets = ds.get_annotations(img_idx)
        img, targets, *_ = ds[img_idx]
        img_dict["height"] = img.shape[-2]
        img_dict["width"] = img.shape[-1]
        image_id = targets["image_id"].item()
        img_dict["id"] = image_id
        dataset["images"].append(img_dict)
        bboxes = targets["boxes"].clone()
        bboxes[:, 2:] -= bboxes[:, :2]
        bboxes = bboxes.tolist()
        labels = targets["labels"].tolist()
        areas = targets["area"].tolist()
        iscrowd = targets["iscrowd"].tolist()
        if "masks" in targets:
            masks = targets["masks"]
            # make masks Fortran contiguous for coco_mask
            masks = masks.permute(0, 2, 1).contiguous().permute(0, 2, 1)
        if "keypoints" in targets:
            keypoints = targets["keypoints"]
            keypoints = keypoints.reshape(keypoints.shape[0], -1).tolist()
        num_objs = len(bboxes)
        for i in range(num_objs):
            ann = {}
            ann["image_id"] = image_id
            ann["bbox"] = bboxes[i]
            ann["category_id"] = labels[i]
            categories.add(labels[i])
            ann["area"] = areas[i]
            ann["iscrowd"] = iscrowd[i]
            ann["id"] = ann_id
            if "masks" in targets:
                ann["segmentation"] = coco_mask.encode(masks[i].numpy())
            if "keypoints" in targets:
                ann["keypoints"] = keypoints[i]
                ann["num_keypoints"] = sum(k != 0 for k in keypoints[i][2::3])
            dataset["annotations"].append(ann)
            ann_id += 1
    dataset["categories"] = [{"id": i} for i in sorted(categories)]
    coco_ds.dataset = dataset
    coco_ds.createIndex()
    return coco_ds


__all__ = [
    "TCommonDetectionOutput",
    "DetectionEvaluator",
    "DetectionMetrics",
    "make_lvis_metrics",
    "get_detection_api_from_dataset",
    "convert_to_coco_api",
]
