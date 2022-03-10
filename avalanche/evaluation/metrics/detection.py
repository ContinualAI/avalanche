import os
import torch
import json
from typing import Any, List, Union
from torch import Tensor
from json import JSONEncoder
from avalanche.evaluation import PluginMetric
from avalanche.evaluation.metric_results import MetricValue
from avalanche.evaluation.metric_utils import get_metric_name
from examples.tvdetection.lvis_eval import LvisEvaluator
from examples.tvdetection.engine import get_detection_api_from_dataset


class TensorEncoder(JSONEncoder):
    def __init__(self, **kwargs):
        super(TensorEncoder, self).__init__(**kwargs)

    def default(self, o: Any) -> Any:
        if isinstance(o, Tensor):
            o = o.detach().cpu().tolist()

        return o


def tensor_decoder(dct):
    for t_name in ['boxes', 'mask', 'scores', 'keypoints', 'labels']:
        if t_name in dct:
            if t_name == 'labels':
                dct[t_name] = torch.as_tensor(dct[t_name], dtype=torch.int64)
            else:
                dct[t_name] = torch.as_tensor(dct[t_name])

            if t_name == 'boxes':
                dct[t_name] = torch.reshape(dct[t_name], shape=(-1, 4))
            # TODO: implement mask shape

    return dct


class LvisMetrics(PluginMetric[dict]):
    """
    Metric used to compute the metrics using the Lvis API.

    Metrics are returned after each evaluation experience.

    This metric can also be used to serialize model outputs to JSON files,
    by producing one file for each evaluation experience. This can be useful
    if outputs have to been processed later (like in a competition).
    """

    def __init__(
            self,
            *,
            save_folder=None,
            filename_prefix='model_output',
            iou_types: Union[str, List[str]] = 'bbox',
            summarize_to_stdout: bool = True):
        """
        Creates an instance of LvisMetrics.

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

        self.iou_types = iou_types
        """
        The IoU types for which metrics will be computed.
        """

        self.summarize_to_stdout = summarize_to_stdout
        """
        If True, a summary of evaluation metrics will be printed to stdout.
        """

        self.lvis_evaluator = None
        """
        Main LVIS evaluator object to compute metrics.
        """

        self.current_filename = None
        """
        File containing current model dump.
        """

        self.current_outputs = []
        """
        List of dictionaries containing the current model outputs.
        """

        self.save = save_folder is not None
        """
        If True, model outputs will be written to file.
        """

    def reset(self) -> None:
        self.current_outputs = []
        self.current_filename = None

    def update(self, res):
        if self.save:
            self.current_outputs.append(res)
        self.lvis_evaluator.update(res)

    def result(self):
        if self.save:
            with open(self.current_filename, 'w') as f:
                json.dump(self.current_outputs, f, cls=TensorEncoder)

        self.lvis_evaluator.evaluate()
        if self.summarize_to_stdout:
            self.lvis_evaluator.summarize()

        result_dict = dict()

        # Encode metrics in CodaLab output format
        for iou, eval_data in self.lvis_evaluator.lvis_eval_per_iou.items():
            result_dict[iou] = dict()
            for key in eval_data.results:
                value = eval_data.results[key]
                result_dict[iou][key] = value

        return result_dict

    def before_eval_exp(self, strategy) -> None:
        super().before_eval_exp(strategy)

        self.reset()
        lvis_api = get_detection_api_from_dataset(
            strategy.experience.dataset)
        self.lvis_evaluator = LvisEvaluator(lvis_api, self.iou_types)
        if self.save:
            self.current_filename = self._get_filename(strategy)

    def after_eval_iteration(self, strategy) -> None:
        super().after_eval_iteration(strategy)
        self.update(strategy.detection_predictions)

    def after_eval_exp(self, strategy):
        super().after_eval_exp(strategy)
        return self._package_result(strategy)

    def _package_result(self, strategy):
        base_metric_name = get_metric_name(
            self, strategy, add_experience=True, add_task=False)
        plot_x_position = strategy.clock.train_iterations
        result_dict = self.result()

        metric_values = []
        for iou, iou_dict in result_dict.items():
            for metric_key, metric_value in iou_dict.items():
                metric_name = base_metric_name + f'/{iou}/{metric_key}'
                metric_values.append(
                    MetricValue(
                        self,
                        metric_name,
                        metric_value,
                        plot_x_position
                    )
                )

        return metric_values

    def _get_filename(self, strategy):
        """e.g. prefix_eval_exp0.json"""
        middle = '_eval_exp'
        if self.filename_prefix == '':
            middle = middle[1:]
        return os.path.join(
            self.save_folder,
            f"{self.filename_prefix}{middle}"
            f"{strategy.experience.current_experience}.json")

    def __str__(self):
        return "LvisMetrics"


__all__ = [
    "LvisMetrics"
]
