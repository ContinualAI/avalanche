import os
import torch
import json
from typing import Any
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


class LvisMetrics(PluginMetric[str]):
    """This metric serializes model outputs to JSON files.
    The metric produces one file for each evaluation experience.
    It also returns the metrics computed by LVIS benchmark based
    on model output. Metrics are returned after each
    evaluation experience."""

    def __init__(self, save_folder=None, filename_prefix='model_output',
                 iou_types=['bbox']):
        """
        :param save_folder: path to the folder where to write model output
            files. None to disable writing to file.
        :param filename_prefix: prefix common to all model outputs files
        :param iou_types: list of iou types
        """
        super().__init__()

        if save_folder is not None:
            os.makedirs(save_folder, exist_ok=True)

        self.save_folder = save_folder
        self.filename_prefix = filename_prefix
        self.iou_types = iou_types

        self.lvis_evaluator = None
        """Main LVIS evaluator object to compute metrics"""

        self.current_filename = None
        """File containing current model dump"""

        self.current_outputs = []
        """List of dictionaries containing the current model outputs"""

        self.no_save = save_folder is None
        """If True, no JSON file will be written"""

    def reset(self) -> None:
        self.current_outputs = []
        self.current_filename = None

    def update(self, res):
        if not self.no_save:
            self.current_outputs.append(res)
        self.lvis_evaluator.update(res)

    def result(self):
        if not self.no_save:
            with open(self.current_filename, 'w') as f:
                json.dump(self.current_outputs, f, cls=TensorEncoder)

        self.lvis_evaluator.evaluate()
        self.lvis_evaluator.summarize()
        # Encode metrics in CodaLab output format
        bbox_eval = self.lvis_eval.lvis_eval_per_iou['bbox']
        score_str = ''
        ordered_keys = sorted(bbox_eval.results.keys())
        for key in ordered_keys:
            value = bbox_eval.results[key]
            score_str += '{}: {:.5f}\n'.format(key, value)
        score_str = score_str[:-1]  # Remove final \n
        print("******* ", score_str)
        return score_str

    def before_eval_exp(self, strategy) -> None:
        super().before_eval_exp(strategy)

        self.reset()
        lvis_api = get_detection_api_from_dataset(
            strategy.experience.dataset)
        self.lvis_evaluator = LvisEvaluator(lvis_api, self.iou_types)
        self.current_filename = self._get_filename(strategy)

    def after_eval_iteration(self, strategy) -> None:
        super().after_eval_iteration(strategy)
        self.update(strategy.res)

    def after_eval_exp(self, strategy):
        super().after_eval_exp(strategy)
        return self._package_result(strategy)

    def _package_result(self, strategy):
        metric_name = get_metric_name(self, strategy, add_experience=True,
                                      add_task=False)
        plot_x_position = strategy.clock.train_iterations
        filename = self.result()
        metric_values = [
            MetricValue(self, metric_name, filename, plot_x_position)
        ]
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
