################################################################################
# Copyright (c) 2022 ContinualAI.                                              #
# Copyrights licensed under the MIT License.                                   #
# See the accompanying LICENSE file for terms.                                 #
#                                                                              #
# Date: 14-02-2022                                                             #
# Author(s): Lorenzo Pellegrini, Antonio Carta                                 #
# E-mail: contact@continualai.org                                              #
# Website: avalanche.continualai.org                                           #
################################################################################
"""
Main script for the CLVision challenge - object detection tracks.
https://sites.google.com/view/clvision2022/challenge?authuser=0
"""
import logging
import math
import sys

from avalanche.models.utils import avalanche_model_adaptation
from examples.detection import split_lvis

sys.path.append('.')

from avalanche.evaluation.metrics import ElapsedTime, LvisMetrics
from avalanche.logging import BaseLogger, InteractiveLogger
from avalanche.training.plugins import LRSchedulerPlugin, EvaluationPlugin
from avalanche.training.templates import BaseSGDTemplate
from examples.tvdetection.utils import SmoothedValue, MetricLogger, reduce_dict
import argparse
import torch
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor


# This sets the root logger to write to stdout (your console).
# Your script/app needs to call this somewhere at least once.
logging.basicConfig(level=logging.NOTSET)


class NaiveObjectDetection(BaseSGDTemplate):
    def __init__(self, scaler=None,
                 **base_kwargs):
        super().__init__(**base_kwargs)
        self.scaler = scaler  # torch.cuda.amp.autocast scaler
        self._images = None
        self._targets = None

        # Object Detection attributes
        self.losses = None
        self.loss_dict = None
        self.res = None  # only for eval loop.

    def make_train_dataloader(self, num_workers=4, **kwargs):
        """Assign dataloader to self.dataloader."""
        self.dataloader = DataLoader(
            self.experience.dataset,
            batch_size=self.train_mb_size,
            shuffle=True,
            drop_last=True,
            num_workers=num_workers,
            collate_fn=detection_collate_fn
        )

    def make_eval_dataloader(self, num_workers=4, **kwargs):
        """Assign dataloader to self.dataloader."""
        self.dataloader = DataLoader(
            self.experience.dataset,
            batch_size=self.eval_mb_size,
            shuffle=False,
            drop_last=False,
            num_workers=num_workers,
            collate_fn=detection_collate_fn
        )

    def make_optimizer(self, **kwargs):
        """Optimizer initialization."""
        # TODO:
        pass  # keep the previous optimizer as is.

    def criterion(self):
        """Compute loss function."""
        if self.is_training:
            self.losses = sum(loss for loss in self.loss_dict.values())
            # reduce losses over all GPUs for logging purposes
            loss_dict_reduced = reduce_dict(self.loss_dict)
            losses_reduced = sum(loss for loss in loss_dict_reduced.values())
            loss_value = losses_reduced.item()

            if not math.isfinite(loss_value):
                print(f"Loss is {loss_value}, stopping training")
                print(loss_dict_reduced)
                sys.exit(1)
            return self.losses
        else:
            # eval does not compute the loss directly.
            # metrics can use self.mb_output and self.res
            self.res = {target["image_id"].item(): output
                        for target, output in zip(self.targets, self.mb_output)}
            return self.res

    def forward(self):
        """Compute the model's output given the current mini-batch."""
        if self.is_training:
            with torch.cuda.amp.autocast(enabled=self.scaler is not None):
                self.loss_dict = self.model(self.images, self.targets)
            return self.loss_dict
        else:
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            outs = self.model(self.images)
            return [{k: v.to('cpu') for k, v in t.items()} for t in outs]

    def _after_eval_forward(self, **kwargs):
        pass

    def model_adaptation(self, model=None):
        """Adapts the model to the current experience."""
        if model is None:
            model = self.model
        avalanche_model_adaptation(model, self.experience.dataset)
        return model.to(self.device)

    @property
    def images(self):
        return self._images

    @property
    def targets(self):
        return self._targets

    def _unpack_minibatch(self,):
        images = self.mbatch[0]
        targets = self.mbatch[1]
        self._images = list(image.to(self.device) for image in images)
        self._targets = [{k: v.to(self.device) for k, v in t.items()} for t in
                         targets]

    def backward(self):
        if self.scaler is not None:
            self.scaler.scale(self.losses).backward()
        else:
            self.losses.backward()

    def optimizer_step(self, **kwargs):
        if self.scaler is not None:
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            self.optimizer.step()


class ModelCheckpoint:
    def after_training_exp(self, strategy):
        """Save the model after each experience."""
        torch.save({
            'epoch': 0,
            'model_state_dict': strategy.model.state_dict(),
            'optimizer_state_dict': strategy.optimizer.state_dict(),
        }, 'model_checkpoint.pth')


class LVISLogger(BaseLogger):
    """TODO: complete logger base on train_one_epoch and `lvsi_evaluate`
    calls."""

    def __init__(self, print_freq):
        super().__init__()
        self.print_freq = print_freq

        self.metric_logger = MetricLogger(delimiter="  ")
        self.metric_logger.add_meter("lr", SmoothedValue(window_size=1,
                                                         fmt="{value:.6f}"))
        # self.header = f"Epoch: [{epoch}]"
        # TODO: LVIS eval mode. See `evaluate`.

    def after_train_iteration(self, strategy, metric_values):
        if strategy.clock.train_exp_iterations % self.print_freq == 0:
            ...
            # TODO: print every print_freq. See MetricLogger.log_every

    def after_train_epoch(self, strategy, metric_values):
        self.metric_logger.update(loss=strategy.losses_reduced, **strategy.loss_dict_reduced)
        self.metric_logger.update(lr=strategy.optimizer.param_groups[0]["lr"])


def main(args):
    # --- CONFIG
    device = torch.device(
        f"cuda:{args.cuda}"
        if torch.cuda.is_available() and args.cuda >= 0
        else "cpu"
    )
    # ---------

    # --- TRANSFORMATIONS
    train_transform = ToTensor()
    test_transform = ToTensor()
    # ---------

    # --- SCENARIO CREATION
    torch.random.manual_seed(1234)
    n_exps = 100  # Keep it high to run a short exp
    benchmark = split_lvis(
        n_experiences=n_exps,
        train_transform=train_transform,
        eval_transform=test_transform)
    # ---------

    # MODEL CREATION
    # load a model pre-trained on COCO
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(
        pretrained=True)

    # Just tune the box predictor
    for p in model.parameters():
        p.requires_grad = False

    # replace the classifier with a new one, that has
    # num_classes which is user-defined
    num_classes = benchmark.n_classes + 1  # N classes + background
    # get number of input features for the classifier
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    # replace the pre-trained head with a new one
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    model = model.to(device)

    # CREATE THE STRATEGY INSTANCE (NAIVE)
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=0.005,
                                momentum=0.9, weight_decay=0.0005)

    train_mb_size = 5
    warmup_factor = 1.0 / 1000
    warmup_iters = min(1000, len(benchmark.train_stream[0].dataset) // train_mb_size - 1)
    lr_scheduler = torch.optim.lr_scheduler.LinearLR(
        optimizer, start_factor=warmup_factor, total_iters=warmup_iters
    )

    lvis_logger = LVISLogger(10)
    cl_strategy = NaiveObjectDetection(
        model=model,
        optimizer=optimizer,
        train_mb_size=train_mb_size,
        train_epochs=1,
        eval_mb_size=train_mb_size,
        device=device,
        plugins=[
            ModelCheckpoint(),
            LRSchedulerPlugin(lr_scheduler)
        ],
        evaluator=EvaluationPlugin(
            ElapsedTime(),
            LvisMetrics(save_folder='./model_outputs'),
            loggers=[lvis_logger, InteractiveLogger()])
    )

    print('Learnable parameters:')
    for n, p in model.named_parameters():
        if p.requires_grad:
            print(n)

    # TRAINING LOOP
    print("Starting experiment...")
    for i, experience in enumerate(benchmark.train_stream):
        print("Start of experience: ", experience.current_experience)
        print('Dataset contains', len(experience.dataset), 'instances')

        cl_strategy.train(experience)
        print("Training completed")

        cl_strategy.eval(benchmark.test_stream[0])
        # evaluate(model, data_loader, device=device)


def detection_collate_fn(batch):
    return tuple(zip(*batch))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--cuda",
        type=int,
        default=0,
        help="Select zero-indexed cuda device. -1 to use CPU.",
    )
    args = parser.parse_args()
    main(args)
