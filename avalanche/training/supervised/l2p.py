from typing import Callable, List, Optional, Union

import numpy as np
import torch
import torch.nn as nn
from avalanche.training.plugins import EvaluationPlugin
from avalanche.training.plugins.strategy_plugin import SupervisedPlugin
from avalanche.training.plugins.evaluation import default_evaluator
from avalanche.training.templates import SupervisedTemplate
from avalanche.models.vit import create_model


class LearningToPrompt(SupervisedTemplate):
    """
    Learning to Prompt (L2P) strategy.

    Technique introduced in:
    "Wang, Zifeng, et al. "Learning to prompt for continual learning."
    Proceedings of the IEEE/CVF Conference on Computer Vision and
    Pattern Recognition. 2022."

    Implementation based on:
    - https://github.com/JH-LEE-KR/l2p-pytorch
    - And implementations by Dario Salvati

    As a model_name, we expect to receive one of the model list in
    avalanche.models.vit

    Those models are based on the library timm.
    """

    def __init__(
        self,
        model_name: str,
        criterion: nn.Module = nn.CrossEntropyLoss(),
        train_mb_size: int = 1,
        train_epochs: int = 1,
        eval_mb_size: Optional[int] = 1,
        device: Union[str, torch.device] = "cpu",
        plugins: Optional[List["SupervisedPlugin"]] = None,
        evaluator: Union[
            EvaluationPlugin, Callable[[], EvaluationPlugin]
        ] = default_evaluator,
        eval_every: int = -1,
        peval_mode: str = "epoch",
        prompt_pool: bool = True,
        pool_size: int = 20,
        prompt_length: int = 5,
        top_k: int = 5,
        lr: float = 0.03,
        sim_coefficient: float = 0.1,
        prompt_key: bool = True,
        pretrained: bool = True,
        num_classes: int = 10,
        drop_rate: float = 0.0,
        drop_path_rate: float = 0.0,
        embedding_key: str = "cls",
        prompt_init: str = "uniform",
        batchwise_prompt: bool = False,
        head_type: str = "prompt",
        use_prompt_mask: bool = False,
        train_prompt_mask: bool = False,
        use_cls_features: bool = True,
        use_mask: bool = True,
        use_vit: bool = True,
        **kwargs,
    ):
        """Init.

        :param model_name: Name of the model to use. For a complete list check \
            models.vit.py
        :param criterion: Loss functions used during training. \
            Default CrossEntropyLoss.
        :param train_mb_size: The train minibatch size. Defaults to 1.
        :param train_epochs: The number of training epochs. Defaults to 1.
        :param eval_mb_size: The eval minibatch size. Defaults to 1.
        :param device: The device to use. Defaults to None (cpu).
        :param plugins: Plugins to be added. Defaults to None.
        :param evaluator: (optional) instance of EvaluationPlugin for logging
            and metric computations.
        :param eval_every: the frequency of the calls to `eval` inside the
            training loop. -1 disables the evaluation. 0 means `eval` is called
            only at the end of the learning experience. Values >0 mean that
            `eval` is called every `eval_every` epochs and at the end of the
            learning experience.
        :param use_cls_features: Use an external pre-trained model to obtained\
             features to obtained the prompts.
        :param use_mask: Use mask to train only classification rows of the \
            classes of the current task. Default True.
        :param use_vit: Boolean to confirm the usage of a visual Transformer.\
            Default True
        """

        if device is None:
            device = torch.device("cpu")

        self.num_classes = num_classes
        self.lr = lr
        self.sim_coefficient = sim_coefficient
        model = create_model(
            model_name=model_name,
            prompt_pool=prompt_pool,
            pool_size=pool_size,
            prompt_length=prompt_length,
            top_k=top_k,
            prompt_key=prompt_key,
            pretrained=pretrained,
            num_classes=num_classes,
            drop_rate=drop_rate,
            drop_path_rate=drop_path_rate,
            embedding_key=embedding_key,
            prompt_init=prompt_init,
            batchwise_prompt=batchwise_prompt,
            head_type=head_type,
            use_prompt_mask=use_prompt_mask,
        )

        for n, p in model.named_parameters():
            if n.startswith(
                tuple(["blocks", "patch_embed", "cls_token", "norm", "pos_embed"])
            ):
                p.requires_grad = False

        model.head = torch.nn.Linear(768, num_classes).to(device)

        optimizer = torch.optim.Adam(
            model.parameters(),
            betas=(0.9, 0.999),
            lr=self.lr,
        )

        super().__init__(
            model,
            optimizer,
            criterion,
            train_mb_size,
            train_epochs,
            eval_mb_size,
            device,
            plugins,
            evaluator,
            eval_every,
            peval_mode,
        )

        self._criterion = criterion
        self.use_cls_features = use_cls_features
        self.train_prompt_mask = train_prompt_mask
        self.use_mask = use_mask
        self.use_vit = use_vit

        if use_cls_features:
            self.original_vit = create_model(
                model_name=model_name,
                pretrained=pretrained,
                num_classes=num_classes,
                drop_rate=drop_rate,
                drop_path_rate=drop_path_rate,
            ).to(device)

            self.original_vit.reset_classifier(0)

            for p in self.original_vit.parameters():
                p.requires_grad = False

    def _before_training_exp(self, **kwargs):
        super()._before_training_exp(**kwargs)
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            betas=(0.9, 0.999),
            lr=self.lr,
        )

    def forward(self):
        assert self.experience is not None
        if self.use_cls_features:
            with torch.no_grad():
                cls_features = self.original_vit(self.mb_x)["pre_logits"]
        else:
            cls_features = None

        if self.use_vit:
            self.res = self.model(
                x=self.mb_x,
                task_id=self.mb_task_id,
                cls_features=cls_features,
                train=self.train_prompt_mask,
            )
        else:
            self.res = {}
            self.res["logits"] = self.model(x=self.mb_x)
            self.res["reduce_sim"] = 0

        logits = self.res["logits"]

        if self.use_mask and self.is_training:
            mask = self.experience.classes_in_this_experience
            not_mask = np.setdiff1d(np.arange(self.num_classes), mask)
            not_mask = torch.tensor(not_mask, dtype=torch.int64).to(self.device)
            logits = logits.index_fill(dim=1, index=not_mask, value=float("-inf"))

        return logits

    def criterion(self):
        loss = self._criterion(self.mb_output, self.mb_y)
        loss = loss - self.sim_coefficient * self.res["reduce_sim"]
        return loss
