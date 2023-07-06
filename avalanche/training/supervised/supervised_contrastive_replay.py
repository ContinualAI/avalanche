from typing import Optional, Sequence

import torch
from torch.nn import Module
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, Lambda

from avalanche.core import BaseSGDPlugin
from avalanche.models import SCRModel
from avalanche.training.losses import SCRLoss
from avalanche.training.plugins import ReplayPlugin
from avalanche.training.plugins.evaluation import default_evaluator
from avalanche.training.storage_policy import ClassBalancedBuffer
from avalanche.training.templates import SupervisedTemplate


class SCR(SupervisedTemplate):
    """
    Supervised Contrastive Replay from https://arxiv.org/pdf/2103.13885.pdf.
    This strategy trains an encoder network in a self-supervised manner to
    cluster together examples of the same class while pushing away examples
    of different classes. It uses the Nearest Class Mean classifier on the
    embeddings produced by the encoder.

    Accuracy cannot be monitored during training (no NCM classifier).
    During training, NCRLoss is monitored, while during eval
    CrossEntropyLoss is monitored.

    The original paper uses an additional fine-tuning phase on the buffer
    at the end of each experience (called review trick, but not mentioned
    in the paper). This implementation does not implement the review trick.
    """

    def __init__(
        self,
        model: SCRModel,
        optimizer: Optimizer,
        augmentations=Compose([Lambda(lambda el: el)]),
        mem_size: int = 100,
        temperature: int = 0.1,
        train_mb_size: int = 1,
        batch_size_mem: int = 100,
        train_epochs: int = 1,
        eval_mb_size: Optional[int] = 1,
        device="cpu",
        plugins: Optional[Sequence["BaseSGDPlugin"]] = None,
        evaluator=default_evaluator,
        eval_every=-1,
        peval_mode="epoch",
    ):
        """
        :param model: an Avalanche model like the avalanche.models.SCRModel,
            where the train classifier uses a projection network (e.g., MLP)
            while the test classifier uses a NCM Classifier.
            Normalization should be applied between feature extractor
            and classifier.
        :param optimizer: PyTorch optimizer.
        :param augmentations: TorchVision Compose Transformations to augment
            the input minibatch. The augmented mini-batch will be concatenated
            to the original one (which includes the memory buffer).
            Note: only augmentations that can be applied to Tensors
            are supported.
        :param mem_size: replay memory size, used also at test time to
            compute class means.
        :param temperature: SCR Loss temperature.
        :param train_mb_size: mini-batch size for training. The default
            dataloader is a task-balanced dataloader that divides each
            mini-batch evenly between samples from all existing tasks in
            the dataset.
        :param batch_size_mem: number of examples drawn from the buffer.
        :param train_epochs: number of training epochs.
        :param eval_mb_size: mini-batch size for eval.
        :param device: PyTorch device where the model will be allocated.
        :param plugins: (optional) list of StrategyPlugins.
        :param evaluator: (optional) instance of EvaluationPlugin for logging
            and metric computations. None to remove logging.
        :param eval_every: the frequency of the calls to `eval` inside the
            training loop. -1 disables the evaluation. 0 means `eval` is called
            only at the end of the learning experience. Values >0 mean that
            `eval` is called every `eval_every` epochs and at the end of the
            learning experience.
        :param peval_mode: one of {'epoch', 'iteration'}. Decides whether the
            periodic evaluation during training should execute every
            `eval_every` epochs or iterations (Default='epoch').
        """

        if not isinstance(model, SCRModel):
            raise ValueError(
                "Supervised Contrastive Replay model "
                "needs to be an instance of avalanche.models.SCRModel."
            )

        self.replay_plugin = ReplayPlugin(
            mem_size,
            batch_size=train_mb_size,
            batch_size_mem=batch_size_mem,
            storage_policy=ClassBalancedBuffer(max_size=mem_size),
        )

        self.augmentations = augmentations
        self.temperature = temperature

        self.train_loss = SCRLoss(temperature=self.temperature)
        self.eval_loss = torch.nn.CrossEntropyLoss()

        if plugins is None:
            plugins = [self.replay_plugin]
        elif isinstance(plugins, list):
            plugins = [self.replay_plugin] + plugins
        else:
            raise ValueError("`plugins` parameter needs to be a list.")
        super().__init__(
            model,
            optimizer,
            SCRLoss(temperature=self.temperature),
            train_mb_size,
            train_epochs,
            eval_mb_size,
            device,
            plugins,
            evaluator,
            eval_every,
            peval_mode,
        )

    def criterion(self):
        if self.is_training:
            return self.train_loss(self.mb_output, self.mb_y)
        else:
            return self.eval_loss(self.mb_output, self.mb_y)

    def _before_forward(self, **kwargs):
        """
        Concatenate together original and augmented examples.
        """
        assert self.is_training
        super()._before_forward(**kwargs)
        mb_x_augmented = self.augmentations(self.mbatch[0])
        # (batch_size*2, input_size)
        self.mbatch[0] = torch.cat([self.mbatch[0], mb_x_augmented], dim=0)

    def _after_forward(self, **kwargs):
        """
        Reshape the model output to have 2 views: one for original examples,
        one for augmented examples.
        """
        assert self.is_training
        super()._after_forward(**kwargs)
        assert self.mb_output.size(0) % 2 == 0
        original_batch_size = int(self.mb_output.size(0) / 2)
        original_examples = self.mb_output[:original_batch_size]
        augmented_examples = self.mb_output[original_batch_size:]
        # (original_batch_size, 2, output_size)
        self.mb_output = torch.stack([original_examples, augmented_examples], dim=1)

    def _after_training_exp(self, **kwargs):
        """Update NCM means"""
        super()._after_training_exp(**kwargs)
        self.model.eval()
        self.compute_class_means()
        self.model.train()

    @torch.no_grad()
    def compute_class_means(self):
        class_means = {}

        # for each class
        for dataset in self.replay_plugin.storage_policy.buffer_datasets:
            dl = DataLoader(
                dataset.eval(),
                shuffle=False,
                batch_size=self.eval_mb_size,
                drop_last=False,
            )
            num_els = 0
            # for each mini-batch in each class
            for x, y, _ in dl:
                num_els += x.size(0)
                # class-balanced buffer, label is the same across mini-batch
                label = y[0].item()
                out = self.model.feature_extractor(x.to(self.device))
                out = torch.nn.functional.normalize(out, p=2, dim=1)
                if label in class_means:
                    class_means[label] += out.sum(0).cpu().detach().clone()
                else:
                    class_means[label] = out.sum(0).cpu().detach().clone()
            class_means[label] /= float(num_els)
            class_means[label] /= class_means[label].norm()

        self.model.eval_classifier.update_class_means_dict(class_means)
