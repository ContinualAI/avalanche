from avalanche.benchmarks.utils.data_loader import ReplayDataLoader
from avalanche.benchmarks.utils import AvalancheDataset
from avalanche.core import SupervisedPlugin
from avalanche.training.templates.supervised import SupervisedTemplate
import torch


class GenerativeReplayPlugin(SupervisedPlugin):
    """
    Experience generative replay plugin.

    Updates the Dataloader of a strategy before training an experience
    by sampling a generator model and weaving the replay data into
    the original training data. 

    The examples in the created mini-batch contain one part of the original data
    and one part of generative data for each class 
    that has been encountered so far.

    :param batch_size: the size of the data batch. If set to `None`, it
        will be set equal to the strategy's batch size.
    :param batch_size_mem: the size of the memory batch. If
        `task_balanced_dataloader` is set to True, it must be greater than or
        equal to the number of tasks. If its value is set to `None`
        (the default value), it will be automatically set equal to the
        data batch size.
    :param task_balanced_dataloader: if True, buffer data loaders will be
            task-balanced, otherwise it will create a single dataloader for the
            buffer samples.
    :param untrained_solver: if True we assume this is the beginning of 
        a continual learning task and add replay data only from the second 
        experience onwards, otherwise we sample and add generative replay data
        before training the first experience. Default to True.
    """

    def __init__(self, generator=None, mem_size: int = 200, 
                 batch_size: int = None,
                 batch_size_mem: int = None,
                 task_balanced_dataloader: bool = False,
                 untrained_solver: bool = True):
        super().__init__()
        self.mem_size = mem_size
        self.batch_size = batch_size
        self.batch_size_mem = batch_size_mem
        self.task_balanced_dataloader = task_balanced_dataloader
        self.generator_strategy = generator
        if self.generator_strategy:
            self.generator = generator.model
        else: 
            self.generator = None
        self.untrained_solver = untrained_solver
        self.classes_until_now = []

    def before_training(self, strategy, *args, **kwargs):
        """Checks whether we are using a user defined external generator 
        or we use the strategy's model as the generator. 
        If the generator is None after initialization 
        we assume that strategy.model is the generator."""
        if not self.generator_strategy:
            self.generator_strategy = strategy
            self.generator = strategy.model

    def before_training_exp(self, strategy: "SupervisedTemplate",
                            num_workers: int = 0, shuffle: bool = True,
                            **kwargs):
        """
        ReplayDataloader to build batches containing examples from both, 
        data sampled from the generator and the training dataset.
        """
        self.classes_until_now.append(
            strategy.experience.classes_in_this_experience)

        if self.untrained_solver:
            # The solver needs to be trained to label generated data
            # the generator needs to be trained before we can sample.
            self.untrained_solver = False
            return

        # Sample data from generator
        memory = self.generator.generate(
            len(strategy.adapted_dataset) *
            (len(self.classes_until_now)-1)).to(strategy.device)
        # Label the generated data using the current solver model
        strategy.model.eval()
        with torch.no_grad():
            memory_output = strategy.model(memory).argmax(dim=-1)
        strategy.model.train()
        # Create an AvalancheDataset from memory data and labels
        memory = AvalancheDataset(torch.utils.data.TensorDataset(
            memory.detach().cpu(), memory_output.detach().cpu()))

        batch_size = self.batch_size
        if batch_size is None:
            batch_size = strategy.train_mb_size

        batch_size_mem = self.batch_size_mem
        if batch_size_mem is None:
            batch_size_mem = strategy.train_mb_size
        # Update strategy's dataloader by interleaving 
        # current experience's data with generated data.
        strategy.dataloader = ReplayDataLoader(
            strategy.adapted_dataset,
            memory,
            batch_size=batch_size,
            batch_size_mem=batch_size_mem*(len(self.classes_until_now)-1),
            task_balanced_dataloader=self.task_balanced_dataloader,
            num_workers=num_workers,
            shuffle=shuffle)


class VAEPlugin(SupervisedPlugin):
    """
    VAEPlugin which facilitates the conventional training of the models.VAE.

    The VAE's forward call computes the representations in the latent space,
    'after_forward' computes the remaining steps of the classic VAE forward.
    """

    def after_forward(
        self, strategy, *args, **kwargs
    ):
        """
        Compute the reconstruction of the input and posterior distribution.
        """
        strategy.mean, strategy.logvar = strategy.model.calc_mean(
            strategy.mb_output), strategy.model.calc_logvar(strategy.mb_output)
        z = strategy.model.sampling(strategy.mean, strategy.logvar)
        strategy.mb_x_recon = strategy.model.decoder(z)

    def after_eval_forward(
        self, strategy, *args, **kwargs
    ):
        """
        Compute the reconstruction of the input and posterior distribution.
        """
        strategy.mean, strategy.logvar = strategy.model.calc_mean(
            strategy.mb_output), strategy.model.calc_logvar(strategy.mb_output)
        z = strategy.model.sampling(strategy.mean, strategy.logvar)
        strategy.mb_x_recon = strategy.model.decoder(z)


class trainGeneratorPlugin(SupervisedPlugin):
    """
    trainGeneratorPlugin makes sure that after each experience of training 
    the solver of a scholar model, we also train the generator on the data 
    of the current experience.
    """

    def after_training_exp(self, strategy: "SupervisedTemplate", **kwargs):
        """
        The training method expects an Experience object 
        with a 'dataset' parameter.
        """
        strategy.plugins[1].generator_strategy.train(strategy.experience) 
