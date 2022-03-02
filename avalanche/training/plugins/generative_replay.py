from avalanche.benchmarks.utils.data_loader import ReplayDataLoader
from avalanche.benchmarks.utils import AvalancheDataset
from avalanche.core import SupervisedPlugin
from avalanche.training.templates.supervised import SupervisedTemplate
import torch


class GenerativeReplayPlugin(SupervisedPlugin):
    """
    Experience replay plugin.

    Handles an external memory filled with randomly selected
    patterns and implementing `before_training_exp` and `after_training_exp`
    callbacks.
    The `before_training_exp` callback is implemented in order to use the
    dataloader that creates mini-batches with examples from both training
    data and external memory. The examples in the mini-batch is balanced
    such that there are the same number of examples for each experience.

    The `after_training_exp` callback is implemented in order to add new
    patterns to the external memory.

    The :mem_size: attribute controls the total number of patterns to be stored
    in the external memory.

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
    :param storage_policy: The policy that controls how to add new exemplars
                           in memory
    """

    def __init__(self, generator, mem_size: int = 200, batch_size: int = None,
                 batch_size_mem: int = None,
                 task_balanced_dataloader: bool = False,
                 untrained_solver: bool = True):
        super().__init__()
        self.mem_size = mem_size
        self.batch_size = batch_size
        self.batch_size_mem = batch_size_mem
        self.task_balanced_dataloader = task_balanced_dataloader
        self.generator_strategy = generator
        self.generator = generator.model
        self.untrained_solver = untrained_solver
        self.classes_until_now = []

    def before_training_exp(self, strategy: "SupervisedTemplate",
                            num_workers: int = 0, shuffle: bool = True,
                            **kwargs):
        """
        Dataloader to build batches containing examples from both memories and
        the training dataset
        """
        self.classes_until_now.append(
            strategy.experience.classes_in_this_experience)

        print("Classes so far: ", self.classes_until_now,
              len(self.classes_until_now))
        if self.untrained_solver:
            # The solver needs to train on the first experience before it can label generated data
            # as well as the generator needs to train first.
            self.untrained_solver = False
            return
        # self.classes_until_now = [class_id for exp_classes in self.classes_until_now for class_id in exp_classes]
        # Sample data from generator
        memory = self.generator.generate(
            len(strategy.adapted_dataset)*(len(self.classes_until_now)-1)).to(strategy.device)
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
        # Update strategies dataloader by mixing current experience's data with generated data.
        strategy.dataloader = ReplayDataLoader(
            strategy.adapted_dataset,
            memory,
            batch_size=batch_size,
            batch_size_mem=batch_size_mem*(len(self.classes_until_now)-1),
            task_balanced_dataloader=self.task_balanced_dataloader,
            num_workers=num_workers,
            shuffle=shuffle)


class VAEPlugin(SupervisedPlugin):

    def after_forward(
        self, strategy, *args, **kwargs
    ):
        # Forward call computes the representations in the latent space. They are stored at strategy.mb_output and can be used here
        strategy.mean, strategy.logvar = strategy.model.calc_mean(
            strategy.mb_output), strategy.model.calc_logvar(strategy.mb_output)
        z = strategy.model.sampling(strategy.mean, strategy.logvar)
        strategy.mb_x_recon = strategy.model.decoder(z)

    def after_eval_forward(
        self, strategy, *args, **kwargs
    ):
        # Forward call computes the representations in the latent space. They are stored at strategy.mb_output and can be used here
        strategy.mean, strategy.logvar = strategy.model.calc_mean(
            strategy.mb_output), strategy.model.calc_logvar(strategy.mb_output)
        z = strategy.model.sampling(strategy.mean, strategy.logvar)
        strategy.mb_x_recon = strategy.model.decoder(z)


class trainGeneratorPlugin(SupervisedPlugin):
    def after_training_exp(self, strategy: "SupervisedTemplate", **kwargs):
        print("Start training of Generator ... ")
        # strategy.generator.train(strategy.dataloader)
        strategy.plugins[1].generator_strategy.train(strategy.experience) 
        # Originally wanted to train directly on strategy.dataloader which already contains generated data
        # However training requires an experience which has an attribute dataset with teh entire dataset.
        # We there do the sampling step again
