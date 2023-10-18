from typing import TYPE_CHECKING

import torch
from avalanche.benchmarks.utils import _make_taskaware_classification_dataset
from avalanche.benchmarks.utils.data_loader import ReplayDataLoader
from avalanche.training.plugins.strategy_plugin import SupervisedPlugin

if TYPE_CHECKING:
    from ..templates import SupervisedTemplate


class GSS_greedyPlugin(SupervisedPlugin):
    """GSSPlugin replay plugin.

    Code adapted from the repository:
    https://github.com/RaptorMai/online-continual-learning
    Handles an external memory fulled with samples selected
    using the Greedy approach of GSS algorithm.
    `before_forward` callback is used to process the current
    sample and estimate a score.
    """

    def __init__(self, mem_size=200, mem_strength=5, input_size=[]):
        """

        :param mem_size: total number of patterns to be stored
            in the external memory.
        :param mem_strength:
        :param input_size:
        """
        super().__init__()
        self.mem_size = mem_size
        self.mem_strength = mem_strength
        self.device = torch.device("cpu")

        self.ext_mem_list_x = torch.FloatTensor(mem_size, *input_size).fill_(0)
        self.ext_mem_list_y = torch.LongTensor(mem_size).fill_(0)
        self.ext_mem_list_current_index = 0

        self.buffer_score = torch.FloatTensor(self.mem_size).fill_(0)

    def before_training(self, strategy: "SupervisedTemplate", **kwargs):
        self.device = strategy.device
        self.ext_mem_list_x = self.ext_mem_list_x.to(strategy.device)
        self.ext_mem_list_y = self.ext_mem_list_y.to(strategy.device)
        self.buffer_score = self.buffer_score.to(strategy.device)

    def cosine_similarity(self, x1, x2=None, eps=1e-8):
        x2 = x1 if x2 is None else x2
        w1 = x1.norm(p=2, dim=1, keepdim=True)

        w2 = w1 if x2 is x1 else x2.norm(p=2, dim=1, keepdim=True)
        sim = torch.mm(x1, x2.t()) / (w1 * w2.t()).clamp(min=eps)
        return sim

    def get_grad_vector(self, pp, grad_dims):
        """
        gather the gradients in one vector
        """
        grads = torch.zeros(sum(grad_dims), device=self.device)
        grads.fill_(0.0)
        cnt = 0
        for param in pp():
            if param.grad is not None:
                beg = 0 if cnt == 0 else sum(grad_dims[:cnt])
                en = sum(grad_dims[: cnt + 1])
                grads[beg:en].copy_(param.grad.data.view(-1))
            cnt += 1
        return grads

    def get_batch_sim(self, strategy, grad_dims, batch_x, batch_y):
        """
        Args:
            buffer: memory buffer
            grad_dims: gradient dimensions
            batch_x: current batch x
            batch_y: current batch y
        Returns: score of current batch, gradient from memory subsets
        """
        mem_grads = self.get_rand_mem_grads(strategy, grad_dims, len(batch_x))
        strategy.model.zero_grad()
        loss = strategy._criterion(strategy.model.forward(batch_x), batch_y)
        loss.backward()
        batch_grad = self.get_grad_vector(
            strategy.model.parameters, grad_dims
        ).unsqueeze(0)
        batch_sim = max(self.cosine_similarity(mem_grads, batch_grad))
        return batch_sim, mem_grads

    def get_rand_mem_grads(self, strategy, grad_dims, gss_batch_size):
        """
        Args:
            buffer: memory buffer
            grad_dims: gradient dimensions
        Returns: gradient from memory subsets
        """
        temp_gss_batch_size = min(gss_batch_size, self.ext_mem_list_current_index)
        num_mem_subs = min(
            self.mem_strength, self.ext_mem_list_current_index // gss_batch_size
        )
        mem_grads = torch.zeros(
            num_mem_subs,
            sum(grad_dims),
            dtype=torch.float32,
            device=self.device,
        )
        shuffeled_inds = torch.randperm(
            self.ext_mem_list_current_index, device=self.device
        )
        for i in range(num_mem_subs):
            random_batch_inds = shuffeled_inds[
                i * temp_gss_batch_size : i * temp_gss_batch_size + temp_gss_batch_size
            ]
            batch_x = self.ext_mem_list_x[random_batch_inds].to(strategy.device)
            batch_y = self.ext_mem_list_y[random_batch_inds].to(strategy.device)
            strategy.model.zero_grad()

            loss = strategy._criterion(strategy.model.forward(batch_x), batch_y)
            loss.backward()
            mem_grads[i].data.copy_(
                self.get_grad_vector(strategy.model.parameters, grad_dims)
            )
        return mem_grads

    def get_each_batch_sample_sim(
        self, strategy, grad_dims, mem_grads, batch_x, batch_y
    ):
        """
        Args:
            buffer: memory buffer
            grad_dims: gradient dimensions
            mem_grads: gradient from memory subsets
            batch_x: batch images
            batch_y: batch labels
        Returns: score of each sample from current batch
        """
        cosine_sim = torch.zeros(batch_x.size(0), device=strategy.device)
        for i, (x, y) in enumerate(zip(batch_x, batch_y)):
            strategy.model.zero_grad()
            ptloss = strategy._criterion(
                strategy.model.forward(x.unsqueeze(0)), y.unsqueeze(0)
            )
            ptloss.backward()
            # add the new grad to the memory grads and add it is cosine
            # similarity
            this_grad = self.get_grad_vector(
                strategy.model.parameters, grad_dims
            ).unsqueeze(0)
            cosine_sim[i] = max(self.cosine_similarity(mem_grads, this_grad))
        return cosine_sim

    def before_training_exp(self, strategy, num_workers=0, shuffle=True, **kwargs):
        """
        Dataloader to build batches containing examples from both memories and
        the training dataset
        """
        if self.ext_mem_list_current_index == 0:
            return

        temp_x_tensors = []
        for elem in self.ext_mem_list_x:
            temp_x_tensors.append(elem.to("cpu"))
        temp_y_tensors = self.ext_mem_list_y.to("cpu")

        memory = list(zip(temp_x_tensors, temp_y_tensors))
        memory_dataset = _make_taskaware_classification_dataset(
            memory, targets=temp_y_tensors.tolist()
        )

        strategy.dataloader = ReplayDataLoader(
            strategy.adapted_dataset,
            memory_dataset,
            oversample_small_tasks=True,
            num_workers=num_workers,
            batch_size=strategy.train_mb_size,
            shuffle=shuffle,
        )

    def after_forward(self, strategy, num_workers=0, shuffle=True, **kwargs):
        """
        After every forward this function select sample to fill
        the memory buffer based on cosine similarity
        """

        strategy.model.eval()

        # Compute the gradient dimension
        grad_dims = []
        for param in strategy.model.parameters():
            grad_dims.append(param.data.numel())

        place_left = self.ext_mem_list_x.size(0) - self.ext_mem_list_current_index
        if place_left <= 0:  # buffer full
            batch_sim, mem_grads = self.get_batch_sim(
                strategy,
                grad_dims,
                batch_x=strategy.mb_x,
                batch_y=strategy.mb_y,
            )

            if batch_sim < 0:
                buffer_score = self.buffer_score[
                    : self.ext_mem_list_current_index
                ].cpu()

                buffer_sim = (buffer_score - torch.min(buffer_score)) / (
                    (torch.max(buffer_score) - torch.min(buffer_score)) + 0.01
                )

                # draw candidates for replacement from the buffer
                index = torch.multinomial(
                    buffer_sim, strategy.mb_x.size(0), replacement=False
                ).to(strategy.device)

                # estimate the similarity of each sample in the received batch
                # to the randomly drawn samples from the buffer.
                batch_item_sim = self.get_each_batch_sample_sim(
                    strategy, grad_dims, mem_grads, strategy.mb_x, strategy.mb_y
                )

                # normalize to [0,1]
                scaled_batch_item_sim = ((batch_item_sim + 1) / 2).unsqueeze(1)
                buffer_repl_batch_sim = ((self.buffer_score[index] + 1) / 2).unsqueeze(
                    1
                )
                # draw an event to decide on replacement decision
                outcome = torch.multinomial(
                    torch.cat((scaled_batch_item_sim, buffer_repl_batch_sim), dim=1),
                    1,
                    replacement=False,
                )
                # replace samples with outcome =1
                added_indx = torch.arange(
                    end=batch_item_sim.size(0), device=strategy.device
                )
                sub_index = outcome.squeeze(1).bool()
                self.ext_mem_list_x[index[sub_index]] = strategy.mb_x[
                    added_indx[sub_index]
                ].clone()
                self.ext_mem_list_y[index[sub_index]] = strategy.mb_y[
                    added_indx[sub_index]
                ].clone()
                self.buffer_score[index[sub_index]] = batch_item_sim[
                    added_indx[sub_index]
                ].clone()
        else:
            offset = min(place_left, strategy.mb_x.size(0))
            updated_mb_x = strategy.mb_x[:offset]
            updated_mb_y = strategy.mb_y[:offset]

            # first buffer insertion
            if self.ext_mem_list_current_index == 0:
                batch_sample_memory_cos = torch.zeros(updated_mb_x.size(0)) + 0.1
            else:
                # draw random samples from buffer
                mem_grads = self.get_rand_mem_grads(
                    strategy=strategy,
                    grad_dims=grad_dims,
                    gss_batch_size=len(strategy.mb_x),
                )

                # estimate a score for each added sample
                batch_sample_memory_cos = self.get_each_batch_sample_sim(
                    strategy, grad_dims, mem_grads, updated_mb_x, updated_mb_y
                )

            curr_idx = self.ext_mem_list_current_index
            self.ext_mem_list_x[curr_idx : curr_idx + offset].data.copy_(updated_mb_x)
            self.ext_mem_list_y[curr_idx : curr_idx + offset].data.copy_(updated_mb_y)
            self.buffer_score[curr_idx : curr_idx + offset].data.copy_(
                batch_sample_memory_cos
            )
            self.ext_mem_list_current_index += offset

        strategy.model.train()
