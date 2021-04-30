from torch.utils.data import random_split, ConcatDataset
import torch
import torch.nn.functional as F
import random
from torch.utils.data import TensorDataset
from avalanche.benchmarks.utils import AvalancheConcatDataset, AvalancheDataset
from avalanche.training.plugins.strategy_plugin import StrategyPlugin
from avalanche.benchmarks.utils.data_loader import \
    MultiTaskJoinedBatchDataLoader
import numpy as np


class GSS_greedyPlugin(StrategyPlugin):
    """
    GSSPlugin replay plugin.

    Handles an external memory fulled with samples selected using the Greedy approach
    of GSS algorithm. 
    `before_forward` callback is used to process the current sample and estimate a score.    

    The :mem_size: attribute controls the total number of patterns to be stored 
    in the external memory.
    """

    def __init__(self, mem_size=200,mem_strength =5):
        super().__init__()
        self.mem_size = mem_size
        self.mem_strength =mem_strength 
        self.ext_mem = {}  # a Dict<task_id, Dataset>

        #TODO
        input_size=[1, 28, 28]
        self.ext_mem_list_x=torch.FloatTensor(mem_size, *input_size).fill_(0)
        self.ext_mem_list_y=torch.LongTensor(mem_size).fill_(0)
        self.ext_mem_list_current_index=0

        self.buffer_score = torch.FloatTensor(self.mem_size).fill_(0)

    def cosine_similarity(self, x1, x2=None, eps=1e-8):
        x2 = x1 if x2 is None else x2
        w1 = x1.norm(p=2, dim=1, keepdim=True)

        w2 = w1 if x2 is x1 else x2.norm(p=2, dim=1, keepdim=True)
        sim= torch.mm(x1, x2.t())/(w1 * w2.t()) #, w1  # .clamp(min=eps), 1/cosinesim

        return sim

    def get_grad_vector(self, pp, grad_dims):
        """
        gather the gradients in one vector
        """
        grads = torch.Tensor(sum(grad_dims))
        grads.fill_(0.0)
        cnt = 0
        for param in pp():
            if param.grad is not None:
                beg = 0 if cnt == 0 else sum(grad_dims[:cnt])
                en = sum(grad_dims[:cnt + 1])
                grads[beg: en].copy_(param.grad.data.view(-1))
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
        loss = strategy.criterion(strategy.model.forward(batch_x), batch_y)
        loss.backward()
        batch_grad = self.get_grad_vector(strategy.model.parameters, grad_dims).unsqueeze(0)
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
        num_mem_subs = min(self.mem_strength, self.ext_mem_list_current_index // gss_batch_size)
        mem_grads = torch.zeros(num_mem_subs, sum(grad_dims), dtype=torch.float32)
        shuffeled_inds = torch.randperm(self.ext_mem_list_current_index)
        for i in range(num_mem_subs):
            random_batch_inds = shuffeled_inds[
                                i * temp_gss_batch_size:i * temp_gss_batch_size + temp_gss_batch_size]
            batch_x = self.ext_mem_list_x[random_batch_inds]
            batch_y = self.ext_mem_list_y[random_batch_inds]
            strategy.model.zero_grad()
            loss = strategy.criterion(strategy.model.forward(batch_x), batch_y)
            loss.backward()
            mem_grads[i].data.copy_(self.get_grad_vector(strategy.model.parameters, grad_dims))
        return mem_grads

    def get_each_batch_sample_sim(self, strategy, grad_dims, mem_grads, batch_x, batch_y):
        """
        Args:
            buffer: memory buffer
            grad_dims: gradient dimensions
            mem_grads: gradient from memory subsets
            batch_x: batch images
            batch_y: batch labels
        Returns: score of each sample from current batch
        """
        cosine_sim = torch.zeros(batch_x.size(0))
        for i, (x, y) in enumerate(zip(batch_x, batch_y)):
            strategy.model.zero_grad()
            ptloss = strategy.criterion(strategy.model.forward(x.unsqueeze(0)), y.unsqueeze(0))
            ptloss.backward()
            # add the new grad to the memory grads and add it is cosine similarity
            this_grad = self.get_grad_vector(strategy.model.parameters, grad_dims).unsqueeze(0)
            cosine_sim[i] = max(self.cosine_similarity(mem_grads, this_grad))
        return cosine_sim

    def before_training_exp(self, strategy, num_workers=0, shuffle=True,
                            **kwargs):
        """
        Dataloader to build batches containing examples from both memories and
        the training dataset
        """
        if len(self.ext_mem) == 0:
            return
        
        memory = AvalancheDataset(self.ext_mem_list_x,targets=self.ext_mem_list_y)

        strategy.dataloader = MultiTaskJoinedBatchDataLoader(
            strategy.adapted_dataset,
            AvalancheConcatDataset(memory),
            oversample_small_tasks=True,
            num_workers=num_workers,
            batch_size=strategy.train_mb_size,
            shuffle=shuffle)

    def after_forward(self, strategy, num_workers=0, shuffle=True, **kwargs):
        """
        After every forward this function select sample to fill the memory buffer based on cosine similarity
        """
        strategy.model.eval()

        # Compute the gradient dimension
        grad_dims = []
        for param in strategy.model.parameters():
            grad_dims.append(param.data.numel())

        place_left = self.ext_mem_list_x.size(0) -self.ext_mem_list_current_index

        if(place_left<=0): #buffer full

            batch_sim, mem_grads = self.get_batch_sim(strategy, grad_dims, batch_x=strategy.mb_x, batch_y=strategy.mb_y)

            if batch_sim < 0:
                buffer_score = self.buffer_score[:self.ext_mem_list_current_index].cpu()
                buffer_sim = ((buffer_score - torch.min(buffer_score)) / \
                             ((torch.max(buffer_score) - torch.min(buffer_score)) + 0.01))+0.01
                # draw candidates for replacement from the buffer
                index = torch.multinomial(buffer_sim, strategy.mb_x.size(0), replacement=False)

                # estimate the similarity of each sample in the recieved batch
                # to the randomly drawn samples from the buffer.
                batch_item_sim = self.get_each_batch_sample_sim(strategy, grad_dims, mem_grads, strategy.mb_x, strategy.mb_y)

                # normalize to [0,1]
                scaled_batch_item_sim = ((batch_item_sim + 1) / 2).unsqueeze(1)
                buffer_repl_batch_sim = ((self.buffer_score[index] + 1) / 2).unsqueeze(1)
                # draw an event to decide on replacement decision
                outcome = torch.multinomial(torch.cat((scaled_batch_item_sim, buffer_repl_batch_sim), dim=1), 1,
                                            replacement=False)
                # replace samples with outcome =1
                added_indx = torch.arange(end=batch_item_sim.size(0))
                sub_index = outcome.squeeze(1).bool()
                self.ext_mem_list_x[index[sub_index]] = strategy.mb_x[added_indx[sub_index]].clone()
                self.ext_mem_list_y[index[sub_index]] = strategy.mb_y[added_indx[sub_index]].clone()
                self.buffer_score[index[sub_index]] = batch_item_sim[added_indx[sub_index]].clone()
        else:
            offset = min(place_left, strategy.mb_x.size(0))
            strategy.mb_x = strategy.mb_x[:offset]
            strategy.mb_y = strategy.mb_y[:offset]
            # first buffer insertion
            if self.ext_mem_list_current_index== 0:
                batch_sample_memory_cos = torch.zeros(strategy.mb_x.size(0)) + 0.1
            else:
                # draw random samples from buffer
                mem_grads = self.get_rand_mem_grads(strategy=strategy, grad_dims=grad_dims,gss_batch_size=len(strategy.mb_x))
                
                # estimate a score for each added sample
                batch_sample_memory_cos = self.get_each_batch_sample_sim(strategy, grad_dims, mem_grads,  strategy.mb_x,  strategy.mb_y)

            self.ext_mem_list_x[self.ext_mem_list_current_index:self.ext_mem_list_current_index + offset].data.copy_(strategy.mb_x)
            self.ext_mem_list_y[self.ext_mem_list_current_index:self.ext_mem_list_current_index + offset].data.copy_(strategy.mb_y)
            self.buffer_score[self.ext_mem_list_current_index:self.ext_mem_list_current_index + offset].data.copy_(batch_sample_memory_cos)
            self.ext_mem_list_current_index += offset
        strategy.model.train()
        

