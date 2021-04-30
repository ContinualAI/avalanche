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

def cosine_similarity(x1, x2=None, eps=1e-8):
    x2 = x1 if x2 is None else x2
    w1 = x1.norm(p=2, dim=1, keepdim=True)

    w2 = w1 if x2 is x1 else x2.norm(p=2, dim=1, keepdim=True)
    sim= torch.mm(x1, x2.t())/(w1 * w2.t()) #, w1  # .clamp(min=eps), 1/cosinesim

    return sim

def get_grad_vector(pp, grad_dims):
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


class GSS_greedy_simplePlugin(StrategyPlugin):
    """
    GSSPlugin replay plugin.

    Handles an external memory fulled with samples selected using the Greedy approach
    of GSS algorithm. 
    `before_forward` callback is used to process the current sample and estimate a score.    

    The :mem_size: attribute controls the total number of patterns to be stored 
    in the external memory.
    """

    def __init__(self, mem_size=200,n=1):
        super().__init__()
        self.mem_size = mem_size
        self.ext_mem = {}  # a Dict<task_id, Dataset>
        self.ext_mem_w_score=[]
        self.n=n

    def before_forward(self, strategy, num_workers=0, shuffle=True, **kwargs):
        """
        Before every forward this function select sample to fill the memory buffer based on cosine similarity
        """
        # Compute the gradient dimension
        grad_dims = []
        for param in strategy.model.parameters():
            grad_dims.append(param.data.numel())

        # For every sample in the minibatch compute score and check if it will be in memory buffer
        for i, sample in enumerate(strategy.mb_x):
            score=0

            # In the first instertion the score is not computed and the sample directly inserted
            if(len(self.ext_mem.values())==0):
                self.ext_mem_w_score.append((sample, strategy.mb_y[i],score))
            else:
                # Extract a random subset from the buffer and compute the gradient
                random_subset=random.choices(self.ext_mem_w_score,k=self.n)
                random_subset_grads=[]
                for random_memory_sample in random_subset:
                    x=random_memory_sample[0]
                    y=random_memory_sample[1]
                    strategy.model.zero_grad()
                    ptloss = strategy.criterion(strategy.model.forward(x.unsqueeze(0)), y.unsqueeze(0))
                    ptloss.backward()

                    grads=get_grad_vector(strategy.model.parameters,  grad_dims).unsqueeze(0)
                    if(len(random_subset_grads)!=0):
                        random_subset_grads = torch.cat((random_subset_grads, grads), dim=0)
                    else:
                        random_subset_grads=grads

                # Conpute the gradient for the current sample
                strategy.model.zero_grad()
                ptloss = strategy.criterion(strategy.model.forward(sample.unsqueeze(0)), strategy.mb_y[i].unsqueeze(0))
                ptloss.backward()
                sample_grad = get_grad_vector(strategy.model.parameters,  grad_dims).unsqueeze(0)
                

                # Compute the score
                score=max(cosine_similarity(random_subset_grads, sample_grad))+1
                

                # If the memory buffer is empty insert directly
                if(len(self.ext_mem.values())<= self.mem_size):
                    self.ext_mem_w_score.append((sample, strategy.mb_y[i],score))
                else:
                    if(score<1):
                        # Get normalizet scores
                        only_scores=[ext_mem_sample[2] for ext_mem_sample in self.ext_mem_w_score]
                        only_scores_normalized=only_scores/(sum(only_scores))

                        # Extract a candidate
                        index=torch.multinomial(only_scores_normalized, 1, replacement=False)
                        r=random.uniform(0, 1)
                        C_i=self.ext_mem_w_score[index][2]

                        #Check for the replacement
                        if(r<C_i/(C_i+score)):
                            self.ext_mem_w_score[index]=(sample,strategy.mb_y[i],score)

        # Concat the dataset
        if(len(self.ext_mem_w_score) == 0):
            return

        ext_mem_samples_x = [current_tuple[0] for current_tuple in self.ext_mem_w_score]
        ext_mem_samples_y = [current_tuple[1] for current_tuple in self.ext_mem_w_score]

        curr_task_id = strategy.experience.task_label
        self.ext_mem[curr_task_id] = AvalancheDataset(ext_mem_samples_x,targets=ext_mem_samples_y)

        strategy.dataloader = MultiTaskJoinedBatchDataLoader(
            strategy.adapted_dataset,
            AvalancheConcatDataset(self.ext_mem.values()),
            oversample_small_tasks=True,
            num_workers=num_workers,
            batch_size=strategy.train_mb_size,
            shuffle=shuffle)
        return
        

