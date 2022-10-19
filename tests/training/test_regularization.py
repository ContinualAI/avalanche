import unittest

import torch
from torch.utils.data import DataLoader

from avalanche.models import SimpleMLP, MTSimpleMLP
from avalanche.models.utils import avalanche_model_adaptation
from avalanche.training.regularization import LearningWithoutForgetting
from tests.test_dataloaders import get_fast_benchmark


class TestLwF(unittest.TestCase):
    def test_lwf(self):
        lwf = LearningWithoutForgetting()
        bm = get_fast_benchmark()

        teacher = SimpleMLP(input_size=6)
        model = SimpleMLP(input_size=6)
        for exp in bm.train_stream:
            mb_x, mb_y, mb_tl = list(DataLoader(exp.dataset))[0]
            mb_pred = model(mb_x)
            loss = lwf(mb_x, mb_pred, model)

            # non-zero loss after first task
            if lwf.expcount == 0:
                assert loss == 0
            else:
                assert loss > 0.0
            lwf.update(exp, teacher)

        lwf = LearningWithoutForgetting()
        teacher = MTSimpleMLP(input_size=6)
        model = MTSimpleMLP(input_size=6)
        for exp in bm.train_stream:
            avalanche_model_adaptation(teacher, exp)
            avalanche_model_adaptation(model, exp)
            mb_x, mb_y, mb_tl = list(DataLoader(exp.dataset))[0]
            mb_pred = model(mb_x, task_labels=mb_tl)
            loss = lwf(mb_x, mb_pred, model)

            # non-zero loss after first task
            if lwf.expcount == 0:
                assert loss == 0
            else:
                assert loss > 0.0

                # non-zero loss for all the previous heads
                loss.backward()
                for tid in lwf.prev_classes_by_task.keys():
                    head = model.classifier.classifiers[str(tid)]
                    weight = head.classifier.weight
                    assert weight.grad is not None
                    assert torch.norm(weight.grad) > 0
                model.zero_grad()

            lwf.update(exp, teacher)
