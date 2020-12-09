from typing import TypeVar

from torch import Tensor

# This list of classes would be greatly simplified if we could use dataclasses.

TElement = TypeVar('TElement')


def _detach_tensor(tensor: TElement) -> TElement:
    if isinstance(tensor, Tensor):
        return tensor.detach().cpu()

    return tensor


class EvalData:
    # TODO: doc

    def __init__(self,
                 step_counter: int,
                 training_step_id: int,
                 training_task_label: int):
        # TODO: doc
        # TODO: fields doc
        self.train_phase: bool = True
        self.step_counter: int = step_counter
        self.training_step_id: int = training_step_id
        self.training_task_label: int = training_task_label

        # TODO: fill and super

    @property
    def test_phase(self):
        return not self.train_phase


class EvalTestData(EvalData):
    # TODO: doc

    def __init__(self,
                 step_counter: int,
                 training_step_id: int,
                 training_task_label: int,
                 test_step_id: int,
                 test_task_label: int):
        # TODO: doc
        # TODO: fields doc
        super().__init__(step_counter, training_step_id, training_task_label)
        self.train_phase: bool = False
        self.test_step_id: int = test_step_id
        self.test_task_label: int = test_task_label


class OnTrainStart(EvalData):
    # TODO: doc

    def __init__(self,
                 step_counter: int,
                 step_id: int,
                 training_task_label: int):
        super().__init__(step_counter, step_id, training_task_label)


class OnTrainEnd(EvalData):
    # TODO: doc

    def __init__(self,
                 step_counter: int,
                 step_id: int,
                 training_task_label: int):
        super().__init__(step_counter, step_id, training_task_label)


class OnTestStart(EvalTestData):
    # TODO: doc

    def __init__(self,
                 step_counter: int,
                 step_id: int,
                 training_task_label: int,
                 test_step_id: int,
                 test_task_label: int):
        super().__init__(step_counter, step_id, training_task_label,
                         test_step_id, test_task_label)


class OnTestEnd(EvalTestData):
    # TODO: doc

    def __init__(self,
                 step_counter: int,
                 step_id: int,
                 training_task_label: int,
                 test_step_id: int,
                 test_task_label: int):
        super().__init__(step_counter, step_id, training_task_label,
                         test_step_id, test_task_label)


class OnTrainEpochStart(EvalData):
    # TODO: doc

    def __init__(self,
                 step_counter: int,
                 step_id: int,
                 training_task_label: int,
                 epoch: int):
        super().__init__(step_counter, step_id, training_task_label)
        self.epoch: int = epoch


class OnTestEpochStart(EvalTestData):
    # TODO: doc

    def __init__(self,
                 step_counter: int,
                 step_id: int,
                 training_task_label: int,
                 test_step_id: int,
                 test_task_label: int,
                 epoch: int):
        super().__init__(step_counter, step_id, training_task_label,
                         test_step_id, test_task_label)
        self.epoch: int = epoch


class OnTrainEpochEnd(EvalData):
    # TODO: doc

    def __init__(self,
                 step_counter: int,
                 step_id: int,
                 training_task_label: int,
                 epoch: int):
        super().__init__(step_counter, step_id, training_task_label)
        self.epoch: int = epoch


class OnTestEpochEnd(EvalTestData):
    # TODO: doc

    def __init__(self,
                 step_counter: int,
                 step_id: int,
                 training_task_label: int,
                 test_step_id: int,
                 test_task_label: int,
                 epoch: int):
        super().__init__(step_counter, step_id, training_task_label,
                         test_step_id, test_task_label)
        self.epoch: int = epoch


class OnTrainIteration(EvalData):
    # TODO: doc

    def __init__(self,
                 step_counter: int,
                 step_id: int,
                 training_task_label: int,
                 epoch: int,
                 iteration: int,
                 ground_truth: Tensor,
                 prediction_logits: Tensor,
                 loss: Tensor):
        super().__init__(step_counter, step_id, training_task_label)
        self.epoch: int = epoch
        self.iteration: int = iteration
        self.ground_truth: Tensor = _detach_tensor(ground_truth)
        self.prediction_logits: Tensor = _detach_tensor(prediction_logits)
        self.loss: Tensor = _detach_tensor(loss)


class OnTestIteration(EvalTestData):
    # TODO: doc

    def __init__(self,
                 step_counter: int,
                 step_id: int,
                 training_task_label: int,
                 test_step_id: int,
                 test_task_label: int,
                 epoch: int,
                 iteration: int,
                 ground_truth: Tensor,
                 prediction_logits: Tensor,
                 loss: Tensor):
        super().__init__(step_counter, step_id, training_task_label,
                         test_step_id, test_task_label)
        self.epoch: int = epoch
        self.iteration: int = iteration
        self.ground_truth: Tensor = _detach_tensor(ground_truth)
        self.prediction_logits: Tensor = _detach_tensor(prediction_logits)
        self.loss: Tensor = _detach_tensor(loss)


__all__ = ['EvalData', 'EvalTestData', 'OnTrainStart', 'OnTrainEnd',
           'OnTestStart', 'OnTestEnd', 'OnTrainEpochStart', 'OnTestEpochStart',
           'OnTrainEpochEnd', 'OnTestEpochEnd', 'OnTrainIteration',
           'OnTestIteration']
