from typing import final

from avalanche.models import avalanche_forward


class SupervisedProblem:
    @property
    def mb_x(self):
        """Current mini-batch input."""
        return self.mbatch[0]

    @property
    def mb_y(self):
        """Current mini-batch target."""
        return self.mbatch[1]

    @property
    def mb_task_id(self):
        """Current mini-batch task labels."""
        assert len(self.mbatch) >= 3
        return self.mbatch[-1]

    def criterion(self):
        """Loss function for supervised problems."""
        # Force self.mb_output and self.mb_y to be from local batch
        with self.use_local_output_batch():
            with self.use_local_input_batch():
                return self._criterion(self.mb_output, self.mb_y)

    @final
    def forward(self):
        """
        Compute the model's output given the current mini-batch.
        This method should not be overridden by child classes.
        Consider overriding :meth:`_forward` instead.
        """
        with self.use_local_input_batch():
            return self._forward()

    def _forward(self):
        """Implementation of the forward pass."""
        return avalanche_forward(self.model, self.mb_x, self.mb_task_id)

    def _check_minibatch(self):
        """Check if the current mini-batch has 3 components."""
        assert len(self.mbatch) >= 3
