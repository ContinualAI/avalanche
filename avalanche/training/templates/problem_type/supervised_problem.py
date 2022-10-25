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
        return self._criterion(self.mb_output, self.mb_y)

    def forward(self):
        """Compute the model's output given the current mini-batch."""
        return avalanche_forward(self.model, self.mb_x, self.mb_task_id)

    def _check_minibatch(self):
        """Check if the current mini-batch has 3 components."""
        assert len(self.mbatch) >= 3
