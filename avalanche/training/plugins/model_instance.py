################################################################################
# Copyright (c) 2021 ContinualAI.                                              #
# Copyrights licensed under the MIT License.                                   #
# See the accompanying LICENSE file for terms.                                 #
#                                                                              #
# Date: 1/12/2021                                                              #
# Author(s): Lorenzo Pellegrini                                                #
# E-mail: contact@continualai.org                                              #
# Website: avalanche.continualai.org                                           #
################################################################################
import torch

from avalanche.distributed import DistributedHelper


class ModelInstance:
    """
    Contains the model used in the :class:`BaseTemplate` strategy template.

    This class is rarely used directly. Most strategies use the child class
    :class:`SGDModelInstance` instead.

    Instances of this class can also carry the distributed (that is, wrapped
    in a PyTorch `DistributedDataParallel`) version of a local model. If no
    distributed model is set, then the model returned by the
    `distributed_model` field will be the local one.

    By setting the `distributed_model` field, the model stored in the
    `local_model` field will be discarded (from that moment, retrieving the
    `local_model` will be the same as obtaining the `distributed_model.module`
    field). Setting the `local_model` will discard the current
    `distributed_model`.
    """

    def __init__(self, initial_model=None):
        """
        Creates a `ModelInstance`.

        :param initial_model: The initial model to use. Defaults to None.
        """
        super().__init__()

        self._local_model = initial_model
        self._distributed_model = None

    @property
    def model(self):
        """
        The current model.

        This is an alias for the `local_model` field.
        """
        return self._distributed_model

    @property
    def local_model(self):
        """
        The current (local) model.

        If a `distributed_model` was set, then the value of the
        `distributed_model.module` field will be returned.
        """
        if self._distributed_model is not None:
            return self._distributed_model.module
        return self._local_model

    @local_model.setter
    def local_model(self, value):
        """
        Sets the local model.

        This will discard the current distributed model.
        """
        self._on_model_reset()
        self._local_model = value
        self._distributed_model = None

    @property
    def distributed_model(self):
        """
        The current (distributed) model.

        If not set (not running a distributed training, or if the wrapped
        model has not been created yet), this is the same as `local_model`.
        """
        if not DistributedHelper.is_distributed:
            return self._local_model

        if self._distributed_model is None:
            return self._local_model

        return self._distributed_model

    @distributed_model.setter
    def distributed_model(self, value):
        """
        Sets the model wrapped by PyTorch `DistributedDataParallel`.

        Setting this field will release the reference to the current local
        model. In that case, the `local_model` field will return
        `distributed_model.module` instead.
        """
        # Prevent alignment and memory issues.
        # The local model will be retrieved from the _distributed_model.
        # This will also reset mb_input, mb_output and loss.
        self.local_model = None

        self._distributed_model = value

    def _on_model_reset(self):
        self._local_model = None
        self._distributed_model = None


class SGDModelInstance(ModelInstance):
    """
    Contains the model and its last input minibatch, output, and loss.
    Used in the :class:`BaseSGDTemplate` strategy.

    Instances of this class can also carry the distributed version of a local
    model. See the description found in :class:`ModelInstance` for more details.
    In addition, the distributed version of the input minibatch, output, and
    loss can be retrieved by accessing the `distributed_*` fields. When not
    running a distributed training, the `distributed_*` fields will act as
    aliases for their `local_*` counterpart.

    Only `local_*` fields can be set. The `distributed_*` versions of the same
    fields are obtained by (lazily) synchronizing the minibatch input, output,
    and loss across processes. The only exception is `distributed_model`, which
    behaves as described in the superclass :class:`ModelInstance` documentation.

    By default, the :class:`BaseSGDTemplate` exposes high-level fields such as
    `model`, `mbatch`, `mb_output`, and `loss` whose getters will retrieve the
    proper `distributed_*` field from an instance of this class (remember
    that, if not running a distributed experiment, this is the same as
    retrieving the proper `local_*` field). However, setters of those
    high-level fields will set the `local_*` field of `model_instance`,
    even when running a distributed training. This follows the
    "set the local value, get the synchronized one" idea, which is useful to
    keep plugins and metrics aligned. The only exception to this idea should
    is the computation of the loss, which should always be computed on local
    minibatch output and ground truth. In addition, the `backward` method
    should be called on the local loss Tensor. The :class:`BaseSGDTemplate`
    already take care of these exceptions.

    If you plan on writing a plugin, metric, or any other element that
    supports distributed training, then explicitly using the `local_*` and
    `distributed_*` fields may make your code more clear and efficient. In
    particular, you should try to minimize the retrieval of `distributed_*`
    values, as accessing these fields requires a synchronization step which
    blocks all processes.

    This class is also in charge of managing the lifetime of model inputs and
    outputs: setting a new value for the model will reset (set to None) the
    input minibatch. Setting a new input minibatch will reset the minibatch
    output. Setting a new minibatch output will reset the last loss.
    """

    def __init__(self, initial_model=None):
        """
        Creates a `SGDModelInstance`.

        :param initial_model: The initial model to use. Defaults to None.
        """
        super().__init__(initial_model=initial_model)

        self._local_mb_input = None
        self._local_mb_output = None
        self._local_loss = None

        self._distributed_mb_input = None
        self._distributed_mb_output = None
        self._distributed_loss = None

    @property
    def mb_input(self):
        """
        The current input minibatch.

        This is an alias for the `distributed_mb_input` field.
        """
        return self.distributed_mb_input

    @property
    def mb_output(self):
        """
        The current minibatch output.

        This is an alias for the `distributed_mb_output` field.
        """
        return self.distributed_mb_output

    @property
    def loss(self):
        """
        The current loss.

        This is an alias for the `distributed_loss` field.
        """
        return self.distributed_loss

    @property
    def local_mb_input(self):
        """
        The current (local) input minibatch.
        """
        return self._local_mb_input

    @local_mb_input.setter
    def local_mb_input(self, value):
        """
        Sets the local input minibatch.

        This will also reset the minibatch output and loss.
        """
        self._on_reset_io()
        self._local_mb_input = value

    @property
    def local_mb_output(self):
        """
        The current (local) output minibatch.
        """
        return self._local_mb_output

    @local_mb_output.setter
    def local_mb_output(self, value):
        """
        Sets the local minibatch output.

        This will also reset the loss.
        """
        self._on_reset_outputs()
        self._local_mb_output = value

    @property
    def local_loss(self):
        """
        The current loss.
        """
        return self._local_loss

    @local_loss.setter
    def local_loss(self, value):
        """
        Sets the local loss.
        """
        self._on_reset_losses()
        self._local_loss = value

    @property
    def distributed_mb_input(self):
        """
        The current (distributed) input minibatch.

        If not running a distributed training, this is the same as
        `local_mb_input`.

        When running a distributed training, this value will be obtained by
        concatenating the input minibatches of all processes. This
        synchronization step is done only if required (lazily).
        """
        if not DistributedHelper.is_distributed:
            return self.local_mb_input

        if self._local_mb_input is None:
            return None

        if self._distributed_model is None:
            # Non-distributed training
            return self._local_mb_input

        if self._distributed_mb_input is None:
            # TODO: implement non-Tensor input synchronization?

            if isinstance(self._local_mb_input, (list, tuple)):
                # Usual minibatch made of at least 2 tensors: x and y
                mb_tuple = []
                for mb_tuple_elem in self._local_mb_input:
                    mb_tuple.append(DistributedHelper.cat_all(mb_tuple_elem))

                self._distributed_mb_input = tuple(mb_tuple)
            else:
                # Single tensor input
                self._distributed_mb_input = \
                    DistributedHelper.cat_all(self._local_mb_input)
        return self._distributed_mb_input

    @property
    def distributed_mb_output(self):
        """
        The current (distributed) minibatch output.

        If not running a distributed training, this is the same as
        `local_mb_output`.

        When running a distributed training, this value will be obtained by
        concatenating the minibatch output of all processes. This
        synchronization step is done only if required (lazily).
        """
        if not DistributedHelper.is_distributed:
            return self.local_mb_output

        if self._local_mb_output is None:
            return None

        if self._distributed_model is None:
            # Non-distributed training
            return self._local_mb_output

        if self._distributed_mb_output is None:
            # TODO: implement non-Tensor output synchronization?

            if isinstance(self._local_mb_output, (list, tuple)):
                # Unusual output made of at more than 1 tensor
                out_tuple = []
                for out_tuple_elem in self._local_mb_output:
                    out_tuple.append(DistributedHelper.cat_all(out_tuple_elem))

                self._distributed_mb_output = tuple(out_tuple)
            else:
                # Usual single-tensor output
                self._distributed_mb_output = \
                    DistributedHelper.cat_all(self._local_mb_output)

        return self._distributed_mb_output

    @property
    def distributed_loss(self):
        """
        The current (distributed) minibatch output.

         If not running a distributed training, this is the same as
        `local_loss`.

        When running a distributed training, this value will be obtained by
        averaging the loss of all processes. This synchronization step is done
        only if required (lazily).

        Please note that the resulting loss Tensor will not be differentiable
        and calling "backward" on it may result in an error.
        """
        if not DistributedHelper.is_distributed:
            return self._local_loss

        if self._local_loss is None:
            return None

        if self._distributed_model is None:
            # Non-distributed training
            return self._local_loss

        if self._distributed_loss is None:
            distributed_loss_not_reduced = \
                DistributedHelper.cat_all(self._local_loss)
            self._distributed_loss = torch.mean(distributed_loss_not_reduced)

        return self._distributed_loss

    def _on_model_reset(self):
        super(SGDModelInstance, self)._on_model_reset()
        self._on_reset_io()

    def _on_reset_io(self):
        # Will also reset other dependent elements such as mb_output and loss
        self._on_reset_inputs()

    def _on_reset_inputs(self):
        self._local_mb_input = None
        self._distributed_mb_input = None

        return self._on_reset_outputs()

    def _on_reset_outputs(self):
        self._local_mb_output = None
        self._distributed_mb_output = None

        return self._on_reset_losses()

    def _on_reset_losses(self):
        self._local_loss = None
        self._distributed_loss = None
