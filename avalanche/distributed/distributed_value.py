from contextlib import contextmanager
from typing import TypeVar, Generic, Optional, Union, Generator, List, \
    Tuple
from abc import ABC, abstractmethod

from avalanche.distributed import DistributedHelper

LocalT = TypeVar('LocalT')
DistributedT = TypeVar('DistributedT')
SwitchableT = TypeVar('SwitchableT', bound='SwitchableDistributedValue')


class DistributedValue(Generic[LocalT, DistributedT], ABC):
    """
    Class used to generically implement values that may need
    a lazy synchronization when running a distributed training.

    When not running a distributed training, this class will act as a
    no-op wrapper.

    This class considers setting the 'value' and 'local_value' as the
    same operation (setting the local value). However, retrieving 'value' will
    trigger the synchronization procedure.

    This class exposes methods that can be customized to define how different
    values should be gathered (and merged) from all processes. For instance,
    loss values should be averaged together, minibatch outputs should be
    concatenated, etcetera.

    Beware that the purpose of this class is to only manage the
    local and distributed values. When implementing the subclass, please do not
    transform the value and/or type of the local and global values. This
    would make it difficult to understand what is going on.

    Also, consider having the same type for the local and distributed value.
    That is, if the local value is a Tensor, the distributed value should be
    a Tensor as well, not a List[Tensor]. This is because local and distributed
    values will be transparently used by users without considering the possibly
    distributed nature of the value.

    Feel free to implement, in subclasses, properties with more readable names.
    For instance 'mb_output', 'local_mb_output', 'loss', 'local_loss', ...
    instead of the default  'value' and 'local_value' already implemented by
    this class.
    """

    def __init__(self, name: str, initial_local_value: LocalT):
        """
        Creates an instance of a distributed value.

        :param name: The name of the value. Also used when obtaining a string
            representation.
        :param initial_local_value: The initial local value.
        """
        self.name: str = name
        self._local_value: LocalT = initial_local_value
        self._distributed_value: Optional[DistributedT] = None
        self._distributed_value_set: bool = False

    @property
    def value(self) -> DistributedT:
        """
        The current value.

        When running a distributed training, this will be the value obtained
        by gathering and merging values coming from all processes.
        """
        return self._get_distributed_value()

    @value.setter
    def value(self, new_value: LocalT):
        """
        Sets the (local) value.

        This will discard the current distributed value.
        """
        self._set_local(new_value)

    @property
    def local_value(self) -> LocalT:
        """
        The current (local) value.

        Even when running a distributed training, this property will always
        contain the local value only.
        """
        return self._local_value

    @local_value.setter
    def local_value(self, new_value: LocalT):
        """
        Sets the (local) value.

        This will discard the current distributed value.
        """
        self._set_local(new_value)

    def _set_local(self, new_local_value: LocalT):
        self._local_value = new_local_value
        self._distributed_value = None
        self._distributed_value_set = False

    def _get_distributed_value(self) -> DistributedT:
        if not DistributedHelper.is_distributed:
            return self._local_value

        if not self._distributed_value_set:
            self._distributed_value = self._synchronize()
            self._distributed_value_set = True

        return self._distributed_value

    @abstractmethod
    def _synchronize(self) -> DistributedT:
        pass

    def __str__(self):
        base_str = f'DistributedObject_{self.name} = {self.local_value}'
        if self._distributed_value_set:
            return base_str + \
                   f' (distributed value = {self.value})'
        else:
            return base_str + \
                   f' (distributed value not synchronized yet)'


class SettableDistributedValue(DistributedValue[LocalT, DistributedT], ABC):
    """
    A version of :class:`DistributedValue` in which the distributed value can be
    set (and reset) externally instead of being synchronized.

    If this class should only allow for distributed values to be set
    externally (that is, synchronization should be disabled), please
    override `_synchronize` to raise an appropriate error.
    In that case, this means this class is mainly used as a switch between a
    local and a distributed value based on whether the distributed value has
    been set or not.
    """

    def __init__(self, name: str, initial_local_value: LocalT):
        super(SettableDistributedValue, self).__init__(
            name, initial_local_value
        )

    @property
    def distributed_value(self) -> DistributedT:
        """
        The current value.

        When running a distributed training, this will be the value obtained
        by gathering and merging values coming from all processes.
        """
        return self._get_distributed_value()

    @distributed_value.setter
    def distributed_value(self, new_distributed_value: DistributedT):
        """
        Set the distributed value.
        """
        self._distributed_value = new_distributed_value
        self._distributed_value_set = True

    def reset_distributed_value(self):
        """
        Discards the distributed value (if set).

        If the distributed value was not set, nothing happens.
        """
        self._distributed_value = None
        self._distributed_value_set = False

    def __str__(self):
        base_str = super(SettableDistributedValue, self).__str__()
        return f'(Settable){base_str}'


class SwitchableDistributedValue(SettableDistributedValue[LocalT, DistributedT],
                                 ABC):
    """
    A version of :class:`SettableDistributedValue` in which the behaviour of
    the `value` property can be switched so that it returns the local value
    instead of the distributed one. The setter behaviour can be customized as
    well.

    Useful for situations in which one has to force components interacting with
    this value to use the local value.Properties whose name feature an explicit
    `local` or `distributed` part are not affected.
    """

    def __init__(self, name: str, initial_local_value: LocalT):
        """
        Creates an instance of a distributed value.

        :param name: The name of the value. Also used when obtaining a string
            representation.
        :param initial_local_value: The initial local value.
        """
        super().__init__(name, initial_local_value)

        self._behaviour_stack: List[Tuple[bool, bool]] = list()
        """
        If greater than 0, the `value` property will return the local value.
        """

    @contextmanager
    def use_local_value(self: SwitchableT, getter=True, setter=True) -> \
            Generator[SwitchableT, None, None]:
        """
        A context manager used to set the behaviour of the value property.

        Please note that in a plain code section (not wrapped by this
        context manager), the default behaviour is that the getter returns the
        distributed value while the setter sets the local value.

        :param getter: If True, the local value will be returned by the getter.
            Defaults to True, which means that the getter behaviour will be
            changed.
        :param setter: If True, the local value will be set by the setter.
            Defaults to True, which means that the setter will behave as usual.
        :return: This object (self).
        """
        self._behaviour_stack.append((getter, setter))
        try:
            yield self
        finally:
            self._behaviour_stack.pop()

    @property
    def value(self) -> Union[LocalT, DistributedT]:
        if self._use_local_getter():
            return self.local_value
        else:
            return self.distributed_value

    @value.setter
    def value(self, new_value):
        if self._use_local_setter():
            self.local_value = new_value
        else:
            self.distributed_value = new_value

    def _use_local_getter(self):
        if len(self._behaviour_stack) == 0:
            return False

        return self._behaviour_stack[-1][0]

    def _use_local_setter(self):
        if len(self._behaviour_stack) == 0:
            return True

        return self._behaviour_stack[-1][1]

    def __str__(self):
        base_str = super(SettableDistributedValue, self).__str__()

        current_get_behaviour = 'local' if self._use_local_getter() \
            else 'distributed'
        current_set_behaviour = 'local' if self._use_local_setter() \
            else 'distributed'

        return f'(fget={current_get_behaviour},' \
               f'fset={current_set_behaviour}){base_str}'


class OptionalDistributedValue(SwitchableDistributedValue[LocalT, LocalT], ABC):
    """
    A version of :class:`SettableDistributedValue` in which the
    'value' property returns the local value if no distributed value has
    been set yet (without attempting a synchronization). Accessing the
    'distributed_value' property will still force a synchronization.

    Beware that, when using this class, the generic types for the local and
    distributed values is enforced to be the same.

    This class is mainly used for managing models wrapped using
    `DistributedDataParallel`.
    """

    def __init__(self, name, initial_local_value):
        super().__init__(name, initial_local_value)

    def _get_distributed_value(self) -> DistributedT:
        if not self._distributed_value_set:
            return self._local_value

        return self._distributed_value


__all__ = [
    'DistributedValue',
    'SettableDistributedValue',
    'SwitchableDistributedValue',
    'OptionalDistributedValue',
    'LocalT',
    'DistributedT'
]
