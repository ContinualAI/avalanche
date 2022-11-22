from contextlib import contextmanager, ExitStack


class DistributedStrategySupport:

    def __init__(self):
        """
        Implements the basic elements needed to support distributed training
        in Avalanche strategies.
        """
        super().__init__()
        self._use_local_contexts = []
        """
        A list of context manager factories to be used in `use_local`.
        """

    @contextmanager
    def use_local(self, *args, **kwargs):
        """
        A context manager used to change the behavior of some property getters.

        When running code in this context, the property getter implementation
        of some distributed-critical fields will return the local value instead
        of the distributed (synchronized) one.

        Examples of distributed-critical fields are `model`, `mbatch`,
        `mb_output`, `loss`.

        Beware that this method will modify the behavior of getters of ALL
        such properties. This may not be desirable. Use the field-specific
        `use_local_*` context managers to control the behavior of these
        fields in a finer way.

        :param args: Passed to all field-specific `use_local_*` context
            managers.
        :param kwargs: Passed to all field-specific `use_local_*` context
            managers.
        :return: The context manager to be used through the `with` syntax.
        """
        with ExitStack() as stack:
            for lcm in self._use_local_contexts:
                stack.enter_context(lcm(*args, **kwargs))
            yield


__all__ = [
    'DistributedStrategySupport'
]
