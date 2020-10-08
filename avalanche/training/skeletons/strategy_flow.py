#!/usr/bin/env python
# -*- coding: utf-8 -*-

################################################################################
# Copyright (c) 2020 ContinualAI Research                                      #
# Copyrights licensed under the CC BY 4.0 License.                             #
# See the accompanying LICENSE file for terms.                                 #
#                                                                              #
# Date: 11-09-2020                                                             #
# Author(s): Lorenzo Pellegrini                                                #
# E-mail: contact@continualai.org                                              #
# Website: clair.continualai.org                                               #
################################################################################


import functools
import sys
import warnings
from collections import OrderedDict
from inspect import signature, Parameter
from typing import Callable, Union, Optional, Sequence, List, Dict, Any, \
    Iterable


def _merge_state_data(a, b, in_place=False):
    """
    Merges two "dictionary" objects together by updating the content of the
    first parameter. By default, creates a new copy of it.

    The first parameter must be a Python dict while the second parameter can be
    a dict, a named tuple or a :class:`StrategyFlow`. If the second parameter
    is a plain object, the ``vars()`` built-in function  will be used to obtain
    a dictionary out of it.

    :param a: The original dictionary object.
    :param b: The update dictionary object.
    :param in_place: If True, a will be updated without making a copy of it.
        Defaults to False.
    :return: A dictionary objects containing the elements from the given
        dictionary parameters. Keys contained in the second parameter take
        precedence over the ones of the first parameter.
    """
    if not in_place:
        a = dict(a)
    if isinstance(b, StrategyFlow):
        a.update(b.extract_self_namespace())
        a.update(b.get_results_namespace())
        a.update(b.get_flattened_kwargs())
    elif isinstance(b, dict):  # Also considers OrderedDict
        a.update(b)
    elif hasattr(b, '_asdict'):  # Manages namedtuple
        a.update(b._asdict())
    else:
        a.update(vars(b))  # Treat as plain object

    return a


def _match_positional_arguments(part: Callable, pos_args: Sequence[Any]):
    """
    Given a function and the list of parameters passed to it in a positional
    way, separates the "positional only" parameters from the "positional or
    keyword" ones.

    :param part: The callable object.
    :param pos_args: The list of positional arguments.
    :return: A tuple containing two elements: the first element is an ordered
        dictionary of "positional only" arguments, with keys being the parameter
        names. The second is an ordered dictionary of "positional or keyword"
        parameters, with keys being the parameter names.
    """
    part_signature = signature(part)
    keyword_params = OrderedDict()
    positional_params = OrderedDict()

    # From the doc of "inspect.Signature.parameters":
    # https://docs.python.org/3/library/inspect.html
    # An ordered mapping of parameters’ names to the corresponding Parameter
    # objects. Parameters appear in strict definition order, including
    # keyword-only parameters.
    for param_idx, parameter_name in enumerate(part_signature.parameters):
        if param_idx >= len(pos_args):
            break
        kind = part_signature.parameters[parameter_name].kind
        if kind == Parameter.POSITIONAL_OR_KEYWORD:
            keyword_params[parameter_name] = pos_args[param_idx]
        elif kind == Parameter.POSITIONAL_ONLY:
            positional_params[parameter_name] = pos_args[param_idx]

    return positional_params, keyword_params


StrategyPart = Callable
StrategyChild = Union['FlowGroup', StrategyPart]
StrategyChildId = Union[StrategyChild, str]


def _is_func_decorated(f, check_flow=None):
    """
    Checks if the given function was decorated using a decorator returned from
    make_strategy_part_decorator.

    :param f: The function to check.
    :param check_flow: If not None, must be the name of a specific instance of
        a decorator returned from make_strategy_part_decorator.
    :return: True if the function is decorated. If check_flow is not None,
        returns True if the function is decorated and one of the decorators
        is the one whose name is specified by the check_flow parameter.
    """
    result = 'flows' in f.__dict__
    if check_flow is not None and result:
        result = check_flow in f.flows
    return result


def make_strategy_part_decorator(flow_field: str):
    """
    Factory for a strategy part decorator.

    :param flow_field: The name of the decorator.
    :return: A decorator that can be used to decorate class methods.
    """
    def call_strategy_part(f):
        # Ok, from now on don't touch anything
        # https://www.python.org/dev/peps/pep-0232/
        use_existing_wrapper = False
        if not _is_func_decorated(f):
            f.flows = set()
        else:
            use_existing_wrapper = True
        if flow_field in f.flows:
            raise ValueError('Error decorating ' + str(f.__name__) +
                             ', ' + str(flow_field) + 'flow was ' +
                             'already added to this method')
        f.flows.add(flow_field)

        if use_existing_wrapper:
            return f.flow_wrapper

        @functools.wraps(f)
        def wrapper(self, *args, **kwargs):
            all_flows = [getattr(self, st_name)
                         for st_name in f.flows]
            flow: Optional[StrategyFlow] = next(
                filter(lambda st: st.is_running(), all_flows), None)
            if flow is None:
                # Method is not being used by a flow, which means that it was
                # called directly.
                return f(self, *args, **kwargs)

            pos_arguments, pos_kw_arguments = _match_positional_arguments(
                f, (self,) + args)

            # pos_kw_arguments and kwargs may contain the same keys
            # (parameter names), but it will be the exact same parameter!
            pos_kw_and_kwargs = {**pos_kw_arguments, **kwargs}

            # Check if we have to call plugins functions
            if not flow.is_plugin_flow():
                kwargs_plugin = dict(pos_kw_and_kwargs)
                kwargs_plugin.pop('self', None)

                # We are in the root flow. This means that the function is
                # being called on the main strategy object. Before calling
                # the actual function we have to execute the methods with
                # the same name found in plugins.
                for plugin in flow.get_strategy_plugins():
                    # First, check if the field with the same name is
                    # actually a plugin's method. Any other callables
                    # are ok too.
                    plugin_callback = getattr(plugin, f.__name__, None)
                    if callable(plugin_callback) and _is_func_decorated(
                            plugin_callback,
                            check_flow=flow.flow_name):
                        # Actually call the plugin function
                        try:
                            getattr(plugin, f.__name__)(**kwargs_plugin)
                        except Exception as _:
                            t, v, tb = sys.exc_info()
                            flow.signal_internal_traceback(tb)
                            raise

            # pos_arguments may contain different parameters with the
            # same keys (names) from pos_kw_and_kwargs.
            # https://www.python.org/dev/peps/pep-0570/#semantic-corner-case
            # Not that we care... we just don't push positional-only arguments
            # to the kwargs stack!
            # Also, positional-only arguments are extremely rare ...
            flow.push_kwargs(pos_kw_and_kwargs)

            try:
                namespace = dict()
                # Python, passes "self" as a positional argument,
                # no matter if the method defines self as
                # POSITIONAL_ONLY or POSITIONAL_OR_KEYWORD
                _merge_state_data(namespace, flow, in_place=True)
                _merge_state_data(namespace, pos_kw_arguments, in_place=True)

                try:
                    ret = _execute_part_call(f, pos_arguments, namespace,
                                             flow)
                except Exception as _:
                    t, v, tb = sys.exc_info()
                    flow.signal_internal_traceback(tb)
                    raise

                # We have to decide if this makes sense!
                # flow.update_results_namespace(ret)
            finally:
                flow.pop_kwargs()

            return ret

        f.flow_wrapper = wrapper
        functools.update_wrapper(wrapper, f)

        return wrapper
    return call_strategy_part


TrainingFlow = make_strategy_part_decorator('training_flow')
TestingFlow = make_strategy_part_decorator('testing_flow')


def _child_name(child: StrategyChildId) -> Optional[str]:
    """
    Returns the flow name of the given part. The part can be a plain python
    function, a FlowGroup or a string.

    :param child: The part to get the name of.
    :return: The part name.
    """
    if child is None:
        return None
    if isinstance(child, str):
        return child
    if isinstance(child, FlowGroup):
        return child.group_name
    return child.__name__


def _all_element_names(parts: Sequence[StrategyChild]) -> List[str]:
    """
    Gets the complete list of children names from a part list.

    :param parts: A list of parts.
    :return: The complete list of children names, including the names
        from the parts listed in the parameter.
    """
    names = [_child_name(child) for child in parts]
    for child in parts:
        if isinstance(child, FlowGroup):
            names += child.child_names

    return names


def _get_part_names(parts: List[StrategyChild]) -> List[str]:
    """
    Gets the names of the given parts.

    :param parts: The parts to get the name of.
    :return: The names of the given parts.
    """
    return [_child_name(part) for part in parts]


def _find_part_group(parts: List[StrategyChild], part_name: str) -> \
        (int, StrategyChild):
    """
    Finds the group the given part belongs to.

    This method searches for the required part in all the elements given by the
    first parameter (including their sub-trees). If the part is found in a
    sub-group, this method will return the index and reference of the relative
    element in the first parameter list.

    :param parts: The list of parts to search.
    :param part_name: The name of the part whose group has to be found.
    :return: The index and reference of its containing group. If the part is
        one of the elements of the "parts" parameter, its index and reference
        will be returned.
    """
    for part_idx, part in enumerate(parts):
        if _child_name(part) == part_name:
            return part_idx, part
        if isinstance(part, FlowGroup) and part_name in part.child_names:
            return part_idx, part
    raise ValueError('No child part found with the given name')


def _contains_duplicates(parts: Sequence[StrategyChild]):
    """
    Checks if the given sequence contains parts with duplicate names.

    :param parts: The list of parts.
    :return: True if a duplicate is found, False otherwise.
    """
    names = [_child_name(child) for child in parts]
    for child in parts:
        if isinstance(child, FlowGroup):
            names = names + list(child.child_names)

    return len(names) != len(set(names))


def _empty_listener(*args):
    pass


def _execute_part_call(part, positional_arguments: Sequence[Any],
                       namespace_dict: Dict[str, Any],
                       strategy_flow: 'StrategyFlow'):
    """
    Calls a strategy part given the positional arguments, the available keyword
    arguments (namespace).

    :param part: The part to be called.
    :param positional_arguments: The positional-only arguments.
    :param namespace_dict: The available namespace elements used to fill
        the remaining arguments.
    :param strategy_flow: The reference to the strategy flow.
    :return: The value returned from the part call.
    """
    if 'flow_listeners' in namespace_dict:
        flow_listeners = namespace_dict['flow_listeners']
    else:
        flow_listeners = _empty_listener

    if not isinstance(flow_listeners, Sequence):
        flow_listeners = [flow_listeners]

    # Get value of required parameters
    selected_parameters = dict()
    part_signature = signature(part)
    for parameter_name in part_signature.parameters:
        if part_signature.parameters[parameter_name].kind == \
                Parameter.VAR_KEYWORD:
            # Function declared the **kwargs argument
            # This means that we pass the entire namespace!
            selected_parameters = dict(namespace_dict)
            break
        elif part_signature.parameters[parameter_name].kind in \
                [Parameter.POSITIONAL_OR_KEYWORD, Parameter.KEYWORD_ONLY]:
            if parameter_name in namespace_dict:
                selected_parameters[parameter_name] = \
                    namespace_dict[parameter_name]

    # TODO: better listeners system
    # Call listeners (before part execution)
    [listener('before', _child_name(part), selected_parameters)
     for listener in flow_listeners]

    # Part execution
    try:
        part_result = part(*positional_arguments, **selected_parameters)
    except Exception as _:
        t, v, tb = sys.exc_info()
        strategy_flow.signal_internal_traceback(tb)
        raise

    # Call listeners (after part execution)
    [listener('after', _child_name(part), selected_parameters)
     for listener in flow_listeners]

    return part_result


class FlowGroup:
    """
    This class defines a flow group.

    Simply put, a flow group contains a sequence of parts that are executed
    one after another when the group's __call__ method is invoked. A FlowGroup
    can be  also flagged as a loop, which means that the last element of the
    parts sequence controls whenever the loops would stop or continue by
    returning True (continue) or False (break). Parts can be class methods
    (usually decorated with @TrainingFlow or @TestingFlow) or another FlowGroup.
    This means that flow groups are organized in a tree where methods are the
    leaves and flow groups are intermediate nodes.

    In a StrategyFlow can't exist parts with duplicate names. A name of a part
    is the method name for class methods or the FlowGroup name for FlowGroups.
    """
    def __init__(self, flow: 'StrategyFlow', elements: List[StrategyChild],
                 group_name: str, is_loop: bool = False):
        """
        Creates a new flow group.

        :param flow: The flow this group belongs to.
        :param elements: The initial sequence of parts. Parts can be methods
            or other FlowGroups. No duplicate names are allowed.
        :param group_name: The group name. Must be unique through the entire
            StrategyFlow tree.
        :param is_loop: If True, the __call__ will loop through the parts
            until the last part return False.
        """
        self.flow: 'StrategyFlow' = flow
        self.children = elements
        self.group_name = group_name
        self.is_loop = is_loop
        self.child_names = []
        self.direct_child_names = []
        self._update_child_names()
        if _contains_duplicates([self]):
            raise ValueError('The elements list contains duplicated elements')
        FlowGroup._set_parts_flow(flow, self.children)

    def __call__(self, *args, **kwargs):
        """
        Executes the contained parts sequentially.

        If this group was created with the is_loop flag set as True, all parts
        will be executed in a loop until the last part returns False.
        :param args: Ignored
        :param kwargs: A dictionary of values to be set in the call namespace.
            Children parts that have any of these parameters in their method
            signature will get the relative value with the same name. This works
            very similar to a dependency injection.

            For instance, calling this group as "group(device="cpu")" will
            expose the "device" value to all children parts.

        :return: The return value of the last part. Please note that if this
            group is a loop, the return value will be True.
        """
        if len(args) > 0:
            warnings.warn('Unnamed arguments will be ignored')

        last_result = None
        continue_loop = True

        while continue_loop:
            for part in self.children:
                # Memory cleanup before next call
                # IDEs may report an unused assignment, but this is correct!
                last_result = None

                # Decorators already manages parameter selection and
                # namespace updates
                try:
                    last_result = part(*args, **kwargs)
                except Exception as _:
                    t, v, tb = sys.exc_info()
                    # Don't worry, the traceback will at least contain one
                    # trace of the previous line (the call to self.root_group of
                    # the StrategyFlow):
                    # "last_result = part(*args, **kwargs)"
                    # Also, the debugger will still correctly display the full
                    # stack!
                    self.flow.signal_internal_traceback(tb)
                    raise

            continue_loop = self.is_loop and bool(last_result)
        return last_result

    def _append_elements(self, to_group: StrategyChildId,
                         after_child: Optional[StrategyChildId],
                         new_elements: List[StrategyChild],
                         at_beginning: bool = False):
        to_group = _child_name(to_group)
        after_child = _child_name(after_child)

        if self.group_name == to_group:
            FlowGroup._check_compatibility(new_elements, [self])
            if at_beginning:
                after_child_idx = 0
            elif after_child is not None:
                if after_child not in self.direct_child_names:
                    raise ValueError('Bad parameter after_child: can\'t '
                                     'find a child with the given name')
                after_child_idx = self.direct_child_names.index(after_child) + 1
            else:
                after_child_idx = len(self.direct_child_names)

            FlowGroup._set_parts_flow(self.flow, new_elements)
            self.children[after_child_idx:after_child_idx] = new_elements
        else:
            child_idx, child = _find_part_group(self.children, to_group)
            if not isinstance(child, FlowGroup):
                raise ValueError('Can only append to groups')
            child._append_elements(to_group, after_child, new_elements,
                                   at_beginning=at_beginning)

        self._update_child_names()

    def _replace(self, to_be_replaced: StrategyChildId,
                 new_element: StrategyChild) -> StrategyChild:
        to_be_replaced = _child_name(to_be_replaced)

        if to_be_replaced in self.direct_child_names:
            child_at = self.direct_child_names.index(to_be_replaced)
            children_without_element = list(self.children)
            del children_without_element[child_at]
            FlowGroup._check_compatibility(
                [new_element], children_without_element)
            replaced_element = self.children[child_at]
            FlowGroup._set_parts_flow(self.flow,
                                      [new_element])
            self.children[child_at] = new_element
        else:
            child_idx, child = _find_part_group(self.children, to_be_replaced)
            # Note: can't be a Callable as they are leaves
            replaced_element = child.replace(to_be_replaced, new_element)

        self._update_child_names()
        return replaced_element

    def _remove(self, to_be_removed: StrategyChildId) -> StrategyChild:
        to_be_removed = _child_name(to_be_removed)

        if to_be_removed in self.direct_child_names:
            child_at = self.direct_child_names.index(to_be_removed)
            removed_element = self.children[child_at]
            del self.children[child_at]
        else:
            _, child = _find_part_group(self.children, to_be_removed)
            # Note: can't be a Callable as they are leaves
            removed_element = child.remove(to_be_removed)
        self._update_child_names()
        return removed_element

    def _update_child_names(self):
        self.child_names = _all_element_names(self.children)
        self.direct_child_names = _get_part_names(self.children)

    @staticmethod
    def _check_compatibility(added_elements: List[StrategyChild],
                             existing_elements: List[StrategyChild]):
        if _contains_duplicates(added_elements):
            raise ValueError('New elements contain duplicates')

        new_names = set(_all_element_names(added_elements))
        existing_names = set(_all_element_names(existing_elements))
        names_intersection = existing_names.intersection(new_names)
        if len(names_intersection) > 0:
            raise ValueError('Element(s) with the same name of one or more of '
                             'the new elements already exist')

    def _change_part_flow(self, flow: 'StrategyFlow'):
        self._set_parts_flow(flow, self.children)
        self.flow = flow

    @staticmethod
    def _set_parts_flow(flow: 'StrategyFlow',
                        parts: Sequence[StrategyChild]):
        for part in parts:
            if isinstance(part, FlowGroup):
                part._change_part_flow(flow)
            else:  # Is a class method (Callable)
                # noinspection PyTypeChecker
                if not _is_func_decorated(part, check_flow=flow.flow_name):
                    raise ValueError(
                        'Error defining group, ' + str(part.__name__) +
                        ' was not meant to be used by the ' +
                        str(flow.flow_name) + '. Did you forget to decorate ' +
                        'the method?')


class StrategyFlow:
    # TODO: manage warning (use of protected methods of FlowGroup)
    # TODO: better group creation and composition syntax
    """
    Implementation of a strategy flow.

    A strategy flow describes the parts of a Continual Learning strategy.
    A parts is usually implemented as a class method of a strategy class.
    Parts can be joined in named groups, which makes it easier to define the
    flow and compose parts in higher-level nodes. Also, groups can be used to
    create loops (usually used to describe epochs or testing steps).

    Considering that parts are strategy class methods, parts are commonly used
    as callback mechanism (for instance "before_training",
    "after_training_epoch", etc.).

    Parts and groups can't be duplicated inside a flow.

    The two default flows are the training and testing flows, which are found
    in every strategy. Methods belonging to the training flow must be
    annotated using the @TrainingFlow decorator while methods belonging to the
    testing flow must be annotated with the @TestingFlow decorator. A method
    can be annotated with more than a flow decorator at the same time.

    Each strategy can also define plugins. Plugins are commonly used to
    modularize common reusable Continual Learning patterns such as multi-head
    management, rehearsal (a.k.a. replay), distillation, ... or even to attach
    the desired metrics system (accuracy, time, ram usage, confusion matrices,
    ...).

    The StrategyFlow is a callable object that, when executed, runs the
    described parts sequentially. Each call to the training flow executes an
    incremental training step on a "batch" (or "task") of new data while the
    testing flow will run an evaluation loop on the test sets. A strategy should
    implement the desired training and testing procedures as parts.

    Each part, being a class method, can define method parameters as usual.
    The flow will inject the correct parameter values by looking at different
    locations:
        - first, a parameter with the same name is searched in the arguments
        passed to the flow. Considering that a strategy part can call other
        parts (to provide a callback or to obtain results), parameters passed to
        previous method calls in the stack are searched for, too. The collection
        of those values is called "arguments namespace";
        - second, a global flow namespace exists where each part may publish its
        results. These values are stored and used for parameter injection and,
        like the arguments passed to the flow, they are discarded after each
        flow execution. Those values form the "results namespace". The good part
        of it is that this namespace is shared with plugins, which makes it
        easier to modularize some common behaviours. When a part publishes a
        result, values with the same name found in the first group are
        discarded;
        - third, fields of the strategy class or any of its attached plugins
        are searched for. This is called "self namespace". It goes without
        saying, those are the only namespace values that are persisted across
        different flow executions. Fields starting with "_" are not considered.
        Also, fields whose values are instances of "StrategyFlow" or "FlowGroup"
        are not considered, as they would pollute the namespace.

    All those values form a global namespace. For values with the same, elements
    or the first group take precedence over the ones in the second group. Values
    from the second group take precedence over the ones in the last group. This
    means that

    This mechanism ensures that any part can have total visibility over the
    state of the strategy.
    """

    def __init__(self, self_ref: Any, flow_name: str, root_group_name: str):
        """
        Creates a flow group.

        :param self_ref: The reference to the object this flow belongs to. This
            is usually the the strategy (or the plugin) object.
        :param flow_name: The name of the flow.
        :param root_group_name: The name of the root group.
        """
        self.self_ref: Any = self_ref
        self.flow_name: str = flow_name
        self.root_flow: Optional['StrategyFlow'] = None
        self.flow_listeners: List[Callable] = []
        self.namespace_dict: Dict[str, Any] = dict()
        self.kwargs_stack: List[Dict[str, Any]] = []
        self._running: bool = False
        self.internal_tracebacks = set()
        # TODO: leaking "self" in constructor
        self.root_group: FlowGroup = FlowGroup(self, [], root_group_name, False)

    def append_part_list(self, parts: Sequence[StrategyChild],
                         to_group: Optional[StrategyChildId] = None,
                         after_part: Optional[StrategyChildId] = None,
                         at_beginning: bool = False):
        """
        Appends a list of parts to a flow group.

        A flow group is a sequence of parts semantically tied together. For
        instance "TrainingEpoch". Parts in a group are executed sequentially.

        :param parts: A list of parts to append to an existing group.
        :param to_group: The name of object reference to the group. Defaults to
            the root group.
        :param after_part: If not None, the parts list will be appended after
            the given part/group (described by name or reference).
        :param at_beginning: If True, the parts will be appended at the
            beginning of the group. Defaults to False, which means that parts
            will be appended at the end. Can't be True at the same time of the
            "after_part" parameter.
        :return: None.
        """
        if after_part is not None and at_beginning:
            raise ValueError('after_part and at_beginning can\'t be set '
                             'at the same time')
        if to_group is None:
            to_group = self.root_group
        parts = list(parts)
        self.root_group._append_elements(to_group, after_part, parts,
                                         at_beginning=at_beginning)

    def append_new_group(self, group_name: str,
                         parts: Sequence[StrategyChild],
                         is_loop: bool = False,
                         to_group: Optional[StrategyChildId] = None,
                         after_part: Optional[StrategyChildId] = None,
                         at_beginning: bool = False):
        """
        Creates a new flow group.

        :param group_name: The group name.
        :param parts: A list of initial parts belonging to the group. Can be
            an empty list.
        :param is_loop: If True, the group will be flagged as a loop, which
            means that its parts will be executed in a loop until the last
            part returns False.
        :param to_group: Appends the newly created group to the group described
            (by name or reference) by this parameter. Defaults to None, which
            means that the root group is used.
        :param after_part: If not None, the group will be appended after
            the given part/group (described by name or reference).
        :param at_beginning: If True, the group will be appended at the
            beginning of the group. Defaults to False, which means that the
            group will be appended at the end. Can't be True at the same time
            of the "after_part" parameter.
        :return: The new group
        """
        new_group = FlowGroup(self, list(parts), group_name, is_loop=is_loop)
        if to_group is None:
            to_group = self.root_group
        self.root_group._append_elements(to_group, after_part, [new_group],
                                         at_beginning=at_beginning)
        return new_group

    def remove_part(self, part_name: StrategyChildId):
        """
        Removes a part (or group) from the flow.

        The part will be searched in any flow sub-group.
        :param part_name: The part name or reference.
        :return: The removed part.
        """
        return self.root_group._remove(part_name)

    def replace_part(self, part_name: StrategyChildId,
                     replacement: StrategyChild):
        """
        Replaces a part with another.

        :param part_name: The part (by name or reference) to replace.
        :param replacement: The replacement.
        :return: The replaced part.
        """
        return self.root_group._replace(part_name, replacement)

    def __call__(self, **kwargs):
        """
        Executes this flow .

        :param kwargs: A dictionary of named parameters that will be passed
            to the parts of the strategy (see class description).
        :return: The return value of the last part.
        """
        if self.root_flow is not None:
            return self.root_flow(**kwargs)

        if self.is_running():
            raise ValueError('Flow already running')

        self._running = True

        self.internal_tracebacks = set()
        self.namespace_dict = dict()
        self.namespace_dict['flow_listeners'] = self.flow_listeners
        self.namespace_dict['strategy'] = self.self_ref
        try:
            result = self.root_group(**kwargs)
        except Exception as _:
            t, v, tb = sys.exc_info()
            # This cleans up the traceback so that decorators and internal
            # procedures don't appear!
            raise v.with_traceback(self._cleanup_traceback(tb.tb_next))

        finally:
            self._running = False
            self.namespace_dict = dict()
            self.internal_tracebacks = set()
        return result

    def is_running(self):
        """
        Checks if this flow is running.

        :return: True if this flow is running, False otherwise.
        """
        if self.is_plugin_flow():
            return self.root_flow.is_running()
        # If the plugins field is not found, return []
        return self._running

    def add_part_change_listener(self, part_change_listener):
        """
        Adds a flow listener.

        :param part_change_listener: The flow listener to attach. The listener
            will receive a callback each time a part is executed.
        :return: None
        """
        if self.root_flow is not None:
            return self.root_flow.add_part_change_listener(part_change_listener)
        self.flow_listeners.append(part_change_listener)

    def remove_part_change_listener(self, part_change_listener):
        """
        Removes a flow listener.

        :param part_change_listener: The flow listener to remove.
        :return: None
        """
        if self.root_flow is not None:
            return self.root_flow.remove_part_change_listener(
                part_change_listener)
        self.flow_listeners.remove(part_change_listener)

    def add_strategy_plugin(self, plugin):
        """
        Adds a plugin to the strategy.

        This will check the plugin for a flow with the same name. If a flow
        with the same name can't be found, an exception is raised. The flow
        found in the plugin is then instrumented to consider this instance as
        the root flow.

        :param plugin: The added plugin.
        :return: None
        """
        if not hasattr(plugin, self.flow_name):
            raise ValueError('Can\'t find flow with name', self.flow_name)
        getattr(plugin, self.flow_name).set_root_flow(self)

    def set_root_flow(self, root_flow):
        """
        Sets the root flow.

        When a root flow is set, the current flow will delegate any operation
        to the root flow. This usually happens in plugins, where the root
        flow is the strategy one.

        :param root_flow: The root flow.
        :return: None
        """
        self.root_flow = root_flow
        self.namespace_dict = dict()

    def get_strategy_plugins(self):
        """
        Returns a list of strategy plugins.

        :return: A list of plugins.
        """
        if self.is_plugin_flow():
            return self.root_flow.get_strategy_plugins()
        # If the plugins field is not found, return []
        return getattr(self.self_ref, 'plugins', [])

    def is_plugin_flow(self):
        """
        Checks if this is a flow belonging to a plugin.

        That is, this returns True if a root flow is set.

        :return: True if this is a plugin flow, False if it's the main
            strategy object flow.
        """
        return self.root_flow is not None

    def extract_self_namespace(self):
        """
        Extracts the "self namespace" from the strategy object and any of its
        plugins. The namespace is then usually searched for when injecting
        parameter values of the parts.

        Simply put, the "self namespace" is the collection of class fields
        belonging to the strategy object and any of its plugins. Fields
        starting with and underscore "_" will be ignored. For more info, refer
        to the class documentation.

        :return: A dictionary containing the self namespace.
        """
        if self.is_plugin_flow():
            return self.root_flow.extract_self_namespace()
        return self._extract_self_data()

    def get_results_namespace(self):
        """
        Returns the "results namespace". The namespace is then usually searched
        for when injecting parameter values of the parts.

        The results namespace is the namespace containing all the results
        published from any part of the strategy or its plugins. This
        namespace is cleaned up after each flow execution.

        For a result to be stored in the "results namespace", the part must
        publish it using the "update_results_namespace" method. However, it is
        recommended to use the "update_namespace" method provided by class
        :class:`StrategySkeleton`, which is easier to use.

        :return: A dictionary containing the results namespace.
        """
        if self.is_plugin_flow():
            return self.root_flow.get_results_namespace()
        return self.namespace_dict

    def update_results_namespace(self, update_dict):
        """
        Updates the "results namespace".

        For a result to be stored in the "results namespace", the part must
        publish it using this method. However, it is recommended to use the
        "update_namespace" method provided by class :class:`StrategySkeleton`,
        which is easier to use.

        :param update_dict: The dictionary or named tuple to merge in the
            results namespace.
        :return: None
        """
        if self.is_plugin_flow():
            return self.root_flow.update_results_namespace(update_dict)
        self._update_namespace_using_return_value(update_dict)

    def get_flattened_kwargs(self) -> Dict[str, Any]:
        """
        Gets the "arguments namespace". This dictionary is then usually
        searched for when injecting parameter values of the parts.

        For more info refer to the class documentation.

        This method looks at the whole call stack on previous strategy parts
        to gather all previous named parameters. Their values are then flattened
        (giving priority at values from the most recent call) in the single
        dictionary returned my this method.

        Not all parameter from previous part calls are included:

        -   Only parameters of part calls from the current call *stack* are
            included, which means that parameters from part calls that already
            returned are not included.
        -   Positional only parameters are not included, while "positional
            or keyword" or "keyword only" parameters are included.
        -   When the "result namespace" is updated using
            "update_results_namespace", any element of the "arguments namespace"
            with a name included in the update dictionary is eliminated (because
            part results take precedence over parameters from the call stack).
            For instance, consider a part that receives as an input a batch,
            executes a data augmentation procedure and returns it. Let's call
            this value "train_batch". It wouldn't make sense if, after that part
            call, the value of "train_batch" pointed to its previous,
            non-augmented, version. This is why result values take precedence
            over values from the arguments namespace.

        :return: The "arguments namespace". That is, a dictionary of keyword
            arguments form previous (stack) part calls.
        """
        if self.is_plugin_flow():
            return self.root_flow.get_flattened_kwargs()

        kwargs_dict = dict()
        for stack_kwargs_dict in self.kwargs_stack:
            _merge_state_data(kwargs_dict, stack_kwargs_dict, in_place=True)
        return kwargs_dict

    def push_kwargs(self, args_dict: Dict[str, Any]) -> None:
        """
        Pushes keyword arguments.

        This is usually automatically done when calling a part decorated
        for the current flow. This updates the "arguments namespace".

        :param args_dict: The keyword arguments.
        :return: None
        """
        if self.is_plugin_flow():
            return self.root_flow.push_kwargs(args_dict)

        args_dict = dict(args_dict)
        args_dict.pop('self', None)

        self.kwargs_stack.append(args_dict)

    def pop_kwargs(self) -> None:
        """
        Pops keyword arguments.

        This is usually automatically done when returning from a part decorated
        for the current flow. This updates the "arguments namespace".

        :return: None
        """
        if self.is_plugin_flow():
            return self.root_flow.pop_kwargs()
        self.kwargs_stack.pop()

    def signal_internal_traceback(self, tb):
        """
        Adds a traceback object to the ignore list.

        This is usually done to prevent the traceback from showing a huge list
        of internal decorator calls.

        :param tb: The traceback object to be ignored.
        :return: None
        """
        if self.is_plugin_flow():
            return self.root_flow.signal_internal_traceback(tb)
        self.internal_tracebacks.add(tb)

    def _update_namespace_using_return_value(self, update_dict):
        """
        Updates the results namespace.

        :param update_dict: The dictionary or named tuple to merge in the
            results namespace.
        :return: The new results namespace.
        """
        if self.is_plugin_flow():
            return self.root_flow._update_namespace_using_return_value(
                update_dict)

        if isinstance(update_dict, dict) or (isinstance(update_dict, tuple) and
                                             hasattr(update_dict, '_asdict')):
            update_dict_as_dict = update_dict

            if isinstance(update_dict, tuple):
                update_dict_as_dict = update_dict._asdict()
            update_dict_as_dict.pop('self', None)

            self._remove_kwargs_from_stack(update_dict_as_dict.keys())

            return _merge_state_data(
                self.namespace_dict, update_dict, in_place=True)
        return self.namespace_dict

    def _cleanup_traceback(self, tb):
        """
        Cleanups the exception traceback.

        This is done to ensure that internal decorator called are not shown
        to the user.

        This method works in-place.

        :param tb: The traceback.
        :return: A cleaned up version of the traceback.
        """
        actual_tb = tb
        while actual_tb is not None:
            if actual_tb.tb_next in self.internal_tracebacks:
                actual_tb.tb_next = actual_tb.tb_next.tb_next
            else:
                actual_tb = actual_tb.tb_next
        return tb

    def _extract_self_data(self):
        """
        Extract all fields from the strategy and any plugin.

        Fields from the main strategy object take precedence over plugins
        ones.

        :return: A list of fields from the strategy and its plugins.
        """
        self_namespace = dict()
        if hasattr(self.self_ref, 'plugins'):
            for plugin in self.self_ref.plugins:
                flow: 'StrategyFlow' = getattr(plugin, self.flow_name)
                self_namespace = _merge_state_data(
                    self_namespace, flow._extract_self_data())

        self_dict = dict(vars(self.self_ref))
        private_self_fields = [field for field in self_dict.keys()
                               if field.startswith('_')]
        flows_and_groups_fields = [field for field in self_dict.keys()
                                   if isinstance(self_dict[field], StrategyFlow)
                                   or isinstance(self_dict[field], FlowGroup)]
        ignored_self_fields = set(private_self_fields + flows_and_groups_fields)
        [self_dict.pop(key) for key in ignored_self_fields]
        return _merge_state_data(self_namespace, self_dict)

    def _remove_kwargs_from_stack(self, arg_names: Iterable[Any]):
        """
        Removes parameters from the call stack.

        This is used to remove elements from the "argument namespace", which
        has to be done when the "results namespace" is updated.

        :param arg_names: The names of the parameters.
        :return: None
        """
        for arg_name in arg_names:
            for stack_kwargs_dict in self.kwargs_stack:
                if arg_name in stack_kwargs_dict:
                    del stack_kwargs_dict[arg_name]


__all__ = ['StrategyPart', 'StrategyChild', 'StrategyChildId',
           'make_strategy_part_decorator', 'TrainingFlow', 'TestingFlow',
           'FlowGroup', 'StrategyFlow']
