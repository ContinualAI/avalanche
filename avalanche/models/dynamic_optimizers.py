################################################################################
# Copyright (c) 2021 ContinualAI.                                              #
# Copyrights licensed under the MIT License.                                   #
# See the accompanying LICENSE file for terms.                                 #
#                                                                              #
# Date: 14-04-2020                                                             #
# Author(s): Antonio Carta                                                     #
# E-mail: contact@continualai.org                                              #
# Website: avalanche.continualai.org                                           #
################################################################################
"""
    Utilities to handle optimizer's update when using dynamic architectures.
    Dynamic Modules (e.g. multi-head) can change their parameters dynamically
    during training, which usually requires to update the optimizer to learn
    the new parameters or freeze the old ones.
"""
from collections import defaultdict

import numpy as np

from avalanche._annotations import deprecated, experimental
from avalanche.benchmarks import CLExperience
from avalanche.core import Adaptable, Agent

colors = {
    "END": "\033[0m",
    0: "\033[32m",
    1: "\033[33m",
    2: "\033[34m",
    3: "\033[35m",
    4: "\033[36m",
}
colors[None] = colors["END"]


@experimental(
    "New dynamic optimizers. The API may slightly change in the next versions."
)
class DynamicOptimizer(Adaptable):
    """Avalanche dynamic optimizer.

    In continual learning, many model architecture may change over time (e.g.
    adding new units to the classifier). This is handled by the `DynamicModule`.
    Changing the model's architecture requires updating the optimizer to add
    the new parameters in the optimizer's state.

    This class provides a simple wrapper to handle the optimizer
    update via the `Adaptable` Protocol.

    This class provides direct methods only for `zero_grad` and `step` to support
    basic training functionality. Other methods of the base optimizers must be
    called by using the base optimizer directly (e.g. `self.optim.add_param_group`).

    NOTE: the model must be adapted *before* calling this method.
    To ensure this, ensure that the optimizer is added to the agent state
    after the model:

    .. code-block::

        agent.model = model
        agent.optimizer = optimizer

        # ... more init code

        # pre_adapt will call the pre_adapt methods in order,
        # first model.pre_adapt, then optimizer.pre_adapt
        agent.pre_adapt(experience)
    """

    def __init__(self, optim, model, reset_state=False, verbose=False):
        self.optim = optim
        self.reset_state = reset_state
        self.verbose = verbose

        # initialize param-id
        self._optimized_param_id = update_optimizer(
            self.optim,
            dict(model.named_parameters()),
            None,
            reset_state=True,
            verbose=self.verbose,
        )

    def zero_grad(self):
        self.optim.zero_grad()

    def step(self):
        self.optim.step()

    def pre_adapt(self, agent: Agent, exp: CLExperience):
        """Adapt the optimizer before training on the current experience."""
        self._optimized_param_id = update_optimizer(
            self.optim,
            dict(agent.model.named_parameters()),
            self._optimized_param_id,
            reset_state=self.reset_state,
            verbose=self.verbose,
        )


def _map_optimized_params(optimizer, parameters, old_params=None):
    """
    Establishes a mapping between a list of named parameters and the parameters
    that are in the optimizer, additionally,
    returns the lists of:

    returns:
        new_parameters: Names of new parameters in the provided "parameters" argument,
                        that are not in the old parameters
        changed_parameters: Names and indexes of parameters that have changed (grown, shrink)
        not_found_in_parameters: List of indexes of optimizer parameters
                                 that are not found in the provided parameters
    """

    if old_params is None:
        old_params = {}

    group_mapping = defaultdict(dict)
    new_parameters = []

    found_indexes = []
    changed_parameters = []
    for group in optimizer.param_groups:
        params = group["params"]
        found_indexes.append(np.zeros(len(params)))

    for n, p in parameters.items():
        gidx = None
        pidx = None

        # Find param in optimizer
        found = False

        if n in old_params:
            search_id = id(old_params[n])
        else:
            search_id = id(p)

        for group_idx, group in enumerate(optimizer.param_groups):
            params = group["params"]
            for param_idx, po in enumerate(params):
                if id(po) == search_id:
                    gidx = group_idx
                    pidx = param_idx
                    found = True
                    # Update found indexes
                    assert found_indexes[group_idx][param_idx] == 0
                    found_indexes[group_idx][param_idx] = 1
                    break
            if found:
                break

        if not found:
            new_parameters.append(n)

        if search_id != id(p):
            if found:
                changed_parameters.append((n, gidx, pidx))

        if len(optimizer.param_groups) > 1:
            group_mapping[n] = gidx
        else:
            group_mapping[n] = 0

    not_found_in_parameters = [np.where(arr == 0)[0] for arr in found_indexes]

    return (
        group_mapping,
        changed_parameters,
        new_parameters,
        not_found_in_parameters,
    )


def _build_tree_from_name_groups(name_groups):
    root = _TreeNode("")  # Root node
    node_mapping = {}

    # Iterate through each string in the list
    for name, group in name_groups.items():
        components = name.split(".")
        current_node = root

        # Traverse the tree and construct nodes for each component
        for component in components:
            if component not in current_node.children:
                current_node.children[component] = _TreeNode(
                    component, parent=current_node
                )
            current_node = current_node.children[component]

        # Update the groups for the leaf node
        if group is not None:
            current_node.groups |= set([group])
            current_node.update_groups_upwards()  # Inform parent about the groups

        # Update leaf node mapping dict
        node_mapping[name] = current_node

    # This will resolve nodes without group
    root.update_groups_downwards()
    return root, node_mapping


def _print_group_information(node, prefix=""):
    # Print the groups for the current node

    if len(node.groups) == 1:
        pstring = (
            colors[list(node.groups)[0]]
            + f"{prefix}{node.global_name()}: {node.groups}"
            + colors["END"]
        )
        print(pstring)
    else:
        print(f"{prefix}{node.global_name()}: {node.groups}")

    # Recursively print group information for children nodes
    for child_name, child_node in node.children.items():
        _print_group_information(child_node, prefix + "  ")


class _ParameterGroupStructure:
    """
    Structure used for the resolution of unknown parameter groups,
    stores parameters as a tree and propagates parameter groups from leaves of
    the same hierarchical level
    """

    def __init__(self, name_groups, verbose=False):
        # Here we rebuild the tree
        self.root, self.node_mapping = _build_tree_from_name_groups(name_groups)
        if verbose:
            _print_group_information(self.root)

    def __getitem__(self, name):
        return self.node_mapping[name]


class _TreeNode:
    def __init__(self, name, parent=None):
        self.name = name
        self.children = {}
        self.groups = set()  # Set of groups (represented by index) this node belongs to
        self.parent = parent  # Reference to the parent node
        if parent:
            # Inform the parent about the new child node
            parent.add_child(self)

    def add_child(self, child):
        self.children[child.name] = child

    def update_groups_upwards(self):
        if self.parent:
            if self.groups != {None}:
                self.parent.groups |= (
                    self.groups
                )  # Update parent's groups with the child's groups
            self.parent.update_groups_upwards()  # Propagate the group update to the parent

    def update_groups_downwards(self, new_groups=None):
        # If you are a node with no groups, absorb
        if len(self.groups) == 0 and new_groups is not None:
            self.groups = self.groups.union(new_groups)

        # Then transmit
        if len(self.groups) > 0:
            for key, children in self.children.items():
                children.update_groups_downwards(self.groups)

    def global_name(self, initial_name=None):
        """
        Returns global node name
        """
        if initial_name is None:
            initial_name = self.name
        elif self.name != "":
            initial_name = ".".join([self.name, initial_name])

        if self.parent:
            return self.parent.global_name(initial_name)
        else:
            return initial_name

    @property
    def single_group(self):
        if len(self.groups) == 0:
            raise AttributeError(
                f"Could not identify group for this node {self.global_name()}"
            )
        elif len(self.groups) > 1:
            raise AttributeError(
                f"No unique group found for this node {self.global_name()}"
            )
        else:
            return list(self.groups)[0]


@deprecated(0.6, "update_optimizer with optimized_params=None is now used instead")
def reset_optimizer(optimizer, model):
    """Reset the optimizer to update the list of learnable parameters.

    .. warning::
        This function fails if the optimizer uses multiple parameter groups.

    :param optimizer:
    :param model:
    :return:
    """
    if len(optimizer.param_groups) != 1:
        raise ValueError(
            "This function only supports single parameter groups."
            "If you need to use multiple parameter groups, "
            "you can override `make_optimizer` in the Avalanche strategy."
        )
    optimizer.state = defaultdict(dict)

    parameters = []
    optimized_param_id = {}
    for n, p in model.named_parameters():
        optimized_param_id[n] = p
        parameters.append(p)

    optimizer.param_groups[0]["params"] = parameters

    return optimized_param_id


def update_optimizer(
    optimizer,
    new_params,
    optimized_params=None,
    reset_state=False,
    remove_params=False,
    verbose=False,
):
    """Update the optimizer by adding new parameters,
    removing removed parameters, and adding new parameters
    to the optimizer, for instance after model has been adapted
    to a new task. The state of the optimizer can also be reset,
    it will be reset for the modified parameters.

    Newly added parameters are added by default to parameter group 0

    WARNING: the first call to `update_optimizer` must be done before
    calling the model's adaptation.

    :param optimizer: the Optimizer object.
    :param new_params: Dict (name, param) of new parameters.
    :param optimized_params: Dict (name, param) of
        currently optimized parameters. In most use cases, it will be `None in
        the first call and the return value of the last `update_optimizer` call
        for the subsequent calls.
    :param reset_state: Whether to reset the optimizer's state (i.e momentum).
                        Defaults to False.
    :param remove_params: Whether to remove parameters that were in the optimizer
                          but are not found in new parameters. For safety reasons,
                          defaults to False.
    :param verbose: If True, prints information about inferred
                    parameter groups for new params.

    :return: Dict (name, param) of optimized parameters
    """
    (
        group_mapping,
        changed_parameters,
        new_parameters,
        not_found_in_parameters,
    ) = _map_optimized_params(optimizer, new_params, old_params=optimized_params)

    # Change reference to already existing parameters
    # i.e growing IncrementalClassifier
    for name, group_idx, param_idx in changed_parameters:
        group = optimizer.param_groups[group_idx]
        old_p = optimized_params[name]
        new_p = new_params[name]
        # Look for old parameter id in current optimizer
        group["params"][param_idx] = new_p
        if old_p in optimizer.state:
            optimizer.state.pop(old_p)
            optimizer.state[new_p] = {}

    # Remove parameters that are not here anymore
    # This should not happend in most use case
    if remove_params:
        for group_idx, idx_list in enumerate(not_found_in_parameters):
            for j in sorted(idx_list, key=lambda x: x, reverse=True):
                p = optimizer.param_groups[group_idx]["params"][j]
                optimizer.param_groups[group_idx]["params"].pop(j)
                if p in optimizer.state:
                    optimizer.state.pop(p)
                del p

    # Add newly added parameters (i.e Multitask, PNN)

    param_structure = _ParameterGroupStructure(group_mapping, verbose=verbose)

    # New parameters
    for key in new_parameters:
        new_p = new_params[key]
        group = param_structure[key].single_group
        optimizer.param_groups[group]["params"].append(new_p)
        optimizer.state[new_p] = {}

    if reset_state:
        optimizer.state = defaultdict(dict)

    return new_params


@deprecated(
    0.6,
    "parameters have to be added manually to the optimizer in an existing or a new parameter group",
)
def add_new_params_to_optimizer(optimizer, new_params):
    """Add new parameters to the trainable parameters.

    :param new_params: list of trainable parameters
    """
    optimizer.add_param_group({"params": new_params})


__all__ = ["add_new_params_to_optimizer", "reset_optimizer", "update_optimizer"]
