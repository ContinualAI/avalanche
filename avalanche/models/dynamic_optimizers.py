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

colors = {
    "END": "\033[0m",
    0: "\033[32m",
    1: "\033[33m",
    2: "\033[34m",
    3: "\033[35m",
    4: "\033[36m",
}
colors[None] = colors["END"]


def map_optimized_params(optimizer, new_params):
    current_parameters_mapping = defaultdict(dict)
    not_found = []
    for n, p in new_params.items():
        g = None
        # Find param in optimizer
        found = False
        for group_idx, group in enumerate(optimizer.param_groups):
            params = group["params"]
            for po in params:
                if id(po) == id(p):
                    g = group_idx
                    found = True
                    break
        if not found:
            not_found.append(n)
        current_parameters_mapping[n] = g
    return current_parameters_mapping, not_found


def build_tree_from_name_groups(name_groups):
    root = TreeNode("")  # Root node
    node_mapping = {}

    # Iterate through each string in the list
    for name, group in name_groups.items():
        components = name.split(".")
        current_node = root

        # Traverse the tree and construct nodes for each component
        for component in components:
            if component not in current_node.children:
                current_node.children[component] = TreeNode(
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


def print_group_information(node, prefix=""):
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
        print_group_information(child_node, prefix + "  ")


class OptimizedParameterStructure:
    """
    Structure holding a tree where each node is a pytorch module
    the tree is linked to the current model parameters
    """

    def __init__(self, new_params, optimizer, verbose=False):
        # Here we rebuild the tree
        name_groups, not_found = map_optimized_params(optimizer, new_params)
        self.root, self.node_mapping = build_tree_from_name_groups(name_groups)
        if verbose:
            print(f"Not found in optimizer: {not_found}")
            print_group_information(self.root)

    def __getitem__(self, name):
        return self.node_mapping[name]


class TreeNode:
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


def compare_keys(old_dict, new_dict):
    not_in_new = list(set(old_dict.keys()) - set(new_dict.keys()))
    in_both = list(set(old_dict.keys()) & set(new_dict.keys()))
    not_in_old = list(set(new_dict.keys()) - set(old_dict.keys()))
    return not_in_new, in_both, not_in_old


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


def update_optimizer(optimizer, new_params, optimized_params, reset_state=False):
    """Update the optimizer by adding new parameters,
    removing removed parameters, and adding new parameters
    to the optimizer, for instance after model has been adapted
    to a new task. The state of the optimizer can also be reset,
    it will be reset for the modified parameters.

    Newly added parameters are added by default to parameter group 0

    :param new_params: Dict (name, param) of new parameters
    :param optimized_params: Dict (name, param) of
        currently optimized parameters (returned by reset_optimizer)
    :param reset_state: Wheter to reset the optimizer's state (i.e momentum).
        Defaults to False.
    :return: Dict (name, param) of optimized parameters
    """
    not_in_new, in_both, not_in_old = compare_keys(optimized_params, new_params)
    # Change reference to already existing parameters
    # i.e growing IncrementalClassifier
    for key in in_both:
        old_p_hash = optimized_params[key]
        new_p = new_params[key]
        # Look for old parameter id in current optimizer
        found = False
        for group in optimizer.param_groups:
            for i, curr_p in enumerate(group["params"]):
                if id(curr_p) == id(old_p_hash):
                    found = True
                    if id(curr_p) != id(new_p):
                        group["params"][i] = new_p
                        optimized_params[key] = new_p
                        optimizer.state[new_p] = {}
                    break
        if not found:
            raise Exception(
                f"Parameter {key} expected but " "not found in the optimizer"
            )

    # Remove parameters that are not here anymore
    # This should not happend in most use case
    keys_to_remove = []
    for key in not_in_new:
        old_p_hash = optimized_params[key]
        found = False
        for i, group in enumerate(optimizer.param_groups):
            keys_to_remove.append([])
            for j, curr_p in enumerate(group["params"]):
                if id(curr_p) == id(old_p_hash):
                    found = True
                    keys_to_remove[i].append((j, curr_p))
                    optimized_params.pop(key)
                    break
        if not found:
            raise Exception(
                f"Parameter {key} expected but " "not found in the optimizer"
            )

    for i, idx_list in enumerate(keys_to_remove):
        for j, p in sorted(idx_list, key=lambda x: x[0], reverse=True):
            del optimizer.param_groups[i]["params"][j]
            if p in optimizer.state:
                optimizer.state.pop(p)

    # Add newly added parameters (i.e Multitask, PNN)
    # by default, add to param groups 0

    param_structure = OptimizedParameterStructure(new_params, optimizer, verbose=True)

    for key in not_in_old:
        new_p = new_params[key]
        group = param_structure[key].single_group
        optimizer.param_groups[group]["params"].append(new_p)
        optimized_params[key] = new_p
        optimizer.state[new_p] = {}

    if reset_state:
        optimizer.state = defaultdict(dict)

    return optimized_params


def add_new_params_to_optimizer(optimizer, new_params):
    """Add new parameters to the trainable parameters.

    :param new_params: list of trainable parameters
    """
    # optimizer.add_param_group({"params": new_params})
    optimizer.param_groups[0]["params"].append(new_params)


__all__ = ["add_new_params_to_optimizer", "reset_optimizer", "update_optimizer"]
