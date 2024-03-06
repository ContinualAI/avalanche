#!/usr/bin/env python3
import copy
from collections import defaultdict

import torch
import torch.nn as nn
from torchvision.models import resnet18

colors = {
    "END": "\033[0m",
    0: "\033[32m",
    1: "\033[33m",
    2: "\033[34m",
    3: "\033[35m",
    4: "\033[36m",
}
colors[None] = colors["END"]


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


def node_iterator(node_mapping):
    for key, node in node_mapping.items():
        assert key == node.name
        yield key, node.param, node.group


class TreeNode:
    def __init__(self, name, param=None, parent=None):
        self.name = name
        self.children = {}
        self.groups = set()  # Set of groups (represented by index) this node belongs to
        self.parent = parent  # Reference to the parent node
        self.param = param
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
        if len(self.groups) == 0:
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
    def group(self):
        if len(self.groups) > 1:
            raise AttributeError(
                f"No unique group found for this node {self.global_name()}"
            )
        else:
            return list(self.groups)[0]


def insert_node(name: str, root: TreeNode, group: int = None, verbose=False):
    """
    Inserts a node in the tree
    """
    components = name.split(".")
    current_node = root
    new_node = None
    for component in components:
        if component in current_node.children:
            current_node = current_node.children[component]
        else:
            new_node = TreeNode(component, parent=current_node)

            if verbose:
                print(
                    f"Creating node {component} with parent node {current_node.global_name()}"
                )

            if group is not None:
                new_node.groups.add(group)
                new_node.update_groups_upwards()

            current_node = new_node

    if new_node is None:
        new_node = current_node

    root.update_groups_downwards()
    return new_node


def build_tree(list_of_strings):
    root = TreeNode("")  # Root node

    # Iterate through each string in the list
    for lst_index, lst in enumerate(list_of_strings):
        groups = {lst_index}  # Each list of strings represents a group
        for string in lst:
            components = string.split(".")
            current_node = root

            # Traverse the tree and construct nodes for each component
            for component in components:
                if component not in current_node.children:
                    current_node.children[component] = TreeNode(
                        component, parent=current_node
                    )
                current_node = current_node.children[component]

            # Update the groups for the leaf node
            current_node.groups |= groups
            current_node.update_groups_upwards()  # Inform parent about the groups
    return root


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


### Tests


def test_tree():
    """
    Tests tree building from text
    """

    # Example usage:
    lists_of_strings = [
        ["layer1.conv1", "layer1.0.conv1", "layer1.1.conv1"],
        ["layer2.conv1", "layer1.clf"],
    ]

    # Build the tree
    root = build_tree(lists_of_strings)

    # Print group information for each node
    print_group_information(root)

    # Define a list of lists of strings
    more_lists_of_strings = [
        ["layer1.conv1", "layer1.0.conv1", "layer1.1.conv1", "layer1.2.conv1"],
        ["layer2.conv1", "layer1.clf", "layer2.clf"],
        ["layer3.conv1", "layer3.0.conv1", "layer4.conv1"],
        ["layer4.conv1", "layer4.0.conv1", "layer4.1.conv1", "layer4.2.conv1"],
    ]

    # Build the tree
    root = build_tree(more_lists_of_strings)

    # Print group information for each node
    print_group_information(root)

    # Build new nodes
    new_nodes = ["layer4.conv1", "layer4.5.conv1", "layer4.2.conv1", "layer5.2.conv1"]
    for node in new_nodes:
        insert_node(node, root)

    print_group_information(root)


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


def test_optimizer():
    """
    Tests group assignment with param ids
    """
    model = resnet18()

    g1 = {"params": [], "lr": 0.1}
    g1["params"] += list(model.layer1.parameters())
    g1["params"] += list(model.layer2.parameters())

    g2 = {"params": [], "lr": 0.05}
    g2["params"] += list(model.layer3.parameters())
    g2["params"] += list(model.layer4.parameters())

    g3 = {"params": [], "lr": 0.01}
    g3["params"] += list(model.fc.parameters())
    g3["params"] += list(model.conv1.parameters())

    optimizer = torch.optim.SGD([g1, g2, g3])

    # Gather all names groups assignments
    name_groups = {}

    for n, p in model.named_parameters():
        g = None
        # Find param in optimizer
        for group_idx, group in enumerate(optimizer.param_groups):
            params = group["params"]
            for po in params:
                if id(po) == id(p):
                    g = group_idx
                    break
        name_groups[n] = g

    # Build tree from name groups assignments
    root, _ = build_tree_from_name_groups(name_groups)
    print_group_information(root)


def compare_keys(old_dict, new_dict):
    not_in_new = list(set(old_dict.keys()) - set(new_dict.keys()))
    in_both = list(set(old_dict.keys()) & set(new_dict.keys()))
    not_in_old = list(set(new_dict.keys()) - set(old_dict.keys()))
    return not_in_new, in_both, not_in_old


def update_optimizer(
    optimizer,
    new_params,
    optimized_params,
    reset_state=False,
):
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

    ## This is what we need to change

    param_structure = OptimizedParameterStructure(new_params, optimizer, verbose=True)

    # Add newly added parameters (i.e Multitask, PNN)
    # by default, add to param groups 0
    for key in not_in_old:
        new_p = new_params[key]
        group = param_structure[key].group
        optimizer.param_groups[group]["params"].append(new_p)
        optimized_params[key] = new_p
        optimizer.state[new_p] = {}

    if reset_state:
        optimizer.state = defaultdict(dict)

    return optimized_params


def test_optimizer_new_param():
    """
    Tests group assignment with param ids
    """
    model = resnet18()

    g1 = {"params": [], "lr": 0.1}
    g1["params"] += list(model.layer1.parameters())
    g1["params"] += list(model.layer2.parameters())

    g2 = {"params": [], "lr": 0.05}
    g2["params"] += list(model.layer3.parameters())
    g2["params"] += list(model.layer4.parameters())

    g3 = {"params": [], "lr": 0.01}
    g3["params"] += list(model.fc.parameters())
    g3["params"] += list(model.conv1.parameters())

    optimizer = torch.optim.SGD([g1, g2, g3])

    # Setup Structure
    param_structure = OptimizedParameterStructure(
        model.named_parameters(), optimizer, verbose=True
    )

    # Add new parameters

    # This one cannot get assigned a unique param group
    model.layerx = nn.Linear(10, 10)

    # This one should get assigned to group 3
    model.fc.external = nn.Linear(10, 10)

    param_structure = OptimizedParameterStructure(
        model.named_parameters(), optimizer, verbose=True
    )


def test_update_optimizer():
    """
    Tests group assignment with param ids
    """
    model = resnet18()

    g1 = {"params": [], "lr": 0.1}
    g1["params"] += list(model.layer1.parameters())
    g1["params"] += list(model.layer2.parameters())

    g2 = {"params": [], "lr": 0.05}
    g2["params"] += list(model.layer3.parameters())
    g2["params"] += list(model.layer4.parameters())

    g3 = {"params": [], "lr": 0.01}
    g3["params"] += list(model.fc.parameters())
    g3["params"] += list(model.conv1.parameters())
    g3["params"] += list(model.bn1.parameters())

    optimizer = torch.optim.SGD([g1, g2, g3])
    optimized_param_id = {n: p for n, p in model.named_parameters()}

    param_structure = OptimizedParameterStructure(
        dict(model.named_parameters()), optimizer, verbose=True
    )

    # Add new parameters

    # This one cannot get assigned a unique param group
    # model.layerx = nn.Linear(10, 10)
    model.layer2.append(nn.Linear(10, 10))

    # This one should get assigned to group 3
    model.fc.external = nn.Linear(10, 10)

    new_params = {n: p for n, p in model.named_parameters()}
    update_optimizer(optimizer, new_params, optimized_param_id)


test_update_optimizer()
