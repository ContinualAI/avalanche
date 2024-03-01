#!/usr/bin/env python3
import copy

import torch
import torch.nn as nn
from torchvision.models import resnet18


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

            current_node.add_child(new_node)

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
    print(f"{prefix}{node.global_name()}: {node.groups}")

    # Recursively print group information for children nodes
    for child_name, child_node in node.children.items():
        print_group_information(child_node, prefix + "  ")


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

    # Iterate through each string in the list
    for name, group in name_groups.items():
        group = set([group])
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
        current_node.groups |= group
        current_node.update_groups_upwards()  # Inform parent about the groups
    return root


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
    root = build_tree_from_name_groups(name_groups)
    print_group_information(root)


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
    root = build_tree_from_name_groups(name_groups)
    print_group_information(root)

    # Add new parameters

    # This one cannot get assigned a unique param group
    model.layerx = nn.Linear(10, 10)

    # This one should get assigned to group 3
    model.fc.external = nn.Linear(10, 10)

    # Update tree
    for n, p in model.named_parameters():
        insert_node(n, root, verbose=True)

    print_group_information(root)



test_optimizer_new_param()
