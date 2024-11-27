"""
This module provides utilities for modifying ONNX graphs, specifically focusing on
ensuring proper connections between input nodes and the first convolution node in the graph.
"""
from collections import defaultdict
from typing import List, Dict, Set
import onnx



# Find all paths from each input to the convolution node
def find_paths(
    current_node_name: str,
    visited: Set[str],
    input_node_names: Set[str],
    node_inputs: Dict[str, List[str]],
    node_outputs: Dict[str, List[str]],
) -> List[List[str]]:
    """
    Recursively finds all paths from a given node to any input node in the graph.

    Args:
        current_node_name (str): The current node being processed.
        visited (set): A set of visited nodes to avoid cycles.
        input_node_names (set): A set of node names that are considered inputs.
        node_inputs (dict): A dictionary where keys are node names and values
            are lists of input node names.
        node_outputs (dict): A dictionary where keys are node names and values
            are lists of output node names.

    Returns:
        list of list of str: A list of paths from the input node to the current node.
    """
    paths = []
    if current_node_name in input_node_names:
        return [[current_node_name]]

    if current_node_name in visited:
        return []  # Avoid cycles

    visited.add(current_node_name)

    if current_node_name in node_outputs:
        producing_nodes = node_outputs[current_node_name]
        for producing_node in producing_nodes:
            for input_name in node_inputs[producing_node]:
                sub_paths = find_paths(
                    input_name, visited, input_node_names, node_inputs, node_outputs
                )
                for sub_path in sub_paths:
                    paths.append(sub_path + [current_node_name])

    visited.remove(current_node_name)
    return paths


def map_outputs_to_nodes(
    paths: List[List[str]], node_outputs: Dict[str, List[str]]
) -> List[List[str]]:
    """
    Converts paths based on output names to paths based on node names.

    Args:
        paths (list of list of str): List of paths where each path is a list of output names.
        node_outputs (dict): Dictionary mapping output names to a list of node names that
            produce these outputs.

    Returns:
        list of list of str: List of paths where each path is a list of node names.
    """
    # Create a reverse mapping from output names to node names
    output_to_node = {}
    for output_name, nodes in node_outputs.items():
        for node in nodes:
            output_to_node[output_name] = node

    # Convert output-based paths to node-based paths
    node_paths = []
    for path in paths:
        node_path = []
        for output_name in path:
            if output_name in output_to_node:
                node_path.append(output_to_node[output_name])
            else:
                # Handle the case where output_name does not map to any node
                node_path.append(output_name)  # Or handle appropriately
        node_paths.append(node_path)

    return node_paths


def modify_graph(
    new_graph: onnx.GraphProto, mapped_paths: List[List[str]]
) -> onnx.GraphProto:
    """
    Modifies the graph by connecting inputs of Concat or Pad nodes to their output nodes
        and removing these nodes.

    Args:
        new_graph (onnx.GraphProto): The ONNX graph to be modified.
        mapped_paths (list of list of str): List of paths where
        each path is a list of node names.

    Returns:
        onnx.GraphProto: The modified ONNX graph.
    """
    nodes_to_remove = set()

    # Collect nodes to remove based on paths
    for path in mapped_paths:
        for node_name in path:
            node = next((n for n in new_graph.node if n.name == node_name), None)
            if node and node.op_type in ["Concat", "Pad"]:
                nodes_to_remove.add(
                    node.name
                )  # Use node name instead of NodeProto object

    # Update the inputs of nodes that use the outputs of the nodes to be removed
    for node_name in nodes_to_remove:
        node_to_remove = next(n for n in new_graph.node if n.name == node_name)
        output_name = node_to_remove.output[0]

        if node_to_remove.op_type == "Concat":
            # Find the node whose output matches one of the inputs of the node_to_remove
            previous_node_output = None
            for input_name in node_to_remove.input:
                for node in new_graph.node:
                    # Check if the input_name is in the new_graph inputs
                    if input_name in [input.name for input in new_graph.input]:
                        previous_node_output = input_name
                        break

                    # Otherwise, find the node whose output matches this input_name
                    for node in new_graph.node:
                        if input_name in node.output and (
                            node.name in path
                            or input_name in [input.name for input in new_graph.input]
                        ):
                            previous_node_output = input_name
                            break
                if previous_node_output:
                    break

            if previous_node_output:
                # Update the inputs of nodes connected to the output of the node_to_remove
                for node in new_graph.node:
                    for i, input_name in enumerate(node.input):
                        if input_name == output_name:
                            node.input[i] = previous_node_output
        else:
            for node in new_graph.node:
                for i, input_name in enumerate(node.input):
                    if input_name == output_name:
                        node.input[i] = node_to_remove.input[0]

        new_graph.node.remove(node_to_remove)

    return new_graph


def path_finder(graph: onnx.GraphProto) -> List[List[str]]:
    """
    Finds and maps paths from input nodes to the first convolution node in the ONNX model graph.

    Args:
        graph (onnx.ModelProto): The original graph.

    Returns:
        List[List[str]]: A list of paths where each path is a list of node names.
    """
    # Find the first convolution node
    first_conv_node = None
    for node in graph.node:
        if node.op_type == "Conv":
            first_conv_node = node
            break

    if first_conv_node is None:
        raise ValueError("First convolution node not found in the graph")

    # Create mappings of node names to their inputs and outputs
    node_inputs = defaultdict(list)
    node_outputs = defaultdict(list)
    for node in graph.node:
        for input_name in node.input:
            node_inputs[node.name].append(input_name)
        for output_name in node.output:
            node_outputs[output_name].append(node.name)

    # Create a set of all input node names
    input_node_names = {node.name for node in graph.input}
    if not input_node_names:
        raise ValueError("No input node found in the graph")

    # Start pathfinding from each input of the convolution node
    all_paths = []
    for input_name in first_conv_node.input:
        if "Constant" not in input_name:
            paths_from_input = find_paths(
                input_name, set(), input_node_names, node_inputs, node_outputs
            )
            for path in paths_from_input:
                all_paths.append(path + [first_conv_node.name])

    mapped_paths = map_outputs_to_nodes(all_paths, node_outputs)

    return mapped_paths


def input_checker(
    model: onnx.ModelProto, new_graph: onnx.GraphProto
) -> onnx.GraphProto:
    """
    Checks and modifies the input paths in the ONNX model graph,
    ensuring the first Conv node is connected properly.

    Args:
        model (onnx.ModelProto): The original ONNX model.
        new_graph (onnx.GraphProto): The modified ONNX graph.

    Returns:
        onnx.GraphProto: The updated ONNX graph.
    """

    mapped_paths = path_finder(model.graph)
    new_graph = modify_graph(new_graph, mapped_paths)

    return new_graph
