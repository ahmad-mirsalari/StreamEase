"""
This module provides functions to modify an ONNX computational graph to
handle multiple ConvTranspose nodes.
It ensures that paths after ConvTranspose nodes are separated to facilitate
streaming approaches.
"""
from collections import defaultdict
from typing import Optional
from typing import List, Tuple, Set, Dict
import onnx

def find_paths(
    current_node_name: str,
    visited: Set[str],
    input_node_names: Set[str],
    node_inputs: Dict[str, List[str]],
    node_outputs: Dict[str, List[str]]
) -> Tuple[List[List[str]], bool]:
    """
    Recursively finds all paths from the current node to any input node in a graph.
    Args:
        current_node_name (str): The name of the current node being processed.
        visited (set): A set of node names that have already been visited to avoid cycles.
        input_node_names (set): A set of node names that are considered input nodes.
        node_inputs (dict): A dictionary where keys are node names and values are lists of
        input node names.
        node_outputs (dict): A dictionary where keys are node names and values are lists of
        output node names.
    Returns:
        tuple: A tuple containing:
            - paths (list of list of str): A list of paths, where each path is a list of node names
            from an input node to the current node.
            - end (bool): A boolean indicating whether an input node has been reached.
    """
    paths: List[List[str]] = []
    end: bool = False
    # Check if the current node is an input node
    if current_node_name in input_node_names:
        return [[current_node_name]], True

    # Check if the current node has been visited to avoid cycles
    if current_node_name in visited:
        return []

    visited.add(current_node_name)

    # Proceed if the current node has outputs
    if current_node_name in node_outputs:
        producing_nodes = node_outputs[current_node_name]

        for producing_node in producing_nodes:

            for input_name in node_inputs[producing_node]:
                # Skip constants as they do not contribute to paths we are interested in
                if "Constant" not in input_name:

                    # Skip ConvTranspose nodes unless they are part of the input nodes
                    if (
                        "ConvTranspose" in input_name
                        and input_name not in input_node_names
                    ):

                        continue
                    # Recursively find paths from the current input node
                    sub_paths, end = find_paths(
                        input_name, visited, input_node_names, node_inputs, node_outputs
                    )
                    for sub_path in sub_paths:
                        paths.append(sub_path + [current_node_name])
                    # If an input node was found, terminate the search
                    if end:
                        return paths, end
    # Unmark the node as visited before returning (for other recursive calls)
    visited.remove(current_node_name)
    return paths, end


def map_outputs_to_nodes(
                        paths: List[List[str]],
                        node_outputs: Dict[str,List[str]]) -> List[List[str]]:
    """
    Converts paths based on output names to paths based on node names.

    Args:
        paths (list of list of str): List of paths where each path is a list of output names.
        node_outputs (dict): Dictionary mapping output names to a list of node names
        that produce these outputs.

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


def find_common_path_nodes(all_paths: List[List[str]]) -> Set[str]:
    """
    Finds the nodes that are shared by more than one path.

    Args:
        all_paths (list of list of str): List of paths where each path is a list of node names.

    Returns:
        Set[str]: Set of node names that are shared between paths.
    """
    node_count = defaultdict(int)

    # Increment the count for each node found in the paths
    for path in all_paths:
        for node_name in path:
            node_count[node_name] += 1

    # Create a set of nodes that appear in more than one path
    common_nodes = {node_name for node_name, count in node_count.items() if count > 1}
    return common_nodes


def get_outputs_from_node_name(node_name, node_outputs):
    """
    Retrieves the output names associated with a specific node name.

    Args:
        node_name (str): The node name to look for.
        node_outputs (defaultdict): A dictionary mapping output names to node names.

    Returns:
        list: A list of output names associated with the node name, or an empty list if not found.
    """
    outputs = []

    # Iterate through the dictionary to find the output names for the given node name
    for output_name, nodes in node_outputs.items():
        if node_name in nodes:
            outputs.append(output_name)

    return outputs


def separate_paths(
                    paths: List[List[str]],
                    new_graph: onnx.GraphProto) -> List[List[str]]:
    """
    Duplicates paths to separate ConvTranspose nodes and creates distinct paths.

    Args:
        paths (list of list of str): List of paths to process.
        new_graph (onnx.GraphProto): The new ONNX graph to be modified.

    Returns:
        list of list of str: The updated paths with duplicates for separation.
    """
    updated_paths: List[List[str]] = []

    # Iterate over each path in the list of paths
    for path in paths:
        # Track already seen nodes for the current path
        seen_nodes: Dict[str, onnx.NodeProto] = {}  # Track already seen nodes
        current_path: List[str] = []

        # Process each node in the path
        for node_name in path:
            if node_name in seen_nodes:
                # If the node has been seen before, duplicate it
                duplicated_node = duplicate_node(new_graph, seen_nodes[node_name])
                current_path.append(
                    duplicated_node.output[0]
                )  # Use the output of the duplicated node
            else:
                # Find the original node in the graph
                node = next((n for n in new_graph.node if n.name == node_name), None)
                if node:
                    seen_nodes[node_name] = node  # Store the original node
                    current_path.append(
                        node.output[0]
                    )  # Add the original node's output

        updated_paths.append(current_path)

    return updated_paths


def find_and_modify_div_mul(
    graph: onnx.GraphProto,
    paths: List[List[str]],
    common_nodes: Set[str]
) -> onnx.GraphProto:
    """
    This function processes a computational graph to identify and modify Div/Mul nodes,
    and handles shared nodes in multiple paths by duplicating them as necessary.
    Args:
        graph (onnx.GraphProto): The computational graph to be modified.
        paths (list of list of str): A list of paths, where each path is a list of node names.
        common_nodes (set of str): A set of node names that are shared across multiple paths.
    Returns:
        onnx.GraphProto: The modified computational graph.
    The function performs the following steps:
    1. Identifies Div/Mul nodes and stores their connection information.
    2. Processes each path to handle Div/Mul nodes and duplicate shared nodes.
    3. Updates the inputs of the next nodes connected to Div/Mul nodes.
    4. Duplicates shared nodes for subsequent paths and updates their inputs.
    5. Updates the graph outputs if any path was modified.
    6. Removes any Div/Mul nodes that were marked for removal.
    """

    # A dictionary to store information about Div/Mul nodes
    div_mul_nodes_info: Dict[str, Dict] = {}
    created_nodes: Dict[str, onnx.NodeProto] = {}

    # Step 1: Identify Div/Mul nodes and store their connection info
    for path in paths:
        for i, node_name in enumerate(path):
            node = next((n for n in graph.node if n.name == node_name), None)
            if node and node.op_type in ["Div", "Mul"]:
                div_mul_nodes_info[node_name] = {
                    "prev_node_names": node.input,  # Store all inputs to the node
                    "next_node_name": path[i + 1] if i + 1 < len(path) else None,
                    "node": node,
                }

    # Step 2: Process each path
    for path_index, path in enumerate(paths):
        path_modified = False
        new_path = []

        for i, node_name in enumerate(path):
            original_node = next(
                (n for n in graph.node if n.name == node_name), None
            )
            if not original_node:
                continue

            # Handle Div/Mul nodes separately
            if node_name in div_mul_nodes_info:
                div_info = div_mul_nodes_info[node_name]
                prev_nodes = [
                    next((n for n in graph.node if n.output[0] == input_name), None)
                    for input_name in div_info["prev_node_names"]
                ]
                next_node = next(
                    (n for n in graph.node if n.name == div_info["next_node_name"]),
                    None,
                )

                if all(prev_nodes) and next_node:
                    # Only update the input of the next node that is connected to the Div/Mul node
                    div_output = original_node.output[0]
                    for j, input_name in enumerate(next_node.input):
                        # Check if this input comes from the Div/Mul node
                        if input_name == div_output:
                            next_node.input[j] = prev_nodes[0].output[
                                0
                            ]  # Connect to the first prev node output

            elif path_index > 0 and node_name in common_nodes:
                # Duplicate the shared node for subsequent ConvTranspose paths
                if node_name not in created_nodes:
                    new_node = duplicate_node(graph, node_name)
                    created_nodes[node_name] = new_node

                    # Update the inputs of the duplicated node
                    new_inputs = []
                    for input_name in original_node.input:
                        # If input is in the current path
                        if input_name in path:
                            prev_node_index = path.index(input_name) - 1
                            if prev_node_index >= 0:
                                prev_node_name = path[prev_node_index]
                                if prev_node_name in created_nodes:
                                    new_inputs.append(
                                        created_nodes[prev_node_name].output[0]
                                    )
                                else:
                                    new_inputs.append(
                                        input_name
                                    )  # Fallback to original input if not created
                        else:
                            # Step 1: Create a mapping of node names to their outputs in the path
                            node_output_mapping = {
                                n.name: n.output[0]
                                for n in graph.node
                                if n.name in path
                            }

                            # Step 2: Check if the node_name is a next_node_name in the Div/Mul node
                            div_mul_info = next(
                                iter(div_mul_nodes_info.values())
                            )  # Get the first (and only) value in the dictionary

                            if node_name == div_mul_info["next_node_name"]:
                                prev_node_names = div_mul_info["prev_node_names"]

                                # Step 3: Check if this input matches any of the previous nodes
                                if input_name in prev_node_names:

                                    # Step 3: Check if any of the previous nodes are in the path
                                    found_valid_input = False
                                    for prev_node_name in prev_node_names:
                                        if (
                                            prev_node_name
                                            in node_output_mapping.values()
                                        ):
                                            new_inputs.append(prev_node_name)
                                            found_valid_input = True
                                            break  # Exit loop after finding the first valid match

                                    if not found_valid_input:
                                        new_inputs.append(
                                            input_name
                                        )  # Fallback to the original input if none found

                                else:
                                    # If not part of the Div/Mul node, keep the original input
                                    new_inputs.append(input_name)
                            else:
                                # If the current node is not the next node of any Div/Mul,
                                # keep the original input
                                new_inputs.append(
                                    input_name
                                )  # Original input if not part of the path

                    # Modify and update the new node's inputs
                    new_node.input[:] = new_inputs

                    # Find the node in the graph and update it
                    for i, node in enumerate(graph.node):
                        if node.name == new_node.name:
                            graph.node[i].input[:] = new_inputs
                            break  # Exit the loop once the node is found and updated

                # Replace the current node in the path with the duplicated one
                new_path.append(created_nodes[node_name].name)
                path_modified = True
            else:
                new_path.append(node_name)

        # If the path was modified, make sure to update the graph outputs
        if path_modified:
            final_node = next(
                (n for n in graph.node if n.name == new_path[-1]), None
            )
            if final_node:
                output_name = final_node.output[0]
                if output_name not in [output.name for output in graph.output]:
                    graph.output.append(
                        onnx.helper.make_tensor_value_info(
                            output_name, onnx.TensorProto.FLOAT, None
                        )
                    )

    # Finally, remove any Div/Mul nodes that were marked for removal
    for node_name in div_mul_nodes_info.keys():
        original_node = next((n for n in graph.node if n.name == node_name), None)
        if original_node:
            graph.node.remove(original_node)

    return graph


def duplicate_node(graph: onnx.GraphProto, node_name: str) -> Optional[onnx.NodeProto]:
    """
    Duplicates a node in an ONNX graph.
    This function searches for a node by its name in the provided ONNX graph.
    If the node is found, it creates a duplicate of the node with the same
    inputs but a different output name, and appends the new node to the graph.
    Args:
        graph (onnx.GraphProto): The ONNX graph in which the node is to be duplicated.
        node_name (str): The name of the node to be duplicated.
    Returns:
        onnx.NodeProto or None: The newly created node if the original node is found,
        otherwise None.
    """

    # Search for the node in the graph by its name
    original_node = next((n for n in graph.node if n.name == node_name), None)

    if original_node:

        # Generate a new output name by appending "_copy" to the original output name
        new_node_output = original_node.output[0] + "_copy"
        new_node_name = node_name + "_copy"

        # Check if the new output name already exists in the graph
        if any(new_node_output in n.output for n in graph.node):
            raise ValueError(f"Output name '{new_node_output}' already exists in the graph.")

        # Check if the new node name already exists in the graph
        if any(new_node_name == n.name for n in graph.node):
            raise ValueError(f"Node name '{new_node_name}' already exists in the graph.")

        # Create a new node with duplicated inputs and a new output
        new_node = onnx.helper.make_node(
            op_type=original_node.op_type,
            inputs=list(original_node.input),  # Copy the inputs from the original node
            outputs=[new_node_output],         # Set the new output for the duplicated node
            name=new_node_name,          # Create a new name for the duplicated node
        )

        # Add the new node to the graph
        graph.node.append(new_node)

        return new_node # Return the newly created node

    return None  # Return None if the original node is not found in the graph


def connect_to_graph_output(
    graph: onnx.GraphProto,
    last_node: onnx.NodeProto,
    original_output_name: str
) -> None:
    """
    Connects the last node in a path to the graph's output.

    Args:
        graph (onnx.GraphProto): The ONNX graph.
        last_node (onnx.NodeProto): The last node in the modified path.
        original_output_name (str): The original output name in the graph.

    Returns:
        None
    """

    # Search for the existing graph output that matches the original output name
    graph_output = next(
        (output for output in graph.output if output.name == original_output_name), None
    )

    if graph_output:
        # If the original output exists, update its name to the output of the last node
        graph_output.name = last_node.output[0]
    else:
        # If the original output does not exist, create a new graph output
        new_output = onnx.helper.make_tensor_value_info(
            last_node.output[0], onnx.TensorProto.FLOAT, None
        )
        graph.output.append(new_output)


def transpose_checker(graph: onnx.GraphProto) -> onnx.GraphProto:
    """
    Checks and modifies the graph to handle ConvTranspose nodes.
    This function identifies ConvTranspose nodes in the given graph and modifies
    the graph to ensure proper handling of these nodes.
    It finds paths from the inputs of the ConvTranspose nodes to the targeted nodes,
    maps the outputs to nodes, and modifies the graph accordingly.
    Args:
        graph (onnx.GraphProto): The graph to be modified.
    Returns:
        onnx.GraphProto: The modified graph.
    Raises:
        ValueError: If no input node is found in the graph or if the targeted node is not found.
    """
    mapped_paths = []
    all_paths = []
    # Iterate through each node in the graph to find ConvTranspose nodes
    for _, node_i in enumerate(graph.node):
        if node_i.op_type == "ConvTranspose":
            conv_node = node_i
            # Find the first convolution node
            # Create a set of all input node names
            output_node_names = {node.name for node in graph.output}
            if not output_node_names:
                raise ValueError("No input node found in the graph")

            # Identify the targeted node connected to the graph output
            targeted_node = None
            for node in graph.node:
                for node_output in node.output:
                    if node_output in output_node_names:
                        targeted_node = node
                        break

            if targeted_node is None:
                raise ValueError("targeted_node node not found in the graph")

            # Create mappings of node names to their inputs and outputs
            node_inputs = defaultdict(list)
            node_outputs = defaultdict(list)
            for node in graph.node:
                for input_name in node.input:
                    node_inputs[node.name].append(input_name)
                for output_name in node.output:
                    node_outputs[output_name].append(node.name)

            input_node_names = [conv_node.output[0]]

            # Pathfinding from each input of the targeted node
            end = False
            for input_name in targeted_node.input:
                if "Constant" not in input_name:
                    paths_from_input, end = find_paths(
                        input_name, set(), input_node_names, node_inputs, node_outputs
                    )
                    for path in paths_from_input:
                        all_paths.append(path + [targeted_node.name])
                    if end:
                        break
            # Map the paths found to the corresponding output nodes
            mapped_paths = map_outputs_to_nodes(all_paths, node_outputs)
    new_graph: onnx.GraphProto = graph
    # Modify the graph by handling Div or Mul nodes
    # Identify common nodes in all paths
    if mapped_paths:
        common_nodes = find_common_path_nodes(mapped_paths)

        # Modify the graph by handling Div or Mul nodes and duplicating common paths
        new_graph = find_and_modify_div_mul(graph, mapped_paths, common_nodes)

        # Connect the final node to the graph output
        for path in mapped_paths:
            last_node_name = path[-1]
            last_node = next(
                (n for n in new_graph.node if n.name == last_node_name), None
            )
            if last_node:
                connect_to_graph_output(new_graph, last_node, targeted_node.output[0])

    return new_graph
