"""_summary_
    This module provides utility functions for identifying and
    removing unused chains of nodes in an ONNX graph.

    Functions:
        identify_disconnected_nodes(graph: onnx.GraphProto)
        -> Tuple[Set[str], Dict[str, List[str]], Dict[str, List[str]]]:

        find_predecessors(node_name: str, graph: onnx.GraphProto,
        node_outputs: Dict[str, List[str]]) -> List[str]:

        build_chain(node_name: str, graph: onnx.GraphProto,
        node_outputs: Dict[str, List[str]]) -> List[str]:

        find_chains_from_disconnected_nodes(graph: onnx.GraphProto,
        disconnected_nodes: Set[str]) -> List[List[str]]:

        remove_nodes_from_graph(graph: onnx.GraphProto,
        nodes_to_remove: Set[str]) -> onnx.GraphProto:

        remove_chains_from_graph(graph: onnx.GraphProto,
        chains: List[List[str]]) -> onnx.GraphProto:
"""

from collections import defaultdict, deque
from typing import List, Set, Tuple, Dict
import onnx

def identify_disconnected_nodes(
    graph: onnx.GraphProto,
) -> Tuple[Set[str], Dict[str, List[str]], Dict[str, List[str]]]:
    """
    Identify nodes that are disconnected from the rest of the ONNX graph.

    Args:
        graph (onnx.GraphProto): The ONNX graph to analyze.

    Returns:
        Tuple[Set[str], Dict[str, List[str]], Dict[str,
        List[str]]]: A set of disconnected node names,
        a mapping of node inputs, and a mapping of node outputs.
    """
    node_inputs = defaultdict(list)
    node_outputs = defaultdict(list)

    # Build mappings for node inputs and outputs
    for node in graph.node:
        for input_name in node.input:
            node_inputs[input_name].append(node.name)
        for output_name in node.output:
            node_outputs[output_name].append(node.name)

    # Determine graph outputs
    graph_outputs = {output.name for output in graph.output}

    # Find disconnected nodes
    disconnected_nodes = set()
    for node in graph.node:
        is_disconnected = True
        for output_name in node.output:
            if output_name in node_inputs or output_name in graph_outputs:
                is_disconnected = False
                break
        if is_disconnected:
            disconnected_nodes.add(node.name)

    return disconnected_nodes, node_inputs, node_outputs


def find_predecessors(
    node_name: str, graph: onnx.GraphProto, node_outputs: Dict[str, List[str]]
) -> List[str]:
    """
    Find predecessors of a node by matching its inputs with the outputs of other nodes.

    Args:
        node_name (str): The name of the node for which to find predecessors.
        graph (onnx.GraphProto): The ONNX graph.
        node_outputs (Dict[str, List[str]]): A mapping of node outputs to node names.

    Returns:
        List[str]: A list of predecessor node names.
    """
    predecessors = []
    node = next((n for n in graph.node if n.name == node_name), None)
    if node:
        for input_name in node.input:
            # Find nodes whose output matches the input of the current node
            for predecessor_name in node_outputs.get(input_name, []):
                # Skip constants except 'ConstantOfShape'
                predecessor_node = next(
                    (n for n in graph.node if n.name == predecessor_name), None
                )
                if (
                    predecessor_node
                    and predecessor_node.op_type not in ["Constant"]
                    or predecessor_node.name == "ConstantOfShape"
                ):
                    predecessors.append(predecessor_name)
    return predecessors


def build_chain(
    node_name: str, graph: onnx.GraphProto, node_outputs: Dict[str, List[str]]
) -> List[str]:
    """
    Build the chain of nodes leading up to a given node, including only valid nodes.

    Args:
        node_name (str): The name of the starting node for the chain.
        graph (onnx.GraphProto): The ONNX graph.
        node_outputs (Dict[str, List[str]]): A mapping of node outputs to node names.

    Returns:
        List[str]: A list of node names representing the chain.
    """
    chain = []
    visited = set()
    queue = deque([node_name])

    while queue:
        current_node = queue.popleft()
        if current_node in visited:
            continue
        visited.add(current_node)
        chain.append(current_node)
        predecessors = find_predecessors(current_node, graph, node_outputs)
        queue.extend(predecessors)

    return chain


def find_chains_from_disconnected_nodes(
    graph: onnx.GraphProto, disconnected_nodes: Set[str]
) -> List[List[str]]:
    """
    Find and return all chains of nodes where the given disconnected nodes are the last nodes.

    Args:
        graph (onnx.GraphProto): The ONNX graph to analyze.
        disconnected_nodes (Set[str]): A set of disconnected node names.

    Returns:
        List[List[str]]: A list of chains (each chain is a list of node names).
    """
    # Build mappings for node inputs and outputs
    node_inputs = defaultdict(list)
    node_outputs = defaultdict(list)

    for node in graph.node:
        for input_name in node.input:
            node_inputs[input_name].append(node.name)
        for output_name in node.output:
            node_outputs[output_name].append(node.name)

    chains = []
    for node_name in disconnected_nodes:
        # Build chain for each disconnected node
        if node_name not in [output.name for output in graph.output]:
            chain = build_chain(node_name, graph, node_outputs)
            if chain:
                chains.append(chain)

    return chains


def remove_nodes_from_graph(
    graph: onnx.GraphProto, nodes_to_remove: Set[str]
) -> onnx.GraphProto:
    """
    Remove nodes from the ONNX graph based on a set of nodes to be removed.

    Args:
        graph (onnx.GraphProto): The ONNX graph to modify.
        nodes_to_remove (Set[str]): A set of node names to remove from the graph.

    Returns:
        onnx.GraphProto: The modified ONNX graph.
    """

    nodes_to_remove_queue = deque(nodes_to_remove)
    removed_nodes = set()

    # Build mappings for node inputs and outputs
    node_inputs = defaultdict(list)
    node_outputs = defaultdict(list)

    for node in graph.node:
        for input_name in node.input:
            node_inputs[input_name].append(node.name)
        for output_name in node.output:
            node_outputs[output_name].append(node.name)

    while nodes_to_remove_queue:
        node_name = nodes_to_remove_queue.popleft()

        if node_name in removed_nodes:
            continue

        # Find the node to remove
        node = next((n for n in graph.node if n.name == node_name), None)
        if not node:
            continue

        # Remove node from the graph
        graph.node.remove(node)
        removed_nodes.add(node_name)

        # Remove the node's outputs from node_outputs
        for output_name in node.output:
            if output_name in node_outputs:
                next_nodes = list(node_outputs[output_name])
                for next_node in next_nodes:
                    if next_node not in removed_nodes:
                        # If the next_node is not already marked for removal, add it to the queue
                        nodes_to_remove_queue.append(next_node)
                del node_outputs[output_name]

        # Remove the node's inputs from node_inputs
        for input_name in node.input:
            if node_name in node_inputs[input_name]:
                node_inputs[input_name].remove(node_name)
                if not node_inputs[input_name]:
                    del node_inputs[input_name]

    # Remove any nodes that are still in the graph but not in the remaining nodes
    remaining_node_names = {node.name for node in graph.node}
    for node_name in remaining_node_names:
        if node_name not in removed_nodes:
            node = next((n for n in graph.node if n.name == node_name), None)
            if node:
                # Remove any remaining inputs/outputs mappings for nodes still in the graph
                for output_name in node.output:
                    if output_name in node_outputs:
                        node_outputs[output_name].remove(node_name)
                        if not node_outputs[output_name]:
                            del node_outputs[output_name]
                for input_name in node.input:
                    if node_name in node_inputs[input_name]:
                        node_inputs[input_name].remove(node_name)
                        if not node_inputs[input_name]:
                            del node_inputs[input_name]

    # Return the modified graph
    return graph


def remove_chains_from_graph(
    graph: onnx.GraphProto, chains: List[List[str]]
) -> onnx.GraphProto:
    """
    Remove all nodes in the given chains from the ONNX graph.

    Args:
        graph (onnx.GraphProto): The ONNX graph to modify.
        chains (List[List[str]]): A list of chains (each chain
        is a list of node names) to be removed.

    Returns:
        onnx.GraphProto: The modified ONNX graph.
    """
    # Collect all nodes to remove
    nodes_to_remove = set()
    for chain in chains:
        nodes_to_remove.update(chain)

    # Remove nodes from the graph
    modified_graph = remove_nodes_from_graph(graph, nodes_to_remove)

    return modified_graph
