"""This module contains helper functions for streaming operations in ONNX models. 
These functions assist in manipulating and analyzing ONNX graphs, particularly 
for tasks related to convolution operations, node sequences, and input/output 
dimension adjustments.
"""

from collections import defaultdict, deque
from typing import Dict, List, Optional, Tuple
import onnx
from onnx import TensorShapeProto, ModelProto, NodeProto, GraphProto
from onnx.helper import make_model, set_model_props, make_graph


def update_output_name(
    graph: onnx.GraphProto, old_name: str, new_name: str
) -> onnx.GraphProto:
    """
    Updates the output name in the ONNX graph when an output node
    is removed or replaced.

    Args:
        graph (onnx.GraphProto): The ONNX graph containing the nodes.
        old_name (str): The old output name to replace.
        new_name (str): The new output name to replace with.

    Returns:
        onnx.GraphProto: The updated ONNX graph with modified output names.
    """
    for node in graph.node:
        # Check if the old name is in the inputs of the node
        if old_name in node.output:
            # Find the index of the old name in the inputs
            # Manually find the index of the old name in the outputs
            output_index = None
            for i, output_name in enumerate(node.output):
                if output_name == old_name:
                    output_index = i
                    break

            # If the old name was found, update it
            if output_index is not None:
                node.output[output_index] = new_name
    return graph


def remove_unused_constants(graph: onnx.GraphProto) -> onnx.GraphProto:
    """
    Removes unused constant nodes from the ONNX graph.

    Args:
        graph (onnx.GraphProto): The ONNX graph containing the nodes.

    Returns:
        onnx.GraphProto: The updated ONNX graph without unused constant nodes.
    """

    # Get the list of Constant nodes
    constant_nodes = [node for node in graph.node if node.op_type == "Constant"]

    # Find the names of all nodes that use Constant nodes
    nodes_with_constants = set()
    for node in graph.node:
        nodes_with_constants.update(node.input)

    # Remove Constant nodes that are not used by any other nodes
    for constant_node in constant_nodes:
        if constant_node.output[0] not in nodes_with_constants:
            # Remove the Constant node
            graph.node.remove(constant_node)
    return graph


def get_conv_attributes(
    graph: onnx.GraphProto, node: onnx.NodeProto
) -> Tuple[int, List[int], List[int], int, int]:
    """
    Retrieves convolution-related attributes from a Conv node.

    Args:
        model (onnx.ModelProto): The ONNX model containing the Conv node.
        node (onnx.NodeProto): The Conv node to extract attributes from.

    Returns:
        Tuple[int, List[int], List[int], int, int]: A tuple containing:
            - conv_len (int): The length of the convolution (number of dimensions).
            - kernel_shape (list of int): The shape of the convolution kernel.
            - dilation_rate (list of int): The dilation rate of the convolution.
            - input_channels (int): The number of input channels.
            - output_channels (int): The number of output channels.
    """
    conv_len = 3
    groups = [1]
    kernel_shape = [1, 1]
    dilation_rate = [1, 1]
    input_channels = 1
    output_channels = 1
    stride = [1, 1]

    # Iterate over attributes of the Conv node
    for attr in node.attribute:
        # Check if the attribute is "kernel_shape"
        if attr.name == "kernel_shape":
            # Check if the kernel_shape has only one element, add a 1 at the beginning
            kernel_shape = (
                list(attr.ints) if len(attr.ints) > 1 else [1] + list(attr.ints)
            )

        # Check if the attribute is "dilations"
        if attr.name == "dilations":
            # Check if the dilation_rate has only one element, add a 1 at the beginning
            dilation_rate = (
                list(attr.ints) if len(attr.ints) > 1 else [1] + list(attr.ints)
            )

        # Check if the attribute is "group"
        if attr.name == "group":
            groups = [attr.i]

        # Check if the attribute is "strides"
        if attr.name == "strides":
            # Check if the stride has only one element, add a 1 at the beginning
            stride = list(attr.ints) if len(attr.ints) > 1 else [1] + list(attr.ints)

    # Find initializer to get input/output channels
    for initializer in graph.initializer:
        if initializer.name == node.input[1]:
            # Assuming the shape of the initializer corresponds to
            # [output_channels, input_channels, kernel_size[0], kernel_size[1]]
            conv_len = len(initializer.dims)
            input_channels = initializer.dims[1]
            output_channels = initializer.dims[0]

    if groups[0] > 1:
        input_channels = groups[0]

    return (
        conv_len,
        kernel_shape,
        dilation_rate,
        input_channels,
        output_channels,
        stride,
    )


# Helper function to check if a node is in the sequence
def is_sequence(node: onnx.NodeProto, op_types: List[str]) -> bool:
    """
    Checks if the node's operation type is part of a given sequence of operation types.

    Args:
        node (onnx.NodeProto): The ONNX node to check.
        op_types (List[str]): List of operation types to check against.

    Returns:
        bool: True if the node's operation type is in the list of operation types, otherwise False.
    """
    return node.op_type in op_types


def find_node_index_by_name(graph: onnx.GraphProto, node_name: str) -> Optional[int]:
    """
    Finds the index of a node in the ONNX graph by its name.

    Args:
        graph (onnx.GraphProto): The ONNX graph containing the nodes.
        node_name (str): The name of the node to find.

    Returns:
        Optional[int]: The index of the node if found, otherwise None.
    """
    for i, node in enumerate(graph.node):
        if node.name == node_name:
            return i
    return None  # Node with the given name not found


def find_pre_node(
    conv_index: int, sequence_op_types: List[str], model: ModelProto
) -> Optional[NodeProto]:
    """
    Finds the first node preceding a convolution node in the model
    that is not part of a given sequence of operation types.

    Args:
        conv_index (int): The index of the convolution node in the graph.
        sequence_op_types (List[str]): A list of operation types that define the sequence.
        model (ModelProto): The ONNX model containing the graph.

    Returns:
        Optional[NodeProto]: The first preceding node that is not in the given sequence of
        operation types, or None if none are found.
    """
    max_index = max(0, conv_index - len(sequence_op_types))

    for j in range(conv_index, max_index - 1, -1):
        if not is_sequence(model.graph.node[j], sequence_op_types):
            break  # Exit loop if the node is not in the sequence
        sequence_node = model.graph.node[j]
    return sequence_node


def is_there_pad_node(
    conv_index: int, sequence_op_types: List[str], model: ModelProto
) -> Optional[NodeProto]:
    """
    Finds if there is a 'Pad' node before the convolution node within a given sequence of
    operation types.

    Args:
        conv_index (int): The index of the convolution node in the graph.
        sequence_op_types (List[str]): A list of operation types that define the sequence.
        model (ModelProto): The ONNX model containing the graph.

    Returns:
        Optional[NodeProto]: The first 'Pad' node encountered in the sequence, or the last
        node in the sequence if no 'Pad' is found.
    """

    max_index = max(0, conv_index - len(sequence_op_types))

    # Initialize sequence_node to the node at max_index
    sequence_node = model.graph.node[max_index]

    for j in range(conv_index, max_index - 1, -1):
        current_node = model.graph.node[j]

        if not is_sequence(current_node, sequence_op_types):
            break  # Exit loop if the node is not in the sequence

        # Update sequence_node to the current node
        sequence_node = current_node

        # If a 'Pad' node is found, break out of the loop and return it
        if current_node.op_type == "Pad":
            break  # Exit loop if the node is "Pad"
    return sequence_node


def build_input_to_nodes_mapping(
    graph: onnx.GraphProto,
) -> Dict[str, List[onnx.NodeProto]]:
    """
    Builds a mapping from input names to the nodes that use these inputs.

    Args:
        graph (onnx.GraphProto): The ONNX graph containing the nodes.

    Returns:
        Dict[str, List[onnx.NodeProto]]: A dictionary mapping input names to lists
        of nodes that use these inputs.
    """
    input_to_nodes = defaultdict(list)
    for node in graph.node:
        for input_name in node.input:
            input_to_nodes[input_name].append(node)
    return input_to_nodes


def find_nodes_using_output(
    output_name: str, input_to_nodes: Dict[str, List[onnx.NodeProto]]
) -> List[onnx.NodeProto]:
    """
    Finds nodes that use a given output name as an input.

    Args:
        output_name (str): The output name to find nodes that use it.
        input_to_nodes (Dict[str, List[onnx.NodeProto]]): A dictionary mapping input names
        to lists of nodes that use these inputs.

    Returns:
        List[onnx.NodeProto]: A list of nodes that use the given output name as an input.
    """
    return input_to_nodes.get(output_name, [])


def find_next_node_not_in_sequence(
    start_node_name: str,
    target_sequence: List[str],
    input_to_nodes: Dict[str, List[onnx.NodeProto]],
) -> Optional[onnx.NodeProto]:
    """
    Finds the first node that is not in the target sequence, following the start node.

    Args:
        start_node_name (str): The name of the node to start tracing from.
        target_sequence (List[str]): List of node types to check in the sequence.
        input_to_nodes (Dict[str, List[onnx.NodeProto]]): A dictionary mapping input names
        to lists of nodes that use these inputs.

    Returns:
        Optional[onnx.NodeProto]: The first node not in the target sequence, or None if all nodes
        are in the sequence.
    """
    nodes_to_trace = deque(find_nodes_using_output(start_node_name, input_to_nodes))
    traced_nodes = set()

    while nodes_to_trace:
        current_node = nodes_to_trace.popleft()
        if current_node.name in traced_nodes:
            continue

        # Mark the node as traced
        traced_nodes.add(current_node.name)

        # Check the current node type
        if current_node.op_type not in target_sequence:
            return current_node

        # Add the nodes that use the current node's outputs
        for output_name in current_node.output:
            if output_name in input_to_nodes:
                for next_node in input_to_nodes[output_name]:
                    if next_node.name not in traced_nodes:
                        nodes_to_trace.append(next_node)

    return None


def is_there_slice_node(
    conv_index: int, target_sequence: List[str], graph: onnx.GraphProto
) -> Optional[onnx.NodeProto]:
    """
    Checks if there is a `Slice` node after the Conv node in the graph.

    Args:
        conv_index (int): The index of the Conv node in the graph.
        target_sequence (List[str]): List of node types to check in the sequence.
        graph (onnx.GraphProto): The ONNX graph containing the nodes.

    Returns:
        Optional[onnx.NodeProto]: The first node that is not in the target sequence,
        or None if all are in the sequence.
    """
    # Build mappings of inputs to nodes
    input_to_nodes = build_input_to_nodes_mapping(graph)

    # Get the ConvTranspose node
    conv_node = graph.node[conv_index]

    # Trace through nodes that use the Conv node's outputs
    for output_name in conv_node.output:
        next_node = find_next_node_not_in_sequence(
            output_name, target_sequence, input_to_nodes
        )
        if next_node:
            return next_node

    return None


def get_next_node(
    conv_index: int, target_sequence: List[str], model: ModelProto
) -> Optional[NodeProto]:
    """
    Finds the next node in the graph that matches the target sequence after the convolution node,
    or returns None if the node is a 'Transpose' or 'Reshape' operation or if it does not match
    the target sequence.

    Args:
        conv_index (int): The index of the convolution node in the graph.
        target_sequence (List[str]): A list of operation types to check against.
        model (ModelProto): The ONNX model containing the graph.

    Returns:
        Optional[NodeProto]: The farthest node that matches the target sequence, or None if no match
        is found.
    """

    # Initialize variables to track the farthest target node and its index
    farthest_target_node = None

    max_index = min(len(model.graph.node), conv_index + len(target_sequence))

    # Iterate over the nodes starting from the index of the Conv node
    for j in range(conv_index, max_index):
        current_node = model.graph.node[j]

        # Check if the current node is in the target sequence
        if current_node.op_type in target_sequence:
            # Update the farthest target node and its index
            farthest_target_node = current_node

        # Break the loop if a node outside the target sequence is encountered
        elif current_node.op_type in ["Transpose", "Reshape"]:
            # Invalidate farthest_target_node if Transpose or Reshape is encountered
            farthest_target_node = None

        else:
            break

    return farthest_target_node


def has_transpose_or_reshape_between(
    graph: GraphProto, start_index: int, end_index: int
) -> bool:
    """
    Checks if there is a 'Transpose' or 'Reshape' node between the specified start and
    end indices in the ONNX graph.

    Args:
        graph (GraphProto): The ONNX graph to search through.
        start_index (int): The starting index of the node range.
        end_index (int): The ending index of the node range.

    Returns:
        bool: True if a 'Transpose' or 'Reshape' node is found between the start and
        end indices, False otherwise.
    """

    for i in range(start_index, end_index):
        node = graph.node[i]
        if node.op_type in ["Transpose", "Reshape"]:
            return True
    return False


def get_framework_from_producer(producer_name: str) -> str:
    """
    Identifies the framework that generated the model based on the producer name.

    Args:
        producer_name (str): The producer name from the ONNX model.

    Returns:
        str: The identified framework ('PyTorch', 'TensorFlow', or 'Unknown').
    """
    if "torch" in producer_name.lower():
        return "PyTorch"
    if "tf" in producer_name.lower():
        return "TensorFlow"
    return "Unknown"


def get_first_conv_info(
    graph: onnx.GraphProto,
) -> Tuple[Optional[List[int]], Optional[List[int]]]:
    """
    Retrieves the kernel shape and strides of the first 'Conv' node related to
    an encoder or STFT in the ONNX graph.

    Args:
        graph (GraphProto): The ONNX graph to search for Conv nodes.

    Returns:
        Tuple[Optional[List[int]], Optional[List[int]]]:
            - A list representing the kernel shape, or None if not found.
            - A list representing the strides, or None if not found.
    """

    # Iterate over the nodes in the graph to find the relevant Conv node
    for node in graph.node:
        # Check if the node is a Conv node
        if node.op_type == "Conv":
            # Get the attributes of the Conv node
            return get_conv_attributes(graph, node)
    return None


def is_numeric_shape(shape: TensorShapeProto) -> bool:
    """
    Checks if all dimensions in the shape are numeric (i.e., have fixed values).

    Args:
        shape (TensorShapeProto): The shape of the tensor to check.

    Returns:
        bool: True if all dimensions are numeric (have fixed values), False otherwise.
    """
    for dim in shape.dim:
        if not dim.dim_value:  # If dim_value is not set, it's not a numeric dimension
            return False
    return True


def get_correct_shapes_framework(
    conv_length: int, first: int, second: int, third: int, forth: int
) -> List[int]:
    """
    Returns the correct shape for the framework based on the convolution length.

    Args:
        conv_length (int): The length of the convolution (number of dimensions).
        first (int): The first dimension of the shape.
        second (int): The second dimension of the shape.
        third (int): The third dimension of the shape.
        forth (int): The fourth dimension of the shape.

    Returns:
        List[int]: The correct shape dimensions for the convolution.
    """
    shape = []
    if conv_length == 3:
        shape = [first, second, forth]
    else:
        shape = [first, second, third, forth]
    return shape


def overwrite_opset(model: onnx.ModelProto) -> onnx.ModelProto:
    """
    Overwrites the opset in the ONNX model without changing any node definitions.

    Args:
        model (onnx.ModelProto): The original ONNX model.

    Returns:
        onnx.ModelProto: The ONNX model with an updated opset.
    """
    graph = make_graph(
        model.graph.node,
        model.graph.name,
        model.graph.input,
        model.graph.output,
        model.graph.initializer,
        sparse_initializer=model.graph.sparse_initializer,
    )

    onnx_model = make_model(graph, functions=model.functions)
    onnx_model.ir_version = model.ir_version
    onnx_model.producer_name = model.producer_name
    onnx_model.producer_version = model.producer_version
    onnx_model.domain = model.domain
    onnx_model.model_version = model.model_version
    onnx_model.doc_string = model.doc_string

    if len(model.metadata_props) > 0:
        values = {p.key: p.value for p in model.metadata_props}
        set_model_props(onnx_model, values)

    del onnx_model.opset_import[:]

    for oimp in model.opset_import:
        op_set = onnx_model.opset_import.add()
        op_set.domain = oimp.domain
        op_set.version = oimp.version
    return onnx_model
