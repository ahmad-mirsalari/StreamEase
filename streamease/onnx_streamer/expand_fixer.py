"""
This module provides utilities for modifying 'Expand' nodes
in an ONNX graph to handle time-step based tensors.

Functions:
    find_source_node(graph: onnx.GraphProto,
    tensor_name: str) -> Union[onnx.NodeProto, onnx.TensorProto, None]:

    modify_expand_nodes(graph: onnx.GraphProto,
    time_steps: int) -> onnx.GraphProto:
"""
import numpy as np
import onnx
from onnx import numpy_helper



def find_source_node(graph: onnx.GraphProto, tensor_name: str):
    """
    Finds the source node or initializer for a given tensor name in the ONNX graph.

    Args:
        graph (onnx.GraphProto): The ONNX graph being processed.
        tensor_name (str): The name of the tensor whose source node is to be found.

    Returns:
        Union[onnx.NodeProto, onnx.TensorProto, None]:
        The source node or initializer if found, otherwise None.
    """
    # Search in node outputs
    for node in graph.node:
        if tensor_name in node.output:
            return node

    # Search in graph initializers (constants)
    for initializer in graph.initializer:
        if initializer.name == tensor_name:
            return initializer  # Found in initializers

    return None  # Not found


def modify_expand_nodes(graph: onnx.GraphProto, time_steps: int) -> onnx.GraphProto:
    """
    Modifies the 'Expand' nodes in the ONNX graph to handle time-step based tensors.
    If the source node is a constant, this function modifies the tensor's shape and replaces
    the old nodes with updated ones, ensuring compatibility with time steps.

    Args:
        graph (onnx.GraphProto): The ONNX graph to be modified.
        time_steps (int): The number of time steps for reshaping the tensors.

    Returns:
        onnx.GraphProto: The modified ONNX graph with adjusted 'Expand' nodes.
    """

    count_expands = 0  # Counter for Expand nodes

    # Dictionary to store the mapping between min_value and constant name
    min_value_to_name = {}

    # Iterate through each node in the graph
    for node in graph.node:
        if node.op_type == "Expand":
            # Skip processing for istft-related nodes
            if "istft" in node.name:
                continue

            first_input_name = node.input[0]

            # Skip nodes where the first input name contains 'bias' or 'gain'
            if "bias" in first_input_name or "gain" in first_input_name:
                continue

            # Directly access nodes and their attributes for modification
            source_node = find_source_node(graph, first_input_name)

            if source_node is None:
                continue  # If source node is not found, skip

            # Handling source nodes based on type (NodeProto or TensorProto)
            if (
                isinstance(source_node, onnx.helper.NodeProto)
                and source_node.op_type == "Constant"
            ):
                # Handle the case when source_node is a node
                if not source_node.attribute:
                    continue  # Skip if there are no attributes

                tensor = source_node.attribute[0].t
                tensor_data = numpy_helper.to_array(tensor)

                # Access the tensor from the first attribute of the Constant node
                tensor = source_node.attribute[0].t
                tensor_data = numpy_helper.to_array(tensor)

                # Modify tensor shape to [1, time_steps] and update values
                new_shape = (1, time_steps)  # Use tuple instead of list
                new_values = tensor_data.flatten()[:time_steps]

                # Ensure the new shape fits the modified values
                if new_shape[1] > len(new_values):
                    new_values = np.pad(
                        new_values, (0, new_shape[1] - len(new_values)), "constant"
                    )

                # Create new tensor with updated shape and values
                new_tensor = numpy_helper.from_array(
                    new_values.astype(np.float32), name=tensor.name
                )
                new_tensor.dims[:] = new_shape  # Update shape using dims

                min_value = int(min(new_values))

                # Check if a constant with the same minimum value has already been added
                if min_value in min_value_to_name:
                    constant_name = min_value_to_name[min_value]
                else:
                    constant_name = f"Expand_{min_value}"
                    # Add the constant name to the dictionary
                    min_value_to_name[min_value] = constant_name
                    new_shape = (1, time_steps)
                    graph.input.extend(
                        [
                            onnx.helper.make_tensor_value_info(
                                constant_name, onnx.TensorProto.FLOAT, new_shape
                            )
                        ]
                    )
                node.input[0] = constant_name
                # Replace the old Constant node with the new one
                graph.node.remove(source_node)
                count_expands += 1

            # Handle the case when source_node is an initializer (constant)
            elif isinstance(source_node, onnx.TensorProto):

                tensor_data = numpy_helper.to_array(source_node)
                min_value = int(min(tensor_data.flatten()))

                # Modify tensor shape to [1, time_steps] and update values
                new_shape = (1, time_steps)  # Use tuple instead of list
                new_values = tensor_data.flatten()[:time_steps]

                # Ensure the new shape fits the modified values
                if new_shape[1] > len(new_values):
                    new_values = np.pad(
                        new_values, (0, new_shape[1] - len(new_values)), "constant"
                    )

                # Create new tensor with updated shape and values
                new_tensor = numpy_helper.from_array(
                    new_values.astype(np.float32), name=source_node.name
                )
                new_tensor.dims[:] = new_shape  # Update shape using dims

                # print(f"Min value: {min_value}")
                # Check if a constant with the same minimum value has already been added
                if min_value in min_value_to_name:
                    constant_name = min_value_to_name[min_value]
                else:
                    constant_name = f"Expand_{min_value}"
                    # Add the constant name to the dictionary
                    min_value_to_name[min_value] = constant_name
                    new_shape = (1, time_steps)
                    graph.input.extend(
                        [
                            onnx.helper.make_tensor_value_info(
                                constant_name, onnx.TensorProto.FLOAT, new_shape
                            )
                        ]
                    )

                node.input[0] = constant_name

                # Replace the old Constant node with the new one
                graph.initializer.remove(source_node)
                graph.initializer.append(new_tensor)

                count_expands += 1

            else:
                continue  # Skip if source node is not a Constant

    return graph
