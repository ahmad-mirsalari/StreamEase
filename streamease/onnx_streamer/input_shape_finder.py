"""
This module provides utility functions for determining and adjusting the 
input shape of an ONNX model. It includes functions to handle various ONNX
operations and trace the input shape backward from a convolution node.
"""

from typing import Dict, List, Tuple
import onnx
from onnx import numpy_helper
from .input_fixer import path_finder
from .helper import get_conv_attributes


def handle_pad(current_order: List[str]) -> List[str]:
    """
    Handles the Pad node without altering the order of dimensions.

    Args:
        current_order (List[str]): Current order of dimensions.

    Returns:
        List[str]: Unchanged dimension order.
    """
    return current_order


def handle_unsqueeze_reverse(
    node: onnx.NodeProto, current_order: List[str], graph: onnx.GraphProto
) -> List[str]:
    """
    Reverses the effect of an Unsqueeze node by removing single-dimensional
    entries based on specified axes.

    Args:
        node (onnx.NodeProto): The Unsqueeze node.
        current_order (List[str]): Current order of dimensions.
        graph (onnx.GraphProto): The ONNX graph containing nodes and initializers.

    Returns:
        List[str]: Updated dimension order after reversing the Unsqueeze operation.
    """

    # Unsqueeze uses the second input tensor to define the axes
    axes_input_name = node.input[1]

    # First, check if the axes are provided as an initializer
    axes_initializer = next(
        (i for i in graph.initializer if i.name == axes_input_name), None
    )

    if axes_initializer is not None:
        # Axes are provided as an initializer (constant)
        axes_tensor = numpy_helper.to_array(axes_initializer)
        # Sort axes in reverse order to remove dimensions from last to first
        for axis in sorted(axes_tensor, reverse=True):
            current_order.pop(axis)  # Remove the dimension at the axis position
    else:
        # Axes might be provided as a dynamic input (not an initializer)
        # Let's search in the inputs of the graph to find the actual tensor for the axes.
        axes_input_node = next(
            (n for n in graph.node if n.output[0] == axes_input_name), None
        )

        if axes_input_node and axes_input_node.op_type == "Constant":
            # The axes are provided dynamically as a constant node in the graph
            axes_tensor = numpy_helper.to_array(axes_input_node.attribute[0].t)
            for axis in sorted(axes_tensor, reverse=True):
                current_order.pop(axis)
        else:
            # If we can't find the axes input, we raise an error
            raise ValueError(f"Axes not found for Unsqueeze node: {node.name}")

    return current_order


def handle_squeeze_reverse(
    node: onnx.NodeProto, current_order: List[str], graph: onnx.GraphProto
) -> List[str]:
    """
    Reverses the effect of the Squeeze node by adding single-dimensional entries
    back at specified axes.

    Args:
        node (onnx.NodeProto): The Squeeze node.
        current_order (List[str]): Current order of dimensions.
        graph (onnx.GraphProto): The ONNX graph containing nodes and initializers.

    Returns:
        List[str]: Updated dimension order after reversing the Squeeze operation.
    """

    axes_input_name = node.input[1]
    axes_initializer = next(
        (i for i in graph.initializer if i.name == axes_input_name), None
    )

    if axes_initializer is not None:
        axes_tensor = numpy_helper.to_array(axes_initializer)
        for axis in sorted(axes_tensor):
            current_order.insert(
                axis, "1"
            )  # Insert a new dimension '1' at the axis position
    else:
        raise ValueError(f"Axes not found for Squeeze node: {node.name}")

    return current_order


def handle_concat_reverse(node: onnx.NodeProto, current_order: List[str]) -> List[str]:
    """
    Concatenation does not change the input order, so we can safely return
    the current order.

    Args:
        node (onnx.NodeProto): The Concat node.
        current_order (List[str]): Current order of dimensions.

    Returns:
        List[str]: Unchanged dimension order.
    """
    return current_order


def handle_reshape_reverse(
    node: onnx.NodeProto, current_order: List[str], graph: onnx.GraphProto
) -> List[str]:
    """
    Handles the Reshape node by adjusting the current order of dimensions
    based on the reshape shape.

    Args:
        node (onnx.NodeProto): The Reshape node.
        current_order (List[str]): Current order of dimensions.
        graph (onnx.GraphProto): The ONNX graph containing nodes and initializers.

    Returns:
        List[str]: Updated dimension order after reversing the Reshape operation.
    """

    # Reshape uses the second input tensor to define the new shape
    shape_input_name = node.input[1]

    # Find the initializer that defines the new shape
    shape_initializer = next(
        (i for i in graph.initializer if i.name == shape_input_name), None
    )

    if shape_initializer is not None:
        new_shape = numpy_helper.to_array(shape_initializer)
        if -1 in new_shape:
            # If -1 exists in the new shape, it means one dimension is inferred based
            # on the size of the tensor. We'll ignore this complexity and assume that
            # the order is preserved, but the size might change.
            print("Warning: -1 found in reshape shape, inferring the dimension size.")

        # Update the current order to reflect the new shape, while maintaining the order
        # of dimensions
        # If the number of dimensions changes, we can't infer much about the order so we
        # will assume it stays the same.
        if len(new_shape) == len(current_order):
            # If the new shape has the same number of dimensions, we keep the same order
            return current_order
        # If the new shape has a different number of dimensions, we update the order
        # accordingly
        # We'll assume that the first dimension (batch size) is preserved, and adjust
        # for other dimensions.
        return current_order[:1] + ["?"] * (len(new_shape) - 1)

    raise ValueError(f"Shape not found for Reshape node: {node.name}")


def get_tensor_shape(graph: onnx.GraphProto, tensor_name: str) -> List[int]:
    """
    Retrieves the shape of a tensor given its name from the graph.

    Args:
        graph (onnx.GraphProto): The ONNX graph containing nodes and initializers.
        tensor_name (str): Name of the tensor to retrieve the shape for.

    Returns:
        List[int]: Shape of the tensor.
    """

    for value_info in graph.input:
        if value_info.name == tensor_name:
            return [dim.dim_value for dim in value_info.type.tensor_type.shape.dim]

    for initializer in graph.initializer:
        if initializer.name == tensor_name:
            return initializer.dims

    raise ValueError(f"Tensor shape not found for: {tensor_name}")


def handle_transpose_reverse(
    node: onnx.NodeProto, current_order: List[str]
) -> List[str]:
    """
    Reverses the effect of the Transpose node by reversing the permutation order.

    Args:
        node (onnx.NodeProto): The Transpose node.
        current_order (List[str]): Current order of dimensions.

    Returns:
        List[str]: Updated dimension order after reversing the Transpose operation.
    """

    for attr in node.attribute:
        if attr.name == "perm":
            perm = list(attr.ints)
            reverse_perm = [perm.index(i) for i in range(len(perm))]
            current_order = [current_order[i] for i in reverse_perm]
            return current_order
    return current_order


def reverse_embedding_or_dense(
    node: onnx.NodeProto, current_order: List[str], graph: onnx.GraphProto
) -> Tuple[List[str], int]:
    """
    Reverses the effect of an Embedding or Dense (Gather) layer, removing the feature dimension.

    Args:
        node (onnx.NodeProto): The Embedding or Dense node.
        current_order (List[str]): Current order of dimensions.
        graph (onnx.GraphProto): The ONNX graph containing nodes and initializers.

    Returns:
        Tuple[List[str], int]: Updated dimension order and the detected input channels.
    """

    # Get the input shape of the "indices" for the Gather node
    indices_input_name = node.input[1]  # The second input to Gather is the indices
    indices_shape = get_tensor_shape(graph, indices_input_name)

    if len(indices_shape) == 2:
        # If the indices input is 2D (N, T), we assume the output will have a feature
        # dimension added
        current_order = ["N", "T"]  # Drop the feature dimension
        embedding_size = get_tensor_shape(graph, node.input[1])[1]  # Get embedding size
        current_channels = embedding_size
        print(f"Embedding size: {embedding_size}")
    else:
        # Handle the case where indices input has more than 2 dimensions
        # This is a more complex gather, but we'll handle it similarly by dropping the
        # added feature dimension
        current_order = ["N"] + ["T" for _ in range(len(indices_shape) - 1)]
        embedding_size = get_tensor_shape(graph, node.input[1])[1]  # Get embedding size
        current_channels = embedding_size

    return current_order, current_channels


def handle_simple_layer(node: onnx.NodeProto, current_order: List[str]) -> List[str]:
    """
    Handles layers that do not change the input order (e.g., Linear, BatchNorm, Dropout).

    Args:
        node (onnx.NodeProto): The node to process.
        current_order (List[str]): Current order of dimensions.

    Returns:
        List[str]: Unchanged dimension order.
    """

    # Linear/Embedding, Dropout, BatchNorm, LayerNorm, Positional Encoding do not change input order
    return current_order


def trace_path_backward(
    graph: onnx.GraphProto, path: List[str], conv_node: onnx.NodeProto
) -> Tuple[List[str], int, List[int], List[int]]:
    """
    Traces the path backward from the Conv node to determine input order and channels.

    Args:
        graph (onnx.GraphProto): The ONNX graph.
        path (List[str]): List of node names in the path to process.
        conv_node (onnx.NodeProto): The Conv node to start tracing from.

    Returns:
        Tuple[List[str], int, List[int], List[int]]: Final input order, input channels,
        stride, and kernel shape.
    """

    # Start from the Conv node and deduce input order based on ONNX convolution standards
    conv_len, kernel_shape, _, input_channels, _, stride = get_conv_attributes(
        graph, conv_node
    )

    # Set the initial input order based on the conv_len
    if conv_len == 3:  # 1D convolution (standard N, C, W)
        current_order = ["N", "C", "W"]
    elif conv_len == 4:  # Extra dimension for single channel
        current_order = ["N", "C", "1", "W"]

    # Process the path in reverse (from conv node to input)
    for node_name in reversed(path):
        node = next((n for n in graph.node if n.name == node_name), None)

        if not node:
            continue

        if node.op_type == "Pad":
            current_order = handle_pad(current_order)

        elif node.op_type == "Unsqueeze":
            current_order = handle_unsqueeze_reverse(node, current_order, graph)

        elif node.op_type == "Transpose":
            current_order = handle_transpose_reverse(node, current_order)

        elif node.op_type == "Reshape":
            current_order = handle_reshape_reverse(node, current_order, graph)

        elif node.op_type == "Squeeze":
            current_order = handle_squeeze_reverse(node, current_order, graph)

        elif node.op_type == "Concat":
            current_order = handle_concat_reverse(node, current_order)
        elif node.op_type in ["Gather", "MatMul"]:  # Handle embedding and dense layers
            current_order, input_channels = reverse_embedding_or_dense(
                node, current_order, graph
            )

        # Handle layers that don't change input order (e.g., Linear, BatchNorm, Dropout, etc.)
        elif node.op_type in [
            "Dropout",
            "BatchNormalization",
            "LayerNormalization",
            "Gemm",
            "Embedding",
        ]:
            current_order = handle_simple_layer(node, current_order)

    return current_order, input_channels, stride, kernel_shape


def get_input_order(
    graph: onnx.GraphProto,
) -> Tuple[List[str], int, List[int], List[int]]:
    """
    Determines the input order, input channels, kernel shape, and stride of the first Conv layer.

    Args:
        graph (onnx.GraphProto): The ONNX graph.

    Returns:
        Tuple[List[str], int, List[int], List[int]]: Final input order, input channels,
        kernel shape,and stride.
    """

    path_to_conv = path_finder(graph)

    # Find the first convolution node in the path (assuming the last node is the Conv node)
    conv_node_name = path_to_conv[0][-1]
    conv_node = next((n for n in graph.node if n.name == conv_node_name), None)

    if not conv_node:
        print("No Conv node found.")

    # Trace the path backward from the Conv node to the input
    final_order, input_channels, stride, kernel_shape = trace_path_backward(
        graph, path_to_conv[0][1:], conv_node
    )

    return final_order, input_channels, kernel_shape, stride


def change_input_dims(
    graph: onnx.GraphProto, time_steps: int
) -> Tuple[Dict[str, Tuple[int, ...]], onnx.GraphProto]:
    """
    Adjusts the input dimensions of the graph based on the expected input order.

    Args:
        graph (onnx.GraphProto): The ONNX graph containing the nodes and initializers.
        time_steps (int): Number of time steps for the input sequence.

    Returns:
        Tuple[Dict[str, Tuple[int, ...]], onnx.GraphProto]: The modified input shapes
        and updated graph.
    """

    final_order, input_c, kernel_shape, stride = get_input_order(graph)

    width = (
        (time_steps - 1) * stride[1] + kernel_shape[1] if stride[1] > 1 else time_steps
    )

    input_shapes = {}
    for input_node in graph.input:
        input_shape = [dim.dim_value for dim in input_node.type.tensor_type.shape.dim]

        # Check if the input shape length matches the final order length
        if len(final_order) != len(input_shape):
            # Print a warning
            print(
                f"The input shape {input_shape} doesn't match model's expected order {final_order}."
            )

            # Ask the user for the correct input shape
            correct_shape = input(
                "Please provide the correct input shape (comma-separated): "
            )

            # Convert the user's input to a list of integers
            correct_shape_list = [
                int(dim) if dim != "?" else None for dim in correct_shape.split(",")
            ]

            # Ensure the input shape and final order lengths match
            if len(correct_shape_list) != len(final_order):
                raise ValueError(
                    f"The provided shape does not match the expected dimensions {final_order}."
                )

            # Update the input shape with the correct user-provided shape
            input_shape = correct_shape_list
            print(f"Using corrected input shape: {input_shape}")

            # Assign or process the input shape for further model adjustments
            input_shapes[input_node.name] = input_shape
        else:

            if len(final_order) == 1:
                if final_order == ["W"]:
                    input_shapes[input_node.name] = width
                elif final_order == ["C"]:
                    input_shapes[input_node.name] = input_c

            elif len(final_order) == 2:
                if final_order == ["N", "W"]:
                    input_shapes[input_node.name] = (1, width)
                elif final_order == ["N", "C"]:
                    input_shapes[input_node.name] = (1, input_c)

                elif final_order == ["W", "C"]:

                    input_shapes[input_node.name] = (
                        width,
                        input_c,
                    )

                elif final_order == ["C", "W"]:
                    input_shapes[input_node.name] = (input_c, width)
                else:
                    raise ValueError(f"Unsupported input order: {final_order}")

            elif len(final_order) == 3:
                if final_order == ["N", "C", "W"]:
                    input_shapes[input_node.name] = (1, input_c, width)
                elif final_order == ["N", "W", "C"]:
                    input_shapes[input_node.name] = (1, width, input_c)
                elif final_order == ["C", "N", "W"]:
                    input_shapes[input_node.name] = (input_c, 1, width)
                elif final_order == ["C", "W", "N"]:
                    input_shapes[input_node.name] = (input_c, width, 1)
                elif final_order == ["W", "N", "C"]:
                    input_shapes[input_node.name] = (width, 1, input_c)
                elif final_order == ["W", "C", "N"]:
                    input_shapes[input_node.name] = (width, input_c, 1)
                else:
                    raise ValueError(f"Unsupported input order: {final_order}")

        if len(input_shape) == 1:
            input_node.type.tensor_type.shape.dim[0].dim_value = input_shapes[
                input_node.name
            ]
        elif len(input_shape) == 2:
            input_node.type.tensor_type.shape.dim[0].dim_value = input_shapes[
                input_node.name
            ][0]
            input_node.type.tensor_type.shape.dim[1].dim_value = input_shapes[
                input_node.name
            ][1]
        else:
            input_node.type.tensor_type.shape.dim[0].dim_value = input_shapes[
                input_node.name
            ][0]
            input_node.type.tensor_type.shape.dim[1].dim_value = input_shapes[
                input_node.name
            ][1]
            input_node.type.tensor_type.shape.dim[2].dim_value = input_shapes[
                input_node.name
            ][2]

        print(f"Input shape for {input_node.name}: {input_shapes[input_node.name]}")

    return input_shapes, graph


def change_output_dims(
    graph: onnx.GraphProto,
) -> Tuple[Dict[str, Tuple[int, ...]], onnx.GraphProto]:
    """
    Adjusts the output dimensions of the graph to support dynamic shapes.

    Args:
        graph (onnx.GraphProto): The ONNX graph containing the nodes and initializers.

    Returns:
        Tuple[Dict[str, Tuple[int, ...]], onnx.GraphProto]:
        The modified output shapes and updated graph.
    """

    orig_output_shape = {}
    for output_node in graph.output:
        output_shape = [dim.dim_value for dim in output_node.type.tensor_type.shape.dim]
        if len(output_shape) == 1:
            output_node.type.tensor_type.shape.dim[0].dim_param = "dim0"
        elif len(output_shape) == 2:
            output_node.type.tensor_type.shape.dim[0].dim_param = "dim0"
            output_node.type.tensor_type.shape.dim[1].dim_param = "dim1"
        elif len(output_shape) == 3:
            output_node.type.tensor_type.shape.dim[0].dim_param = "dim0"
            output_node.type.tensor_type.shape.dim[1].dim_param = "dim1"
            output_node.type.tensor_type.shape.dim[2].dim_param = "dim2"

    return orig_output_shape, graph
