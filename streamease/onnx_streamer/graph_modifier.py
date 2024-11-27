"""
    This module provides functionality to modify
    an ONNX model's graph to support streaming or time-step-based operations.

    Functions:
        modify_graph_for_streaming(model:
        onnx.ModelProto, new_graph: onnx.GraphProto,
        time_steps: int) -> onnx.GraphProto
"""

import onnx
from .helper import (
    get_correct_shapes_framework,
    find_node_index_by_name,
    is_there_slice_node,
    is_there_pad_node,
    get_conv_attributes,
)


def modify_graph_for_streaming(
    model: onnx.ModelProto, new_graph: onnx.GraphProto, time_steps: int
) -> onnx.GraphProto:
    """
    Modifies the given ONNX model's graph to support streaming or time-step-based operations.

    Args:
        model (onnx.ModelProto): The original ONNX model to modify.
        new_graph (onnx.GraphProto): The new graph where modifications are applied.
        time_steps (int): The number of time steps for streaming.

    Returns:
        onnx.GraphProto: The modified ONNX graph with streaming adjustments.
    """
    conv_node_counter = 0  # to count the number of conv nodes
    cumsum_node_counter = 0  # to count the number of cumsum nodes

    for i, node in enumerate(model.graph.node):

        # Modify the CumSum node behavior
        if node.op_type == "CumSum":

            cumsum_node_counter += 1
            buffer_name = f"CumSum_buffer_{cumsum_node_counter}"
            buffer_dim = [1, time_steps]

            # Add a buffer node to the graph for the CumSum operation
            new_graph.input.extend(
                [
                    onnx.helper.make_tensor_value_info(
                        buffer_name, onnx.TensorProto.FLOAT, buffer_dim
                    )
                ]
            )

            add_node_name = f"Add_CumSum_{cumsum_node_counter}"
            add_node_output = f"Add_CumSum_{cumsum_node_counter}_Output"

            add_node_for_cumsum = onnx.helper.make_node(
                "Add",
                inputs=[buffer_name, node.input[0]],
                outputs=[add_node_output],
                name=add_node_name,
            )

            # Find the node in the new graph that corresponds to the original CumSum node
            sequence_node_index, sequence_node = next(
                ((i, n) for i, n in enumerate(new_graph.node) if n.name == node.name),
                (None, None),
            )

            if sequence_node is None:
                sequence_node_index = i

            # # Insert Add node just before the current sequence node (keep the order of the nodes)
            new_graph.node.insert(sequence_node_index - 1, add_node_for_cumsum)

            # Update the original node's input to use the Add node's output
            for graph_node in new_graph.node:
                if graph_node.name == node.name:
                    graph_node.input[0] = add_node_output

            # Extend the output of the graph with the updated node output shape
            node_shape = [1, time_steps]
            new_graph.output.extend(
                [
                    onnx.helper.make_tensor_value_info(
                        node.output[0], onnx.TensorProto.FLOAT, node_shape
                    )
                ]
            )

        # Handling ConvTranspose or Conv Nodes (for encoder or STFT parts)
        elif node.op_type == "ConvTranspose" or (
            node.op_type == "Conv"
            and (("encoder" in node.name) or ("stft" in node.name))
        ):

            # Find the corresponding node in the modified graph by name
            sequence_node = next(
                (n for n in new_graph.node if n.name == node.name), None
            )

            # Modify the dilation attribute if present
            if sequence_node is not None:
                for attr in sequence_node.attribute:
                    if attr.name == "dilations":
                        # Modify the dilation values as needed
                        attr.ints[:] = (
                            [1] if len(attr.ints) == 1 else [1, 1]
                        )  # Set dilation to 1

            conv_node_counter += 1
            continue

        # Handling Conv Layers (General Conv Layers)
        elif node.op_type == "Conv":

            # Retrieve convolution attributes like kernel size, dilation, etc.
            conv_len, kernel_shape, dilation_rate, input_channels, _, _ = (
                get_conv_attributes(model.graph, node)
            )

            # Skip processing for 1x1 convolutions
            if kernel_shape == [1, 1]:
                conv_node_counter += 1
                continue

            # Calculate buffer dimensions depending on the time steps and dilation rate
            if time_steps > dilation_rate[1]:
                buffer_dim = get_correct_shapes_framework(
                    conv_len,
                    1,
                    input_channels,
                    1,
                    dilation_rate[1] * (kernel_shape[1] - 1),
                )
                updated_dilation_rate = dilation_rate[1]
            else:
                buffer_dim = get_correct_shapes_framework(
                    conv_len, 1, input_channels, 1, time_steps * (kernel_shape[1] - 1)
                )
                updated_dilation_rate = time_steps

            # Find the corresponding node in the new graph
            sequence_node = next(
                (n for n in new_graph.node if n.name == node.name),
                None,
            )

            # Find the index of the node in the new_graph
            node_index = find_node_index_by_name(new_graph, node.name)

            # Calculate the reshape dimensions for the buffer
            reshape_dim = get_correct_shapes_framework(
                conv_len, 1, input_channels, 1, time_steps
            )

            # Ensure the Conv node's input is added to the graph outputs if not already present
            if sequence_node.input[0] not in [
                output.name for output in new_graph.output
            ]:
                if node.input[0] not in [inputs.name for inputs in new_graph.input]:
                    new_graph.output.extend(
                        [
                            onnx.helper.make_tensor_value_info(
                                sequence_node.input[0],
                                onnx.TensorProto.FLOAT,
                                reshape_dim,
                            )
                        ]
                    )

            # Create a buffer name for the current convolutional layer
            buffer_name = (
                f"buffer_{conv_node_counter}_k{kernel_shape[1]}_d"
                f"{dilation_rate[1]}_i{input_channels}"
            )

            concat_node_name = f"Concat_{conv_node_counter}"
            concat_node_output = f"Concat_{conv_node_counter}_Output"

            # Create Concat node to concatenate the new input with the original input
            concat_node = onnx.helper.make_node(
                "Concat",
                inputs=[buffer_name, sequence_node.input[0]],
                outputs=[concat_node_output],
                axis=conv_len
                - 1,  # Concatenate along the last dimension (e.g., time steps)
                name=concat_node_name,
            )

            # Insert the Concat node before the sequence node
            new_graph.node.insert(node_index, concat_node)

            # Update the Conv node to take the output of the Concat node as its input
            for graph_node in new_graph.node:
                if graph_node.name == node.name:
                    graph_node.input[0] = concat_node_output

            # # Update dilation rates in the Conv node attributes
            for attr in sequence_node.attribute:
                if attr.name == "dilations":
                    attr.ints[:] = (
                        [updated_dilation_rate]
                        if len(attr.ints) == 1
                        else [1, updated_dilation_rate]
                    )

            # Handle padding attribute if present
            for attr in sequence_node.attribute:
                if attr.name == "strides":
                    attr.ints[:] = [1] if len(attr.ints) == 1 else [1, 1]

            # Handle padding attribute if present
            pads_attr = next(
                (attr for attr in node.attribute if attr.name == "pads"), None
            )

            # Remove the "pads" attribute
            if pads_attr:
                sequence_node.attribute.remove(pads_attr)

            # Add buffer node to the graph's input for handling time steps
            new_graph.input.extend(
                [
                    onnx.helper.make_tensor_value_info(
                        buffer_name, onnx.TensorProto.FLOAT, buffer_dim
                    )
                ]
            )

            # Handling Slice nodes in the sequence
            target_sequence = [
                "Conv",
                "Add",
                "Relu",
                "Squeeze",
                "Constant",
                "LayerNormalization",
                "Gemm",
                "Transpose",
                "Unsqueeze",
            ]
            is_slice_node = is_there_slice_node(i, target_sequence, model.graph)

            if is_slice_node and is_slice_node.op_type == "Slice":

                # Update nodes that use the output of the Slice node to bypass it
                for graph_node in new_graph.node:
                    for i, input_name in enumerate(graph_node.input):
                        if input_name == is_slice_node.output[0]:
                            graph_node.input[i] = is_slice_node.input[0]

            # Handling Pad nodes in the sequence
            # Look backward to find the first node in the sequence
            sequence_op_types = ["Conv", "Pad", "Unsqueeze", "Transpose"]

            is_pad_node = is_there_pad_node(i, sequence_op_types, model)

            if is_pad_node.op_type == "Pad":

                # Iterate over the nodes in the new graph to update inputs
                for node_itr in new_graph.node:
                    if is_pad_node.output[0] in node_itr.input:
                        # Update nodes that use the output of the Pad node to bypass it
                        for i, input_name in enumerate(node_itr.input):
                            if input_name == is_pad_node.output[0]:
                                node_itr.input[i] = is_pad_node.input[0]
                                break
                # Remove the Pad node from the graph
                new_graph.node.remove(is_pad_node)

            conv_node_counter += 1

    return new_graph
