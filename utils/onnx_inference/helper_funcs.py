"""
This module provides utility functions for network inference using ONNX models. The functions
include creating and initializing buffers, retrieving bias values, reading input names, updating
buffers, extracting relevant buffer portions, preparing input dictionaries, and updating outputs.
"""

import re
from typing import Dict, Tuple, List, Union, Optional, Any
import numpy as np
import torch
import onnx
from streamease.onnx_streamer.helper import (
    get_conv_attributes,
    get_correct_shapes_framework,
)


def create_buffers(model_path: str, time_steps: int = 1) -> Tuple[
    Dict[str, np.ndarray],
    Dict[str, int],
    Dict[str, int],
    Dict[str, int],
    Dict[str, Dict[str, Union[int, str]]],
]:
    """
    Creates and initializes buffers for each Conv and CumSum layer in the ONNX model.

    Args:
        model_path (str): The path to the ONNX model.
        time_steps (int): The number of time steps for the buffer.

    Returns:
        Tuple[Dict[str, np.ndarray], Dict[str, int], Dict[str, int], Dict[str, int], Dict[str,
        Dict[str, Union[int, str]]]]:

        Returns dictionaries storing buffers, kernel sizes, dilation rates, convolution lengths,
        and buffer indices.
    """

    current_buffers = {}  # Dictionary to store buffers for each layer
    kernel_sizes = {}  # Dictionary to store kernel size for each layer
    dilations_rates = {}  # Dictionary to store dilations rate for each layer
    conv_lengths = {}  # Dictionary to store dilations rate for each layer
    buffer_index = (
        {}
    )  # Dictionary to store the current network's output for each buffer

    onnx_model = onnx.load(model_path)

    count_convs = 0
    count_cumsums = 0
    count_buffer = 0
    total_params = 0
    gap_params = 0

    # Iterate over the nodes in the original graph
    for i, node in enumerate(onnx_model.graph.node):
        # Check if the node is a Conv node
        if node.op_type == "Conv":
            # print(f"Found Conv node {node.name}")

            conv_len, kernel_shape, _, input_channels, _, _ = get_conv_attributes(
                onnx_model.graph, node
            )

            dilation_rate, buffer_name = get_dilation_rate_from_buffer(
                onnx_model.graph, count_convs
            )

            if dilation_rate is None:
                count_convs += 1
                continue

            # Compute buffer size based on the layer's dilation rate
            buffer_size = (kernel_shape[1] - 1) * dilation_rate  # + 1

            buffer_dim = get_correct_shapes_framework(
                conv_len, 1, input_channels, 1, buffer_size
            )  # [1, input_channels, 1, kernel_shape[1]-1]

            total_params += np.prod(buffer_dim)

            # Check if the buffer is enough to store the intermediate output
            if time_steps * (kernel_shape[1] - 1) > buffer_dim[-1]:
                gap_buffer_dim = list(buffer_dim)
                gap_buffer_dim[-1] = time_steps
                gap_params += np.prod(gap_buffer_dim)

            # We need only a part of the buffer
            else:
                gap_buffer_dim = list(buffer_dim)
                gap_buffer_dim[-1] = time_steps * (kernel_shape[1] - 1)
                gap_params += np.prod(gap_buffer_dim)

            # Convert the list to a tuple
            buffer_dim = tuple(buffer_dim)

            # Initialize buffer for the current layer
            buffer = np.zeros(buffer_dim)

            # Add the buffer to the dictionary
            current_buffers[buffer_name] = buffer

            kernel_sizes[f"conv_{count_buffer}"] = kernel_shape[1]
            dilations_rates[f"conv_{count_buffer}"] = dilation_rate
            conv_lengths[f"conv_{count_buffer}"] = conv_len

            b_index, b_name = find_concat_input_index(onnx_model.graph, count_convs)
            buffer_index[f"conv_{count_buffer}"] = {"name": b_name, "index": b_index}
            count_convs += 1
            count_buffer += 1

        elif node.op_type == "CumSum":
            count_cumsums += 1
            buffer_dim = (1, time_steps)
            buffer_dim = tuple(buffer_dim)
            buffer = np.zeros(buffer_dim)
            current_buffers[f"CumSum_buffer_{count_cumsums}"] = buffer

            b_index, b_name = find_cumsum_input_index(onnx_model.graph, count_cumsums)
            buffer_index[f"CumSum_buffer_{count_cumsums}"] = {
                "name": b_name,
                "index": b_index,
            }

    # Check if an Expand node is present in the input of the graph
    for input_info in onnx_model.graph.input:

        if "Expand" in input_info.name:

            buffer_dim = (1, time_steps)
            buffer_dim = tuple(buffer_dim)
            buffer = np.zeros(buffer_dim)

            min_value = extract_number(input_info.name)
            for i in range(time_steps):
                buffer[:, i] = min_value * (i + 1)

            current_buffers[input_info.name] = buffer
            buffer_index[input_info.name] = {"name": None, "index": None}
            count_buffer += 1
            total_params += np.prod(buffer_dim)
            gap_params += np.prod(buffer_dim)

    print(f"Total number of values (Buffers): {total_params}")
    return current_buffers, kernel_sizes, dilations_rates, conv_lengths, buffer_index


def get_conv_transpose_bias(model_path: str) -> List[float]:
    """
    Retrieves the bias values for ConvTranspose nodes associated
    with decoding layers in an ONNX model.

    Specifically searches for ConvTranspose nodes that contain
    'decoder' or 'istft' in their names and returns
    the corresponding bias values.

    Args:
        model_path (str): The path to the ONNX model file.

    Returns:
        List[float]: The bias values associated with the ConvTranspose layers.
        If no biases are found, returns [0].
    """
    bias_value = [0]  # Default bias value if none is found
    onnx_model = onnx.load(model_path)

    # Iterate over the nodes in the ONNX model graph
    for node in onnx_model.graph.node:
        # Check if the node is a ConvTranspose node with 'decoder' or 'istft' in its name
        if (
            (node.op_type == "ConvTranspose")
            and ("decoder" in node.name)
            or (node.op_type == "ConvTranspose")
            and ("istft" in node.name)
        ):
            # The bias name is often in the form of "decoder.bias"
            bias_name = "decoder.bias"
            for initializer in onnx_model.graph.initializer:
                if initializer.name == bias_name:
                    # Extract bias values from the initializer
                    bias_value = np.frombuffer(initializer.raw_data, dtype=np.float32)
                    break  # Stop searching if we find the matching initializer
    return bias_value


def read_input_name(model_path: str) -> Optional[str]:
    """
    Reads the input name of an ONNX model. Assumes the first input is
    the primary input tensor for the model, which is typical for non-streaming models.

    Args:
        model_path (str): The file path to the ONNX model.

    Returns:
        Optional[str]: The name of the primary input tensor, or None if no inputs are found.
    """

    # Load the ONNX model
    model = onnx.load(model_path)

    # Get the list of input names from the model
    input_names = [input.name for input in model.graph.input]

    # Return the first input name if available, otherwise return None
    return input_names[0] if input_names else None


def get_dilation_rate_from_buffer(
    graph: onnx.GraphProto, index: int
) -> Tuple[Optional[int], Optional[str]]:
    """
    Retrieves the dilation rate and corresponding buffer name from the graph using the given index.

    Args:
        graph (onnx.GraphProto): The ONNX graph containing the model nodes and inputs.
        index (int): The index used to locate the buffer name within the graph inputs.

    Returns:
        Tuple[Optional[int], Optional[str]]: A tuple containing:
            - dilation_rate (Optional[int]): The dilation rate if found, otherwise None.
            - input_name (Optional[str]): The buffer name if found, otherwise None.
    """

    # Construct the expected buffer name using the index
    buffer_name = f"buffer_{index}_k"

    # Iterate over the inputs in the graph to find a match with the buffer name
    for input_info in graph.input:
        input_name = input_info.name
        if buffer_name in input_name:
            # Extract the dilation rate from the input name after the "_d" marker
            start_index = input_name.find("_d") + 2
            end_index = input_name.find("_", start_index)

            # Ensure indices are valid before extracting the substring
            if start_index != -1 and end_index != -1:
                dilation_rate_str = input_name[start_index:end_index]

                try:
                    # Convert the extracted string to an integer dilation rate
                    dilation_rate = int(dilation_rate_str)
                    return dilation_rate, input_name
                except ValueError:
                    print(
                        f"Warning: Unable to parse dilation rate from '{dilation_rate_str}'."
                    )
                    return None, None

    # Return None for both values if no matching buffer is found
    return None, None


def find_concat_input_index(
    graph: onnx.GraphProto, concat_index: int
) -> Tuple[Optional[str], Optional[str]]:
    """
    Finds the index and name of the second input to the Concat node within the ONNX graph.

    Args:
        graph (onnx.GraphProto): The ONNX graph containing nodes and inputs/outputs.
        concat_index (int): The index used to locate the specific Concat node in the graph.

    Returns:
        Tuple[Optional[str], Optional[str]]: A tuple containing:
            - index (Optional[str]): The index of the Concat input, either as an integer
            or a string "input#".
            - name (Optional[str]): The name of the second input to the Concat node, if found.
            If no match is found, returns (None, None).
    """

    # Construct the expected name for the Concat node based on the given index
    concat_name = f"Concat_{concat_index}"
    concat_node = None

    # Locate the Concat node in the graph
    for node in graph.node:
        if node.op_type == "Concat" and concat_name in node.name:
            concat_node = node
            break

    if concat_node is None:
        return None, None

    # Retrieve the second input to the Concat node if it exists
    if len(concat_node.input) > 1:
        concat_input_name = concat_node.input[1]

        # Search through graph outputs for a match with concat_input_name
        for i, output_info in enumerate(graph.output):
            if output_info.name == concat_input_name:
                return i, output_info.name

        # Find the index of the concat_input_name in graph.input
        for i, input_info in enumerate(graph.input):
            if input_info.name == concat_input_name:
                return f"input{i}", input_info.name

    # If no valid second input is found, return None for both values
    return None, None


def find_cumsum_input_index(
    graph: onnx.GraphProto, cumsum_index: int
) -> Tuple[Optional[int], Optional[str]]:
    """
    Finds the index and name of the output of the CumSum node in the ONNX graph based on
    a specified index.

    Args:
        graph (onnx.GraphProto): The ONNX graph containing the nodes.
        cumsum_index (int): The 1-based index used to locate the specific CumSum node in the graph.

    Returns:
        Tuple[Optional[int], Optional[str]]: A tuple containing:
            - index (Optional[int]): The index of the CumSum node's output within the graph outputs,
            if found.
            - name (Optional[str]): The name of the CumSum node's output, if found.
            If no match is found, returns (None, None).
    """

    # Locate the CumSum node in the graph based on the cumsum_index
    cumsum_node = None
    itr = 0
    for node in graph.node:
        if node.op_type == "CumSum":
            itr += 1
            if itr == cumsum_index:
                cumsum_node = node
                break

    if cumsum_node is None:
        return None, None

    # Retrieve the output name of the identified CumSum node
    cumsum_output_name = cumsum_node.output[0]

    # Find the index of the CumSum output within the graph outputs
    for i, output_info in enumerate(graph.output):
        if output_info.name == cumsum_output_name:
            return i, output_info.name

    # If no match is found in the graph outputs, return None for both values
    return None, None


def update_buffers(
    current_buffers: Dict[str, np.ndarray],
    new_states: Dict[str, np.ndarray],
    current_seq: np.ndarray,
    conv_lengths: Dict[str, int],
    buffer_index: Dict[str, Dict[str, Any]],
    time_steps: int = 1,
    receptive_field: int = 1,
    reset: bool = False,
) -> Dict[str, np.ndarray]:
    """
    Updates the current buffers based on new states from the convolutional layers.
    This function manages the buffers for different layers including CumSum, Expand,
    and Conv layers.

    Args:
        current_buffers (Dict[str, np.ndarray]): The current buffer states for each layer.
        new_states (Dict[str, np.ndarray]): The new states produced by the model layers.
        current_seq (np.ndarray): The current input sequence being processed.
        conv_lengths (Dict[str, int]): Dictionary containing the convolution lengths for each layer.
        buffer_index (Dict[str, Dict[str, Any]]): Dictionary mapping buffer indices and names for
        each layer.
        time_steps (int, optional): Number of time steps for which to update the buffer.
        Defaults to 1.
        receptive_field (int, optional): Receptive field of the model for CumSum layers.
        Defaults to 1.
        reset (bool, optional): Whether to reset the buffers for the current input sequence.
        Defaults to False.

    Returns:
        Dict[str, np.ndarray]: Updated buffer states for each layer.
    """

    i = 0  # Track Conv layer index
    j = 1  # Track CumSum layer index

    for key in current_buffers.keys():

        # Handle CumSum layers
        if "CumSum" in key:
            if reset:  # Reset the buffer
                current_buffers[key][:, :] = 0
            else:
                correct_index = buffer_index[f"CumSum_buffer_{j}"]["index"]
                current_buffers[key] = np.roll(
                    current_buffers[key], shift=-time_steps, axis=1
                )
                current_buffers[key][:, -time_steps:] = new_states[correct_index][
                    :, -time_steps:
                ]
            j += 1

        # Handle Expand layers
        elif "Expand" in key:
            min_value = extract_number(key)
            if reset:
                for i in range(time_steps):
                    current_buffers[key][:, i] = min_value * (i + 1)
            else:
                current_buffers[key][:, :] = current_buffers[key][:, :] + (
                    min_value * time_steps
                )

        # Handle Conv layers
        else:
            conv_dim = conv_lengths[f"conv_{i}"]
            axis = 2 if conv_dim == 3 else 3
            updated_len = min(time_steps, current_buffers[key].shape[axis])
            # Roll the buffer and update with new states or current sequence
            current_buffers[key] = np.roll(
                current_buffers[key], shift=-updated_len, axis=axis
            )
            correct_index = buffer_index[f"conv_{i}"]["index"]
            target = (
                current_seq
                if correct_index == "input0"
                else new_states[int(correct_index)]
            )
            if conv_dim == 3:
                current_buffers[key][:, :, -updated_len:] = target[:, :, -updated_len:]
            else:
                current_buffers[key][:, :, :, -updated_len:] = (
                    target[:, :, :, -updated_len:]
                    if target.ndim == 4
                    else target[:, :, -updated_len:]
                )

            i += 1

    return current_buffers


def extract_number(name: str) -> Optional[int]:
    """
    Extracts a number following an underscore in the provided string.
    If no such pattern is found, returns None.

    Args:
        name (str): The string from which to extract the number.

    Returns:
        Optional[int]: The extracted number as an integer, or None if no number is found.
    """

    # Use a regex pattern to search for an underscore followed by digits
    match = re.search(r"_(\d+)", name)
    if match:
        # If a match is found, return the first group as an integer
        return int(match.group(1))
    # If no match is found, return None
    return None


def shift_last_to_first(
    buffer: Union[np.ndarray, torch.Tensor]
) -> Union[np.ndarray, torch.Tensor]:
    """
    Shifts the last element of a 2-dimensional buffer to the first position and fills
    the remaining positions with zeros. The function supports both NumPy arrays and
    PyTorch tensors.

    Args:
        buffer (Union[np.ndarray, torch.Tensor]): A 2-dimensional buffer to be processed.

    Returns:
        Union[np.ndarray, torch.Tensor]: A buffer with the last element shifted to the
        first position and the rest filled with zeros. Returns a NumPy array if the input
        is a NumPy array, otherwise returns a PyTorch tensor.

    Raises:
        TypeError: If the buffer is not a NumPy array or PyTorch tensor.
        ValueError: If the buffer is not 2-dimensional.

    Example:
        >>> buffer = np.array([[1, 2, 3], [4, 5, 6]])
        >>> shift_last_to_first(buffer)
        array([[3, 0, 0],
            [6, 0, 0]])

        >>> buffer = torch.tensor([[1, 2, 3], [4, 5, 6]])
        >>> shift_last_to_first(buffer)
        tensor([[3, 0, 0],
                [6, 0, 0]])
    """

    # Convert NumPy array to PyTorch tensor if needed
    if isinstance(buffer, np.ndarray):
        buffer = torch.from_numpy(buffer)
    elif not isinstance(buffer, torch.Tensor):
        raise TypeError("Input buffer must be a NumPy array or a PyTorch tensor.")

    # Ensure buffer is 2-dimensional
    if buffer.dim() != 2:
        raise ValueError("Input buffer must be 2-dimensional.")

    # Shift last element to first position and fill rest with zeros
    shifted_buffer = np.zeros_like(buffer)
    shifted_buffer[:, 0] = buffer[:, -1]

    return shifted_buffer.numpy() if isinstance(buffer, np.ndarray) else shifted_buffer


def get_relevant_buffer(
    current_buffers: Dict[str, np.ndarray],
    kernel_sizes: Dict[str, int],
    dilations_rates: Dict[str, int],
    conv_lengths: Dict[str, int],
    time_steps: int = 1,
) -> Dict[str, np.ndarray]:
    """
    Extracts the relevant portions of buffers based on kernel sizes, dilation rates,
    and convolution lengths. This function supports various layers, including CumSum
    and Expand layers.

    Args:
        current_buffers (Dict[str, np.ndarray]): A dictionary containing the buffers for
            each layer as NumPy arrays.
        kernel_sizes (Dict[str, int]): A dictionary containing kernel sizes for convolution layers.
        dilations_rates (Dict[str, int]): A dictionary containing dilation rates for convolution
        layers.
        conv_lengths (Dict[str, int]): A dictionary containing the length dimensions
        (e.g., 3 for 1D conv) for convolution layers.
        time_steps (int): The number of time steps to consider. Defaults to 1.

    Returns:
        Dict[str, np.ndarray]: A dictionary with layer names as keys and the relevant portions
        of buffers as values in NumPy array format.

    Raises:
        KeyError: If expected layer information (kernel size, dilation rate, conv length) is not
        available.
    """
    relevant_buffers = {}
    i = 0  # Index for convolution layers
    j = 1  # Index for CumSum layers

    for key in current_buffers.keys():
        if "CumSum" in key:  # CumSum layer
            buffer = current_buffers[key]
            relevant_buffer = buffer[:, -time_steps:]

            # Shift last elements to first position and fill rest with zeros
            relevant_buffer = shift_last_to_first(relevant_buffer)
            relevant_buffers[key] = relevant_buffer.astype(np.float32)
            j += 1

        elif "Expand" in key:  # Expand layer
            # Expand layer (assume full buffer is relevant)
            buffer = current_buffers[key]
            relevant_buffer = buffer
            relevant_buffers[key] = relevant_buffer.astype(np.float32)

        else:  # Convolution layer

            dilation_rate = dilations_rates[f"conv_{i}"]
            kernel_size = kernel_sizes[f"conv_{i}"]
            conv_len = conv_lengths[f"conv_{i}"]
            num_rows = kernel_size - 1
            buffer = current_buffers[key]

            # Determine axis based on conv_len and set up buffer slicing
            axis = 2 if conv_len == 3 else 3

            if time_steps * (kernel_size - 1) > buffer.shape[axis]:
                # Use all elements if the calculated length exceeds buffer size
                relevant_indices = range(0, buffer.shape[axis])

            else:
                # Calculate starting point and gather indices based on kernel size and dilation rate
                start_row = buffer.shape[axis] - (num_rows * dilation_rate)
                relevant_indices = [
                    idx
                    for j in range(num_rows)
                    for idx in range(
                        start_row + j * dilation_rate,
                        start_row + j * dilation_rate + time_steps,
                    )
                ]

            # Select the relevant indices based on conv_len (1D or other)
            if conv_len == 3:
                relevant_buffer = buffer[:, :, relevant_indices]
            else:
                relevant_buffer = buffer[:, :, :, relevant_indices]

            # Store the relevant buffer after slicing
            relevant_buffers[key] = relevant_buffer.astype(np.float32)

            i += 1

    return relevant_buffers


def prepare_input_dictionary(
    r_buffers: Dict[str, np.ndarray], input_data: np.ndarray, input_name: str
) -> Dict[str, np.ndarray]:
    """
    Prepares the input dictionary for the ONNX model by combining the input data
    and relevant buffers.

    Args:
        r_buffers (Dict[str, np.ndarray]): A dictionary of buffer arrays where keys
        are buffer names.
        input_data (np.ndarray): The input data for the model, typically in the format
        expected by the
        first input layer.
        input_name (str): The name of the main input tensor in the ONNX model.

    Returns:
        Dict[str, np.ndarray]: A dictionary containing the main input data and the
        associated buffers
        for streaming inference.
    """

    # Initialize the input dictionary
    input_dict: Dict[str, np.ndarray] = {}

    # Add the main input data to the dictionary, casting it to float32 if necessary
    input_dict[input_name] = input_data.astype(np.float32)

    for key in r_buffers.keys():
        input_dict[key] = r_buffers[key]

    return input_dict


def update_output(
    net_output: Union[List[float], np.ndarray],
    current_output: Union[List[float], np.ndarray],
    iteration: int,
    frame_length: int,
    stride: int,
    bias: Union[List[float], np.ndarray],
) -> Union[List[float], np.ndarray]:
    """
    Updates the output buffer with the current output values, handling overlap
    between consecutive outputs by summing overlapping values and subtracting the bias.

    Args:
        net_output (Union[List[float], np.ndarray]): The existing output buffer that stores
        the aggregated output values.
        current_output (Union[List[float], np.ndarray]): The current output values to be added
        to the output buffer.
        iteration (int): The current iteration index.
        output_length (int): The length of the current output segment.
        stride (int): The stride or step size for the output update.
        bias (Union[List[float], np.ndarray]): The bias value(s) to subtract during overlap
        resolution.

    Returns:
        Union[List[float], np.ndarray]: The updated output buffer with aggregated values.
    """

    # Handle the first segment without overlap
    if iteration == 0:
        net_output[:frame_length] = current_output[:frame_length]
    else:
        # Calculate the overlap length, ensuring it doesn't exceed the net_output bounds
        overlap_length = min(
            frame_length - stride, len(net_output) - iteration * stride
        )

        net_output[iteration * stride : iteration * stride + overlap_length] = [
            a + b - bias[0]
            for a, b in zip(
                net_output[iteration * stride : iteration * stride + overlap_length],
                current_output[:overlap_length],
            )
        ]

        # Update the non-overlapping part with the remaining current output values
        net_output[
            iteration * stride + overlap_length : iteration * stride + frame_length
        ] = current_output[overlap_length:frame_length]
    return net_output
