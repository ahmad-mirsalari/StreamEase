"""
This module provides utility functions for preparing input data and extracting model outputs
for neural network inference.
"""

from typing import Dict, List, Union, Optional
import numpy as np


def prepare_input_list(
    r_buffers: Dict[str, np.ndarray], input_data: np.ndarray
) -> List[np.ndarray]:
    """
    Prepares a list of input data for model inference by appending buffers.

    Args:
        r_buffers (dict): A dictionary where keys are buffer names and values are numpy arrays.
        input_data (np.ndarray): The main input data for the model.

    Returns:
        list: A list containing the main input data followed by buffer data.
    """

    input_list = []
    input_list.append(input_data.astype(np.float32))
    for key in r_buffers.keys():
        input_list.append(r_buffers[key])

    return input_list


def extract_outputs_list(
    prediction: List[np.ndarray],
    bmodel: dict,
    buffer_index: Optional[Dict[str, Dict[str, Union[str, int]]]] = None,
    use_onnx_names: bool = False,
) -> List[Optional[np.ndarray]]:
    """
    Extracts and organizes the model's prediction outputs into a list, including buffer outputs
    if necessary.

    Args:
        prediction (list): The list of predictions from the model.
        bmodel (dict): The model's output nodes, mapping output names to their details.
        buffer_index (dict, optional): A dictionary with buffer information. Defaults to None.
        use_onnx_names (bool): Flag indicating whether to use ONNX names for output handling.
        Defaults to True.

    Returns:
        list: A list of outputs organized by buffer and non-buffer outputs.
    """

    if use_onnx_names:
        # we need to find the index that we must append the output to the needed_outputs list
        # for buffers
        # ( it is equal to number of speakers or the number of outputs of non-streaming model)
        if buffer_index is None:  # Check if buffer_index is empty or None
            first_buffer_index = 1
        else:
            # Extract the 'index' values from the nested dictionaries and find the minimum
            # Filter out None values for the 'index' and find the minimum
            valid_indices = [
                value["index"]
                for value in buffer_index.values()
                if value["index"] is not None and "input" not in value["name"]
            ]

            first_buffer_index = (
                int(min(valid_indices)) if valid_indices else 1
            )  # Default to 1 if no valid indices

        output_names = [i.name for i in list(bmodel.output_nodes())]

        # Initialize needed_outputs with None for each name
        needed_outputs = [None] * len(output_names)

        # we need to append the outputs for buffers to the needed_outputs list
        for i in range(len(output_names) - first_buffer_index):
            output_name = output_names[i]
            if output_name in bmodel:
                output_index = bmodel[output_name].step_idx
                # Loop through the values in buffer_index to find the index
                buffer_idx = next(
                    (
                        value["index"]
                        for value in buffer_index.values()
                        if value["name"] == output_name
                    ),
                    None,
                )

                if buffer_idx is not None:
                    needed_outputs[int(buffer_idx)] = prediction[output_index][0]
            else:
                continue

        # Handle non-buffer outputs at the start of the needed_outputs list
        for i in range(first_buffer_index):
            output_name = output_names[len(output_names) - i - 1]
            if output_name in bmodel:
                output_index = bmodel[output_name].step_idx
                # needed_outputs.insert(i, prediction[output_index][0])
                needed_outputs[i] = prediction[output_index][0]
            else:
                continue

    else:
        needed_outputs = []
        # prediction[bmodel["output_2"].step_idx]
        for i in range(len(bmodel)):
            output_name = f"output_{i}"
            if output_name in bmodel:
                output_index = bmodel[output_name].step_idx
                needed_outputs.append(prediction[output_index][0])
            else:
                continue
    return needed_outputs
