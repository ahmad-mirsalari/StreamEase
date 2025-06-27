"""ONNX runtime utilities for encoder, main model, and decoder stages in TCN Denoiser."""

import numpy as np


def preprocessing(
    input_signal: np.ndarray, ort_sess_enc: any
) -> tuple[np.ndarray, int, np.ndarray]:
    """
    Run the encoder ONNX model on input signal and prepare for main model.

    Args:
        input_signal (np.ndarray): 1D input audio signal.
        ort_sess_enc (any): ONNX encoder session.

    Returns:
        tuple: Model output, original input length, and expanded input array.
    """
    input_data = input_signal

    original_length = input_data.shape[0]
    if input_data.ndim == 1:
        input_data = np.expand_dims(input_data, axis=(0, 1))

    input_name = ort_sess_enc.get_inputs()[0].name
    output = ort_sess_enc.run(None, {input_name: input_data})

    return output, original_length, input_data


def run_main_model(input_data: np.ndarray, ort_sess_main: any) -> np.ndarray:
    """
    Run the main ONNX model on encoder output.

    Args:
        input_data (np.ndarray): Encoder output tensor.
        ort_sess_main (any): ONNX main session.

    Returns:
        np.ndarray: Model output tensor.
    """

    input_tensor = input_data[0]
    input_name = ort_sess_main.get_inputs()[0].name
    output = ort_sess_main.run(None, {input_name: input_tensor})
    return output[0]


def post_processing(
    input_data: np.ndarray, main_outputs: np.ndarray, ort_sess_encdec: any
) -> np.ndarray:
    """
    Run encoder-decoder ONNX model for final output stage.

    Args:
        input_data (np.ndarray): Current audio input tensor.
        main_outputs (np.ndarray): Output from main model stage.
        ort_sess_encdec (any): ONNX encoder-decoder session.

    Returns:
        np.ndarray: Final model output tensor.
    """

    if input_data.ndim == 1:
        input_data = np.expand_dims(input_data, axis=(0, 1))

    input_name_main = ort_sess_encdec.get_inputs()[0].name
    input_name_data = ort_sess_encdec.get_inputs()[1].name

    final_output = ort_sess_encdec.run(
        None, {input_name_data: input_data, input_name_main: main_outputs}
    )
    return final_output
