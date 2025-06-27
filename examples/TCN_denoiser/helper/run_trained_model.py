"""Run inference on a trained ONNX model with input data processing."""

import onnxruntime as ort
import numpy as np


def run_onnx_model(model_path: str, input_data: np.ndarray) -> np.ndarray:
    """
    Run an ONNX model for inference with fixed-size input processing.

    Args:
        model_path (str): Path to the ONNX model.
        input_data (np.ndarray): Input audio data (1D array).

    Returns:
        np.ndarray: Model output, truncated to original input length.
    """

    session_options = ort.SessionOptions()
    # session_options.enable_profiling = True
    ort_sess = ort.InferenceSession(model_path, sess_options=session_options)

    # ort_sess.enable_profiling = True

    len_input = input_data.shape[0]
    if input_data.shape[0] < 64000:
        input_data = np.pad(
            input_data, (0, 64000 - len(input_data)), "constant", constant_values=(0, 0)
        )

    elif input_data.shape[0] > 64000:
        input_data = input_data[:64000]

    if input_data.ndim == 1:
        input_data = np.expand_dims(input_data, axis=(0, 1))

    # Run the model
    input_name = ort_sess.get_inputs()[0].name
    output = ort_sess.run(None, {input_name: input_data})[0]

    output = np.transpose(output, (2, 0, 1))
    output = np.squeeze(output)
    output = np.squeeze(output)

    output = output[:len_input]  # Truncate the output to the length of the input

    return output
