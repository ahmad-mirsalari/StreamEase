"""Streaming ONNX conversion utilities for TCN Denoiser."""

import os
import sys
import onnx
from streamease.onnx_streamer.streamer import StreamingConverter as converter

# Ensure parent directory is in path for streamease import
UTILS_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, UTILS_PATH)


def get_streaming_model(
    path: str, streaming_folder: str, time_steps: int
) -> onnx.ModelProto:
    """
    Load an ONNX model, convert it to streaming mode, and return the streaming model.

    Args:
        path (str): Path to the trained ONNX model.
        streaming_folder (str): Directory where streaming model will be saved.
        time_steps (int): Number of time steps for streaming configuration.

    Returns:
        onnx.ModelProto: Converted streaming ONNX model.
    """
    model = onnx.load(path)
    streaming = converter(model, time_steps=time_steps)

    streaming.run()

    file_name = "torch_streaming_model.onnx"
    streaming.save_streaming_onnx(streaming_folder, onnx_filename=file_name)

    # Construct the full file path
    streaming_path = os.path.join(streaming_folder, file_name)
    print("Streaming model is saved in ", streaming_path)

    return onnx.load(streaming_path)
