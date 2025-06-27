"""Streaming inference runner for ONNX and NNTool models in TCN Denoiser."""
import os
import sys
from typing import Optional, Any
import numpy as np


# Add utils directory to sys.path for external inference utilities
UTILS_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
sys.path.insert(0, UTILS_PATH)

from utils.nntool_inference.streaming_inference import Inference as nntool_inference
from utils.onnx_inference.streaming_inference import Inference as onnx_inference


def get_pad_info(x_input: np.ndarray, args: Any) -> tuple[int, int, int]:
    """
    Calculate padding information based on input shape and model parameters.

    Args:
        x_input (np.ndarray): Input audio signal.
        args (Any): Argument namespace with model parameters.

    Returns:
        tuple[int, int, int]: Tuple containing first_axis_pad, second_axis_pad, rest.
    """
    window = args.L
    stride = args.S
    num_samples = x_input.shape[0]
    rest = window - (stride + num_samples % window) % window

    first_axis_pad = stride
    second_axis_pad = rest + stride

    return window, 700, 0


def run_streaming_model(
    model_nn: Optional[Any],
    streaming_model_path: str,
    input_data: np.ndarray,
    encoder: Optional[str] = None,
    enc_dec: Optional[str] = None,
    use_onnx_names: bool = False,
    args: Optional[Any] = None,
) -> np.ndarray:
    """
    Run streaming inference for ONNX or NNTool models with padding and post-processing.

    Args:
        model_nn (Optional[Any]): NNTool model graph, required for nntool modes.
        streaming_model_path (str): Path to the ONNX streaming model.
        input_data (np.ndarray): Input signal array.
        encoder (Optional[str]): Encoder model path for separated networks.
        enc_dec (Optional[str]): Encoder-decoder model path.
        use_onnx_names (bool): Whether to use ONNX naming in NNTool.
        args (Optional[Any]): Argument namespace with runtime configurations.

    Returns:
        np.ndarray: Model output signal, post-processed.
    """

    if args.mode in ("streaming-onnx-main", "streaming-nntool-main"):
        full_network = False
    elif args.mode == "streaming-onnx-full":
        full_network = True
    else:
        full_network = False

    original_length = input_data.shape[0]
    print(f"Input shape is {input_data.shape}")

    window = args.L
    stride = args.S
    time_steps = args.time_steps
    receptive_field = args.receptive_field

    quantization = bool(args.quantization)
    int8_quant = "int8" in args.q_type

    if args.mode in ("streaming-nntool-main", "streaming-nntool-full"):
        s_test = nntool_inference(
            model_nn,
            streaming_model_path,
            receptive_field=receptive_field,
            causal=True,
            time_steps=time_steps,
            full_network=full_network,
            enc_dec_model=True,
            encoder_path=encoder,
            decoder_path=enc_dec,
            int8_q=int8_quant,
            quantization=quantization,
            use_onnx_names=use_onnx_names,
        )
    else:
        s_test = onnx_inference(
            streaming_model_path,
            receptive_field=receptive_field,
            causal=True,
            time_steps=time_steps,
            full_network=full_network,
            enc_dec_model=True,
            encoder_path=encoder,
            decoder_path=enc_dec,
        )
    s_test.init_buffers()

    first_axis_pad, second_axis_pad, _ = get_pad_info(input_data, args)

    if input_data.ndim == 2:
        x_pad = np.pad(input_data, ((0, 0), (0, 0)), mode="constant", constant_values=0)
    elif input_data.ndim == 3:
        x_pad = np.pad(
            input_data,
            ((0, 0), (0, 0), (first_axis_pad, second_axis_pad)),
            mode="constant",
            constant_values=0,
        )
    else:
        x_pad = np.pad(
            input_data,
            (first_axis_pad, second_axis_pad),
            mode="constant",
            constant_values=0,
        )

    transpose_output = args.mode in ("streaming-nntool-main", "streaming-nntool-full")

    input_data_flat = np.reshape(x_pad, (x_pad.shape[-1]))
    str_output = s_test.run_audio(
        input_data_flat,
        frame_length=window,
        stride=stride,
        transpose=transpose_output,
        dim=3,
    )

    output_1 = str_output[0][
        window : original_length + window
    ]  # [stride:-(rest + stride)]
    output_2 = str_output[1][
        window : original_length + window
    ]  # [stride:-(rest + stride)]

    if full_network:
        final_output = output_1 / output_2
    else:
        final_output = output_2 / output_1

    return final_output
