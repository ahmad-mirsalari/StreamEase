"""Real-time inference runner for TCN Denoiser with ONNX model components."""

import time
import numpy as np
from .onnx_runner import preprocessing, run_main_model, post_processing
from .helper_funcs import update_output


def run_realtime_model(
    noisy_data_norm: np.ndarray,
    ort_sess_enc: any,
    ort_sess_main: any,
    ort_sess_encdec: any,
    args: any,
) -> np.ndarray:
    """
    Run real-time inference using separated encoder, main, and decoder ONNX components.

    Args:
        noisy_data_norm (np.ndarray): Normalized input signal.
        ort_sess_enc (any): ONNX encoder session.
        ort_sess_main (any): ONNX main session.
        ort_sess_encdec (any): ONNX encoder-decoder session.
        args (any): Argument namespace with runtime parameters.

    Returns:
        np.ndarray: Final processed output signal.
    """

    receptive_field = args.receptive_field
    features = args.features
    time_steps = args.time_steps
    method = args.method
    stride = args.S
    frame_len = args.L
    frame_mts = frame_len + (time_steps - 1) * stride  # Multi time steps frame length

    input_length = noisy_data_norm.shape[0]
    total_window = ((receptive_field - 1) * stride + frame_len) + (
        (time_steps - 1) * stride
    )
    num_time_steps = (len(noisy_data_norm) - frame_len) // stride + 1

    remainder = num_time_steps % time_steps
    if remainder != 0:
        pad_width = (time_steps - remainder) * stride
        noisy_data_norm = np.pad(noisy_data_norm, ((0, pad_width)), mode="constant")

    noisy_data_norm = np.pad(
        noisy_data_norm, (frame_len, 0), mode="constant", constant_values=0
    )

    num_time_steps = (len(noisy_data_norm) - frame_len) // stride + 2

    model_output = np.zeros((len(noisy_data_norm)), dtype=np.float32)
    if method == 2:
        conv_transpose1 = np.zeros((len(noisy_data_norm) + frame_len), dtype=np.float32)
        conv_transpose2 = np.zeros((len(noisy_data_norm) + frame_len), dtype=np.float32)

    noisy_data_norm = np.pad(
        noisy_data_norm,
        (total_window - frame_mts, 700),
        mode="constant",
        constant_values=0,
    )

    print(
        f"Input size after padding: {noisy_data_norm.shape}, Time steps: {num_time_steps}"
    )
    print("Please wait, the model is running...")

    last_dim = receptive_field + time_steps - 1
    main_outputs = np.zeros((1, features, last_dim), dtype=np.float32)

    total_time = 0.0
    for i in range(0, num_time_steps, time_steps):
        window = noisy_data_norm[i * stride : i * stride + total_window]

        if window.shape[0] < total_window:
            rest = total_window - window.shape[0]
            window = np.pad(window, (0, rest), mode="constant", constant_values=0)

        sqrt_output, _, inp = preprocessing(window, ort_sess_enc)

        # Capture the start time
        start_time = time.time()

        output = run_main_model(sqrt_output, ort_sess_main)

        # Calculate the time taken for this iteration
        iteration_time = time.time() - start_time

        # Accumulate the total time
        total_time += iteration_time

        # Update the last #time_steps elements of the buffer with the output
        main_outputs[:, :, -time_steps:] = output

        if method == 1:
            out_dec = post_processing(
                inp,
                main_outputs[..., -(receptive_field + time_steps - 1) :],
                ort_sess_encdec,
            )
            out_dec = np.squeeze(np.squeeze(np.transpose(out_dec[0], (2, 0, 1))))
            model_output[i * stride : i * stride + frame_mts] = out_dec[-frame_mts:]
            # Shift main_outputs to the left
            main_outputs = np.roll(main_outputs, -time_steps, axis=-1)

        else:
            correct_inp = inp[:, :, -frame_mts:]
            out_dec = post_processing(correct_inp, output, ort_sess_encdec)

            out_dec = np.squeeze(out_dec)
            out_dec = np.squeeze(out_dec)
            conv_transpose1 = update_output(
                conv_transpose1, out_dec[0], i, frame_mts, stride, [0]
            )
            conv_transpose2 = update_output(
                conv_transpose2, out_dec[1], i, frame_mts, stride, [0]
            )

    if method == 1:
        f_model_output = model_output[frame_len : input_length + frame_len]
    else:
        conv_transpose_2 = conv_transpose2[frame_len : input_length + frame_len]
        conv_transpose_1 = conv_transpose1[frame_len : input_length + frame_len]
        f_model_output = conv_transpose_2 / conv_transpose_1
    # Calculate the average time per iteration
    avg_time = total_time / num_time_steps
    print(
        f"Total time: {total_time:.3f} s, Iterations: {num_time_steps}, "
        f"Avg time per iteration: {avg_time:.5f} s"
    )

    return f_model_output
