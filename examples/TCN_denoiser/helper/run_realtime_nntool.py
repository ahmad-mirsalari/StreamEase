"""Real-time inference runner using NNTool models for TCN Denoiser."""

import numpy as np
from .helper_funcs import update_output
from .onnx_runner import preprocessing, post_processing


def run_realtime_model_nntool(
    noisy_data_norm: np.ndarray,
    model_nn: any,
    ort_sess_enc: any,
    ort_sess_encdec: any,
    args: any,
) -> np.ndarray:
    """
    Run real-time inference using NNTool and ONNX models with streaming input.

    Args:
        noisy_data_norm (np.ndarray): Normalized input signal.
        model_nn (any): NNTool model graph.
        ort_sess_enc (any): ONNX encoder session.
        ort_sess_encdec (any): ONNX encoder-decoder session.
        args (any): Argument namespace with runtime configurations.

    Returns:
        np.ndarray: Final processed output signal.
    """
    q_type = args.q_type
    receptive_field = args.receptive_field
    features = args.features
    time_steps = args.time_steps
    method = args.method
    stride = args.S
    frame_len = args.L
    frame_len_mts = (
        frame_len + (time_steps - 1) * stride
    )  # Frame length for multiple time steps
    quantization = args.quantization

    input_length = noisy_data_norm.shape[0]
    total_window = ((receptive_field - 1) * stride + frame_len) + (
        (time_steps - 1) * stride
    )
    num_time_steps = (len(noisy_data_norm) - frame_len) // stride + 1

    print(
        f"Input size: {noisy_data_norm.shape}, Time steps: {num_time_steps}, "
        f"Receptive field: {total_window} samples."
    )

    remainder = num_time_steps % time_steps

    if remainder != 0:
        pad_width = (time_steps - remainder) * stride
        noisy_data_norm = np.pad(noisy_data_norm, (0, pad_width), mode="constant")
        print(f"Padded input to shape: {noisy_data_norm.shape}")

    noisy_data_norm = np.pad(noisy_data_norm, (frame_len, 0), mode="constant")

    model_output = np.zeros((len(noisy_data_norm)), dtype=np.float32)

    if method == 2:
        conv_transpose_1 = np.zeros(
            (len(noisy_data_norm) + frame_len), dtype=np.float32
        )
        conv_transpose_2 = np.zeros(
            (len(noisy_data_norm) + frame_len), dtype=np.float32
        )

    num_time_steps = (len(noisy_data_norm) - frame_len) // stride + 2

    noisy_data_norm = np.pad(
        noisy_data_norm,
        (total_window - frame_len_mts, 700),
        mode="constant",
        constant_values=0,
    )
    print(
        f"Input after final padding: {noisy_data_norm.shape}, Time steps: {num_time_steps}"
    )
    print("Running model...")

    last_dim = receptive_field + time_steps - 1
    main_outputs = np.zeros((1, features, last_dim), dtype=np.float32)

    for i in range(0, num_time_steps, time_steps):
        window = noisy_data_norm[i * stride : i * stride + total_window]

        if window.shape[0] < total_window:
            rest = total_window - window.shape[0]
            window = np.pad(window, (0, rest), mode="constant")

        sqrt_output, _, inp = preprocessing(window, ort_sess_enc)

        if quantization:
            # if it is quantized to int 8, you need to transpose the input
            nntool_inp = (
                sqrt_output[0].transpose(0, 2, 1)
                if "int8" in q_type
                else sqrt_output[0]
            )
            output = model_nn.execute(nntool_inp, quantize=True, dequantize=True)
        else:
            nntool_inp = sqrt_output[0]
            output = model_nn.execute(nntool_inp, output_dict=False)

        output = output[model_nn["output_1"].step_idx][0]

        if method == 1:
            main_outputs[:, :, -time_steps:] = output
            out_dec = post_processing(
                inp,
                main_outputs[..., -(receptive_field + time_steps - 1) :],
                ort_sess_encdec,
            )[0]
            out_dec = out_dec[0]

            out_dec = np.squeeze(np.squeeze(np.transpose(out_dec, (2, 0, 1))))
            model_output[i * stride : i * stride + frame_len_mts] = out_dec[
                -frame_len_mts:
            ]
            main_outputs = np.roll(main_outputs, -time_steps, axis=-1)

        else:
            correct_inp = inp[:, :, -frame_len_mts:]

            out_dec = post_processing(correct_inp, output, ort_sess_encdec)
            out_dec = np.squeeze(out_dec)
            out_dec = np.squeeze(out_dec)
            conv_transpose_1 = update_output(
                conv_transpose_1, out_dec[0], i, frame_len_mts, stride, [0]
            )
            conv_transpose_2 = update_output(
                conv_transpose_2, out_dec[1], i, frame_len_mts, stride, [0]
            )

    print("Model run completed.")
    if method == 1:
        final_output = model_output[frame_len : input_length + frame_len]
    else:
        output_2 = conv_transpose_2[frame_len : input_length + frame_len]
        output_1 = conv_transpose_1[frame_len : input_length + frame_len]
        final_output = output_2 / output_1

    return final_output
