"""Main entry point for TCN Denoiser model testing with real-time and streaming options."""

from __future__ import print_function
import os
import argparse

from helper.test_accuracy import test_accuracy
from helper.nntool_graph import nntool_load_graph
from helper.quantization import nntool_quantization
from helper.run_on_gap import run_on_gap
from helper.streaming import get_streaming_model

if __name__ == "__main__":

    # Mode options:
    # trained - test the trained model using onnxruntime
    # real-time-onnx - test the real-time model using onnxruntime
    # real-time-nntool - test the real-time model using nntool
    # streaming-onnx-full - full network in streaming mode using onnxruntime
    # streaming-onnx-main - main part of the network in streaming mode using onnxruntime
    # streaming-nntool-main - main part of the network in streaming mode using nntool because
    #                     nntool doesn't support some nodes of the encoder-decoder model

    MODE = "real-time-nntool"  # Execution mode
    USE_ONNX_NAMES = False  # True: if you want to use the onnx names in nntool
    QUANTIZATION = False  # True: if you want to quantize the model using nntool

    if QUANTIZATION and ("trained" in MODE or "onnx" in MODE):
        raise ValueError("Quantization is only available for nntool modes.")

    QUANTIZATION_MODE = "streaming" if "streaming" in MODE else "real-time"

    Q_TYPE = "int8_f32_m"  # Quantization type

    # Quantization types:
    # Fixed:
    #   int8        - Quantize to int8
    #   bfloat16    - Quantize to bfloat16 (expr nodes in float32 due to nntool limitation)
    #   float16     - Quantize to float16
    # Mixed:
    #   int8_bf16_m - Conv layers to int8, others to bfloat16
    #   int8_fp16_m - Conv layers to int8, others to float16
    #   int8_f32_m  - Conv layers to int8, others to float32
    #   bf16_f32_m  - Conv layers to bfloat16, others to float32
    #   fp16_f32_m  - Conv layers to float16, others to float32
    #   fp16_bf16_m - Conv layers to float16, others to bfloat16

    CHECK_ON_GAP = False  # True: if you want to run the model on gap
    CHECK_ACCURACY = True  # True: if you want to test the model accuracy
    TIME_STEPS = 3

    if CHECK_ON_GAP:
        QUANTIZATION = True

    #####################################

    METHOD = 2
    # 1: append the previous main outputs to the current output then run the decoder
    # 2: use the current main network output as the input to the decoder network

    RECEPTIVE_FIELD = 187  # Receptive field of the model
    FEATURES = 257  # Number of features in the input data
    L = 400  # Number of samples in the input data
    S = 100  # Stride of the model

    NOISY_DATASET = "dataset/dataset-test10/noisy"
    CLEAN_DATASET = "dataset/dataset-test10/clean"
    CAL_DATA_PATH = "dataset/dataset-cal/"
    SAMPLERATE = 16000

    # set all the paths
    """
    basically we need one path for encoder, one for main and one for encoder decoder
    """
    STREAMING_FOLDER = "models/streaming-model"
    FILE_NAME = "torch_streaming_model.onnx"

    REALTIME_MAIN_PATH = f"models/realtime-model/real-time-main-convtasnet_stft_session_2_{TIME_STEPS}.onnx"
    ONNX_MODEL_PATH = "models/trained-model/convtasnet_stft_session_2.onnx"
    MODEL_ENCODER_PATH = (
        f"models/realtime-model/enc_convtasnet_stft_session_2_{TIME_STEPS}.onnx"
    )

    if METHOD == 1:
        MODEL_ENCDEC_PATH = (
            f"models/realtime-model/encdec_convtasnet_stft_session_2_{TIME_STEPS}.onnx"
        )
    else:
        MODEL_ENCDEC_PATH = (
            f"models/realtime-model/encdec1_convtasnet_stft_session_2_{TIME_STEPS}.onnx"
        )

    ONNX_ENCODER_PART_PATH = "models/realtime-model/enc_convtasnet_stft_session_2.onnx"
    ONNX_MAIN_PART_PATH = "models/realtime-model/main-convtasnet_stft_session_2.onnx"
    ONNX_ENCDEC_PART_PATH = (
        "models/realtime-model/encdec1_convtasnet_stft_session_2.onnx"
    )

    STREAMING_PATH = os.path.join(STREAMING_FOLDER, FILE_NAME)

    parser = argparse.ArgumentParser(description="TCN Denoiser")

    parser.add_argument(
        "--receptive_field",
        type=int,
        default=RECEPTIVE_FIELD,
        help="Receptive field of the model",
    )
    parser.add_argument(
        "--features",
        type=int,
        default=FEATURES,
        help="Number of features",
    )
    parser.add_argument(
        "--L",
        type=int,
        default=L,
        help="Frame length (number of samples)",
    )
    parser.add_argument(
        "--S",
        type=int,
        default=S,
        help="Stride",
    )
    parser.add_argument(
        "--time_steps",
        type=int,
        default=TIME_STEPS,
        help="Number of time steps",
    )
    parser.add_argument(
        "--method",
        type=int,
        default=METHOD,
        help="Method: 1 for appended output, 2 for direct decoder input",
    )
    parser.add_argument(
        "--noisy_dataset",
        type=str,
        default=NOISY_DATASET,
        help="Path to noisy dataset",
    )
    parser.add_argument(
        "--clean_dataset",
        type=str,
        default=CLEAN_DATASET,
        help="Path to clean dataset",
    )
    parser.add_argument(
        "--mode",
        type=str,
        default=MODE,
        help="Execution mode",
    )
    parser.add_argument(
        "--quantization",
        type=bool,
        default=QUANTIZATION,
        help="Enable quantization (nntool only)",
    )
    parser.add_argument(
        "--samplerate",
        type=int,
        default=SAMPLERATE,
        help="Sample rate",
    )
    parser.add_argument(
        "--realtime_main_path",
        type=str,
        default=REALTIME_MAIN_PATH,
        help="Path to real-time main ONNX model",
    )
    parser.add_argument(
        "--model_encoder_path",
        type=str,
        default=MODEL_ENCODER_PATH,
        help="Path to encoder ONNX model",
    )
    parser.add_argument(
        "--model_encdec_path",
        type=str,
        default=MODEL_ENCDEC_PATH,
        help="Path to encoder-decoder ONNX model",
    )
    parser.add_argument(
        "--trained_model_path",
        type=str,
        default=ONNX_MODEL_PATH,
        help="Path to trained ONNX model",
    )
    parser.add_argument(
        "--onnx_encoder_part_path",
        type=str,
        default=ONNX_ENCODER_PART_PATH,
        help="Path to ONNX encoder part",
    )
    parser.add_argument(
        "--onnx_main_part_path",
        type=str,
        default=ONNX_MAIN_PART_PATH,
        help="Path to ONNX main part",
    )
    parser.add_argument(
        "--onnx_encdec_part_path",
        type=str,
        default=ONNX_ENCDEC_PART_PATH,
        help="Path to ONNX encoder-decoder part",
    )
    parser.add_argument(
        "--cal_data_path",
        type=str,
        default=CAL_DATA_PATH,
        help="Path to calibration dataset",
    )
    parser.add_argument(
        "--quantization_mode",
        type=str,
        default=QUANTIZATION_MODE,
        help="Quantization mode: 'real-time' or 'streaming'",
    )
    parser.add_argument(
        "--q_type",
        type=str,
        default=Q_TYPE,
        help="Quantization type (int8, float16, mixed, etc.)",
    )
    parser.add_argument(
        "--streaming_path",
        type=str,
        default=STREAMING_PATH,
        help="Path to streaming ONNX model",
    )

    args = parser.parse_args()

    STREAMING_MODEL = None
    MODEL_NN = None
    # Load model graphs based on execution mode
    if args.mode == "real-time-nntool":
        MODEL_NN = nntool_load_graph(
            args.realtime_main_path, use_onnx_names=USE_ONNX_NAMES
        )
        STREAMING_MODEL = None
    elif args.mode == "streaming-onnx-full":
        STREAMING_MODEL = get_streaming_model(
            args.trained_model_path, STREAMING_FOLDER, args.time_steps
        )
        MODEL_NN = None
    elif args.mode == "streaming-nntool-main":
        STREAMING_MODEL = get_streaming_model(
            args.onnx_main_part_path, STREAMING_FOLDER, args.time_steps
        )
        MODEL_NN = nntool_load_graph(args.streaming_path, use_onnx_names=USE_ONNX_NAMES)
    elif args.mode == "streaming-onnx-main":
        STREAMING_MODEL = get_streaming_model(
            args.onnx_main_part_path, STREAMING_FOLDER, args.time_steps
        )
        MODEL_NN = None
    else:
        STREAMING_MODEL = None
        MODEL_NN = None

    # Quantization and evaluation logic
    if MODEL_NN is not None:
        if CHECK_ON_GAP:
            MODEL_NN = nntool_quantization(
                MODEL_NN, use_onnx_names=USE_ONNX_NAMES, args=args
            )
            result = run_on_gap(MODEL_NN)
            print(result.print_basic_mem_infos())
            result.plot_memory_boxes("L2")

        if args.quantization:
            model_nn = nntool_quantization(
                MODEL_NN, use_onnx_names=USE_ONNX_NAMES, args=args
            )

        if CHECK_ACCURACY:
            test_accuracy(MODEL_NN, use_onnx_names=USE_ONNX_NAMES, args=args)
    else:
        if CHECK_ACCURACY:
            test_accuracy(STREAMING_MODEL, use_onnx_names=USE_ONNX_NAMES, args=args)

    print("Done")
