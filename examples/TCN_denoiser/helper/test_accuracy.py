"""Accuracy evaluation for TCN Denoiser models with PESQ and STOI metrics."""

import os
import numpy as np
import pandas as pd
from .helper_funcs import (
    open_wav,
    normalize_audio,
    compute_pesq,
    compute_stoi,
    save_wav,
)
from .run_realtime import run_realtime_model
from .run_realtime_nntool import run_realtime_model_nntool
from .run_trained_model import run_onnx_model
from .model_loaders import model_loader
from .run_streaming import run_streaming_model


def load_networks(args):
    """Load encoder, main, and encoder-decoder ONNX sessions"""
    return model_loader(args)


def summarize_metrics(metric_list):
    """Compute average PESQ and STOI."""
    pesq_total = sum(m[1] for m in metric_list)
    stoi_total = sum(m[2] for m in metric_list)
    count = len(metric_list)
    return pesq_total / count, stoi_total / count


def test_accuracy(model_nn, use_onnx_names, args):
    """
    Evaluate model performance on test dataset.

    Supports real-time, streaming, and trained model modes. Calculates PESQ and STOI metrics.
    """
    noisy_dataset = args.noisy_dataset
    clean_dataset = args.clean_dataset
    sample_rate = args.samplerate

    # paths
    trained_model_path = args.trained_model_path

    files = os.listdir(noisy_dataset)
    metric = []

    if args.mode in ("streaming-onnx-full", "trained"):
        ort_sess_enc = None
        ort_sess_main = None
        ort_sess_encdec = None
    else:
        ort_sess_enc, ort_sess_main, ort_sess_encdec = load_networks(args)

    for filename in files:

        noisy_file = os.path.join(noisy_dataset, filename)
        noisy_data = open_wav(noisy_file, expected_sr=sample_rate, verbose=False)

        clean_filename = filename

        clean_file = os.path.join(clean_dataset, clean_filename)
        print("*" * 40)
        print("Processing file", filename, "clean file", clean_file)
        print("*" * 40)

        clean_data = open_wav(clean_file, expected_sr=sample_rate, verbose=False)
        clean_data = clean_data.astype(np.float32)
        noisy_data = noisy_data.astype(np.float32)
        noisy_data_norm = normalize_audio(noisy_data)

        if args.mode == "trained":
            model_output = run_onnx_model(trained_model_path, noisy_data_norm)

        elif args.mode == "real-time-nntool":
            model_output = run_realtime_model_nntool(
                noisy_data_norm, model_nn, ort_sess_enc, ort_sess_encdec, args
            )

        elif args.mode in (
            "streaming-onnx-full",
            "streaming-onnx-main",
            "streaming-nntool-main",
        ):
            model_output = run_streaming_model(
                model_nn,
                args.streaming_path,
                noisy_data_norm,
                ort_sess_enc,
                ort_sess_encdec,
                use_onnx_names,
                args,
            )
        else:
            model_output = run_realtime_model(
                noisy_data_norm, ort_sess_enc, ort_sess_main, ort_sess_encdec, args
            )

        # Save model output as WAV file
        os.makedirs("output_wav_files", exist_ok=True)
        save_wav(
            os.path.join("output_wav_files", filename), model_output, sample_rate=sample_rate
        )

        # Check if any NaN values are present in model_output or clean_data
        if np.isnan(model_output).any() or np.isnan(clean_data).any():
            print("Model output or clean data contains NaN values")
            continue  # Skip further processing or handle the NaN values

        pesq_val = compute_pesq(clean_data, model_output, sample_rate)  # Compute PESQ
        stoi_val = compute_stoi(clean_data, model_output, sample_rate)  # Compute STOI
        metric.append([filename, pesq_val, stoi_val])
        print("PESQ=", pesq_val, "STOI=", stoi_val)

    final_pesq, final_stoi = summarize_metrics(metric)
    
    print("*" * 40)
    print(
        "Test set performance: PESQ=",
        final_pesq,
        "STOI=",
        final_stoi,
        "over",
        len(metric),
        "samples",
    )
    metric.append(["average", final_pesq, final_stoi])
    metric_df = pd.DataFrame(metric, columns=["filename", "PESQ", "STOI"])
    print(metric_df)
