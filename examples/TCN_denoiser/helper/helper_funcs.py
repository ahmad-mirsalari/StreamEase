"""Helper functions for audio processing, evaluation metrics, and network output updates."""

import numpy as np
from scipy.io import wavfile
from pesq import pesq
from pystoi import stoi


def open_wav(file_path: str, expected_sr: int, verbose: bool = False) -> np.ndarray:
    """
    Read a WAV file and resample if needed.

    Args:
        file_path (str): Path to the WAV file.
        expected_sr (int): Expected sample rate.
        verbose (bool): Print resampling info if True.

    Returns:
        np.ndarray: Audio data array.
    """
    sample_rate, data = wavfile.read(file_path)
    if sample_rate != expected_sr:
        if verbose:
            print(f"Resampling from {sample_rate} to {expected_sr}")
        num_samples = round(len(data) * float(expected_sr) / sample_rate)
        data = np.interp(
            np.linspace(0, len(data), num_samples), np.arange(len(data)), data
        )
    return data


def save_wav(file_path: str, data: np.ndarray, sample_rate: int) -> None:
    """
    Save audio data to WAV file.

    Args:
        file_path (str): Output file path.
        data (np.ndarray): Audio data array.
        sample_rate (int): Sample rate.
    """
    wavfile.write(file_path, sample_rate, data)


def normalize_audio(audio: np.ndarray) -> np.ndarray:
    """
    Normalize audio signal to [-1, 1] range.

    Args:
        audio (np.ndarray): Input audio signal.

    Returns:
        np.ndarray: Normalized audio.
    """
    max_val = np.max(np.abs(audio))
    return audio / max_val if max_val > 0 else audio


def compute_stoi(
    clean_data: np.ndarray, noisy_data: np.ndarray, sample_rate: int
) -> float:
    """
    Compute STOI (Short-Time Objective Intelligibility) score.

    Args:
        clean_data (np.ndarray): Clean reference audio.
        noisy_data (np.ndarray): Noisy audio signal.
        sample_rate (int): Sample rate.

    Returns:
        float: STOI score.
    """
    min_length = min(len(clean_data), len(noisy_data))
    clean_data = clean_data[:min_length].astype(np.float32)
    noisy_data = noisy_data[:min_length].astype(np.float32)

    return stoi(clean_data, noisy_data, sample_rate)


def compute_pesq(
    clean_data: np.ndarray, noisy_data: np.ndarray, sample_rate: int
) -> float:
    """
    Compute PESQ (Perceptual Evaluation of Speech Quality) score.

    Args:
        clean_data (np.ndarray): Clean reference audio.
        noisy_data (np.ndarray): Noisy audio signal.
        sample_rate (int): Sample rate.

    Returns:
        float: PESQ score.
    """
    min_length = min(len(clean_data), len(noisy_data))
    clean_data = clean_data[:min_length].astype(np.float32)
    noisy_data = noisy_data[:min_length].astype(np.float32)

    return pesq(sample_rate, clean_data, noisy_data, "nb")


def update_output(
    net_output: np.ndarray,
    current_output: np.ndarray,
    index: int,
    frame_length: int,
    stride: int,
    bias: float,
) -> np.ndarray:
    """
    Update the final output with the current network output using overlap-add.

    Args:
        net_output (np.ndarray): Full output buffer.
        current_output (np.ndarray): Current frame output.
        index (int): Time step index.
        frame_length (int): Frame size (kernel size).
        stride (int): Stride or hop size.
        bias (float): Bias value to subtract for overlapping regions.

    Returns:
        np.ndarray: Updated output buffer.
    """

    start = index * stride
    end = start + frame_length
    if index == 0:
        net_output[start:end] = current_output[:frame_length]
    else:
        common_part = min(frame_length - stride, len(net_output) - start)
        net_output[start : start + common_part] = (
            net_output[start : start + common_part]
            + current_output[:common_part]
            - bias
        )
        net_output[start + common_part : end] = current_output[common_part:frame_length]

    return net_output
