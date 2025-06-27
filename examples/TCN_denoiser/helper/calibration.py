"""Real-time calibration data reader for collecting quantization statistics."""

import os
import numpy as np
import onnxruntime as ort
from scipy.io import wavfile


class MyCalibrationDataReader:

    """
    Data reader for streaming-mode quantization calibration using ONNX encoder output.
    """

    def __init__(
        self,
        cal_dataset_path: str,
        chunk_size: int = 19000,
        max_index: int = 1000,
        args: any = None,
    ):
        """
        Initialize the streaming calibration reader.

        Args:
            cal_dataset_path (str): Path to calibration audio dataset.
            model_nn (any): Unused placeholder for NNTool compatibility.
            chunk_size (int): Length of extracted audio chunks.
            max_index (int): Maximum number of chunks to yield.
            use_onnx_names (bool): Placeholder flag for interface consistency.
            args (any): Argument namespace with model paths and configurations.
        """
        self.cal_dataset_path = cal_dataset_path
        self.chunk_size = chunk_size
        self.stride_size = args.S
        self.files = os.listdir(cal_dataset_path)
        self.counter = 0
        self.max_index = max_index  # Limit the number of chunks to process
        self.file_index = 0  # Track which file we are on
        self.chunk_index = 0  # Track where we are in the current file

        self.sample_rate = args.samplerate
        model_path_enc = args.model_encoder_path
        self.ort_sess = ort.InferenceSession(model_path_enc)
        self.input_name = self.ort_sess.get_inputs()[
            0
        ].name  # Adjust this to match your model's input name

    def get_next(self) -> np.ndarray | None:
        """
        Retrieve the next preprocessed chunk of audio.

        Returns:
            np.ndarray or None: Processed audio chunk, or None if finished.
        """

        if self.counter >= self.max_index or self.file_index >= len(self.files):
            return None

        noisy_file = os.path.join(self.cal_dataset_path, self.files[self.file_index])

        noisy_data = self._open_wav(
            noisy_file, expected_sr=self.sample_rate, verbose=False
        )
        noisy_data = self._normalize_audio(noisy_data)

        if self.chunk_index >= len(noisy_data) // 8:
            # Move to the next file if we've exhausted the current one
            self.file_index += 1
            self.chunk_index = 0
            return self.get_next()

        # Get the chunk starting at the current chunk_index
        chunk = self._get_chunk(noisy_data)
        processed_chunk = self._preprocess(chunk)

        # Move the chunk_index by the stride_size
        self.chunk_index += self.stride_size
        self.counter += 1

        return processed_chunk[0]

    def _open_wav(
        self, file_path: str, expected_sr: int = 16000, verbose: bool = False
    ) -> np.ndarray:
        """
        Read and optionally resample an audio file.

        Args:
            file_path (str): Path to the WAV file.
            expected_sr (int): Expected sample rate.

        Returns:
            np.ndarray: Audio data array.
        """
        sr, data = wavfile.read(file_path)
        if sr != expected_sr:
            if verbose:
                print(f"Resampling from {sr} to {expected_sr}")
            number_of_samples = round(len(data) * float(expected_sr) / sr)
            data = np.interp(
                np.linspace(0, len(data), number_of_samples), np.arange(len(data)), data
            )
        return data

    def _normalize_audio(self, audio: np.ndarray) -> np.ndarray:
        """
        Normalize audio signal to [-1, 1] range.

        Args:
            audio (np.ndarray): Input audio.

        Returns:
            np.ndarray: Normalized audio.
        """
        max_val = np.max(np.abs(audio))
        if max_val > 0:
            return audio / max_val
        return audio

    def _get_chunk(self, data: np.ndarray) -> np.ndarray:
        """
        Extract a fixed-size audio chunk with padding if needed.

        Args:
            data (np.ndarray): Full audio signal.

        Returns:
            np.ndarray: Audio chunk.
        """
        start = self.chunk_index
        end = start + self.chunk_size
        if end > len(data):
            end = len(data)
            start = max(0, end - self.chunk_size)

        chunk = data[start:end]

        # If the chunk is smaller than chunk_size, pad it
        if len(chunk) < self.chunk_size:
            chunk = np.pad(
                chunk,
                (0, self.chunk_size - len(chunk)),
                mode="constant",
                constant_values=0,
            )

        return chunk

    def _preprocess(self, chunk: np.ndarray) -> np.ndarray:
        """
        Preprocess a chunk through the ONNX encoder.

        Args:
            chunk (np.ndarray): Raw audio chunk.

        Returns:
            np.ndarray: Encoder output.
        """
        chunk = chunk.astype(np.float32)
        if chunk.ndim == 1:
            chunk = np.expand_dims(chunk, axis=(0, 1))
        return self.ort_sess.run(None, {self.input_name: chunk})

    def get_calibration_data(self):
        """
        Generator yielding calibration data for quantization statistics collection.

        Yields:
            np.ndarray: Processed audio chunk.
        """
        while self.counter < self.max_index:
            data = self.get_next()
            if data is None:
                break
            yield data
