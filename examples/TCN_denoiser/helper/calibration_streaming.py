"""Streaming calibration data reader for buffer-aware quantization in TCN Denoiser."""

import os
import sys
import numpy as np
import onnxruntime as ort
from scipy.io import wavfile

utils_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, utils_path)

from utils.onnx_inference.helper_funcs import (
    create_buffers,
    extract_number,
    get_relevant_buffer,
    update_buffers,
)
from utils.nntool_inference.helper_funcs import (
    prepare_input_list,
    extract_outputs_list,
)


class MyStreamingCalibrationDataReader:
    """
    Data reader for streaming calibration with buffer management and ONNX inference.
    """

    def __init__(
        self,
        noisy_dataset_path: str,
        model_nn: any,
        chunk_size: int = 19000,
        max_index: int = 2000,
        use_onnx_names: bool = False,
        args: any = None,
    ):
        """
        Initialize calibration reader for streaming models.

        Args:
            noisy_dataset_path (str): Path to noisy dataset.
            model_nn (any): NNTool model for streaming execution.
            chunk_size (int): Size of audio chunks.
            max_index (int): Max number of chunks to yield.
            use_onnx_names (bool): ONNX naming flag for compatibility.
            args (any): Argument namespace with configs.
        """

        self.current_buffers = {}
        self.kernel_sizes = {}
        self.buffer_index = {}
        self.dilations_rates = {}
        self.conv_lengths = None

        self.streaming_model_path = args.streaming_path
        self.receptive_field = args.receptive_field

        self.noisy_dataset_path = noisy_dataset_path
        self.chunk_size = chunk_size
        self.stride_size = args.S

        self.model_nn = model_nn

        self.use_onnx_names = use_onnx_names
        self.files = os.listdir(noisy_dataset_path)
        self.counter = 0
        self.current_file_counter = 0
        self.max_index = max_index  # Limit the number of chunks to process
        self.file_index = 0  # Track which file we are on
        self.chunk_index = 0  # Track where we are in the current file
        self.time_steps = args.time_steps
        self.sample_rate = args.samplerate
        model_path_enc = "models/streaming-model/enc_convtasnet_stft_session_2_str.onnx"  # Adjust this to match your model's path
        self.ort_sess = ort.InferenceSession(model_path_enc)

        self.input_name = self.ort_sess.get_inputs()[
            0
        ].name  # Adjust this to match your model's input name
        self.init_buffers()

    def init_buffers(self) -> None:
        """Initialize streaming model buffers."""
        (
            self.current_buffers,
            self.kernel_sizes,
            self.dilations_rates,
            self.conv_lengths,
            self.buffer_index,
        ) = create_buffers(self.streaming_model_path, self.time_steps)
        print("Buffers are initialized")

    def zero_buffers(self) -> None:
        """Zero out the buffers for the next iteration."""
        for key, value in self.current_buffers.items():
            if "Expand" in key:
                min_value = extract_number(key)
                for i in range(self.time_steps):
                    self.current_buffers[key][:, i] = min_value * (i + 1)
            else:
                self.current_buffers[key] = np.zeros_like(value)

    def get_next(self) -> np.ndarray | None:
        """
        Get next processed chunk with updated buffers.

        Returns:
            np.ndarray or None: Prepared model input, or None if done.
        """
        if self.counter >= self.max_index or self.file_index >= len(self.files):
            return None

        noisy_file = os.path.join(self.noisy_dataset_path, self.files[self.file_index])
        noisy_data = self._open_wav(
            noisy_file, expected_sr=self.sample_rate, verbose=False
        )
        noisy_data = self._normalize_audio(noisy_data)
        noisy_data = np.pad(noisy_data, (400, 0), mode="constant", constant_values=0)

        if self.chunk_index >= len(noisy_data) - self.stride_size:
            # Move to the next file if we've exhausted the current one
            self.file_index += 1
            self.chunk_index = 0
            self.current_file_counter = 0
            return self.get_next()

        print(
            f"Counter: {self.counter}, File: {self.file_index}, Chunk: {self.chunk_index}"
        )
        # Get the chunk starting at the current chunk_index
        chunk = self._get_chunk(noisy_data)
        processed_chunk = self._preprocess(chunk)

        # Move the chunk_index by the stride_size
        self.chunk_index += self.stride_size
        self.counter += 1

        # here we need to read the buffers
        r_buffers = get_relevant_buffer(
            self.current_buffers,
            self.kernel_sizes,
            self.dilations_rates,
            self.conv_lengths,
            self.time_steps,
        )
        test_sequence = processed_chunk[0]

        correct_input_data = prepare_input_list(r_buffers, test_sequence)

        # Next step is updating the buffers for the next iteration
        prediction = self.model_nn.execute(correct_input_data, output_dict=False)

        outputs = extract_outputs_list(
            prediction, self.model_nn, self.buffer_index, self.use_onnx_names
        )

        transpose = True
        num_speaker = 1
        if transpose:
            for o_net in range(num_speaker, len(outputs)):
                shape = outputs[o_net].shape
                # print(f"shape{o_net} :{shape}")
                if len(shape) == 1:
                    # Shape is (128,) - Reshape to (1, 128, 1)
                    outputs[o_net] = np.reshape(outputs[o_net], (1, shape[0], 1))
                if len(shape) == 2:
                    if shape[1] == self.time_steps:
                        # Shape is (128, 2), reshape to (1, 128, timestep)
                        outputs[o_net] = np.reshape(
                            outputs[o_net], (1, shape[0], self.time_steps)
                        )
                    else:
                        # Shape is (2, 128), reshape to (1, 128, timestep)
                        outputs[o_net] = np.reshape(
                            outputs[o_net], (1, shape[1], self.time_steps)
                        )

                elif len(shape) == 3:
                    if shape[-1] == self.time_steps:
                        # Timestep is already the last dimension but may need reshaping
                        # Check if the first dimension is 1, if not, adjust
                        if shape[0] == 1:
                            # Shape is (1, X, timestep), no reshaping needed
                            continue
                        else:
                            # Shape is (Y, 1, timestep), reshape to (1, Y, timestep)
                            outputs[o_net] = np.reshape(
                                outputs[o_net], (1, shape[0], self.time_steps)
                            )
                    elif shape[1] == self.time_steps:
                        if shape[0] == 1:
                            # Timestep is in the middle, e.g., (1, timestep, X)
                            outputs[o_net] = np.reshape(
                                outputs[o_net], (1, shape[2], self.time_steps)
                            )
                        else:
                            # Timestep is in the middle, e.g., (X, timestep, 1) then reshape to (1, X, timestep)
                            outputs[o_net] = np.reshape(
                                outputs[o_net], (1, shape[0], self.time_steps)
                            )

                    elif shape[0] == self.time_steps:
                        if shape[1] == 1:
                            # Timestep is in the middle, e.g., (timestep, 1, X)
                            outputs[o_net] = np.reshape(
                                outputs[o_net], (1, shape[2], self.time_steps)
                            )
                        else:
                            # Timestep is in the middle, e.g., (timestep, X, 1) then reshape to (1, X, timestep)
                            outputs[o_net] = np.reshape(
                                outputs[o_net], (1, shape[1], self.time_steps)
                            )

                # print(f"Output {o_net}: {outputs[o_net].shape}")

        if self.current_file_counter + self.time_steps > self.receptive_field:
            reset = True
            self.current_file_counter = 0
        else:
            reset = False
            self.current_file_counter += self.time_steps

        self.current_buffers = update_buffers(
            self.current_buffers,
            outputs,
            test_sequence,
            self.conv_lengths,
            self.buffer_index,
            self.time_steps,
            self.receptive_field,
            reset,
        )

        return correct_input_data

    def _open_wav(
        self, file_path: str, expected_sr: int = 16000, verbose: bool = False
    ):
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
        """Normalize audio to [-1, 1] range."""
        max_val = np.max(np.abs(audio))
        if max_val > 0:
            return audio / max_val
        return audio

    def _get_chunk(self, data: np.ndarray) -> np.ndarray:
        """Extract and pad audio chunk."""
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
        """Run encoder preprocessing."""
        chunk = chunk.astype(np.float32)
        if chunk.ndim == 1:
            chunk = np.expand_dims(chunk, axis=(0, 1))
        return self.ort_sess.run(None, {self.input_name: chunk})

    def get_calibration_data(self):
        """Generator yielding calibration chunks."""
        while self.counter < self.max_index:
            data = self.get_next()
            if data is None:
                break
            yield data
