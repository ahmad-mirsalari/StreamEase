"""
This module provides the Inference class for performing streaming inference using ONNX models.
"""

import numpy as np
import onnxruntime as ort
from .helper_funcs import (
    create_buffers,
    get_relevant_buffer,
    prepare_input_dictionary,
    read_input_name,
    update_buffers,
    get_conv_transpose_bias,
    update_output,
)


class Inference:
    """
    Initializes the inference class with model parameters and paths.

    Args:
        streaming_model_path (str): Path to the streaming model.
        time_steps (int): Time steps for streaming.
        receptive_field (int): Receptive field size.
        causal (bool): If True, model operates in causal mode.
        full_network (bool): If True, loads the full network.
        enc_dec_model (bool): If True, specifies the encoder-decoder model.
        encoder_path (str): Path to the encoder model.
        decoder_path (str): Path to the decoder model.
    """

    def __init__(
        self,
        streaming_model_path: str,
        time_steps: int = 1,
        receptive_field: int = 15,
        causal: bool = True,
        full_network: bool = True,
        enc_dec_model: bool = False,
        encoder_path: str = None,
        decoder_path: str = None,
    ) -> None:

        self.current_buffers: dict = {}
        self.kernel_sizes: dict = {}
        self.buffer_index: dict = {}
        self.dilations_rates: dict = {}
        self.conv_lengths = None
        self.streaming_model_path = streaming_model_path
        self.receptive_field = receptive_field
        self.causal = causal
        self.time_steps = time_steps
        self.full_network = full_network

        if not full_network:
            if enc_dec_model:
                self.ort_sess_enc = encoder_path
                self.ort_sess_encdec = decoder_path
            else:
                self.load_encoder_decoder(encoder_path, decoder_path)

    def init_buffers(self) -> None:
        """
        Initializes the buffers by creating them based on the streaming model.
        """
        (
            self.current_buffers,
            self.kernel_sizes,
            self.dilations_rates,
            self.conv_lengths,
            self.buffer_index,
        ) = create_buffers(self.streaming_model_path, self.time_steps)

    def zero_buffers(self) -> None:
        """
        Resets the buffers to zero, except for buffers related to "Expand".
        """

        for key, value in self.current_buffers.items():
            if "Expand" not in key:
                self.current_buffers[key] = np.zeros_like(value)

    def load_encoder_decoder(self, encoder_path: str, decoder_path: str) -> None:
        """
        Loads the encoder and decoder models if not using the full network.

        Args:
            encoder_path (str): Path to the encoder model.
            decoder_path (str): Path to the decoder model.
        """

        self.ort_sess_enc = ort.InferenceSession(encoder_path)
        self.ort_sess_encdec = ort.InferenceSession(decoder_path)

    def post_processing(
        self, input_data: np.ndarray, main_outputs: np.ndarray, len_input: int
    ) -> np.ndarray:
        """
        Post-processes the output from the model using the decoder.

        Args:
            input_data (np.ndarray): Input data for processing.
            main_outputs (np.ndarray): Main outputs from the encoder.
            len_input (int): Length of the input data.

        Returns:
            np.ndarray: Final processed output from the decoder.
        """

        if len(input_data.shape) == 1:
            input_data = np.expand_dims(np.expand_dims(input_data, axis=0), axis=0)

        # Run the model
        input_name1 = self.ort_sess_encdec.get_inputs()[0].name
        input_name2 = self.ort_sess_encdec.get_inputs()[1].name
        final_output_model = self.ort_sess_encdec.run(
            None, {input_name2: input_data, input_name1: main_outputs}
        )
        return final_output_model

    def preprocessing(self, x: np.ndarray) -> tuple[np.ndarray, int, np.ndarray]:
        """
        Preprocesses the input data using the encoder.

        Args:
            x (np.ndarray): Input data.

        Returns:
            Tuple[np.ndarray, int, np.ndarray]: Processed output, input length, and input data.
        """
        input_data = x
        len_input = input_data.shape[0]
        if len(input_data.shape) == 1:
            input_data = np.expand_dims(np.expand_dims(input_data, axis=0), axis=0)

        # Run the model
        input_name = self.ort_sess_enc.get_inputs()[0].name
        sqrt_output = self.ort_sess_enc.run(None, {input_name: input_data})
        return sqrt_output[0], len_input, input_data

    def run(self, input_data: np.ndarray, transpose: bool = False) -> np.ndarray:
        """
        Runs inference on the input data.

        Args:
            input_data (np.ndarray): Input data for streaming inference.
            transpose (bool): Whether to transpose the input data.

        Returns:
            np.ndarray: Decoded samples array containing the streaming output.
        """

        input_name = read_input_name(self.streaming_model_path)
        sess = ort.InferenceSession(self.streaming_model_path)

        # Check if the number of rows is a multiple of the time steps
        remainder = input_data.shape[1] % self.time_steps
        if remainder != 0:
            pad_width = self.time_steps - remainder
            input_data = np.pad(
                input_data, ((0, 0), (0, pad_width), (0, 0)), mode="constant"
            )
        if self.causal:
            decoded_samples_array = np.zeros(
                (input_data.shape[0], input_data.shape[1], input_data.shape[2])
            )
        else:
            decoded_samples_array = np.zeros(
                (
                    input_data.shape[0],
                    input_data.shape[1] - self.receptive_field + self.time_steps,
                    input_data.shape[2],
                )
            )

        self.zero_buffers()

        for idx, sample in enumerate(input_data):
            accumulated_rows = np.zeros((self.time_steps, input_data.shape[2]))
            row_counter = 0

            for num in range(0, len(sample), self.time_steps):
                row = sample[num : num + self.time_steps]
                accumulated_rows[: self.time_steps] = row

                if row_counter >= 0:
                    test_sequence = np.array([accumulated_rows]).astype(np.float32)
                    if transpose:
                        test_sequence = np.transpose(test_sequence, (0, 2, 1))

                    r_buffers = get_relevant_buffer(
                        self.current_buffers,
                        self.kernel_sizes,
                        self.dilations_rates,
                        self.conv_lengths,
                        self.time_steps,
                    )

                    correct_input_data = prepare_input_dictionary(
                        r_buffers, test_sequence, input_name
                    )

                    output = sess.run(None, correct_input_data)

                    self.current_buffers = update_buffers(
                        self.current_buffers,
                        output,
                        test_sequence,
                        self.conv_lengths,
                        self.buffer_index,
                        self.time_steps,
                        self.receptive_field,
                    )

                    if not self.causal and (
                        num < self.receptive_field - self.time_steps + 1
                    ):  # output is not valid yet
                        continue

                    if transpose:
                        output = np.transpose(output[0][-1], (1, 0))
                        decoded_samples_array[
                            idx, row_counter : row_counter + self.time_steps
                        ] = output
                    else:
                        decoded_samples_array[
                            idx, row_counter : row_counter + self.time_steps
                        ] = output[0][-1]
                row_counter += self.time_steps

        if not self.causal:
            """
            find the index of the first output of the network based on the receptive
            field and time steps
            """
            first_output_index = self.receptive_field % self.time_steps

            # shift the decoded samples array to the left by the first output index
            decoded_samples_array = np.roll(
                decoded_samples_array, -first_output_index, axis=1
            )
        return decoded_samples_array

    def run_audio(
        self,
        input_data: np.ndarray,
        frame_length: int,
        stride: int,
        transpose: bool = False,
        dim: int = 2,
    ) -> np.ndarray:
        """
        Runs streaming inference on audio data using the streaming model.

        Args:
            input_data (np.ndarray): The input audio data array.
            L (int): The frame length.
            S (int): The step size.
            transpose (bool): If True, transposes the input sequence.
            dim (int): Dimension for input reshaping.

        Returns:
            np.ndarray: The reconstructed output audio data.
        """

        input_name = read_input_name(self.streaming_model_path)
        bias = get_conv_transpose_bias(self.streaming_model_path)
        sess = ort.InferenceSession(self.streaming_model_path)

        # Compute necessary data length for multiple time steps
        length_mts = frame_length + (self.time_steps - 1) * stride

        number_of_samples = input_data.shape[0]
        number_of_ts = (number_of_samples - frame_length) // stride + 1

        # Pad input data if it is not a multiple of the time steps
        remainder = number_of_ts % self.time_steps
        if remainder != 0:
            pad_width = (self.time_steps - remainder) * stride
            input_data = np.pad(input_data, ((0, pad_width)), mode="constant")

        number_of_samples = input_data.shape[0]
        number_of_ts = (number_of_samples - frame_length) // stride + 1

        decoded_samples_array = np.zeros((2, number_of_ts, frame_length))

        str_output = np.zeros((2, number_of_samples))

        self.zero_buffers()

        row_counter = 0

        reset = False

        counter = 0

        for i in range(0, number_of_ts, self.time_steps):
            row = np.zeros((length_mts))
            row = input_data[i * stride : i * stride + length_mts]

            if row_counter >= 0:
                input_sequence = np.array([row]).astype(np.float32)
                if transpose:
                    input_sequence = np.transpose(input_sequence, (0, 2, 1))

                r_buffers = get_relevant_buffer(
                    self.current_buffers,
                    self.kernel_sizes,
                    self.dilations_rates,
                    self.conv_lengths,
                    self.time_steps,
                )

                input_sequence = np.reshape(
                    input_sequence,
                    (
                        (length_mts,)
                        if dim == 1
                        else (1, length_mts) if dim == 2 else (1, 1, length_mts)
                    ),
                )
                test_sequence = (
                    self.preprocessing(input_sequence)[0]
                    if not self.full_network
                    else input_sequence
                )
                correct_input_data = prepare_input_dictionary(
                    r_buffers, test_sequence, input_name
                )

                output = sess.run(None, correct_input_data)

                if counter + self.time_steps > self.receptive_field:
                    reset = True
                    counter = 0
                else:
                    reset = False
                    counter += self.time_steps

                self.current_buffers = update_buffers(
                    self.current_buffers,
                    output,
                    test_sequence,
                    self.conv_lengths,
                    self.buffer_index,
                    self.time_steps,
                    self.receptive_field,
                    reset,
                )

                if not self.causal and (
                    i < self.receptive_field - 1
                ):  # output is not valid yet
                    continue

                if transpose:
                    decoded_samples_array[0, i] = output[0][-1].flatten()
                else:
                    out_index = row_counter
                    if not self.full_network:
                        output_processed = self.post_processing(
                            input_sequence, output[0], length_mts
                        )
                        output_reshaped = output_processed[0].reshape(-1)
                        output_reshaped_1 = output_processed[1].reshape(-1)
                    else:
                        output_reshaped = (
                            output[0].reshape(-1)
                            if len(output[0].shape) == 3
                            else output[0]
                        )
                        output_reshaped_1 = (
                            output[-1].reshape(-1)
                            if len(output[-1].shape) == 3
                            else output[-1]
                        )

                    str_output[0] = update_output(
                        str_output[0],
                        output_reshaped,
                        out_index,
                        length_mts,
                        stride,
                        bias,
                    )
                    # str_output[1] = update_output(
                    # str_output[1], output_reshaped_1, out_index, length_mts, stride, bias
                    # )
            row_counter += self.time_steps

        return str_output
