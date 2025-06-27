from typing import Tuple, Optional
import numpy as np
import onnxruntime as ort
from utils.onnx_inference.helper_funcs import (
    create_buffers,
    get_relevant_buffer,
    update_buffers,
    get_conv_transpose_bias,
    update_output,
)
from .helper_funcs import prepare_input_list, extract_outputs_list


class Inference:
    def __init__(
        self,
        streaming_model: ort.InferenceSession,
        streaming_model_path: str,
        time_steps: int = 1,
        receptive_field: int = 15,
        causal: bool = True,
        full_network: bool = True,
        enc_dec_model: bool = False,
        encoder_path: Optional[str] = None,
        decoder_path: Optional[str] = None,
        int8_q: bool = False,
        quantization: bool = False,
        use_onnx_names: bool = False,
    ) -> None:
        """
        Initializes the inference class for a streaming model.

        Args:
            streaming_model (ort.InferenceSession): The ONNX runtime session for
            streaming model inference.
            streaming_model_path (str): Path to the streaming model file.
            time_steps (int): Number of time steps for each streaming inference.
            receptive_field (int): Receptive field of the model in time steps.
            causal (bool): Whether to run inference in causal mode.
            full_network (bool): Whether to load the full network or encoder-decoder separately.
            enc_dec_model (bool): Whether the model has separate encoder-decoder paths.
            encoder_path (str, optional): Path to the encoder model file. Defaults to None.
            decoder_path (str, optional): Path to the decoder model file. Defaults to None.
            int8_q (bool): Use int8 quantization if available.
            quantization (bool): Whether quantization is enabled.
            use_onnx_names (bool): Whether to use ONNX names in the model.
        """
        self.current_buffers = {}
        self.kernel_sizes = {}
        self.dilations_rates = {}
        self.conv_lengths = None
        self.streaming_model = streaming_model
        self.streaming_model_path = streaming_model_path
        self.buffer_index = {}
        self.receptive_field = receptive_field
        self.causal = causal
        self.time_steps = time_steps
        self.full_network = full_network
        self.int8_q = int8_q
        self.quantization = quantization
        self.use_onnx_names = use_onnx_names
        if not full_network:
            if enc_dec_model:
                self.ort_sess_enc = encoder_path
                self.ort_sess_encdec = decoder_path
            else:
                self.load_encoder_decoder(encoder_path, decoder_path)

    def init_buffers(self) -> None:
        """
        Initializes buffers required for model inference.
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
        Resets the buffers to zero, except for buffers labeled as "Expand".
        """
        for key, value in self.current_buffers.items():
            if "Expand" not in key:
                self.current_buffers[key] = np.zeros_like(value)

    def load_encoder_decoder(self, encoder_path: str, decoder_path: str) -> None:
        """
        Loads the encoder and decoder sessions if using separate encoder-decoder models.

        Args:
            encoder_path (str): Path to the encoder model file.
            decoder_path (str): Path to the decoder model file.
        """
        self.ort_sess_enc = ort.InferenceSession(encoder_path)

        self.ort_sess_encdec = ort.InferenceSession(decoder_path)

    def post_processing(
        self, input_data: np.ndarray, main_outputs: np.ndarray, len_input: int
    ) -> np.ndarray:
        """
        Post-processes the outputs of the streaming model using the decoder.

        Args:
            input_data (np.ndarray): Input data fed into the decoder.
            main_outputs (np.ndarray): Main outputs from the encoder.
            len_input (int): Length of the input data.

        Returns:
            np.ndarray: Final model output after decoding.
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

    def preprocessing(self, x: np.ndarray) -> Tuple[np.ndarray, int, np.ndarray]:
        """
        Prepares the input data for model inference.

        Args:
            x (np.ndarray): Raw input data.

        Returns:
            Tuple[np.ndarray, int, np.ndarray]: Preprocessed output, input length,
            and reshaped input data.
        """
        input_data = x
        len_input = input_data.shape[0]
        if len(input_data.shape) == 1:
            input_data = np.expand_dims(np.expand_dims(input_data, axis=0), axis=0)

        # Run the model
        input_name = self.ort_sess_enc.get_inputs()[0].name
        output_result = self.ort_sess_enc.run(None, {input_name: input_data})[0]
        return output_result, len_input, input_data

    def _reshape_output(self, output: np.ndarray) -> np.ndarray:
        """
        Reshapes the given output based on its shape to ensure compatibility
        with the expected input structure.

        Args:
            output (np.ndarray): The output array from the model that needs reshaping.

        Returns:
            np.ndarray: The reshaped output array.
        """
        shape = output.shape
        if len(shape) == 1:
            # Case for shape (128,) - Reshape to (1, 128, 1)
            output = np.reshape(output, (1, shape[0], 1))
        elif len(shape) == 2:
            if shape[1] == self.time_steps:
                # Case for shape (128, 2) - Reshape to (1, 128, timestep)
                output = np.reshape(output, (1, shape[0], self.time_steps))
            else:
                # Case for shape (2, 128) - Reshape to (1, 128, timestep)
                output = np.reshape(output, (1, shape[1], self.time_steps))
        elif len(shape) == 3:
            if shape[-1] == self.time_steps:
                if shape[0] == 1:
                    return output  # Already in the correct shape (1, X, time_steps)
                output = np.reshape(output, (1, shape[0], self.time_steps))
            elif shape[1] == self.time_steps:
                if shape[0] == 1:
                    output = np.reshape(output, (1, shape[2], self.time_steps))
                else:
                    output = np.reshape(output, (1, shape[0], self.time_steps))
            elif shape[0] == self.time_steps:
                if shape[1] == 1:
                    output = np.reshape(output, (1, shape[2], self.time_steps))
                else:
                    output = np.reshape(output, (1, shape[1], self.time_steps))

        return output

    def run(
        self,
        input_data: np.ndarray,
        i_transpose: bool = False,
        o_transpose: bool = False,
    ) -> np.ndarray:
        """
        Runs streaming inference on the input data.

        Args:
            input_data (np.ndarray): The input data for streaming inference.
            i_transpose (bool): If True, transpose the input data.
            o_transpose (bool): If True, transpose the output data.

        Returns:
            np.ndarray: Decoded samples after streaming inference.
        """

        # Check if the number of rows is a multiple of the time steps
        remainder = input_data.shape[1] % self.time_steps
        if remainder != 0:
            pad_width = self.time_steps - remainder
            input_data = np.pad(
                input_data, ((0, 0), (pad_width, 0), (0, 0)), mode="constant"
            )
        decoded_samples_array = (
            np.zeros(
                (
                    input_data.shape[0],
                    input_data.shape[1] - self.receptive_field + self.time_steps,
                    input_data.shape[2],
                )
            )
            if not self.causal
            else np.zeros(input_data.shape)
        )

        for idx, sample in enumerate(input_data):
            accumulated_rows = np.zeros((self.time_steps, input_data.shape[2]))
            self.zero_buffers()
            row_counter = 0

            for num in range(0, len(sample), self.time_steps):
                row = sample[num : num + self.time_steps]
                accumulated_rows[: self.time_steps] = row

                test_sequence = np.array([accumulated_rows]).astype(np.float32)
                test_sequence = (
                    np.transpose(test_sequence, (2, 0, 1))
                    if i_transpose
                    else np.transpose(test_sequence, (0, 2, 1))
                )

                r_buffers = get_relevant_buffer(
                    self.current_buffers,
                    self.kernel_sizes,
                    self.dilations_rates,
                    self.conv_lengths,
                    self.time_steps,
                )

                correct_input_data = prepare_input_list(r_buffers, test_sequence)

                prediction = self.streaming_model.execute(
                    correct_input_data, output_dict=False
                )

                outputs = extract_outputs_list(
                    prediction, self.streaming_model, self.buffer_index, self.use_onnx_names
                )

                if o_transpose:
                    outputs = [
                        np.reshape(out, (1, out.shape[0], out.shape[1]))
                        for out in outputs
                    ]

                self.current_buffers = update_buffers(
                    self.current_buffers,
                    outputs,
                    test_sequence,
                    self.conv_lengths,
                    self.buffer_index,
                    self.time_steps,
                )
                output = outputs[0]

                output = (
                    np.transpose(
                        np.reshape(output, (self.time_steps, input_data.shape[2])),
                        (0, 1),
                    )
                    if len(output.shape) == 3
                    else np.transpose(output, (1, 0))
                )

                if not self.causal and (
                    num < self.receptive_field - self.time_steps
                ):  # output is not valid yet
                    continue

                decoded_samples_array[
                    idx, row_counter : row_counter + self.time_steps
                ] = output

                row_counter += self.time_steps
        if not self.causal:
            # find the index of the first output of the network
            first_output_index = self.receptive_field % self.time_steps
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
        Run audio inference for streaming model on a given input audio data.

        Args:
            input_data (np.ndarray): The input audio data as a NumPy array.
            frame_length (int): The frame length required by the model.
            stride (int): The stride length used for overlapping frames.
            transpose (bool, optional): Whether to transpose the input data for processing.
            Defaults to False.
            dim (int, optional): Dimension for reshaping input data. Defaults to 2.

        Returns:
            np.ndarray: The processed audio output as a NumPy array.
        """

        num_speaker = 1
        number_outputs = 2

        # Load bias from the decoder layer and subtract it from the output
        bias = get_conv_transpose_bias(self.streaming_model_path)

        # Determine the total length of input data required for multiple time steps
        frame_l_mts = frame_length + (self.time_steps - 1) * stride

        number_of_samples = input_data.shape[0]
        number_of_ts = (number_of_samples - frame_length) // stride + 1

        # Check if the number of rows is a multiple of the time steps
        remainder = number_of_ts % self.time_steps
        if remainder != 0:
            pad_width = (self.time_steps - remainder) * stride
            input_data = np.pad(input_data, ((0, pad_width)), mode="constant")

        # Recalculate the number of timesteps after padding
        number_of_samples = input_data.shape[0]
        number_of_timesteps = (number_of_samples - frame_length) // stride + 1

        str_output = np.zeros(
            (number_outputs, number_of_samples)
        )  # for the string output of the model

        self.zero_buffers()
        counter = 0
        row_counter = 0
        reset = False

        for ts in range(0, number_of_timesteps, self.time_steps):

            # Extract the input segment based on the current time step and frame length
            row = input_data[ts * stride : ts * stride + frame_l_mts]

            # Prepare the input sequence as a NumPy array
            input_sequence = np.array([row]).astype(np.float32)

            r_buffers = get_relevant_buffer(
                self.current_buffers,
                self.kernel_sizes,
                self.dilations_rates,
                self.conv_lengths,
                self.time_steps,
            )

            # Reshape input sequence based on specified dimension
            if dim == 1:
                input_sequence = np.reshape(input_sequence, (frame_l_mts))
            elif dim == 2:
                input_sequence = np.reshape(input_sequence, (1, frame_l_mts))
            else:
                input_sequence = np.reshape(input_sequence, (1, 1, frame_l_mts))

            if not self.full_network:
                test_sequence, _, _ = self.preprocessing(input_sequence)
            else:
                test_sequence = input_sequence

            # Apply transpose if necessary for quantization
            if self.quantization:
                if self.int8_q:
                    test_sequence = np.transpose(test_sequence, (0, 2, 1))

            # Prepare input data for the model execution
            correct_input_data = prepare_input_list(r_buffers, test_sequence)

            # Execute the model with quantization if enabled
            if self.quantization:
                prediction = self.streaming_model.execute(
                    correct_input_data,
                    output_dict=False,
                    quantize=True,
                    dequantize=True,
                )
            else:
                prediction = self.streaming_model.execute(
                    correct_input_data, output_dict=False
                )

            # Extract outputs and reshape if necessary
            outputs = extract_outputs_list(
                prediction, self.streaming_model, self.buffer_index, self.use_onnx_names
            )

            if transpose:
                for i in range(num_speaker, len(outputs)):
                    outputs[i] = self._reshape_output(outputs[i])

            # Reset Expand buffers as needed based on receptive field and time steps
            reset = counter + self.time_steps > self.receptive_field
            counter = 0 if reset else counter + self.time_steps

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

            # Process the output for each speaker
            for o_spk in range(num_speaker):
                output = outputs[o_spk]

                if self.full_network:

                    output = np.reshape(output, (frame_l_mts))
                    # print(f"final output is {output}")
                    str_output[o_spk] = update_output(
                        str_output[o_spk], output, ts, frame_l_mts, stride, bias
                    )
                else:
                    output_processed = self.post_processing(
                        input_sequence, output, frame_length
                    )

                    str_output[0] = update_output(
                        str_output[0],
                        output_processed[0].reshape(-1),
                        ts,
                        frame_l_mts,
                        stride,
                        bias,
                    )
                    str_output[1] = update_output(
                        str_output[1],
                        output_processed[1].reshape(-1),
                        ts,
                        frame_l_mts,
                        stride,
                        bias,
                    )
            row_counter += self.time_steps

        return str_output
