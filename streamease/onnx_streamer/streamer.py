"""
This module defines the StreamingConverter class, which is used to convert a given ONNX model 
into a streaming model by modifying its graph. The class provides methods to adjust input 
and output dimensions, modify graph structures, remove unused constants, and save the 
final streaming model. 
"""

import os
import onnx
import onnxoptimizer
from .convtranspose_fixer import transpose_checker
from .expand_fixer import modify_expand_nodes
from .graph_modifier import modify_graph_for_streaming
from .helper import (
    get_framework_from_producer,
    overwrite_opset,
    remove_unused_constants,
)
from .input_shape_finder import change_input_dims, change_output_dims
from .input_fixer import input_checker
from .remove_unused_chains import (
    find_chains_from_disconnected_nodes,
    remove_chains_from_graph,
    identify_disconnected_nodes,
)
from .shape_fixer import shape_checker


class StreamingConverter:
    """
    A class to convert a given ONNX model into a streaming model by modifying its graph.
    Attributes:
        model (onnx.ModelProto): The original ONNX model to be converted.
        input_shapes (dict[str, List[int]]): Dictionary to store input shapes.
        output_shapes (dict[str, List[int]]): Dictionary to store output shapes.
        streaming_graph (Optional[onnx.GraphProto]): The modified streaming graph.
        streaming_model (Optional[onnx.ModelProto]): The modified streaming model.
        framework (Optional[str]): The framework from which the model was produced.
        time_steps (int): The number of time steps for streaming.
    Methods:
        __init__(model: onnx.ModelProto, time_steps: int = 1) -> None:
            Initializes the StreamingConverter with the given model and time steps.
        remove_unused_initializers(model_path: str, output_path: str) -> None:
            Removes unused initializers from the ONNX model and saves the optimized model.
        copy_all_graph_info() -> None:
            Copies all the original model's information to the streaming graph.
        copy_specific_graph_info() -> None:
            Copies specific information from the original model to the streaming model.
        input_output_dims() -> None:
            Changes the input and output dimensions of the streaming graph.
        run() -> None:
            Executes the conversion process by modifying the graph and removing unused constants.
        save_streaming_onnx(path: str = ".", onnx_filename: str = "streaming_model") -> None:
            Saves the streaming ONNX model to the specified path and removes unused initializers.
        print_info() -> None:
            Prints the input and output names of the new streaming graph.
    """

    def __init__(self, model: onnx.ModelProto, time_steps: int = 1) -> None:
        self.model: onnx.ModelProto = model
        self.input_shapes: dict[str, list[int]] = {}
        self.output_shapes: dict[str, list[int]] = {}
        self.streaming_graph: onnx.GraphProto = None
        self.streaming_model: onnx.ModelProto = None
        self.framework: str = None
        self.time_steps: int = time_steps

    def remove_unused_initializers(self, model_path: str, output_path: str) -> None:
        """
        Removes unused initializers from an ONNX model and saves the optimized model.

        Args:
            model_path (str): The file path to the input ONNX model.
            output_path (str): The file path where the optimized ONNX model will be saved.

        Returns:
            None
        """

        onnx_model: onnx.ModelProto = onnx.load(model_path)
        passes: list[str] = ["eliminate_unused_initializer"]

        optimized_model: onnx.ModelProto = onnxoptimizer.optimize(onnx_model, passes)

        onnx.save(optimized_model, output_path)

    def copy_all_graph_info(self) -> None:
        """
        Copies all the original model's information to the streaming graph.
        This method performs the following actions:
        1. Copies the original model to the streaming model using the `overwrite_opset` function.
        2. Determines the framework from the producer name of the original model
        using the `get_framework_from_producer` function.
        3. Assigns the graph of the streaming model to `self.streaming_graph`.

        Attributes:
            self.model: The original model to be copied.
            self.streaming_model: The model after copying the original model.
            self.framework: The framework determined from the producer name of the original model.
            self.streaming_graph: The graph of the streaming model.
        """

        # at the beginning, copy all the original model's info to the streaming graph
        self.streaming_model: onnx.ModelProto = overwrite_opset(self.model)

        # Get the framework from the producer name
        self.framework: str = get_framework_from_producer(self.model.producer_name)

        self.streaming_graph: onnx.GraphProto = self.streaming_model.graph

    def copy_specific_graph_info(self) -> None:
        """
        Copies specific information from the original model to the streaming model.
        This includes copying the modified graph, relevant model properties, metadata properties,
        and opset imports from the original model to the new streaming model.
        """

        # Initialize a new ONNX model to store the streaming model
        self.streaming_model: onnx.ModelProto = onnx.ModelProto()

        # Copy the modified streaming graph to the new model
        self.streaming_model.graph.CopyFrom(self.streaming_graph)

        # Copy relevant model properties from the original model to the streaming model
        self.streaming_model.ir_version = self.model.ir_version
        self.streaming_model.producer_name = self.model.producer_name
        self.streaming_model.producer_version = self.model.producer_version
        self.streaming_model.domain = self.model.domain
        self.streaming_model.model_version = self.model.model_version
        self.streaming_model.doc_string = self.model.doc_string

        # Copy metadata properties from the original model to the streaming model
        self.streaming_model.metadata_props.extend(self.model.metadata_props)

        # Clear the existing opset imports in the streaming model
        del self.streaming_model.opset_import[:]

        # Copy the opset imports from the original model to the streaming model
        self.streaming_model.opset_import.extend(self.model.opset_import)

    def input_output_dims(self) -> None:
        """
        Adjusts the input and output dimensions of the streaming graph.
        This method modifies the input and output shapes of the streaming graph
        based on the framework and time steps provided. It uses the
        `change_input_dims` and `change_output_dims` functions to perform these
        modifications.

        Attributes:
            input_shapes (list): The shapes of the inputs after modification.
            output_shapes (list): The shapes of the outputs after modification.
            streaming_graph (onnx.GraphProto): The streaming graph with updated dimensions.
            framework (str): The framework being used (e.g., TensorFlow, PyTorch).
            time_steps (int): The number of time steps for the streaming graph.
        """

        # Modify the input shapes of the streaming graph based on the framework and time steps
        # The function change_input_dims returns the modified input shapes and the updated graph
        self.input_shapes, self.streaming_graph = change_input_dims(
            self.streaming_graph, self.time_steps
        )

        # Modify the output shapes of the streaming graph based on the framework and time steps
        # The function change_output_dims returns the modified output shapes and the updated graph
        self.output_shapes, self.streaming_graph = change_output_dims(
            self.streaming_graph
        )

    def run(self) -> None:
        """
        Executes the streaming graph modification process.
        This method performs a series of steps to modify the streaming graph of a model.
        The steps include copying graph information, adjusting input/output dimensions,
        applying custom modifications, checking inputs and shapes, handling ConvTranspose nodes,
        removing unused constants, identifying and removing disconnected node chains, modifying
        Expand nodes, and finalizing the streaming model.
        Steps:

        1. Copy all relevant information from the original model to the streaming graph.
        2. Adjust the input and output dimensions of the streaming graph.
        3. Modify the streaming graph using a custom modifier function.
        4. Perform input checking to ensure there aren't any Pads on the input path.
        5. Perform shape checking to ensure the graph is in the correct shape.
        6. Check if there is more than one ConvTranspose node in the graph.
        7. Remove unused constants, such as those created by slice operations.
        8. Identify and remove disconnected chains of nodes in the graph.
        9. Ensure any remaining unused constants are removed.
        10. Modify the Expand nodes to handle streaming.
        11. Final clean-up to remove any unused constants introduced during modifications.
        12. Copy specific graph information to finalize the streaming model.
        """

        # Step 1: Copy all relevant information from the original model to the streaming graph
        self.copy_all_graph_info()

        # Step 2: Adjust the input and output dimensions of the streaming graph
        self.input_output_dims()

        # Step 3: Modify the streaming graph using a custom modifier function
        self.streaming_graph = modify_graph_for_streaming(
            self.model, self.streaming_graph, self.time_steps
        )

        # Step 4: Perform input checking to ensure there isn't any Pads on the input path
        self.streaming_graph = input_checker(self.model, self.streaming_graph)

        # Step 5: Perform shape checking to ensure the graph is in the correct shape
        # Specifically, check for ConvTranspose nodes and Slice nodes
        self.streaming_graph = shape_checker(self.model, self.streaming_graph)

        # Step 6: check if there is more than one ConvTranspose node in the graph
        # If there is, the paths must be separated because of the streaming affect
        self.streaming_graph = transpose_checker(self.streaming_graph)

        # Step 7: Remove unused constants, such as those created by slice operations
        self.streaming_graph = remove_unused_constants(self.streaming_graph)

        # Step 8: Identify and remove disconnected chains of nodes in the graph
        disconnected_nodes, _, _ = identify_disconnected_nodes(self.streaming_graph)
        chains = find_chains_from_disconnected_nodes(
            self.streaming_graph, disconnected_nodes
        )
        self.streaming_graph = remove_chains_from_graph(self.streaming_graph, chains)

        # Step 9: Ensure any remaining unused constants are removed
        self.streaming_graph = remove_unused_constants(self.streaming_graph)

        # Step 10: Modify the Expand nodes to handle streaming
        self.streaming_graph = modify_expand_nodes(
            self.streaming_graph, self.time_steps
        )

        # Step 11: Final clean-up to remove any unused constants introduced during modifications
        self.streaming_graph = remove_unused_constants(self.streaming_graph)

        # Step 12: Copy specific graph information to finalize the streaming model
        self.copy_specific_graph_info()

    def save_streaming_onnx(
        self, path: str = ".", onnx_filename: str = "streaming_model"
    ) -> None:
        """
        Saves the streaming ONNX model to the specified path with the provided filename.

        This method ensures that the ONNX model is saved with the correct file extension
        and removes unused initializers after saving the model.

        Args:
            path (str): The directory where the ONNX model will be saved.
            onnx_filename (str): The desired name for the ONNX model file.
        """

        # Append the '.onnx' extension if it is not already included in the filename
        if not onnx_filename.endswith(".onnx"):
            onnx_filename += ".onnx"

        # This combines the provided directory path with the filename to create a full file path
        full_path = os.path.join(path, onnx_filename)

        # Save the ONNX model to the specified path
        onnx.save(self.streaming_model, full_path)

        # This function will clean up the model by removing any initializers that are not used
        # and will overwrite the model at the same path with the optimized version
        self.remove_unused_initializers(full_path, full_path)

    def print_info(self) -> None:
        """
        Prints the names of the inputs and outputs of the new streaming graph.

        This method prints the input and output node names from the streaming model's graph.
        It provides a quick overview of the graph's interface after the conversion process.
        """

        # Extract the names of all output nodes in the streaming graph
        output: list[str] = [node.name for node in self.streaming_model.graph.output]

        # Extract the names of all input nodes in the streaming graph
        input_all: list[str] = [node.name for node in self.streaming_model.graph.input]

        # Print the names of all input nodes
        print("Inputs: ", input_all)

        # Print the names of all output nodes
        print("Outputs: ", output)
