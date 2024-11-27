
"""
A function to verify and modify the shapes of certain nodes in the graph,
particularly ConvTranspose and Slice nodes.
"""

import onnx
from .helper import is_there_slice_node, update_output_name

def shape_checker(
    model: onnx.ModelProto, new_graph: onnx.GraphProto
) -> onnx.GraphProto:
    """
    A function to verify and modify the shapes of certain nodes in the graph,
    particularly ConvTranspose and Slice nodes.
    It removes redundant or unnecessary Slice nodes
    based on their usage and modifies the input/output accordingly.

    Args:
        model (onnx.ModelProto): The original ONNX model.
        new_graph (onnx.GraphProto): The modified ONNX graph.

    Returns:
        onnx.GraphProto: The updated ONNX graph after
        shape verification and Slice node removal.
    """

    # Iterate through the nodes in the model graph
    for i, node in enumerate(model.graph.node):

        # Handle ConvTranspose nodes
        if node.op_type == "ConvTranspose":

            # Check if there is a Slice node related to this ConvTranspose
            target_sequence = [
                "ConvTranspose",
                "Add",
                "Div",
                "Relu",
                "Squeeze",
                "Constant",
            ]

            is_slice_node = is_there_slice_node(i, target_sequence, model.graph)

            # Handle the case if a Slice node is found
            if is_slice_node and is_slice_node.op_type == "Slice":

                # Update inputs of nodes using the output of the removed "Slice" node
                for graph_node in new_graph.node:
                    for idx, input_name in enumerate(graph_node.input):
                        if input_name == is_slice_node.output[0]:
                            if graph_node.op_type == "Slice":
                                # If the current node is another Slice, remove both Slice nodes
                                for inner_node in new_graph.node:
                                    for j, inner_input in enumerate(inner_node.input):
                                        if inner_input == graph_node.output[0]:
                                            inner_node.input[j] = is_slice_node.input[0]
                            else:
                                graph_node.input[idx] = is_slice_node.input[0]
                # The actual removal of the Slice node will be handled later
                continue

        # Handle Slice nodes
        if node.op_type == "Slice":

            remove_slice = True

            # Check if the Slice node is used to generate masks,
            for input_name in node.input:
                if "gen_masks" in input_name:  # or 'decoder' in input_name:
                    remove_slice = False
                    break

            # If the Slice node is not related to mask generation, further analyze its usage
            if remove_slice:
                slice_is_used = False

                # Check if the output of the Slice node is used in other nodes
                for graph_node in new_graph.node:
                    for input_name in graph_node.input:
                        if node.output[0] == input_name:
                            if graph_node.op_type == "Slice":
                                # Skip if the output is used by another Slice node
                                slice_is_used = False
                                break
                            # If the output is used in other nodes, don't remove the Slice node
                            slice_is_used = True
                            break
                if not slice_is_used:
                    # Update graph output if the Slice node's output is part
                    # of the graph's final output
                    for idx, output in enumerate(new_graph.output):
                        if output.name == node.output[0]:
                            new_graph = update_output_name(
                                new_graph, node.input[0], output.name
                            )

                    # Remove the Slice node from the graph
                    new_graph.node.remove(node)
    return new_graph
