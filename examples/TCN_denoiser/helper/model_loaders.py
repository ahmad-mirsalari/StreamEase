"""ONNX model preparation utilities for TCN Denoiser streaming and real-time modes."""

import onnx
import onnxoptimizer
import onnxruntime as ort


def remove_unused_constants(graph: onnx.GraphProto) -> onnx.GraphProto:
    """
    Remove unused constant nodes from the ONNX graph.

    Args:
        graph (onnx.GraphProto): The ONNX graph.

    Returns:
        onnx.GraphProto: Updated graph with unused constants removed.
    """
    constant_nodes = [node for node in graph.node if node.op_type == "Constant"]

    # Find the names of all nodes that use Constant nodes
    used_inputs = set()

    for node in graph.node:
        used_inputs.update(node.input)

    # Remove Constant nodes that are not used by any other nodes
    for const_node in constant_nodes:
        if const_node.output[0] not in used_inputs:
            # Remove the Constant node
            graph.node.remove(const_node)
    return graph


def remove_unused_initializers(model_path: str, output_path: str) -> None:
    """
    Optimize an ONNX model by removing unused initializers.

    Args:
        model_path (str): Path to input ONNX model.
        output_path (str): Path to save optimized ONNX model.
    """

    onnx_model = onnx.load(model_path)
    passes = ["eliminate_unused_initializer"]

    optimized_model = onnxoptimizer.optimize(onnx_model, passes)

    onnx.save(optimized_model, output_path)


def streaming_encoder_decoder(
    args,
) -> tuple[ort.InferenceSession, ort.InferenceSession]:
    """
    Prepare encoder and encoder-decoder ONNX models for streaming mode.

    Args:
        args: Argument namespace with ONNX model paths and parameters.

    Returns:
        Tuple of ONNX InferenceSession for encoder and encoder-decoder.
    """

    # Example usage
    remove_unused_initializers(args.onnx_encoder_part_path, args.onnx_encoder_part_path)

    enc_model = onnx.load(args.onnx_encoder_part_path)
    enc_graph = enc_model.graph

    enc_graph = remove_unused_constants(enc_graph)

    number_samples = args.L + (args.time_steps - 1) * args.S

    for graph_input in enc_graph.input:
        if graph_input.name == enc_graph.input[0].name:
            graph_input.type.tensor_type.shape.dim[2].dim_param = str(number_samples)
    encoder_path_str = "models/streaming-model/enc_convtasnet_stft_session_2_str.onnx"
    onnx.save(enc_model, encoder_path_str)

    num_ts = (number_samples - args.L) // args.S + 1

    remove_unused_initializers(args.onnx_encdec_part_path, args.onnx_encdec_part_path)

    encdec_model = onnx.load(args.onnx_encdec_part_path)
    encdec_graph = encdec_model.graph
    encdec_graph = remove_unused_constants(encdec_graph)

    input_name1 = encdec_graph.input[0].name
    input_name2 = encdec_graph.input[1].name

    for graph_input in encdec_graph.input:
        if graph_input.name == input_name2:
            graph_input.type.tensor_type.shape.dim[2].dim_param = str(number_samples)
        elif graph_input.name == input_name1:
            graph_input.type.tensor_type.shape.dim[2].dim_param = str(num_ts)
    enc_dec_path_str = (
        "models/streaming-model/encdec_convtasnet_stft_session_2_str.onnx"
    )
    onnx.save(encdec_model, enc_dec_path_str)

    srt_encoder_model = load_model_from_onnx(encoder_path_str)
    srt_encdec_model = load_model_from_onnx(enc_dec_path_str)

    return srt_encoder_model, srt_encdec_model


def load_model_from_onnx(model_path: str) -> ort.InferenceSession:
    """
    Load and optimize an ONNX model for runtime inference.

    Args:
        model_path (str): Path to the ONNX model.

    Returns:
        onnxruntime.InferenceSession: Loaded and optimized inference session.
    """

    session_options = ort.SessionOptions()
    session_options.graph_optimization_level = (
        ort.GraphOptimizationLevel.ORT_ENABLE_EXTENDED
    )
    session_options.optimized_model_filepath = model_path.replace(".onnx", "_opt.onnx")

    print(
        f"Optimized model will be saved to: {session_options.optimized_model_filepath}"
    )
    return ort.InferenceSession(model_path, session_options)


def model_loader(
    args,
) -> tuple[ort.InferenceSession, ort.InferenceSession, ort.InferenceSession]:
    """
    Load encoder, main, and encoder-decoder ONNX models based on the execution mode.

    Args:
        args: Argument namespace with ONNX model paths and parameters.

    Returns:
        Tuple of encoder session, main session, encoder-decoder session.
    """

    if args.mode in ("streaming-onnx-main", "streaming-nntool-main"):
        encoder_session, encdec_session = streaming_encoder_decoder(args)
        main_session = None
    else:

        print("Loading the models...")
        print(f"Encoder path: {args.model_encoder_path}")
        print(f"Main path: {args.realtime_main_path}")
        print(f"Encoder decoder path: {args.model_encdec_path}")

        encoder_session = (
            load_model_from_onnx(args.model_encoder_path)
            if args.model_encoder_path
            else None
        )
        main_session = (
            load_model_from_onnx(args.realtime_main_path)
            if args.realtime_main_path
            else None
        )
        encdec_session = (
            load_model_from_onnx(args.model_encdec_path)
            if args.model_encdec_path
            else None
        )

    return encoder_session, main_session, encdec_session
