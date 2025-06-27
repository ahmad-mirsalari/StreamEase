"""Model quantization utilities for NNTool with real-time and streaming configurations."""
import os
import pickle
import re
from nntool.api import NNGraph
from nntool.api.utils import quantization_options
from nntool.api.types import ExpressionFusionNode, ConvFusionNode, Conv2DNode
from .calibration import MyCalibrationDataReader
from .calibration_streaming import MyStreamingCalibrationDataReader


def nntool_quantization(model_nn: NNGraph, use_onnx_names: bool, args) -> NNGraph:
    """
    Quantize an NNTool model with support for int8, float16, bfloat16, and mixed schemes.

    Args:
        model_nn (NNGraph): Loaded NNTool model graph.
        use_onnx_names (bool): Whether to use ONNX naming conventions.
        args: Argument namespace with quantization configuration.

    Returns:
        NNGraph: Quantized NNTool model.
    """

    in_buffers_names = [
        n.name for n in model_nn.input_nodes() if n.name.startswith("buffer")
    ]
    out_buffers_names = [
        n.name
        for n in model_nn.output_nodes()
        if re.match(r"_TCN_TCN_\d+_reg\d+_Add_\d+_output_\d+", n.name)
    ]
    q_type = args.q_type
    stats = None

    if "int8" in q_type:
        print("Quantizing model to int8 — collecting statistics...")
        stats_file = (
            f"stats_r_{args.time_steps}.pickle"
            if args.quantization_mode == "real-time"
            else f"stats_s_{args.time_steps}.pickle"
        )

        if os.path.exists(stats_file):
            print(f"Loading statistics from {stats_file}")
            with open(stats_file, "rb") as file_pointer:
                stats = pickle.load(file_pointer)
        else:
            if args.quantization_mode == "real-time":

                chunk_size = ((args.receptive_field - 1) * args.S + args.L) + (
                    (args.time_steps - 1) * args.S
                )
                print(
                    "Collecting statistics for real-time quantization with chunk size",
                    chunk_size,
                )
                calibration_data_reader = MyCalibrationDataReader(
                    args.cal_data_path, chunk_size=chunk_size, args=args
                )
                stats = model_nn.collect_statistics(
                    calibration_data_reader.get_calibration_data()
                )
            else:

                chunk_size = args.L + ((args.time_steps - 1) * args.S)
                print(
                    "Collecting statistics for streaming quantization with chunk size",
                    chunk_size,
                )
                calibration_data_reader = MyStreamingCalibrationDataReader(
                    args.cal_data_path,
                    model_nn,
                    chunk_size=chunk_size,
                    use_onnx_names=use_onnx_names,
                    args=args,
                )
                stats = model_nn.collect_statistics(
                    calibration_data_reader.get_calibration_data()
                )
            print(f"Saving statistics to {stats_file}")
            with open(stats_file, "wb") as file_pointer:
                pickle.dump(stats, file_pointer, protocol=pickle.HIGHEST_PROTOCOL)

        print(" Stats is collected")

    if q_type == "int8":
        print("Applying int8 quantization...")
        model_nn.quantize(
            statistics=stats,
            graph_options=quantization_options(scheme="scaled", use_ne16=True),
        )

    elif q_type == "bfloat16":
        print("Applying bfloat16 quantization...")
        node_opts = {
            n.name: quantization_options(scheme="FLOAT", float_type="float32")
            for n in model_nn.nodes((ExpressionFusionNode))
        }

        model_nn.quantize(
            graph_options=quantization_options(scheme="FLOAT", float_type="bfloat16"),
            # Select specific nodes and move to different quantization Scheme - TOTAL FLEXIBILITY
            node_options=node_opts,
        )

    elif q_type == "float16":
        print("Applying float16 quantization...")
        model_nn.quantize(
            graph_options=quantization_options(scheme="FLOAT", float_type="float16"),
        )

    elif q_type in ["bf16_f32_m", "f16_f32_m"]:
        float_type = "bfloat16" if "bf16" in q_type else "float16"
        print(
            f"Mixed quantization: {float_type} for conv layers, float32 for the rest."
        )
        node_opts_names = (
            [n.name for n in model_nn.nodes((ConvFusionNode, Conv2DNode))]
            + in_buffers_names
            + out_buffers_names
        )
        node_opts = {
            n: quantization_options(scheme="FLOAT", float_type=float_type)
            for n in node_opts_names
        }

        # Update the first dictionary with the second one
        node_opts.update(
            {
                n.name: quantization_options(scheme="FLOAT", float_type="float32")
                for n in model_nn.nodes((ExpressionFusionNode))
            }
        )
        model_nn.quantize(
            graph_options=quantization_options(scheme="FLOAT", float_type="float32"),
            # Select specific nodes and move to different quantization Scheme - TOTAL FLEXIBILITY
            node_options=node_opts,
        )

    elif q_type in ["int8_f16_m", "int8_bf16_m", "int8_f32_m"]:
        float_type = (
            "bfloat16"
            if "bf16" in q_type
            else ("float16" if "f16" in q_type else "float32")
        )
        print(f"Mixed quantization: int8 for conv layers, {float_type} for the rest.")

        node_opts_names = (
            [n.name for n in model_nn.nodes((ConvFusionNode, Conv2DNode))]
            + in_buffers_names
            + out_buffers_names
        )
        node_opts = {
            n: quantization_options(scheme="scaled", use_ne16=True)
            for n in node_opts_names
        }

        model_nn.quantize(
            statistics=stats,
            graph_options=quantization_options(scheme="FLOAT", float_type=float_type),
            # Select specific nodes and move to different quantization Scheme - TOTAL FLEXIBILITY
            node_options=node_opts,
        )

    return model_nn
