"""NNTool graph loading utility for TCN Denoiser."""

from nntool.api import NNGraph

def nntool_load_graph(model_path: str, use_onnx_names: bool) -> NNGraph:
    """
    Load and prepare an NNTool graph from the specified model file.

    Args:
        model_path (str): Path to the saved NNTool model.
        use_onnx_names (bool): Whether to preserve ONNX node naming conventions.

    Returns:
        NNGraph: Loaded and prepared NNTool graph.
    """
    
    model_nn = NNGraph.load_graph(model_path, use_onnx_names=use_onnx_names)
    model_nn.adjust_order()
    model_nn.fusions('scaled_match_group', exclude_match_names=["fuse_op_activation_scale8"])
    # model_nn.draw() # Optional visualization
    return model_nn
