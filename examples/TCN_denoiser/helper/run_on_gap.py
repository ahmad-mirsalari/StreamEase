"""Run NNTool model on GAP hardware target with configured execution settings."""
from nntool.api import NNGraph, CompletedProcess
from nntool.api.utils import model_settings


def run_on_gap(model_nn: NNGraph) -> CompletedProcess:
    """
    Execute NNTool model on GAP hardware with predefined memory and performance settings.

    Args:
        model_nn (NNGraph): Loaded NNTool model graph.

    Returns:
        CompletedProcess: Execution result with performance and memory details.
    """

    res = model_nn.execute_on_target(
        at_log=True,
        settings=model_settings(
            tensor_directory="tensors",
            model_directory="model",
            l1_size=128000,
            l2_size=1200000,
            graph_const_exec_from_flash=True,
            l3_flash_device="AT_MEM_L3_MRAMFLASH",
            graph_async_fork=True,
            graph_l1_promotion=2,
            graph_size_opt=2,
        ),
        dont_run=False,
        performance=True,
        print_output=True,
        directory="/tmp/convtasnet",
        at_loglevel=1,
    )
    return res
