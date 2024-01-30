import logging
import os
import sys

import torch

sys.path.append(
    os.path.join(
        os.path.dirname(os.path.realpath(__file__)),
        "..",
        "..",
        "..",
        "..",
        "..",
        "..",
        "machop",
    )
)
from chop.dataset import MaseDataModule, get_dataset_info
from chop.tools.logger import set_logging_verbosity

from chop.passes.graph.analysis import (
    report_node_meta_param_analysis_pass,
    profile_statistics_analysis_pass,
)
from chop.passes.graph import (
    add_common_metadata_analysis_pass,
    init_metadata_analysis_pass,
    add_software_metadata_analysis_pass,
)
from chop.tools.get_input import InputGenerator
from chop.ir.graph.mase_graph import MaseGraph

from chop.models import get_model_info, get_model


def compare_quantization_result(ori_graph, graph):
    set_logging_verbosity("info")

    batch_size = 8
    model_name = "jsc-tiny"
    dataset_name = "jsc"


    data_module = MaseDataModule(
        name=dataset_name,
        batch_size=batch_size,
        model_name=model_name,
        num_workers=0,
        # custom_dataset_cache_path="../../chop/dataset"
    )
    data_module.prepare_data()
    data_module.setup()

    for inputs in data_module.train_dataloader():
        xs, ys = inputs
        preds_ori = ori_graph.model(xs)
        preds_new = graph.model(xs)

        print(preds_ori, preds_new)
    
    return graph,{}