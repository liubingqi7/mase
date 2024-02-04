from torch import nn
from chop.passes.graph.utils import get_parent_name

from copy import copy, deepcopy
import logging
import torch
from chop.passes.graph.interface.save_and_load import load_mase_graph_interface_pass

from ...utils import (
    deepcopy_mase_graph,
    get_mase_op,
    get_mase_type,
    get_node_actual_target,
    get_parent_name,
    get_similar_node_actual_target,
    match_a_pattern,
    get_node_target_by_name,
)

LINEAR_OP = (
    "linear",
    "relu",
    "output_only",
    "both",
    "input_only",
)

def get_config(config: dict, name: str):
    if name in config:
        return config[name]["config"]
    else:
        return config["default"]["config"]
    
def instantiate_linear(in_features, out_features, bias):
    if bias is not None:
        bias = True
    return nn.Linear(
        in_features=in_features,
        out_features=out_features,
        bias=bias)

def redefine_linear_transform_pass(graph, pass_args=None):
    main_config = pass_args
    default = pass_args.pop('default', None)
    if default is None:
        raise ValueError(f"default value must be provided.")
    i = 0
    father_node = None
    for node in graph.fx_graph.nodes:
        i += 1
        # if node name is not matched, it won't be tracked
        config = main_config.get(node.name, default)['config']
        father_config = main_config.get(father_node.name, default)['config'] if father_node is not None else default['config']
        name = config.get("name", None)

        if name in LINEAR_OP:
            ori_module = graph.modules[node.target]
            if name == "relu":
                new_module = nn.ReLU(father_node.meta["mase"].parameters["common"]["args"]["data_out_0"]["shape"][1] * father_config["channel_multiplier"])
            elif name == "linear":
                # in_features = ori_module.in_features
                out_features = ori_module.out_features
                bias = ori_module.bias

                in_features = father_node.meta["mase"].parameters["common"]["args"]["data_out_0"]["shape"][1] * father_config["channel_multiplier"]
                out_features = out_features * config["channel_multiplier"]
                new_module = instantiate_linear(in_features, out_features, bias)
            parent_name, name = get_parent_name(node.target)
            setattr(graph.modules[parent_name], name, new_module)
            father_node = node
    return graph, {}