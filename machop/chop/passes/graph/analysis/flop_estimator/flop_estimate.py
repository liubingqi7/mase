import logging
import math
from typing import Any

from tqdm import tqdm
from .calculator import calc_funcs, calc_modules
from chop.passes.graph.analysis.utils import fetch_attr, load_arg

def flops_statistics_analysis_pass(graph):

    total = 0
    for (i, node) in enumerate(graph.fx_graph.nodes):
        mase_meta = node.meta["mase"].parameters
        mase_op = mase_meta["common"]["mase_op"]
        mase_type = mase_meta["common"]["mase_type"]
        # print(mase_meta["common"]["args"]["data_in_0"]["type"])        
        if mase_type in ["module", "module_related_func"]:
            data_in_0 = mase_meta["common"]["args"]["data_in_0"]["value"]
            data_out_0 = mase_meta["common"]["results"]["data_out_0"]["value"]
            # print(node.meta["mase"].module, data_in_0,data_out_0)
            count = calc_modules.calculate_modules(node.meta["mase"].module, [data_in_0], [data_out_0])
            if mase_meta["common"]["args"]["data_in_0"]["type"] == "float":
                total += count["computations"]

        # if mase_type in ["builtin_func"]:
        #     func = node.target
        #     data_in_0 = mase_meta["common"]["args"]["data_in_0"]["value"]
        #     data_out_0 = mase_meta["common"]["results"]["data_out_0"]["value"]
        #     count = calc_funcs.calculate_funcs(func, [data_in_0], [data_out_0])
        #     if mase_meta["common"]["args"]["data_in_0"]["type"] == "float":
        #         total += count["computations"]

    print(int(total))
    
    return graph,{"total_flops": total}


def bits_statistics_analysis_pass(graph):

    total = 0
    for (i, node) in enumerate(graph.fx_graph.nodes):
        mase_meta = node.meta["mase"].parameters
        mase_op = mase_meta["common"]["mase_op"]
        mase_type = mase_meta["common"]["mase_type"]
        # print(mase_meta["common"]["args"]["data_in_0"]["type"])        
        if mase_type in ["module", "module_related_func"]:
            data_in_0 = mase_meta["common"]["args"]["data_in_0"]["value"]
            data_out_0 = mase_meta["common"]["results"]["data_out_0"]["value"]
            
            precision_in = mase_meta["common"]["args"]["data_in_0"]["precision"][0]
            if mase_op == "linear" or mase_op == "conv1d":    
                precision_weight = mase_meta["common"]["args"]["weight"]["precision"][0]
                precision_bias = mase_meta["common"]["args"]["bias"]["precision"][0]
            # print(node.meta["mase"].module, data_in_0,data_out_0)
            count = calc_modules.calculate_modules(node.meta["mase"].module, [data_in_0], [data_out_0])
            if mase_op == "linear":
                total += count["computations"] * precision_in * precision_weight
            elif mase_op == "conv1d":
                total += count["computations"] * precision_in * precision_weight
        # if mase_type in ["builtin_func"]:
        #     func = node.target
        #     data_in_0 = mase_meta["common"]["args"]["data_in_0"]["value"]
        #     data_out_0 = mase_meta["common"]["results"]["data_out_0"]["value"]
        #     count = calc_funcs.calculate_funcs(func, [data_in_0], [data_out_0])
        #     if mase_meta["common"]["args"]["data_in_0"]["type"] == "float":
        #         total += count["computations"]

    print(int(total))
    
    return graph,{"total_flops": total}
            