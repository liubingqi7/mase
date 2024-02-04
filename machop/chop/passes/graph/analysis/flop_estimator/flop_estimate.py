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
            if  mase_meta["common"]["args"]["data_in_0"]["type"] == "float":
                total += count["computations"]

        if mase_type in ["builtin_func"]:
            print(mase_meta["common"]["args"])

    print(int(total))
    
    return graph,{"total_flops": total}

# def bitwise_ops_analysis_pass(graph):
#     total = 0
#     for (i, node) in enumerate(graph.fx_graph.nodes):
#         mase_meta = node.meta["mase"].parameters
#         mase_op = mase_meta["common"]["mase_op"]
#         mase_type = mase_meta["common"]["mase_type"]
#         if mase_type in ["module", "module_related_func"]:
            