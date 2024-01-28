import logging
import math
from typing import Any

from tqdm import tqdm
from .calculator import calc_funcs, calc_modules
from chop.passes.graph.analysis.utils import fetch_attr, load_arg

def flops_statistics_analysis_pass(graph):
    env = {}

    model, fx_graph, modules = graph.model, graph.fx_graph, graph.modules
    for node in tqdm(
        graph.fx_graph.nodes,
        total=len(list(graph.fx_graph.nodes)),
        desc="Profiling weight statistics",
    ):
        if node.op == "call_module":
            args = load_arg(node.args, env)
            kwargs = load_arg(node.kwargs, env)
            result = modules[node.target](*args, **kwargs)
            name = node.target
            count = calc_funcs.calculate_modules(node.meta["mase"].module, node.target, result)

            print(count["computations"])
        env[node.name] = result

    return graph