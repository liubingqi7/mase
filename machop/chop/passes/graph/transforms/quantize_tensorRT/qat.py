import os
from datetime import datetime as dt
from glob import glob

from copy import copy, deepcopy
import logging

import cv2
import numpy as np
import pytorch_quantization.calib as calib
import pytorch_quantization.nn as qnn
import tensorrt as trt
import torch
import torch.nn.functional as F
from cuda import cudart
from pytorch_quantization import quant_modules
from pytorch_quantization.tensor_quant import QuantDescriptor
from torch.autograd import Variable    
import time        

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

from .utils import create_new_module

QUANTIZEABLE_OP = (
    # "add",
    # "bmm",
    # "conv1d",
    "conv2d",
    # "matmul",
    # "mul",
    "linear",
    # "relu",
    # "sub",
)

logger = logging.getLogger(__name__)

def get_config(config: dict, name: str):
    if name in config:
        return config[name]["config"]
    else:
        return config["default"]["config"]


def graph_fake_quantize_by_type(graph, config: dict):
    for node in graph.fx_graph.nodes:
        if get_mase_op(node) not in QUANTIZEABLE_OP:
            continue
        node_config = get_config(config, get_mase_op(node))
        if node_config["name"] is None:
            continue
        # node_config = parse_node_config(node_config, get_mase_op(node))
        # if get_mase_type(node) == "module":
        if node.op == "call_module":
            ori_module = get_node_actual_target(node)
            new_module = create_new_module(
                get_mase_op(node),
                ori_module,
                node_config,
            )
            parent_name, name = get_parent_name(node.target)
            setattr(graph.modules[parent_name], name, new_module)
            # update precision and type in meta.parameters["common"]
            # update_quant_meta_param(node, node_config, get_mase_op(node))
        # elif get_mase_type(node) in [
        #     "builtin_func",
        #     "module_related_func",
        # ]:
        #     new_f, args, kwargs = create_new_fn(node, node_config)
        #     with graph.fx_graph.inserting_before(node):
        #         new_node = graph.fx_graph.call_function(new_f, args, kwargs)
        #         new_node.name = node.name
        #         new_node.meta["mase"] = copy(node.meta["mase"])
        #         # new_node.meta["mase"].node -> new_node
        #         relink_node_meta(new_node, model=graph.model)
        #         update_quant_meta_param(new_node, node_config, get_mase_op(node))
        #         node.replace_all_uses_with(new_node)
        #     graph.fx_graph.erase_node(node)
    return graph


def graph_fake_quantize_by_name(graph, config: dict):
    quant_modules.initialize()
    for node in graph.fx_graph.nodes:
        if get_mase_op(node) not in QUANTIZEABLE_OP:
            continue
        node_config = get_config(config, node.name)
        # print(node_config)
        if node_config["name"] is None:
            continue
        if node.op == "call_module":
            ori_module = get_node_actual_target(node)
            new_module = create_new_module(
                get_mase_op(node),
                ori_module,
                node_config,
            )
            parent_name, name = get_parent_name(node.target)
            setattr(graph.modules[parent_name], name, new_module)
            # update_quant_meta_param(node, node_config, get_mase_op(node))
            logger.debug(f"Quantized module: {node.target} with config: {node_config}")
        # elif get_mase_type(node) in [
        #     "builtin_func",
        #     "module_related_func",
        # ]:
        #     new_f, args, kwargs = create_new_fn(node, node_config)
        #     with graph.fx_graph.inserting_before(node):
        #         new_node = graph.fx_graph.call_function(new_f, args, kwargs)
        #         new_node.name = node.name
        #         new_node.meta["mase"] = copy(node.meta["mase"])
        #         relink_node_meta(new_node, model=graph.model)
        #         update_quant_meta_param(new_node, node_config, get_mase_op(node))
        #         node.replace_all_uses_with(new_node)
        #     graph.fx_graph.erase_node(node)
        #     logger.debug(
        #         f"Quantized function: {node.target} with config: {node_config}"
        #     )
        else:
            raise ValueError(
                "Unsupported node type for quantisation: {}".format(get_mase_type(node))
            )
    
    quant_modules.deactivate()
    return graph


def fake_quantize_transform_pass(graph, pass_args=None):
    """
    This function applies the fake quantization transform pass to the graph.
    """

    by = pass_args.pop("by")
    match by:
        case "type":
            graph = graph_fake_quantize_by_type(graph, pass_args)
        case "name":
            graph = graph_fake_quantize_by_name(graph, pass_args)
        case _:
            raise ValueError(f'Unsupported quantize "by": {by}')

    # graph.model = torch.fx.GraphModule(graph.model, graph.fx_graph)
    return graph


def export_quantized_to_onnx(graph, dataloader, onnxFile):
    """
    This function exports the fake quantized model to ONNX format.
    """

    device = torch.device('cpu')
    dummy_in, _ = next(iter(dataloader()))
    torch.onnx.export(graph.model.to(device), dummy_in.to(device), onnxFile, export_params=True, opset_version=13, do_constant_folding=True, \
                      input_names = ['input'], output_names = ['output'], dynamic_axes={'input' : {0 : 'batch_size'}, 'output' : {0 : 'batch_size'}})

    return graph


def build_trt_engine_from_onnx(onnxFile, engineFile, dataloader):
    """
    This function builds a TensorRT engine from the ONNX file.
    """
    logger = trt.Logger(trt.Logger.ERROR)
    builder = trt.Builder(logger)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    profile = builder.create_optimization_profile()

    config = builder.create_builder_config()
        
    parser = trt.OnnxParser(network, logger)
    if not os.path.exists(onnxFile):
        print("Failed finding ONNX file!")
        exit()
    print("Succeeded finding ONNX file!")
    with open(onnxFile, "rb") as model:
        if not parser.parse(model.read()):
            print("Failed parsing .onnx file!")
            for error in range(parser.num_errors):
                print(parser.get_error(error))
            exit()
        print("Succeeded parsing .onnx file!")

    inputTensor = network.get_input(0)
    profile.set_shape(inputTensor.name, (1,) + inputTensor.shape[1:], (8,) + inputTensor.shape[1:], (32,) + inputTensor.shape[1:])
    config.add_optimization_profile(profile)

    engineString = builder.build_serialized_network(network, config)
    if engineString == None:
        print("Failed building engine!")
        exit()
    print("Succeeded building engine!")
    with open(engineFile, "wb") as f:
        f.write(engineString)

    engine = trt.Runtime(logger).deserialize_cuda_engine(engineString)

    nIO = engine.num_io_tensors
    lTensorName = [engine.get_tensor_name(i) for i in range(nIO)]
    nInput = [engine.get_tensor_mode(lTensorName[i]) for i in range(nIO)].count(trt.TensorIOMode.INPUT)

    context = engine.create_execution_context()

    dataiter = iter(dataloader())
    input, labels = next(dataiter)
    input_shape = input.shape
    context.set_input_shape(lTensorName[0], input_shape)
    for i in range(nIO):
        print("[%2d]%s->" % (i, "Input " if i < nInput else "Output"), engine.get_tensor_dtype(lTensorName[i]), engine.get_tensor_shape(lTensorName[i]), context.get_tensor_shape(lTensorName[i]), lTensorName[i])

    execute_time = []
    accuracy = []
    for data, label in dataloader():
        bufferH = []
        bufferH.append(np.ascontiguousarray(data))
        for i in range(nInput, nIO):
            bufferH.append(np.empty(context.get_tensor_shape(lTensorName[i]), dtype=trt.nptype(engine.get_tensor_dtype(lTensorName[i]))))
        bufferD = []
        for i in range(nIO):
            bufferD.append(cudart.cudaMalloc(bufferH[i].nbytes)[1])

        for i in range(nInput):
            cudart.cudaMemcpy(bufferD[i], bufferH[i].ctypes.data, bufferH[i].nbytes, cudart.cudaMemcpyKind.cudaMemcpyHostToDevice)

        for i in range(nIO):
            context.set_tensor_address(lTensorName[i], int(bufferD[i]))

        start_time = time.time()
        context.execute_async_v3(0)
        execute_time.append(time.time() - start_time)
    
        for i in range(nInput, nIO):
            cudart.cudaMemcpy(bufferH[i].ctypes.data, bufferD[i], bufferH[i].nbytes, cudart.cudaMemcpyKind.cudaMemcpyDeviceToHost)
            
            categories = np.argmax(bufferH[nInput], axis=1)
            # print(categories, label)
            acc = np.sum(categories == np.array(label)) / len(label)
            # print("Accuracy: %.2f%%" % (acc * 100))
            accuracy.append(acc)
        
        # for i in range(nIO):
        #     print(lTensorName[i])
        #     print(bufferH[i])
        #     print(categories, label)

        for b in bufferD:
            cudart.cudaFree(b)
    print("Succeeded running model in TensorRT!")
    print("Average execute time for one batch: %.2fms" % (sum(execute_time) / len(execute_time) * 1000))
    print("Total accuracy: %.2f%%" % (sum(accuracy) / len(accuracy) * 100))


def evaluate_fake_quantize_pass(graph, pass_args=None):
    """
    This function evaluates the performance of the fake quantized model.
    """

    # quant_modules.initialize()
    # qnn.TensorQuantizer.use_fb_fake_quant = True
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    graph.model.to(device)
    val_loader = pass_args["data_module"].test_dataloader()

    with torch.no_grad():
        acc = 0
        n = 0
        execute_time = []
        for data, target in val_loader:
            data, target = data.to(device), target.to(device)
            start_time = time.time()
            output = graph.model(data)
            execute_time.append(time.time() - start_time)
            acc += (output.argmax(dim=1) == target).sum().item()
            n += data.size(0)
        acc /= n
        print("Average execute time for one batch: %.2fms" % (sum(execute_time) / len(execute_time) * 1000))
        print("Total accuracy: %.2f%%" % (acc * 100))

    return graph


def fake_quantize_to_trt_pass(graph, pass_args=None):
    """
    This function applies the fake quantization to TensorRT pass to the graph.
    """

    onnxFile = pass_args.pop("onnxFile")
    engineFile = pass_args.pop("engineFile")
    dataloader = pass_args.pop("dataloader")

    export_quantized_to_onnx(graph, dataloader, onnxFile)
    build_trt_engine_from_onnx(onnxFile, engineFile, dataloader)

    return graph