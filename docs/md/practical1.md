## Lab1
####1. What is the impact of varying batch sizes and why?

With the current settings, different batch sizes have a obvious impact on the training results.

| Batch Size | Learning Rate | Max Epochs | Training Loss | Validation Loss | Validation Accuracy |
|:----------:|:----------:|:----------:|:----------:|:----------:|:----------:|
| 32 | 0.001 | 10 | 1.019 | 1.081  | 0.6875 |
| 64 | 0.001 | 10 | 0.8894 | 0.8839  | 0.709 |
| 128 | 0.001 | 10 | 0.8871 | 0.8603   | 0.711 |
| 256 | 0.001 | 10 | 0.8354 |  0.8493  | 0.7132 |
| 512 | 0.001 | 10 | 1.105 | 1.035   | 0.5996 |

According to the table, the larger the batch size, the faster the training process will be. However, the larger the batch size, the more memory will be required to store the training data, which may cause the out-of-memory error. 

In addition, as the batch size increases, the curve of training and validation loss becomes more smooth and less noisy. Thus larger batch size leads to a increase in the performance of the model. But when batch size reach 512, the validation accuracy starts to decrease, which may indicate that the model stuck in a local minimum.


####2. What is the impact of varying maximum epoch number?

Fixing learning rate to 0.00001 and varying maximum epoch number, we can see that the validation accuracy increases as the maximum epoch number increases.

| Max Epochs | Learning Rate | Batch Size | Training Loss | Validation Loss | Validation Accuracy |
|:----------:|:----------:|:----------:|:----------:|:----------:|:----------:|
| 5 | 0.00001 | 256 | 1.422 | 1.413 | 0.4677 |
| 10 | 0.00001 | 256 | 1.24 | 1.33 | 0.513 |
| 20 | 0.00001 | 256 | 1.24 | 1.256 | 0.5354 |
| 30 | 0.00001 | 256 | 1.259 | 1.212 | 0.5471 |
| 50 | 0.00001 | 256 | 1.214 | 1.090 | 0.618 |

According to the table, as maximum epoch number increases, the model can achieve better performance on the validation set.

####3. What is happening with a large learning and what is happening with a small learning rate and why? What is the relationship between learning rates and batch sizes?

Fixing maximum epoch number to 10 and varying learning rate, the result is shown in the following table.
 
| Learning Rate | Max Epochs | Batch Size | Training Loss | Validation Loss | Validation Accuracy |
|:----------:|:----------:|:----------:|:----------:|:----------:|:----------:|
| 0.00001 | 10 | 256 | 1.24 | 1.33 | 0.513 |
| 0.0001 | 10 | 256 | 0.9919 | 0.9922 | 0.6805 |
| 0.001 | 10 | 256 | 0.8354 | 0.8459 | 0.7132 |
| 0.01 | 10 | 256 | 1.064 | 1.174 | 0.5732 |
| 0.1 | 10 | 256 | 1.099 | 1.195 | 0.5669 |
| 1 | 10 | 256 | 1.609 | 1.609 | 0.2022 |

According to the table, as the learning rate increases, the training and validation loss first decrease, and then increase. This is because the small learning rate leads to slow convergence, and the model cannot converge to the optimal solution within 10 epochs. Also, from 0.001 to 1, the result shows that large learing rate leads to a decrease in the model performance.

Theoretically, the larger batch size is, the larger learning rate should be, since larger batch size leads to a decrease in the variance of the gradient. In practice, the combination of learning rate and batch size of 0.001 and 256 may be the best choice.

####4. Implement a network that has in total around 10x more parameters than the toy network.

To implement a network that has in total around 10x more parameters than the toy network, we enlarged the number of hidden layers and nodes in each layer. The new network is shown in the following code:

```python
class JSC_Lab1(nn.Module):
    def __init__(self, info):
        super(JSC_Lab1, self).__init__()
        self.config = info
        self.num_features = self.config.num_features
        self.num_classes = self.config.num_classes
        hidden_layers = [64, 32]
        self.num_neurons = [self.num_features] + hidden_layers + [self.num_classes]
        layer_list = []
        for i in range(1, len(self.num_neurons)):
            in_features = self.num_neurons[i - 1]
            out_features = self.num_neurons[i]
            bn = nn.BatchNorm1d(out_features)
            layer = []
            if i == 1:
                bn_in = nn.BatchNorm1d(in_features)
                in_act = nn.ReLU()
                fc = nn.Linear(in_features, out_features)
                out_act = nn.ReLU()
                layer = [bn_in, in_act, fc, bn, out_act]
            elif i == len(self.num_neurons) - 1:
                fc = nn.Linear(in_features, out_features)
                out_act = nn.ReLU()
                layer = [fc, bn, out_act]
            else:
                fc = nn.Linear(in_features, out_features)
                out_act = nn.ReLU()
                layer = [fc, out_act]
            layer_list = layer_list + layer
        self.module_list = nn.ModuleList(layer_list)

    def forward(self, x):
        x = self.bn1(x)
        x = x.view(x.size(0), 1, -1)
        x = self.conv1(x)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        # x = self.fc2(self.relu(self.bn2(x)))
        x = self.fc3(self.relu(self.bn2(x)))

        return x

```

In this network, we added two more hidden layers with 64 and 32 neurons, respectively. The total number of parameters is around 10x more than the toy network.

## Lab2

####1. Explain the functionality of `report_graph_analysis_pass` and its printed jargons such as `placeholder`, `get_attr` … You might find the doc of torch.fx useful.

The fuctionality of `report_graph_analysis_pass` is to discribe the structure of a graph. Sepecifically, it takes in a mase graph, and report its structure overview and also all the layers. It's printed jargons are:
- Placeholder refers to the number of nodes corresponding to function input
- Get_attr means the number of nodes that retrieve a parameter from module
- Call_function refers to the number of nodes that apply functions like add or sum on some values
- Call_method refers to the number of nodes applying a method of the node like relu()
- Call_module mears the number of nodes representing a call to modules, usually a custom and complicated one,
- Output refers to the number of nodes corresponding to output

The function also report layer types, which shows the mase operation of those node calling a module.

####2. What are the functionalities of `profile_statistics_analysis_pass` and `report_node_meta_param_analysis_pass` respectively?

The `profile_statistics_analysis_pass` is to add statistics profile to a graph. It acquire the weight and bias of target node, and also the statistic of activation nodes such as its input data, then calculate the mean, variance, and standard deviation of the target parameters. The results of this pass are saved in the node's meta-parameters as `['stats']`.

The `report_node_meta_param_analysis_pass` present the statistics of the graph's nodes. It prints the meta-parameters of each node, including the `stats` parameter added by `profile_statistics_analysis_pass`.

####3. Explain why only 1 OP is changed after the `quantize_transform_pass`.

The `quantize_transform_pass` is to transform the target float nodes to a quantized nodes. It firstly replace the float nodes with quantized nodes. In the config `pass_args` of the pass, it only included linear type nodes, and there is only one linear type node in the graph. Therefore, only one node is transformed.

####4. Write some code to traverse both `mg` and `ori_mg`, check and comment on the nodes in these two graphs. You might find the source code for the implementation of `summarize_quantization_analysis_pass` useful.

We could traverse the graph with the following code:

```python
for node in mg.fx_graph.nodes:
    print(node.meta["mase"].parameters["common"]['args'])
```

With this code, we could check the common parameters of each node in the `mg` graph, where `['args']` shows the type and precision of the data and weights of the node.

The difference between `mg` and `ori_mg` is only in the linear type nodes. The output of `mg` is as follows:

```
{'data_in_0': {'shape': [8, 16], 'torch_dtype': torch.float32, 'type': 'float', 'precision': [32], 'value': tensor([[1.5354e+00, 3.2755e-01, 1.6167e+00, 2.5895e+00, 0.0000e+00, 2.3793e+00,
         0.0000e+00, 1.4324e-01, 2.7070e-01, 2.3151e+00, 2.5446e-01, 0.0000e+00,
         1.5589e+00, 1.1185e+00, 8.8014e-03, 2.4488e-01],
        [0.0000e+00, 1.8628e+00, 1.4624e+00, 2.7711e+00, 0.0000e+00, 1.2928e+00,
         0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 1.6329e+00, 2.6245e+00,
         5.0323e-01, 1.6869e+00, 0.0000e+00, 0.0000e+00],
        [1.3121e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00,
         2.0893e-01, 2.8518e+00, 4.2125e-01, 0.0000e+00, 0.0000e+00, 6.4965e-02,
         0.0000e+00, 0.0000e+00, 0.0000e+00, 1.7421e+00],
        [1.7589e+00, 6.4707e-01, 7.8538e-01, 0.0000e+00, 1.8676e+00, 1.2592e+00,
         2.1673e+00, 0.0000e+00, 1.8058e-01, 0.0000e+00, 8.7075e-01, 0.0000e+00,
         1.0469e+00, 0.0000e+00, 8.6397e-01, 1.6284e+00],
        [3.1909e-01, 1.2861e+00, 2.7058e+00, 4.7340e+00, 0.0000e+00, 3.1973e+00,
         0.0000e+00, 0.0000e+00, 9.5881e-01, 7.7429e-01, 1.6506e+00, 9.7668e-01,
         2.0915e+00, 3.0070e+00, 0.0000e+00, 0.0000e+00],
        [0.0000e+00, 3.0906e+00, 2.1444e+00, 5.0514e+00, 0.0000e+00, 1.7380e+00,
         0.0000e+00, 0.0000e+00, 1.6019e+00, 5.4670e+00, 0.0000e+00, 0.0000e+00,
         1.2392e+00, 3.2003e+00, 0.0000e+00, 0.0000e+00],
        [4.3466e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 2.4490e+00, 0.0000e+00,
         2.7520e+00, 2.3833e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00,
         0.0000e+00, 0.0000e+00, 0.0000e+00, 1.0718e+00],
        [2.7307e-03, 0.0000e+00, 0.0000e+00, 0.0000e+00, 1.8134e+00, 0.0000e+00,
         1.8700e+00, 1.6795e+00, 0.0000e+00, 0.0000e+00, 2.7236e+00, 3.4775e+00,
         0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00]],
       grad_fn=<ReluBackward0>)}, 'weight': {'type': 'float', 'precision': [32], 'shape': [5, 16], 'from': None, 'value': Parameter containing:
tensor([[-0.9025,  0.0131, -0.7339,  0.3845,  0.3396, -0.5754,  0.2347,  0.4795,
         -0.2110,  0.4487, -0.0667,  0.7702, -0.7697,  0.4298, -0.0358, -0.4824],
        [-0.4588,  0.3892, -0.8250,  0.5773,  0.0326, -0.7123,  0.4457,  0.5097,
          0.0374,  0.1699,  0.2789, -0.3091, -0.1081, -0.0183, -0.3997,  0.5226],
        [ 0.5685,  0.2991,  0.7887, -2.7264, -1.0735,  0.3732,  0.4806, -0.1174,
          0.2069, -1.5488,  0.6335, -0.7720, -0.0457,  1.9870,  0.8680, -0.2580],
        [ 0.3483, -2.1224,  0.1490,  1.6927,  0.9415,  0.3009, -1.2247, -0.2052,
          0.2834, -0.9039,  0.2664, -0.3870,  0.2838, -2.8530, -0.3925, -0.1363],
        [-0.2659,  0.4980, -0.6214,  0.0284, -0.7932,  0.7361,  1.1090, -1.1837,
         -0.7962,  0.4715, -0.6206,  0.1053,  0.1889,  0.1293, -0.3074, -0.2899]],
       requires_grad=True)}, 'bias': {'type': 'float', 'precision': [32], 'shape': [5], 'from': None, 'value': Parameter containing:
tensor([ 0.1483, -0.1248, -0.1439, -0.1684,  0.1843], requires_grad=True)}}
```

The output of `ori_mg` is as follows:

```
{'data_in_0': {'shape': [8, 16], 'torch_dtype': torch.float32, 'type': 'integer', 'precision': [8, 4], 'value': tensor([[1.5354e+00, 3.2755e-01, 1.6167e+00, 2.5895e+00, 0.0000e+00, 2.3793e+00,
         0.0000e+00, 1.4324e-01, 2.7070e-01, 2.3151e+00, 2.5446e-01, 0.0000e+00,
         1.5589e+00, 1.1185e+00, 8.8014e-03, 2.4488e-01],
        [0.0000e+00, 1.8628e+00, 1.4624e+00, 2.7711e+00, 0.0000e+00, 1.2928e+00,
         0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 1.6329e+00, 2.6245e+00,
         5.0323e-01, 1.6869e+00, 0.0000e+00, 0.0000e+00],
        [1.3121e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00,
         2.0893e-01, 2.8518e+00, 4.2125e-01, 0.0000e+00, 0.0000e+00, 6.4965e-02,
         0.0000e+00, 0.0000e+00, 0.0000e+00, 1.7421e+00],
        [1.7589e+00, 6.4707e-01, 7.8538e-01, 0.0000e+00, 1.8676e+00, 1.2592e+00,
         2.1673e+00, 0.0000e+00, 1.8058e-01, 0.0000e+00, 8.7075e-01, 0.0000e+00,
         1.0469e+00, 0.0000e+00, 8.6397e-01, 1.6284e+00],
        [3.1909e-01, 1.2861e+00, 2.7058e+00, 4.7340e+00, 0.0000e+00, 3.1973e+00,
         0.0000e+00, 0.0000e+00, 9.5881e-01, 7.7429e-01, 1.6506e+00, 9.7668e-01,
         2.0915e+00, 3.0070e+00, 0.0000e+00, 0.0000e+00],
        [0.0000e+00, 3.0906e+00, 2.1444e+00, 5.0514e+00, 0.0000e+00, 1.7380e+00,
         0.0000e+00, 0.0000e+00, 1.6019e+00, 5.4670e+00, 0.0000e+00, 0.0000e+00,
         1.2392e+00, 3.2003e+00, 0.0000e+00, 0.0000e+00],
        [4.3466e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 2.4490e+00, 0.0000e+00,
         2.7520e+00, 2.3833e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00,
         0.0000e+00, 0.0000e+00, 0.0000e+00, 1.0718e+00],
        [2.7307e-03, 0.0000e+00, 0.0000e+00, 0.0000e+00, 1.8134e+00, 0.0000e+00,
         1.8700e+00, 1.6795e+00, 0.0000e+00, 0.0000e+00, 2.7236e+00, 3.4775e+00,
         0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00]],
       grad_fn=<ReluBackward0>)}, 'weight': {'type': 'integer', 'precision': [8, 4], 'shape': [5, 16], 'from': None, 'value': Parameter containing:
tensor([[-0.9025,  0.0131, -0.7339,  0.3845,  0.3396, -0.5754,  0.2347,  0.4795,
         -0.2110,  0.4487, -0.0667,  0.7702, -0.7697,  0.4298, -0.0358, -0.4824],
        [-0.4588,  0.3892, -0.8250,  0.5773,  0.0326, -0.7123,  0.4457,  0.5097,
          0.0374,  0.1699,  0.2789, -0.3091, -0.1081, -0.0183, -0.3997,  0.5226],
        [ 0.5685,  0.2991,  0.7887, -2.7264, -1.0735,  0.3732,  0.4806, -0.1174,
          0.2069, -1.5488,  0.6335, -0.7720, -0.0457,  1.9870,  0.8680, -0.2580],
        [ 0.3483, -2.1224,  0.1490,  1.6927,  0.9415,  0.3009, -1.2247, -0.2052,
          0.2834, -0.9039,  0.2664, -0.3870,  0.2838, -2.8530, -0.3925, -0.1363],
        [-0.2659,  0.4980, -0.6214,  0.0284, -0.7932,  0.7361,  1.1090, -1.1837,
         -0.7962,  0.4715, -0.6206,  0.1053,  0.1889,  0.1293, -0.3074, -0.2899]],
       requires_grad=True)}, 'bias': {'type': 'integer', 'precision': [8, 4], 'shape': [5], 'from': None, 'value': Parameter containing:
tensor([ 0.1483, -0.1248, -0.1439, -0.1684,  0.1843], requires_grad=True)}}
```

Here we could see that the precision and type of the weights, bias and data_in changed from float32 to integer.

####5. Perform the same quantisation flow to the bigger JSC network that you have trained in lab1. You must be aware that now the `pass_args` for your custom network might be different if you have used more than the `Linear layer` in your network.

The bigger JSC network include 2 linear layer and a `conv1d` layer. The `pass_args` for this network should be:

```python
pass_args = {
    "by": "type",
    "default": {"config": {"name": None}},
    "linear": {
        "config": {
            "name": "integer",
            # data
            "data_in_width": 8,
            "data_in_frac_width": 4,
            # weight
            "weight_width": 8,
            "weight_frac_width": 4,
            # bias
            "bias_width": 8,
            "bias_frac_width": 4,
        }
    },
    "conv1d": {
        "config": {
            "name": "integer",
            # data
            "data_in_width": 8,
            "data_in_frac_width": 4,
            # weight
            "weight_width": 8,
            "weight_frac_width": 4,
            # bias
            "bias_width": 8,
            "bias_frac_width": 4,
        }
    
    },
}
```

Then by executing the same quantization code `mg, _ = quantize_transform_pass(mg, pass_args)`, we can get the quantized graph.

####6. Write code to show and verify that the weights of these layers are indeed quantised. You might need to go through the source code of the implementation of the quantisation pass and also the implementation of the Quantized Layers.

We could verify the weights are indeed quantised by comparing the output of both models given the same input. The code for generating input data and running forward is as follows:

```python
inputs, labels = next(iter(data_module.val_dataloader()))
ori_mg.model(inputs)
mg.model(inputs)
```

The output of both models is shown below:

```python
# original model
tensor([[  3.7678,   1.7272, -16.1090, -17.6071,   7.3943],
        [ -1.5366,  -1.3909,   1.0792,   0.7184,  -0.9956],
        [ -0.9890,   0.9364,   0.2796,   0.0480,  -3.5545],
        [ -1.0388,  -0.9393,  -1.7128,   1.2209,  -1.6869],
        [  3.8044,  -0.4426,  -3.2139,  -1.9470,  -1.3612],
        [ -1.0286,   1.6458,  -0.9528,   0.5494,  -4.1782],
        [ -2.3108,  -1.6260,  -4.4452,  -0.1040,   1.9220],
        [ -4.5382,  -0.6942,   1.6112,   1.5397,  -1.3274]],
       grad_fn=<AddmmBackward0>)

# quantized model
tensor([[  3.1523,   0.9688, -13.7188, -19.9180,   6.8203],
        [ -1.5117,  -1.0508,   1.4102,   0.3047,  -0.6367],
        [ -1.0820,   0.1953,  -0.0820,   1.0547,  -3.5664],
        [ -0.8711,  -0.9766,  -2.1328,   1.6289,  -1.9844],
        [  3.4531,  -0.4492,  -2.8906,  -1.7578,  -1.1641],
        [ -0.7891,   1.8906,  -0.8789,   0.2500,  -3.8164],
        [ -2.4922,  -1.4961,  -2.3086,   1.2656,   1.1133],
        [ -3.9922,  -0.2422,   0.7773,   1.6719,  -1.3633]],
       grad_fn=<AddmmBackward0>)
```

We can see that the output of the quantized model is changed slightly.

####7. Load your own pre-trained JSC network, and perform perform the quantisation using the command line interface.

## Optional Task: Write your own pass

#### Many examples of existing passes are in the source code, the test files for these passes also contain useful information on helping you to understand how these passes are used.
#### Implement a pass to count the number of FLOPs (floating-point operations) and BitOPs (bit-wise operations).

To implement a pass to count the number of flops, we could use the mase build in module `/home/qizhu/Desktop/Work/mase/machop/chop/passes/graph/analysis/flop_estimator/calculator/calc_modules.py` and `/home/qizhu/Desktop/Work/mase/machop/chop/passes/graph/analysis/flop_estimator/calculator/calc_funcs.py`

By traversing the graph, we calculate the number of flops for each call_module and call_function node, and add them up to get the total flops for the graph. Code for this pass is as follows:

```python
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
```

To implement a pass to count the number of BitOPs, we could further calculate the number of bitwise operations based on the precision and the nuber of original flop operations. Specifically, when adding two float32 numbers, we need to perform 32 bitwise operations, and when multiplying two float32 numbers, we need to perform $32^2$ bitwise operations approximately. Based on this, we can calculate the number of bitwise operations for each node in the graph following the code below:

```python
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
```

According to the above code, we can count the number of flops and bitops for the JSC-Tiny network in lab1. Before quantization, the flops and bitops are 1320 and 655360. After quantization, the flops and bitops are 680 and 40960.