## Lab3
####1. Explore additional metrics that can serve as quality metrics for the search process. For example, you can consider metrics such as latency, model size, or the number of FLOPs (floating-point operations) involved in the model.

To evaluate a model, we could also consider the following metrics:
- Latency: The time it takes to perform a prediction on a sample.
- Model size: The number of parameters and memory usage.
- FLOPs: The number of floating-point operations required to perform a prediction.
- BitOPs: The number of bitwise operations required to perform a prediction.

####2. Implement some of these additional metrics and attempt to combine them with the accuracy or loss quality metric. It’s important to note that in this particular case, accuracy and loss actually serve as the same quality metric (do you know why?).

Latency implemenation:
```python
import time

xs, ys = inputs
start_time = time.time()
preds = mg.model(xs)
end_time = time.time()
latency = end_time - start_time
```
For FLOPs and BitOPs implementation, we could call the pass defined in Lab2 and return the number of FLOPs and BitOPs.

In this particular case, the loss refers to cross-entropy loss, which represents the difference between the predicted and actual labels. At the same time, accuracy represents the percentage of correct prediction, therefore they are actually evaluating the same thing.

####3. Implement the brute-force search as an additional search method within the system, this would be a new search strategy in MASE.

To implement the brute-force search, we can use the implicit sampler `Optuma.BruteForceSampler` provided by the Optuna library. We could add the sample to the optuna.py file and modify the config of searching strategy to include the brute-force search.

The code added to the optuna.py file:

```python
def sampler_map(self, name):
        match name.lower():
            case "bruteforce":
                sampler = optuna.samplers.BruteForceSampler()
```
The modified config of searching strategy:
```
[search.strategy.setup]
n_jobs = 1
n_trials = 20
timeout = 20000
sampler = "BruteForce"
```

####4. Compare the brute-force search with the TPE based search, in terms of sample efficiency. Comment on the performance difference between the two search methods.

The output of both search setup is shown below:
![Brute-force search output](https://github.com/liubingqi7/mase/blob/main/docs/md/bruteforce_search_quantize.png?raw=true)

TPE based search output:
![TPE based search output](https://github.com/liubingqi7/mase/blob/main/docs/md/tpe_search_quantize.png?raw=true)

Comparing the brute-force search with TPE based search, the b
ruteforce only select 18 trials, which is caused by the restriction of the search space. The TPE goes throught the whole search space with repeated trials. In this case, the brute-force search is faster than the TPE based search due to the small search space, but the TPE based search can find better results in the case that large scale of choices are involved.

<!-- Based on the given code, we can implement the brute-force search as follows:

```python
recorded_accs = []
recorded_time = []
recorded_loss = []
for i, config in enumerate(search_spaces):
    mg, _ = quantize_transform_pass(mg, config)
    j = 0

    # this is the inner loop, where we also call it as a runner.
    acc_avg, loss_avg = 0, 0
    accs, losses = [], []
    times = []
    for inputs in data_module.train_dataloader():
        xs, ys = inputs
        start_time = time.time()
        preds = mg.model(xs)
        end_time = time.time()
        loss = torch.nn.functional.cross_entropy(preds, ys)
        acc = metric(preds, ys)
        accs.append(acc)
        losses.append(loss)
        times.append(end_time - start_time)
        if j > num_batchs:
            break
        j += 1
    time_avg = sum(times) / len(times)
    acc_avg = sum(accs) / len(accs)
    loss_avg = sum(losses) / len(losses)
    recorded_accs.append(acc_avg)
    recorded_time.append(time_avg)
    recorded_loss.append(loss_avg)
    total_metric = acc_avg - loss_avg + time_avg * 0.1
    print(f"Config {i}: acc={acc_avg}, loss={loss_avg}, time={time_avg}, total_metric={total_metric}")  -->
<!-- ``` -->

## Lab4
####1. Can you edit your code, so that we can modify the above network to have layers expanded to double their sizes? Note: you will have to change the `ReLU` also.

Actually, the input size of fully connected layers is determined by the output size of the previous layer, while the relu layer could be ReLU() with regards to any input sizes. Therefore, we can only modify the output size and thus we could change the redine pass and pass_config as follows:

```python
LINEAR_OP = (
    "linear",
    "relu",
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
    last_linear_multi = 1
    for node in graph.fx_graph.nodes:
        i += 1
        # if node name is not matched, it won't be tracked
        config = main_config.get(node.name, default)['config']
        name = config.get("name", None)

        mase_meta = node.meta["mase"].parameters
        mase_op = mase_meta["common"]["mase_op"]
        if mase_op in LINEAR_OP:
            ori_module = graph.modules[node.target]
            if mase_op == "relu":
                new_module = nn.ReLU()
            elif name == "linear":
                in_features = ori_module.in_features * last_linear_multi
                out_features = ori_module.out_features * config["channel_multiplier"]
                bias = ori_module.bias
                last_linear_multi = config["channel_multiplier"]
                new_module = instantiate_linear(in_features, out_features, bias)
            else:
                in_features = ori_module.in_features * last_linear_multi
                out_features = ori_module.out_features
                bias = ori_module.bias
                last_linear_multi = 1
                new_module = instantiate_linear(in_features, out_features, bias)
            parent_name, name = get_parent_name(node.target)
            setattr(graph.modules[parent_name], name, new_module)
            
    return graph, {}
```
The config of the search space becomes:
```
pass_config = {
"by": "name",
"default": {"config": {"name": None}},
"seq_blocks_2": {
    "config": {
        "name": "linear",
        # weight
        "channel_multiplier": 2,
        }
    },
"seq_blocks_4": {
    "config": {
        "name": "linear",
        # "input_channel_multiplier": 2,
        "channel_multiplier": 4,
        }
    },
"seq_blocks_6": {
    "config": {
        "name": "linear",
        "channel_multiplier": 1,
        }
    },
}
```

####2. In lab3, we have implemented a grid search, can we use the grid search to search for the best channel multiplier value?

Task 2, task 3 and task 4 are related to the same question, we can use the grid search to search for the best channel multiplier value. The implementation is shown in task 4.

####3. You may have noticed, one problem with the channel multiplier is that it scales all layers uniformly. Can you then design a search so that it can reach a network that can have this kind of structure?

As discussed in Task 1, the input channel size of the fully connected layers is determined by the output size of the previous layer. Therefore, the code in Task 1 is already designed to have the function of scaling all layers differently.

####4. Integrate the search to the `chop` flow, so we can run it from the command line.

To integrate the search to the `chop` flow, we need to modify the following parts:
- Add the new search space to `machop/chop/actions/search/search_space`.
- Add the new search strategy to `machop/chop/actions/search/strategies`. The original search strategy `basic_train` does not suit the jsc architecture, so we need to create a new training runner for jsc.

Specifically, we need to add or modify the following files:
- `machop/chop/actions/search/search_space/model_arch/graph.py`: add the new search space class, where we modified the `rebuild_model` fuction compared to the quantize graph file. Use the `redefine_linear_transform_pass` function to modify the linear layers.
- `machop/chop/actions/search/strategies/runners/software/train.py`: modified the `__call__` function of the class, add a new training process for jsc.
- `machop/configs/examples/jsc_arch_search.toml`: add a new search space config for jsc-three-linear network.

After this, we could run the search from the command line.

