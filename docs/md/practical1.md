### Lab1
['_Node__update_args_kwargs', '__class__', '__delattr__', '__dict__', '__dir__', '__doc__', '__eq__', '__format__', '__ge__', '__getattribute__', '__gt__', '__hash__', '__init__', '__init_subclass__', '__le__', '__lt__', '__module__', '__ne__', '__new__', '__reduce__', '__reduce_ex__', '__repr__', '__setattr__', '__sizeof__', '__str__', '__subclasshook__', '__weakref__', '_args', '_erased', '_input_nodes', '_kwargs', '_next', '_pretty_print_target', '_prev', '_remove_from_list', '_rename', '_repr_fn', 'all_input_nodes', 'append', 'args', 'format_node', 'graph', 'insert_arg', 'is_impure', 'kwargs', 'meta', 'name', 'next', 'normalized_arguments', 'op', 'prepend', 'prev', 'replace_all_uses_with', 'replace_input_with', 'stack_trace', 'target', 'type', 'update_arg', 'update_kwarg', 'users']

{'seq_blocks_1': {'config': {'name': [None]}}, 'seq_blocks_2': {'config': {'name': ['integer'], 'data_in_width': [4, 8], 'data_in_frac_width': [None], 'weight_width': [2, 4, 8], 'weight_frac_width': [None], 'bias_width': [2, 4, 8], 'bias_frac_width': [None]}}, 'seq_blocks_3': {'config': {'name': [None]}}}

{'seq_blocks_1/config/name': [None], 'seq_blocks_2/config/name': ['integer'], 'seq_blocks_2/config/data_in_width': [4, 8], 'seq_blocks_2/config/data_in_frac_width': [None], 'seq_blocks_2/config/weight_width': [2, 4, 8], 'seq_blocks_2/config/weight_frac_width': [None], 'seq_blocks_2/config/bias_width': [2, 4, 8], 'seq_blocks_2/config/bias_frac_width': [None], 'seq_blocks_3/config/name': [None]}

pass args

['__class__', '__deepcopy__', '__delattr__', '__dict__', '__dir__', '__doc__', '__eq__', '__format__', '__ge__', '__getattribute__', '__gt__', '__hash__', '__init__', '__init_subclass__', '__le__', '__lt__', '__module__', '__ne__', '__new__', '__reduce__', '__reduce_ex__', '__repr__', '__setattr__', '__sizeof__', '__str__', '__subclasshook__', '__weakref__', '_co_fields', '_codegen', '_graph_namespace', '_insert', '_len', '_owning_module', '_python_code', '_root', '_target_to_str', '_tracer_cls', '_tracer_extras', '_used_names', 'call_function', 'call_method', 'call_module', 'create_node', 'eliminate_dead_code', 'erase_node', 'get_attr', 'graph_copy', 'inserting_after', 'inserting_before', 'lint', 'node_copy', 'nodes', 'on_generate_code', 'output', 'owning_module', 'placeholder', 'print_tabular', 'process_inputs', 'process_outputs', 'python_code', 'set_codegen']

model attr
['__class__', '__delattr__', '__dict__', '__dir__', '__doc__', '__eq__', '__format__', '__ge__', '__getattribute__', '__gt__', '__hash__', '__init__', '__init_subclass__', '__le__', '__lt__', '__module__', '__ne__', '__new__', '__reduce__', '__reduce_ex__', '__repr__', '__setattr__', '__sizeof__', '__str__', '__subclasshook__', '__weakref__', 'cf_args', 'draw', 'fx_graph', 'implicit_nodes', 'meta', 'model', 'modules', 'nodes', 'tracer']

data_module
['CHECKPOINT_HYPER_PARAMS_KEY', 'CHECKPOINT_HYPER_PARAMS_NAME', 'CHECKPOINT_HYPER_PARAMS_TYPE', '__annotations__', '__class__', '__delattr__', '__dict__', '__dir__', '__doc__', '__eq__', '__format__', '__ge__', '__getattribute__', '__gt__', '__hash__', '__init__', '__init_subclass__', '__jit_unused_properties__', '__le__', '__lt__', '__module__', '__ne__', '__new__', '__reduce__', '__reduce_ex__', '__repr__', '__setattr__', '__sizeof__', '__str__', '__subclasshook__', '__weakref__', '_log_hyperparams', '_set_hparams', '_to_hparams_dict', 'allow_zero_length_dataloader_with_multiple_devices', 'batch_size', 'dataset_info', 'from_datasets', 'hparams', 'hparams_initial', 'load_from_cache_file', 'load_from_checkpoint', 'load_state_dict', 'max_token_len', 'model_name', 'name', 'num_workers', 'on_after_batch_transfer', 'on_before_batch_transfer', 'pred_dataloader', 'pred_dataset', 'predict_dataloader', 'prepare_data', 'prepare_data_per_node', 'save_hyperparameters', 'setup', 'state_dict', 'teardown', 'test_dataloader', 'test_dataset', 'tokenizer', 'train_dataloader', 'train_dataset', 'trainer', 'transfer_batch_to_device', 'val_dataloader', 'val_dataset']

max_epochs = self.config["max_epochs"]
            optimizer=self.config["optimizer"]

            train_dataloader = data_module.train_dataloader()

            match optimizer:
                case "adamw":
                    optimizer = torch.optim.AdamW(
                        model.parameters(), lr=self.config["learning_rate"] 
                    )
                case "adam":
                    optimizer = torch.optim.Adam(
                        model.parameters(), lr=self.config["learning_rate"]
                    )
                case "sgd":
                    optimizer = torch.optim.SGD(
                        model.parameters(), lr=self.config["learning_rate"]
                    )
                case _:
                    raise ValueError(f"Unsupported optimizer: {optimizer}")
            
            for epoch in range(max_epochs):


            
if search_space.__class__.__name__ == "ArchitectureSearchSpace":
            linear_indexes = -1
            for name, length in search_space.choice_lengths_flattened.items():
                if linear_indexes == -1:
                    linear_indexes = trial.suggest_int("linear", 0, length - 1)
                sampled_indexes[name] = linear_indexes
            sampled_config = search_space.flattened_indexes_to_config(sampled_indexes)
            is_eval_mode = self.config.get("eval_mode", False)

'plt_trainer_args': {'max_epochs': 5, 'max_steps': -1, 'devices': 1, 'num_nodes': 1, 'accelerator': 'auto', 'strategy': 'auto', 'fast_dev_run': False, 'precision': '16-mixed', 'accumulate_grad_batches': 1, 'log_every_n_steps': 50}

{'model': JSC_Three_Linear_Layers(
  (seq_blocks): Sequential(
    (0): BatchNorm1d(16, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (1): ReLU(inplace=True)
    (2): Linear(in_features=16, out_features=16, bias=True)
    (3): ReLU(inplace=True)
    (4): Linear(in_features=16, out_features=16, bias=True)
    (5): ReLU(inplace=True)
    (6): Linear(in_features=16, out_features=5, bias=True)
    (7): ReLU(inplace=True)
  )
), 'model_info': MaseModelInfo(name='jsc-three-linear', model_source=<ModelSource.PHYSICAL: 'physical'>, task_type=<ModelTaskType.PHYSICAL: 'physical'>, image_classification=False, physical_data_point_classification=True, sequence_classification=False, seq2seqLM=False, causal_LM=False, is_quantized=False, is_lora=False, is_sparse=False, is_fx_traceable=True), 'data_module': <chop.dataset.MaseDataModule object at 0x7f9f8d6c49d0>, 'dataset_info': MaseDatasetInfo(name='jsc', dataset_source=<DatasetSource.MANUAL: 'manual'>, available_splits=(<DatasetSplit.TRAIN: 'train'>, <DatasetSplit.VALIDATION: 'validation'>, <DatasetSplit.TEST: 'test'>), preprocess_one_split_for_all=True, data_collator_cls=None, image_classification=False, physical_data_point_classification=True, sequence_classification=False, causal_LM=False, seq2seqLM=False, num_classes=5, image_size=None, num_features=16, nerf_config=None), 'task': 'classification', 'optimizer': 'adam', 'learning_rate': 0.001, 'weight_decay': 0, 'plt_trainer_args': {'max_epochs': 5, 'max_steps': -1, 'devices': 1, 'num_nodes': 1, 'accelerator': 'auto', 'strategy': 'auto', 'fast_dev_run': False, 'precision': '16-mixed', 'accumulate_grad_batches': 1, 'log_every_n_steps': 50}, 'auto_requeue': False, 'save_path': '/home/qizhu/Desktop/Work/mase/mase_output/jsc-three-linear_classification_jsc_2024-02-06/software/training_ckpts', 'visualizer': <lightning.pytorch.loggers.tensorboard.TensorBoardLogger object at 0x7fa0ef945b40>, 'load_name': None, 'load_type': 'mz'}

{'model': GraphModule(
  (seq_blocks): Module(
    (0): BatchNorm1d(16, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (1): ReLU()
    (2): Linear(in_features=16, out_features=16, bias=True)
    (3): ReLU()
    (4): Linear(in_features=16, out_features=32, bias=True)
    (5): ReLU()
    (6): Linear(in_features=32, out_features=5, bias=True)
    (7): ReLU()
  )
), 'model_info': MaseModelInfo(name='jsc-three-linear', model_source=<ModelSource.PHYSICAL: 'physical'>, task_type=<ModelTaskType.PHYSICAL: 'physical'>, image_classification=False, physical_data_point_classification=True, sequence_classification=False, seq2seqLM=False, causal_LM=False, is_quantized=False, is_lora=False, is_sparse=False, is_fx_traceable=True), 'data_module': <chop.dataset.MaseDataModule object at 0x7fef1defc5e0>, 'dataset_info': MaseDatasetInfo(name='jsc', dataset_source=<DatasetSource.MANUAL: 'manual'>, available_splits=(<DatasetSplit.TRAIN: 'train'>, <DatasetSplit.VALIDATION: 'validation'>, <DatasetSplit.TEST: 'test'>), preprocess_one_split_for_all=True, data_collator_cls=None, image_classification=False, physical_data_point_classification=True, sequence_classification=False, causal_LM=False, seq2seqLM=False, num_classes=5, image_size=None, num_features=16, nerf_config=None), 'task': 'classification', 'optimizer': 'adam', 'learning_rate': 0.001, 'weight_decay': 0.0, 'plt_trainer_args': {'max_epochs': 3, 'max_steps': 30000, 'devices': 1, 'num_nodes': 1, 'accelerator': 'cuda', 'strategy': 'auto', 'fast_dev_run': False, 'precision': '16-mixed', 'accumulate_grad_batches': 1, 'log_every_n_steps': 10000000}, 'auto_requeue': False, 'save_path': '/home/qizhu/Desktop/Work/mase/mase_output/jsc-three-linear_modified_classification_jsc_2024-02-06/software/training_ckpts', 'visualizer': None, 'load_name': None, 'load_type': 'mz'}

{'model': GraphModule(
  (seq_blocks): Module(
    (0): BatchNorm1d(16, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (1): ReLU()
    (2): Linear(in_features=16, out_features=16, bias=True)
    (3): ReLU()
    (4): Linear(in_features=16, out_features=16, bias=True)
    (5): ReLU()
    (6): Linear(in_features=16, out_features=5, bias=True)
    (7): ReLU()
  )
), 'model_info': MaseModelInfo(name='jsc-three-linear', model_source=<ModelSource.PHYSICAL: 'physical'>, task_type=<ModelTaskType.PHYSICAL: 'physical'>, image_classification=False, physical_data_point_classification=True, sequence_classification=False, seq2seqLM=False, causal_LM=False, is_quantized=False, is_lora=False, is_sparse=False, is_fx_traceable=True), 'data_module': <chop.dataset.MaseDataModule object at 0x7fad9ea2c580>, 'dataset_info': MaseDatasetInfo(name='jsc', dataset_source=<DatasetSource.MANUAL: 'manual'>, available_splits=(<DatasetSplit.TRAIN: 'train'>, <DatasetSplit.VALIDATION: 'validation'>, <DatasetSplit.TEST: 'test'>), preprocess_one_split_for_all=True, data_collator_cls=None, image_classification=False, physical_data_point_classification=True, sequence_classification=False, causal_LM=False, seq2seqLM=False, num_classes=5, image_size=None, num_features=16, nerf_config=None), 'task': 'classification', 'optimizer': 'adam', 'learning_rate': 0.001, 'weight_decay': 0.0, 'plt_trainer_args': {'max_epochs': 3, 'max_steps': 30000, 'devices': 1, 'num_nodes': 1, 'accelerator': 'cuda', 'strategy': 'auto', 'fast_dev_run': False, 'precision': '16-mixed', 'accumulate_grad_batches': 1, 'log_every_n_steps': 10000000}, 'auto_requeue': False, 'save_path': '/home/qizhu/Desktop/Work/mase/mase_output/jsc-three-linear_modified_classification_jsc_2024-02-06/software/training_ckpts', 'visualizer': None, 'load_name': None, 'load_type': 'mz'}