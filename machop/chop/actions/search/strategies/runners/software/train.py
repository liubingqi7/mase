import math
import torch
from torchmetrics.classification import MulticlassAccuracy
from torchmetrics.text import Perplexity
from torchmetrics import MeanMetric
from transformers import get_scheduler

import torch.distributed as dist
import torch.nn as nn
import torch.optim as optim
import torch.multiprocessing as mp

from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import DataLoader

from tqdm import tqdm

from .base import SWRunnerBase


def get_optimizer(model, optimizer: str, learning_rate, weight_decay=0.0):
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [
                p
                for n, p in model.named_parameters()
                if not any(nd in n for nd in no_decay)
            ],
            "weight_decay": weight_decay,
        },
        {
            "params": [
                p
                for n, p in model.named_parameters()
                if any(nd in n for nd in no_decay)
            ],
            "weight_decay": 0.0,
        },
    ]
    match optimizer:
        case "adamw":
            optimizer = torch.optim.AdamW(
                optimizer_grouped_parameters, lr=learning_rate
            )
        case "adam":
            optimizer = torch.optim.Adam(optimizer_grouped_parameters, lr=learning_rate)
        case "sgd":
            optimizer = torch.optim.SGD(optimizer_grouped_parameters, lr=learning_rate)
        case _:
            raise ValueError(f"Unsupported optimizer: {optimizer}")

    return optimizer


class RunnerBasicTrain(SWRunnerBase):
    available_metrics = ("loss", "accuracy", "perplexity")

    def _post_init_setup(self) -> None:
        self.loss = MeanMetric().to(self.accelerator)
        self._setup_metric()

    def _setup_metric(self):
        if self.model_info.is_vision_model:
            match self.task:
                case "classification" | "cls":
                    self.metric = MulticlassAccuracy(
                        num_classes=self.dataset_info.num_classes
                    ).to(self.accelerator)
                case _:
                    raise ValueError(f"task {self.task} is not supported.")
        elif self.model_info.is_nlp_model:
            match self.task:
                case "classification" | "cls":
                    self.metric = MulticlassAccuracy(
                        num_classes=self.dataset_info.num_classes
                    ).to(self.accelerator)
                case "language_modeling" | "lm":
                    self.metric = Perplexity().to(self.accelerator)
                case _:
                    raise ValueError(f"task {self.task} is not supported.")
        elif self.model_info.physical_data_point_classification:
            match self.task:
                case "classification" | "cls":
                    self.metric = MulticlassAccuracy(
                        num_classes=self.dataset_info.num_classes
                    ).to(self.accelerator)
                case _:
                    raise ValueError(f"task {self.task} is not supported.")
        else:
            raise ValueError(f"model type {self.model_info} is not supported.")

    def nlp_cls_forward(self, batch, model):
        batch.pop("sentence")
        batch = {
            k: v.to(self.accelerator) if isinstance(v, torch.Tensor) else v
            for k, v in batch.items()
        }
        outputs = model(**batch)
        loss = outputs["loss"]
        logits = outputs["logits"]
        labels = batch["labels"]
        labels = labels[0] if len(labels) == 1 else labels.squeeze()
        self.metric(logits, labels)
        self.loss(loss)
        return loss

    def nlp_lm_forward(self, batch, model):
        raise NotImplementedError()

    def vision_cls_forward(self, batch, model):
        raise NotImplementedError()
    
    def physical_cls_forward(self, batch, model):
        pass

    def forward(self, task: str, batch: dict, model):
        if self.model_info.is_vision_model:
            match self.task:
                case "classification" | "cls":
                    loss = self.vision_cls_forward(batch, model)
                case _:
                    raise ValueError(f"task {self.task} is not supported.")
        elif self.model_info.is_nlp_model:
            match self.task:
                case "classification" | "cls":
                    loss = self.nlp_cls_forward(batch, model)
                case "language_modeling" | "lm":
                    loss = self.nlp_lm_forward(batch, model)
                case _:
                    raise ValueError(f"task {self.task} is not supported.")
        else:
            raise ValueError(f"model type {self.model_info} is not supported.")

        return loss

    def compute(self) -> dict[str, float]:
        reduced = {"loss": self.loss.compute().item()}
        if isinstance(self.metric, Perplexity):
            reduced["perplexity"] = self.metric.compute().item()
        elif isinstance(self.metric, MulticlassAccuracy):
            reduced["accuracy"] = self.metric.compute().item()
        else:
            raise ValueError(f"metric {self.metric} is not supported.")
        return reduced

    def __call__(self, data_module, model, sampled_config) -> dict[str, float]:    
        if not data_module.dataset_info.physical_data_point_classification:
            num_samples = self.config["num_samples"]
            max_epochs = self.config["max_epochs"]

            assert not (
                num_samples == -1 and max_epochs == -1
            ), "num_samples and max_epochs cannot be both -1"

            num_batches_per_epoch = len(data_module.train_dataloader())
            num_samples = min(
                max(num_samples, 1),
                num_batches_per_epoch * data_module.batch_size * max_epochs,
            )
            num_batches = math.ceil(num_samples / data_module.batch_size)

            # ddp_sampler = DistributedSampler(data_module.train_dataset)

            train_dataloader = data_module.train_dataloader()
            steps_per_epoch = len(train_dataloader)

            # torch 2.0
            # model = torch.compile(model)

            optimizer = get_optimizer(
                model=model,
                optimizer=self.config["optimizer"],
                learning_rate=self.config["learning_rate"],
                weight_decay=self.config.get("weight_decay", 0.0),
            )
            lr_scheduler = get_scheduler(
                name=self.config["lr_scheduler"],
                optimizer=optimizer,
                num_warmup_steps=self.config["num_warmup_steps"],
                num_training_steps=steps_per_epoch * self.config["max_epochs"],
            )

            grad_accumulation_steps = self.config.get("gradient_accumulation_steps", 1)
            assert grad_accumulation_steps > 0, "num_accumulation_steps must be > 0"

            train_iter = iter(train_dataloader)
            for step_i in range(num_batches):
                if step_i > num_batches:
                    break

                try:
                    batch = next(train_iter)
                except StopIteration:
                    train_iter = iter(train_dataloader)
                    batch = next(train_iter)

                model.train()
                loss_i = self.forward(self.task, batch, model)
                loss_i = loss_i / grad_accumulation_steps
                loss_i.backward()

                if (step_i + 1) % grad_accumulation_steps == 0 or step_i == num_batches - 1:
                    optimizer.step()
                    lr_scheduler.step()
                    optimizer.zero_grad()

        else:
            max_epochs = self.config["max_epochs"]
            optimizer=self.config["optimizer"]

            model.model.to(self.accelerator)
            train_dataloader = data_module.train_dataloader()

            match optimizer:
                case "adamw":
                    optimizer = torch.optim.AdamW(
                        model.parameters(), lr=self.config["learning_rate"] 
                    )
                case "adam":
                    optimizer = torch.optim.Adam(
                        model.model.parameters(), lr=self.config["learning_rate"]
                    )
                case "sgd":
                    optimizer = torch.optim.SGD(
                        model.parameters(), lr=self.config["learning_rate"]
                    )
                case _:
                    raise ValueError(f"Unsupported optimizer: {optimizer}")
            
            for epoch in tqdm(range(max_epochs), desc="Epochs Progress"):
                for data, labels in tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{max_epochs}", leave=False):
                    data, labels = data.to(self.accelerator), labels.to(self.accelerator)
                    optimizer.zero_grad()
                    outputs = model.model(data)
                    loss = nn.CrossEntropyLoss()(outputs, labels)
                    loss.backward()
                    optimizer.step()

            # train_iter = iter(train_dataloader)
            # batch.to(self.accelerator)
            # optimizer.zero_grad()
            # outputs = model(**batch)
            val_dataloader = data_module.val_dataloader()
            data, labels = next(iter(val_dataloader))
            data, labels = data.to(self.accelerator), labels.to(self.accelerator)
            outputs = model.model(data)
            loss = nn.CrossEntropyLoss()(outputs, labels)
            self.loss(loss)
            # self.compute()
                

        return self.compute()
