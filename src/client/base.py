from collections import OrderedDict
from copy import deepcopy
from typing import Dict, List, OrderedDict, Tuple

import torch
import numpy as np
from path import Path
from rich.console import Console
from torch.utils.data import Subset, DataLoader

_CURRENT_DIR = Path(__file__).parent.abspath()

import sys

sys.path.append(_CURRENT_DIR.parent)

from data.utils.util import get_dataset


class ClientBase:
    def __init__(
        self,
        backbone: torch.nn.Module,
        dataset: str,
        batch_size: int,
        local_epochs: int,
        local_lr: float,
        logger: Console,
        gpu: int,
    ):
        print(f"Initializing client on gpu {gpu}")
        print(f"CUDA is available: {torch.cuda.is_available()}")
        print(f"GPU: {not not torch.cuda.is_available()}")
        print((not gpu is None) and torch.cuda.is_available())
        self.device = torch.device(
            f"cuda:{gpu}" if not gpu is None and torch.cuda.is_available() else "cpu"
        )
        print(f"The following device is selected: {self.device}")
        self.client_id: int = None
        self.valset: Subset = None
        self.trainset: Subset = None
        self.testset: Subset = None
        self.model: torch.nn.Module = deepcopy(backbone).to(self.device)
        self.optimizer: torch.optim.Optimizer = torch.optim.SGD(
            self.model.parameters(), lr=local_lr
        )
        self.dataset = dataset
        self.batch_size = batch_size
        self.local_epochs = local_epochs
        self.local_lr = local_lr
        self.criterion = torch.nn.CrossEntropyLoss()
        self.logger = logger
        self.untrainable_params: Dict[int, Dict[str, torch.Tensor]] = {}

    @torch.no_grad()
    def evaluate(self, use_valset=True):
        self.model.eval()
        criterion = torch.nn.CrossEntropyLoss(reduction="sum")
        loss= 0#torch.Tensor = torch.Tensor(0).to(self.device)
        correct = 0#torch.Tensor(0).to(self.device)
        dataloader = DataLoader(self.valset if use_valset else self.testset, 32)
        for x, y in dataloader:
            x, y = x.to(self.device), y.to(self.device)
            logits = self.model(x).to(self.device)
            loss += criterion(logits, y)
            pred = torch.softmax(logits, -1).argmax(-1)#.to(self.device)
            correct += (pred == y).int().sum()
        return loss.item(), correct.item()

    def train(
        self,
        client_id: int,
        model_params: OrderedDict[str, torch.Tensor],
        evaluate=True,
        verbose=False,
        use_valset=True,
    ) -> Tuple[Tuple[List[torch.Tensor], int], dict[str, int|float]]:
        self.client_id = client_id
        self.set_parameters(model_params)
        self.get_client_local_dataset()
        res, stats = self._log_while_training(evaluate, verbose, use_valset)()
        return res, stats

    def _train(self):
        self.model.train()
        for _ in range(self.local_epochs):
            x, y = self.get_data_batch()
            logits = self.model(x)
            loss = self.criterion(logits, y)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
        return (
            list(self.model.state_dict(keep_vars=True).values()),
            len(self.trainset.dataset),
        )

    def test(
        self,
        client_id: int,
        model_params: OrderedDict[str, torch.Tensor],
    ):
        self.client_id = client_id
        self.set_parameters(model_params)
        self.get_client_local_dataset()
        loss, correct = self.evaluate()
        stats = {"loss": loss, "correct": correct, "size": len(self.testset)}
        return stats

    def get_client_local_dataset(self):
        datasets = get_dataset(
            self.dataset,
            self.client_id,
        )
        self.trainset = datasets["train"]
        self.valset = datasets["val"]
        self.testset = datasets["test"]

    def _log_while_training(self, evaluate=True, verbose=False, use_valset=True):
        def _log_and_train(*args, **kwargs):
            loss_before = 0
            loss_after = 0
            correct_before = 0
            correct_after = 0
            num_samples = len(self.valset)
            if evaluate:
                loss_before, correct_before = self.evaluate(use_valset)

            res = self._train(*args, **kwargs)

            if evaluate:
                loss_after, correct_after = self.evaluate(use_valset)

            if verbose:
                self.logger.log(
                    "client [{}]   [bold red]loss: {:.4f} -> {:.4f}    [bold blue]accuracy: {:.2f}% -> {:.2f}%".format(
                        self.client_id,
                        loss_before / num_samples,
                        loss_after / num_samples,
                        correct_before / num_samples * 100.0,
                        correct_after / num_samples * 100.0,
                    )
                )

            stats = {
                "correct": correct_before,
                "size": num_samples,
            }
            return res, stats

        return _log_and_train

    def set_parameters(self, model_params: OrderedDict):
        self.model.load_state_dict(model_params, strict=False)
        if self.client_id in self.untrainable_params.keys():
            self.model.load_state_dict(
                self.untrainable_params[self.client_id], strict=False
            )

    def get_data_batch(self):
        batch_size = (
            self.batch_size
            if self.batch_size > 0
            else int(len(self.trainset) / self.local_epochs)
        )
        indices = torch.from_numpy(
            np.random.choice(self.trainset.indices, batch_size)
        ).long()
        data, targets = self.trainset.dataset[indices]
        return data.to(self.device), targets.to(self.device)
