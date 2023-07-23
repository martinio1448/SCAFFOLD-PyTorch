from collections import OrderedDict
from copy import deepcopy
from typing import Dict, List, OrderedDict

CUDA_VISIBLE_DEVICES=[6]

import torch
import os
from rich.console import Console

from .base import ClientBase


class SCAFFOLDClient(ClientBase):
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
        super(SCAFFOLDClient, self).__init__(
            backbone,
            dataset,
            batch_size,
            local_epochs,
            local_lr,
            logger,
            gpu,
        )
        self.c_local: Dict[int, List[torch.Tensor]] = {}
        self.c_diff: List[torch.Tensor] = []

    def train(
        self,
        client_id: int,
        model_params: OrderedDict[str, torch.Tensor],
        c_global: List[torch.Tensor],
        round_number: int,
        evaluate=True,
        verbose=True,
        use_valset=True,
    ):
        self.client_id = client_id
        self.set_parameters(model_params)
        self.get_client_local_dataset()
        if self.client_id not in self.c_local.keys():
            self.c_diff = c_global
        else:
            self.c_diff = []
            # c_l: List[torch.Tensor]
            # c_g: List[torch.Tensor]
            for c_l, c_g in zip(self.c_local[self.client_id], c_global):
                self.c_diff.append(-c_l + c_g)
        _, stats = self._log_while_training(evaluate, verbose, use_valset)()
        # update local control variate
        with torch.no_grad():
            trainable_parameters = filter(
                lambda p: p.requires_grad, model_params.values()
            )

            if self.client_id not in self.c_local.keys():
                self.c_local[self.client_id] = [
                    torch.zeros_like(param, device=self.device)
                    for param in self.model.parameters()
                ]

            y_delta = []
            c_plus = []
            c_delta = []

            # compute y_delta (difference of model before and after training)
            for param_l, param_g in zip(self.model.parameters(), trainable_parameters):
                y_delta.append(param_l - param_g)

            # compute c_plus
            coef = 1 / (self.local_epochs * self.local_lr)
            for c_l, c_g, diff in zip(self.c_local[self.client_id], c_global, y_delta):
                c_plus.append(c_l - c_g - coef * diff)

            path="./data/control_variates"
            if not(os.path.exists(path)):
                os.makedirs(path)

            
            torch.save(c_plus, f"{path}/control_variates_c{client_id}_r{round_number}.pt")

            # compute c_delta
            for c_p, c_l in zip(c_plus, self.c_local[self.client_id]):
                c_delta.append(c_p - c_l)

            self.c_local[self.client_id] = c_plus

        if self.client_id not in self.untrainable_params.keys():
            self.untrainable_params[self.client_id] = {}
        for name, param in self.model.state_dict(keep_vars=True).items():
            if not param.requires_grad:
                self.untrainable_params[self.client_id][name] = param.clone()

        return (y_delta, c_delta), stats

    def _train(self):
        self.model.train()
        for _ in range(self.local_epochs):
            x, y = self.get_data_batch()
            logits = self.model(x)
            loss = self.criterion(logits, y)
            self.optimizer.zero_grad()
            loss.backward()
            for param, c_d in zip(self.model.parameters(), self.c_diff):
                param.grad += c_d.data
            self.optimizer.step()
