from collections import OrderedDict
from copy import deepcopy
from typing import Dict, List, OrderedDict, Tuple
import tqdm
import math
import torch
from torch.utils.data import RandomSampler, BatchSampler
import os
import random
from rich.console import Console
from rich.progress import Progress

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
        output_dir: str,
        num_client: int,
        augment: bool,
        writer: torch.utils.tensorboard.SummaryWriter,
    ):
        super(SCAFFOLDClient, self).__init__(
            backbone,
            dataset,
            batch_size,
            local_epochs,
            local_lr,
            logger,
            gpu,
            output_dir,
            num_client,
            augment,
            writer
        )
        self.c_local: Dict[int, List[torch.Tensor]] = {}
        self.c_diff: List[torch.Tensor] = []
        

    def train(
        self,
        client_id: int,
        model_params: OrderedDict[str, torch.Tensor],
        c_global: List[torch.Tensor],
        round_number: int,
        profiler: torch.profiler.profiler.profile,
        progress_tracker: Progress,
        prev_acc: Tuple[int] = False,
        evaluate=True,
        verbose=True,
        use_valset=True,
    ):
        self.client_id = client_id
        # print(f"Train triggered for client {client_id} in round {round_number}")
        self.set_parameters(model_params)

        transforms = None
        if(self.augment):
            transforms = self.get_transforms(round_number)
        
        self.set_transforms_for_global_dataset(transforms)
        self.get_client_local_dataset(transforms)
        if self.client_id not in self.c_local.keys():
            self.c_diff = c_global
        else:
            self.c_diff = []
            # c_l: List[torch.Tensor]
            # c_g: List[torch.Tensor]
            for c_l, c_g in zip(self.c_local[self.client_id], c_global):
                self.c_diff.append(-c_l + c_g)
        # ds = self.get_client_global_dataset()
        
        _, stats = self._log_while_training(round_number, progress_tracker, evaluate, verbose, use_valset, self.global_dataset["val"], prev_acc, profiler=profiler)(round_number=round_number)
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

            
            torch.save(c_plus, f"{self.output_dir}/control_variates_c{client_id}_r{round_number}.pt")

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

    def _train(self, round_number:int):
        self.model.train()
        batchsize = self.get_batch_size()
        # sampler = torch.utils.data.DataLoader(self.trainset, batch_size = 2500, shuffle=True)
        sampler = BatchSampler(RandomSampler(self.trainset), 2500, drop_last=False)
        loader = torch.utils.data.DataLoader(self.trainset, sampler=sampler)
        total_steps = math.ceil(len(self.trainset)/batchsize)*self.local_epochs
        export_img: List[torch.Tensor] = []
        # with tqdm.tqdm(total=total_steps, position=0, leave=True,  desc=f"Training model {self.client_id}") as pbar:
        for current_epoch in range(self.local_epochs):
            # print(f"Getting data for train of local epoch {current_epoch}")
            for batch_num, idx in enumerate(sampler):
                x,y = self.trainset[idx]
                x, y = (x.to(self.device), y.to(self.device))
                export_index = random.sample(range(0, x.shape[0]), 1)[0]
                export_img.append(x[export_index])
                # print(f"Finished getting data for train of local epoch {current_epoch}")
                # if(len(self.model._parameters) == 0):
                #     print(f"Model has no params for client {self.client_id}")
                logits = self.model(x)
                loss = self.criterion(logits, y)
                if(math.isnan(loss.item())):
                    print(f"Encountered nan loss for client {self.client_id}!")
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                # pbar.update()

            for param, c_d in zip(self.model.parameters(), self.c_diff):
                param.grad += c_d.data
 
        self.writer.add_images(f"client_{self.client_id}", torch.stack(export_img),  round_number)
        
                # if(len(self.model._parameters) == 0):
                #     print(f"Model has no params for client {self.client_id}")
