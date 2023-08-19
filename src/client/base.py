from collections import OrderedDict
from copy import deepcopy
from typing import Dict, List, OrderedDict, Tuple
import tqdm
import math
import torch
import numpy as np
from path import Path
from rich.console import Console
from rich.progress import Progress
from torch.utils.data import Subset, DataLoader, BatchSampler, SequentialSampler
from torchvision import transforms
from torch.utils.tensorboard import SummaryWriter
import random


from data.utils.augmentations.MultiConcatDataset import MultiConcatDataset

_CURRENT_DIR = Path(__file__).parent.abspath()

import sys

sys.path.append(_CURRENT_DIR.parent)

from data.utils.util import get_cached_datasets, get_dataset
from data.utils.augmentations.AugmentedDataset import AugSet
from data.utils.augmentations.colorcycle import CycleColor
from data.utils.augmentations.CyclicDeform import CyclicDeform

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
        output_dir: str,
        num_clients: int,
        augment: bool,
        writer: SummaryWriter

    ):
        self.writer = writer
        print(f"Initializing client on gpu {gpu}")
        print(f"CUDA is available: {torch.cuda.is_available()}")
        print(f"GPU: {not not torch.cuda.is_available()}")
        print((not gpu is None) and torch.cuda.is_available())
        self.device = torch.device(
            f"cuda:{gpu}" if not gpu is None and torch.cuda.is_available() else "cpu"
        )
        print(f"The following device is selected: {self.device}")
        self.augment : augment
        self.output_dir = output_dir
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
        self.num_clients = num_clients
        self.client_datasets, self.global_dataset = get_cached_datasets(self.dataset, self.device)
        self.augment = augment

    @torch.no_grad()
    def evaluate(self, use_valset=True, dataset: Subset = None, epoch: int = None, output_tag: str = None):
        self.model.eval()
        criterion = torch.nn.CrossEntropyLoss(reduction="sum")
        loss= 0#torch.Tensor = torch.Tensor(0).to(self.device)
        correct = 0#torch.Tensor(0).to(self.device)
        if output_tag is None:
            output_tag = f"client_{self.client_id}"
        if dataset is None:
            if use_valset:
                dataset = self.valset 
                f"client_{self.client_id}"
            else:
                dataset = self.testset
        if dataset.transform is None and self.augment:
            if epoch is None:
                epoch = 0
            dataset.set_transform(self.get_transforms(epoch))
        sampler = BatchSampler(SequentialSampler(dataset), 5000, drop_last=False)
        dataloader = DataLoader(dataset, sampler=sampler, batch_size=5000, drop_last=False)
        l = len(sampler)
        export_imgs = []
        # with tqdm.tqdm(total=l, desc=f"Evaluating for client {self.client_id}", leave=True, position=1) as pbar:

        for batch_num,idx in enumerate(sampler):
            x,y = dataset[idx]
            x, y = x.to(self.device), y.to(self.device)
            export_index = random.sample(range(0, x.shape[0]), 1)[0]
            export_imgs.append(x[export_index])
            # if(len(self.model._parameters) == 0):
            #     print(f"Model has no params for client {self.client_id}")
            logits = self.model(x).to(self.device)
            loss += criterion(logits, y)
            pred = torch.softmax(logits, -1).argmax(-1)#.to(self.device)
            correct += (pred == y).int().sum()
            if(math.isnan(loss.item())):
                print(f"Encountered nan loss for client {self.client_id}!")
            # pbar.update()
        
        self.writer.add_images(output_tag, torch.stack(export_imgs), epoch)

        return loss.item(), correct.item()

    def train(
        self,
        client_id: int,
        model_params: OrderedDict[str, torch.Tensor],
        global_epoch: int,
        profiler: torch.profiler.profiler,
        progress_tracker: Progress,
        evaluate=True,
        verbose=False,
        use_valset=True,
        prev_acc = Tuple[int],
    ) -> Tuple[Tuple[List[torch.Tensor], int], dict[str, int|float]]:
        self.client_id = client_id
        self.set_parameters(model_params)


        self.get_client_local_dataset()

        res, stats = self._log_while_training(evaluate, progress_tracker, verbose, use_valset, self.global_dataset["val"], prev_acc=prev_acc)()
        return res, stats

    def _train(self, round_number: int):
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
        # global_dataset: bool = False
    ):
        self.client_id = client_id
        self.set_parameters(model_params)
        # if global_dataset:
        #     self.get_client_global_dataset()
        # else:
        self.get_client_local_dataset()
        loss, correct = self.evaluate()
        stats = {"loss": loss, "correct": correct, "size": len(self.testset)}
        return stats

    def set_transforms_for_global_dataset(self, transforms):
        self.global_dataset["train"].set_transform(transforms)
        self.global_dataset["val"].set_transform(transforms)
        self.global_dataset["test"].set_transform(transforms)

    def get_client_global_dataset(self):
        all_datasets = [get_dataset(self.dataset, i) for i in range(0, self.num_clients)]
        all_train = MultiConcatDataset([ds["train"] for ds in all_datasets])
        all_test = MultiConcatDataset([ds["test"] for ds in all_datasets])
        all_val = MultiConcatDataset([ds["val"] for ds in all_datasets])

    #     return {"train": AugSet(all_train), "test": AugSet(all_test), "val": AugSet(all_val)}

    def get_transforms(self, epoch):
        data_transforms = transforms.Compose([
            CyclicDeform(epoch=epoch, cycle= 100, img_size= (28,28), stretch_intensity=0.35, device=self.device),
            CycleColor(epoch = epoch, cycle= 100, tolerance=0.1, device=self.device),
            # transforms.ToTensor()
        ])

        return data_transforms
        
    def get_client_local_dataset(self, transforms=None):
        datasets = self.client_datasets[self.client_id]
        self.trainset = AugSet(datasets["train"], transforms)
        self.valset = AugSet(datasets["val"], transforms)
        self.testset = AugSet(datasets["test"], transforms)

    def _log_while_training(self, global_epoch:int, progress_tracker: Progress, evaluate=True, verbose=False, use_valset=True, testset: Subset = None, prev_acc: Tuple[int] = None, profiler: torch.profiler.profiler.profile = None):
        def _log_and_train(*args, **kwargs):
            current_global_epoch = global_epoch
            # print(f"Log and train in epoch {global_epoch}")
            the_set = testset
            loss_before = 0
            loss_after = 0
            correct_before = 0
            correct_after = 0
            train_loss_after = 0
            train_correct_before = 0
            train_loss_before = 0
            train_loss_after = 0
            if the_set is None:
                the_set = self.valset

            num_samples = len(the_set)
            
            task_progress = progress_tracker.add_task(f"Train client {self.client_id}", total=5)
            
            if evaluate:
                train_loss_before, train_loss_before = self.evaluate(use_valset=False)
                progress_tracker.advance(task_progress, 1)
                if (prev_acc is None):
                    loss_before, correct_before = self.evaluate(use_valset, the_set, current_global_epoch, f"client_{self.client_id}_eval")
                else:
                    loss_before, correct_before = prev_acc
                progress_tracker.advance(task_progress, 1)
            else:
                progress_tracker.advance(task_progress, 2)
            

            res = self._train(*args, **kwargs)
            progress_tracker.advance(task_progress, 1)

            if(profiler is not None):
                # print("taking profiler step!")
                profiler.step()
            else:
                print("Profile is None!")


            if evaluate:
                loss_after, correct_after = self.evaluate(use_valset, the_set, current_global_epoch, f"client_{self.client_id}_eval")
                progress_tracker.advance(task_progress, 1)

                train_loss_after, train_correct_after= self.evaluate(use_valset=False)
                progress_tracker.advance(task_progress, 1)
            else:
                progress_tracker.advance(task_progress, 2)

            self.writer

            if verbose:
                self.logger.log(
                    "client [{}]   [bold red]test loss: {:.4f} -> {:.4f}    [bold blue]test accuracy: {:.2f}% -> {:.2f} [bold red]train loss: {:.4f} -> {:.4f}   [bold blue]train accuracy: {:.2f}% -> {:.2f}%".format(
                        self.client_id,
                        loss_before / num_samples,
                        loss_after / num_samples,
                        correct_before / num_samples * 100.0,
                        correct_after / num_samples * 100.0,
                        train_loss_before / len(self.testset),
                        train_loss_after / len(self.testset),
                        train_correct_before / len(self.testset) * 100.0,
                        train_correct_after / len(self.testset) * 100.0,
                    )
                )

            stats = {
                "train_correct_before": train_correct_before,
                "train_acc_before": train_correct_before / len(self.testset),
                "train_loss_before": train_loss_before,
                "train_loss_after": train_loss_after,
                "train_correct_after": train_correct_before,
                "train_acc_after": train_correct_after / len(self.testset),
                "loss_before": loss_before,
                "loss_after": loss_after,
                "correct_after": correct_after,
                "acc_after": correct_after / num_samples,
                "correct": correct_before,
                "size": num_samples,
            }

            self.writer.add_scalars(f"client_{self.client_id}", stats, global_step=current_global_epoch)
            progress_tracker.update(task_progress, visible=False)
            return res, stats

        return _log_and_train

    def set_parameters(self, model_params: OrderedDict):
        self.model.load_state_dict(model_params, strict=False)
        if self.client_id in self.untrainable_params.keys():
            self.model.load_state_dict(
                self.untrainable_params[self.client_id], strict=False
            )

    def get_batch_size(self):
        batch_size = (
            self.batch_size
            if self.batch_size > 0
            else int(len(self.trainset) / self.local_epochs)
        )

        return batch_size

    def get_data_batch(self):
        batch_size = self.get_batch_size()
        sampler = torch.utils.data.DataLoader(self.trainset, batch_size = batch_size, shuffle=True)
        data, targets = next(iter(sampler))
        # indices = torch.from_numpy(
        #     np.random.choice(self.trainset.subset.indices, batch_size)
        # ).long()
        # data, targets = self.trainset[indices]
        return data.to(self.device), targets.to(self.device)
