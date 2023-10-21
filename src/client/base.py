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
from data.utils.augmentations.LabelSwap import LabelSwap


from data.utils.augmentations.MultiConcatDataset import MultiConcatDataset

_CURRENT_DIR = Path(__file__).parent.abspath()

import sys

sys.path.append(_CURRENT_DIR.parent)

from data.utils.util import get_cached_datasets, get_dataset
from data.utils.augmentations.AugmentedDataset import AugSet
from data.utils.augmentations.colorcycle import CycleColor
from data.utils.augmentations.CyclicDeform import CyclicDeform
from data.utils.augmentations.ExpandToRGB import ExpandToRGB

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
    def evaluate(self, epoch:int, use_valset=True, dataset: Subset = None, output_tag: str = None, transforms: transforms.Compose = None):
        self.model.eval()
        criterion = torch.nn.CrossEntropyLoss(reduction="sum")
        loss= 0#torch.Tensor = torch.Tensor(0).to(self.device)
        correct = 0#torch.Tensor(0).to(self.device)
        if output_tag is None:
            output_tag = f"client_{self.client_id}"
        if dataset is None:
            if use_valset:
                dataset = self.valset 
            else:
                dataset = self.testset
        if transforms is not None:
            dataset.set_transform(transforms)
        sampler = BatchSampler(SequentialSampler(dataset), 5000, drop_last=False)
        dataloader = DataLoader(dataset, sampler=sampler, batch_size=5000, drop_last=False)
        l = len(sampler)
        export_imgs = []
        # with tqdm.tqdm(total=l, desc=f"Evaluating for client {self.client_id}", leave=True, position=1) as pbar:

        for batch_num,idx in enumerate(sampler):
            x,y = dataset[idx]
            x, y = x.to(self.device), y.to(self.device)
            export_index = random.sample(range(0, x.shape[0]), 1)[0]
            export_imgs.append(x[export_index].to("cpu"))
            # if(len(self.model._parameters) == 0):
            #     print(f"Model has no params for client {self.client_id}")
            logits = self.model(x).to(self.device)
            loss += criterion(logits, y)
            pred = torch.softmax(logits, -1).argmax(-1)#.to(self.device)
            correct += (pred == y).int().sum()
            if(math.isnan(loss.item())):
                print(f"Encountered nan loss for client {self.client_id}!")
            
            del x,y
        
        self.writer.add_images(output_tag, torch.stack(export_imgs), epoch)
        for pic in export_imgs:
            del pic
        del export_imgs
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
        loss, correct = self.evaluate(-1, transforms=self.get_test_transforms(0))
        stats = {"loss": loss, "correct": correct, "size": len(self.testset)}
        return stats

    def set_transforms_for_global_dataset(self, transforms):
        self.global_dataset["train"].set_transform(transforms)
        self.global_dataset["val"].set_transform(transforms)
        self.global_dataset["test"].set_transform(transforms)

    def get_test_transforms(self, epoch):
        data_transforms = transforms.Compose([
            ExpandToRGB(),
            transforms.Resize((100,100), antialias=False)

        ])

        return data_transforms, None
        

    def get_train_transforms(self, epoch):
        # if(self.client_id == 4 or self.client_id == 8) and epoch >= 300:
        #     epoch = epoch + 50
        group = np.ceil((self.client_id+1)/2)-1
        shift = int(group*10*2)
        print(f"Shifting client {self.client_id} to group {group} with shift {shift}")
        data_transforms = transforms.Compose([
            CyclicDeform(epoch=shift, cycle= 100, img_size= (28,28), stretch_intensity=0.2, device=self.device),
            CycleColor(epoch = shift, cycle= 100, background_tolerance=0.1, device=self.device, generation_range=10, style_count=500),
            # transforms.Normalize(mean=(0.3798, 0.3760, 0.3695) , std=(0.0461, 0.0469, 0.0455)),
            transforms.Resize((100,100), antialias=False)
        ])

        label_transforms = transforms.Compose([
            LabelSwap(group=group)
        ])

        return data_transforms, label_transforms
        
    def get_client_local_dataset(self, transforms=None):
        datasets = self.client_datasets[self.client_id]
        self.trainset = AugSet(datasets["train"], transforms)
        self.valset = AugSet(datasets["val"], transforms)
        self.testset = AugSet(datasets["test"], transforms)

    def _log_while_training(self, global_epoch:int, progress_tracker: Progress, evaluate=True, verbose=False, use_valset=True, testset: Subset = None, prev_acc: Tuple[int] = None, profiler: torch.profiler.profiler.profile = None):
        def _log_and_train(*args, **kwargs):
            current_global_epoch = global_epoch
            # print(f"Log and train in epoch {global_epoch}")
            global_testset = testset
            loss_before = 0
            loss_after = 0
            correct_before = 0
            correct_after = 0
            train_loss_after = 0
            train_correct_before = 0
            train_loss_before = 0
            train_loss_after = 0
            if global_testset is None:
                global_testset = self.valset

            num_samples = len(global_testset)
            
            task_progress = progress_tracker.add_task(f"Train client {self.client_id}", total=5)
            

            if evaluate:

                #Test global model on local test data
                train_loss_before, train_correct_before = self.evaluate(dataset=self.testset, epoch=current_global_epoch, output_tag=f"client_{self.client_id}_local_eval_before")
                progress_tracker.advance(task_progress, 1)
                if (prev_acc is None):
                    #test global model on global testset --> Only do this once per round, then pass on the values
                    loss_before, correct_before = self.evaluate(dataset = global_testset, epoch=current_global_epoch, output_tag= f"client_{self.client_id}_global_eval_before")
                else:
                    loss_before, correct_before = prev_acc
                progress_tracker.advance(task_progress, 1)
            else:
                progress_tracker.advance(task_progress, 2)
            
            
            #train model on local data
            res = self._train(*args, task_progress=task_progress, **kwargs)
            progress_tracker.update(task_progress, total=3)

            if evaluate:
                #test local model on local train data
                train_loss_after, train_correct_after= self.evaluate(dataset=self.testset, epoch=current_global_epoch, output_tag=f"client_{self.client_id}_local_eval_after")
                progress_tracker.advance(task_progress, 1)

                #Test local model on global data
                loss_after, correct_after = self.evaluate(dataset=global_testset,epoch=current_global_epoch, output_tag= f"client_{self.client_id}_global_eval_after")
                progress_tracker.advance(task_progress, 1)
            else:
                progress_tracker.advance(task_progress, 2)

            if verbose:
                self.logger.log(
                    "client [{}]   [bold red]test loss: {:.4f} -> {:.4f}    [bold blue]test accuracy: {:.2f}% -> {:.2f}% [bold red]train loss: {:.4f} -> {:.4f}   [bold blue]train accuracy: {:.2f}% -> {:.2f}%".format(
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

            if(profiler is not None):
                # print("taking profiler step!")
                profiler.step()
                # print("Profile is None!")

            self.writer.add_scalars(f"client_{self.client_id}", stats, global_step=current_global_epoch)
            self.writer.flush()
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
