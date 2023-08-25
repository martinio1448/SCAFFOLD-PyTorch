import random
from argparse import ArgumentParser, Namespace
from collections import OrderedDict
from typing import OrderedDict, Union

import numpy as np
import torch
from path import Path

PROJECT_DIR = Path(__file__).parent.parent.parent.abspath()
LOG_DIR = PROJECT_DIR / "logs"
TEMP_DIR = PROJECT_DIR / "temp"
DATA_DIR = PROJECT_DIR / "data"


def fix_random_seed(seed: int) -> None:
    torch.cuda.empty_cache()
    torch.random.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


def clone_parameters(
    src: Union[OrderedDict[str, torch.Tensor], torch.nn.Module]
) -> OrderedDict[str, torch.Tensor]:
    if isinstance(src, OrderedDict):
        return OrderedDict(
            {
                name: param.clone().detach().requires_grad_(param.requires_grad)
                for name, param in src.items()
            }
        )
    if isinstance(src, torch.nn.Module):
        return OrderedDict(
            {
                name: param.clone().detach().requires_grad_(param.requires_grad)
                for name, param in src.state_dict(keep_vars=True).items()
            }
        )


def get_args() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument("--global_epochs", type=int, default=100)
    parser.add_argument("--local_epochs", type=int, default=10)
    parser.add_argument("--local_lr", type=float, default=1e-2)
    parser.add_argument("--global_lr", type=float, default=1)
    parser.add_argument("--verbose_gap", type=int, default=20)
    parser.add_argument(
        "--dataset",
        type=str,
        choices=["mnist", "cifar10", "cifar100", "emnist", "fmnist"],
        default="mnist",
    )
    parser.add_argument("--batch_size", type=int, default=-1)
    parser.add_argument("--gpu", type=int, default=6)
    parser.add_argument("--log", type=int, default=0)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--client_num_per_round", type=int, default=2)
    parser.add_argument("--save_period", type=int, default=20)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--colorized", type=bool, default=False)
    parser.add_argument("--augment", type=bool, default=False)
    return parser.parse_args()
