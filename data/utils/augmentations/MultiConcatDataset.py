import bisect
import warnings
import torch
from typing import (
    Iterable,
    List,
    TypeVar,
)

T_co = TypeVar('T_co', covariant=True)

from torch.utils.data import Dataset, IterableDataset


import bisect
import warnings
from typing import (
    Iterable,
    List,
    TypeVar,
)

T_co = TypeVar('T_co', covariant=True)

from torch.utils.data import Dataset, IterableDataset


class MultiConcatDataset(Dataset[T_co]):
    r"""Dataset as a concatenation of multiple datasets.

    This class is useful to assemble different existing datasets.

    Args:
        datasets (sequence): List of datasets to be concatenated
    """
    datasets: List[Dataset[T_co]]
    cumulative_sizes: List[int]

    @staticmethod
    def cumsum(sequence):
        r, s = [], 0
        for e in sequence:
            l = len(e)
            r.append(l + s)
            s += l
        return r

    def __init__(self, datasets: Iterable[Dataset], device: torch.cuda.Device) -> None:
        super().__init__()
        self.device = device
        self.datasets = list(datasets)
        assert len(self.datasets) > 0, 'datasets should not be an empty iterable'  # type: ignore[arg-type]
        for d in self.datasets:
            assert not isinstance(d, IterableDataset), "ConcatDataset does not support IterableDataset"
        self.cumulative_sizes = self.cumsum(self.datasets)
        self.t_cumulative_sizes = torch.as_tensor(self.cumulative_sizes).to(device)

    def __len__(self):
        return self.cumulative_sizes[-1]

    def __getitem__(self, idx):
        if isinstance(idx, slice):
            start = idx.start
            stop = idx.stop
            step = idx.step
            if start is None:
                start = 0
            if stop is None:
                stop = self.__len__()
            if step is None:
                step = 1

            idx = range(start, stop, step)

        t_idx = torch.as_tensor(idx).to(self.device)
        dataset_idx = torch.searchsorted(self.t_cumulative_sizes, t_idx, side="right")
        sample_idx = t_idx.clone()
        sample_idx[dataset_idx != 0] = t_idx[dataset_idx != 0] - self.t_cumulative_sizes[dataset_idx[dataset_idx != 0] -1]
        d2_indices = torch.stack((dataset_idx, sample_idx, t_idx, torch.arange(0, len(idx)).to(self.device)))

        mapped_sample_indices = []
        targets = []    
        inputs = []
        # print(f"Retrieving samples from datasets with ids {torch.unique(dataset_idx)}")
        for dataset_id in torch.unique(dataset_idx):
            dataset_samples = d2_indices[0]==dataset_id
            sample_indices = d2_indices[:,dataset_samples][1]

            selected_data = self.datasets[dataset_id][sample_indices.tolist()]
            selected_ids = d2_indices[:,dataset_samples]

            mapped_sample_indices.append(selected_ids)      
            inputs.append(selected_data[0])
            targets.append(selected_data[1])

        tensor_targets = torch.concat(targets, dim=0)
        tensor_inputs = torch.concat(inputs, dim=0)
        tensor_indices_mapping = torch.concat(mapped_sample_indices, dim=1)
        test_index = tensor_indices_mapping[3].argsort()
        sorting_index = test_index
        sorted_indices = tensor_indices_mapping[:,sorting_index]

        tensor_targets = tensor_targets[sorting_index]
        tensor_inputs = tensor_inputs[sorting_index]

        return (tensor_inputs, tensor_targets)

    @property
    def cummulative_sizes(self):
        warnings.warn("cummulative_sizes attribute is renamed to "
                      "cumulative_sizes", DeprecationWarning, stacklevel=2)
        return self.cumulative_sizes
