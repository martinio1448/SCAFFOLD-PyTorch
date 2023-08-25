from torch.utils.data import Dataset

class CudaDataset(Dataset):
    def __init__(self, internal_dataset: Dataset, device):
        inputs, targets = internal_dataset[:]
        self.inputs, self.targets = inputs.to(device), targets.to(device)
        del inputs, targets, internal_dataset
    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, i):
        input = self.inputs[i]
        target = self.targets[i]
        return (input, target)